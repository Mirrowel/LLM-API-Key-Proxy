# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Best-effort session inference for sticky credential routing.

The tracker intentionally does not try to prove session identity. API clients may
send random conversation IDs, tool calls can be pruned, and compaction can rewrite
the visible context. Instead, it accumulates scoped evidence anchors over time and
only continues sticky routing when the evidence is strong enough.

Two identifiers are kept separate:

- ``session_id`` is the live sticky scope used by sequential credential routing.
- ``affinity_key`` is a deterministic placement hint used for the first pick of a
  new session when the request contains enough stable evidence.

Compression/compaction is tracked as lineage telemetry, not as a hard sticky
continuation. A compacted context is often a genuinely new live context, but the
parent relation is useful for debugging and future policy experiments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils.resilient_io import ResilientStateWriter, safe_read_json

lib_logger = logging.getLogger("rotator_library")


@dataclass(frozen=True)
class SessionAnchor:
    """A single piece of evidence that may identify a live conversation.

    Strength is deliberately coarse. One strong anchor is enough to continue a
    session; medium anchors compound; weak anchors can help with telemetry but do
    not create sticky continuation by themselves.
    """

    value: str
    strength: str = "medium"  # "strong", "medium", or "weak"
    source: str = "generic"
    group: Optional[str] = None


@dataclass
class SessionTrackingHints:
    """Provider-supplied tracking evidence.

    Providers should return evidence, not routing decisions. Core routing still
    owns credential selection. Provider anchors and affinity are automatically
    qualified by provider and optional ``session_scope``; global anchors are
    reserved for proxy-owned/client-global identity such as Responses IDs.
    """

    strong_anchors: List[str] = field(default_factory=list)
    medium_anchors: List[str] = field(default_factory=list)
    weak_anchors: List[str] = field(default_factory=list)
    global_strong_anchors: List[str] = field(default_factory=list)
    global_medium_anchors: List[str] = field(default_factory=list)
    global_weak_anchors: List[str] = field(default_factory=list)
    affinity_key: Optional[str] = None
    # Partitions provider-native anchors only; it never changes logical identity.
    session_scope: Optional[str] = None


@dataclass
class SessionInference:
    """Result of request session inference.

    ``lineage_parent_session_id`` is informational. It is populated when a request
    looks like a compacted descendant of a known session, but the tracker chose not
    to keep sticky routing because compaction mutates the live context too much.
    """

    session_id: Optional[str]
    affinity_key: Optional[str] = None
    confidence: str = "none"  # "strong", "probable", "weak", or "none"
    match_score: int = 0
    possible_compaction: bool = False
    lineage_parent_session_id: Optional[str] = None
    tracking_namespace: Optional[str] = None


@dataclass
class _SessionState:
    session_id: str
    namespace: str
    expires_at: float
    affinity_key: Optional[str] = None
    anchors: set[str] = field(default_factory=set)
    last_seen: float = 0.0
    history_signatures: tuple[str, ...] = field(default_factory=tuple)
    loaded_from_persistence: bool = False


@dataclass
class _AnchorRecord:
    session_id: str
    namespace: str
    strength: str
    source: str
    group: Optional[str]
    expires_at: float
    last_seen: float


@dataclass
class _MatchCandidate:
    session_id: str
    score: int = 0
    strong_matches: int = 0
    medium_matches: int = 0
    weak_matches: int = 0
    medium_groups: set[str] = field(default_factory=set)
    provider_matches: int = 0
    response_matches: int = 0
    response_groups: set[str] = field(default_factory=set)
    request_groups: set[str] = field(default_factory=set)
    matched_probe_groups: set[str] = field(default_factory=set)
    response_probe_groups: set[str] = field(default_factory=set)
    last_seen: float = 0.0

    @property
    def confidence(self) -> str:
        if self.strong_matches > 0:
            return "strong"
        if self.score >= 70 and self.medium_matches >= 2 and self.has_diverse_medium_evidence:
            return "probable"
        if self.score > 0:
            return "weak"
        return "none"

    @property
    def has_diverse_medium_evidence(self) -> bool:
        """Avoid treating one repeated long prompt as a whole conversation."""
        return (
            len(self.medium_groups) >= 2
            or self.provider_matches > 0
            or self.response_matches > 0
        )

    @property
    def is_sticky_match(self) -> bool:
        return self.confidence in {"strong", "probable"}


@dataclass(frozen=True)
class _CompactionDecision:
    """Validated structural replacement of a known parent history."""

    parent_session_id: Optional[str] = None
    marker_compaction: bool = False
    retained_history_ratio: Optional[float] = None
    response_group_count: int = 0
    context_probe_groups: frozenset[str] = frozenset()

    @property
    def possible_compaction(self) -> bool:
        return self.parent_session_id is not None


class SessionTracker:
    """TTL-based session inference with scoped, compounding evidence anchors.

    The implementation favors conservative correctness over perfect continuity:
    it keeps sticky routing when evidence compounds, but it starts a new session
    when the request only has weak/noisy signals. Future expansion can tune the
    scoring constants, add tenant-aware namespaces, or expose lineage events via a
    status endpoint without changing the routing API.
    """

    _STRONG_SCORE = 100
    _MEDIUM_SCORE = 35
    _WEAK_SCORE = 5
    _PERSISTENCE_SCHEMA_VERSION = 3
    _COMPACTION_MAX_RETAINED_HISTORY_RATIO = 0.5
    _MIN_UNMARKED_RESPONSE_GROUPS = 2
    _COMPACTION_PROBE_ROLES = {"user", "system", "developer"}
    _MAX_PERSISTED_HISTORY_SIGNATURES = 4096
    _MAX_PERSISTED_FILE_BYTES = 16 * 1024 * 1024
    _MAX_PERSISTED_SESSIONS = 10000
    _MAX_PERSISTED_STRING_LENGTH = 1024
    _PERSISTED_ANCHOR_SOURCES = {
        "compaction_context",
        "compaction_replay",
        "explicit",
        "first_user",
        "global_hint",
        "message",
        "provider",
        "response",
        "tool",
        "tool_event",
        "window",
    }

    def __init__(
        self,
        ttl_seconds: int = 3600,
        persist_to_disk: bool = False,
        persistence_path: Optional[Path] = None,
        persistence_flush_interval_seconds: float = 5.0,
        max_anchor_records: int = 10000,
        max_anchors_per_session: int = 256,
        trusted_explicit_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.ttl_seconds = max(1, ttl_seconds)
        self.persist_to_disk = persist_to_disk
        self.persistence_path = persistence_path
        self.persistence_flush_interval_seconds = max(0.0, persistence_flush_interval_seconds)
        self.max_anchor_records = max(100, max_anchor_records)
        self.max_anchors_per_session = max(16, max_anchors_per_session)
        if trusted_explicit_fields is None:
            trusted_explicit_fields = self._trusted_fields_from_env()
        self.trusted_explicit_fields = {field for field in trusted_explicit_fields if field}
        self._anchors: Dict[str, _AnchorRecord] = {}
        self._sessions: Dict[str, _SessionState] = {}
        self._dirty = False
        self._dirty_generation = 0
        self._last_persisted_generation = 0
        self._last_save_attempt = 0.0
        self._writer: Optional[ResilientStateWriter] = None
        self._lock = threading.RLock()
        self._save_io_lock = threading.Lock()
        if self.persist_to_disk:
            self._load()
            if self.persistence_path:
                self._writer = ResilientStateWriter(
                    self.persistence_path,
                    lib_logger,
                    serializer=lambda data: json.dumps(data, indent=2, sort_keys=True),
                )

    def infer_session_id(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Compatibility wrapper for older callers/tests."""
        return self.infer_session(request_data).session_id

    def infer_session(
        self,
        request_data: Dict[str, Any],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        scope_key: Optional[str] = None,
        hints: Optional[Any] = None,
        _trusted_isolation_key: bool = False,
    ) -> SessionInference:
        """Infer live session and deterministic affinity from a request payload."""
        with self._lock:
            result = self._infer_session_locked(
                request_data,
                provider=provider,
                model=model,
                scope_key=scope_key,
                hints=hints,
                trusted_isolation_key=_trusted_isolation_key,
            )
            save_job = self._prepare_save_locked()
        self._write_save_job(save_job)
        return result

    def _infer_session_locked(
        self,
        request_data: Dict[str, Any],
        *,
        provider: Optional[str],
        model: Optional[str],
        scope_key: Optional[str],
        hints: Optional[Any],
        trusted_isolation_key: bool,
    ) -> SessionInference:
        now = time.time()
        self._prune(now)

        hints = self._coerce_hints(hints)
        namespace = self._namespace(
            provider,
            scope_key=scope_key,
            trusted_isolation_key=trusted_isolation_key,
        )
        history_signatures = self._request_history_signatures(request_data)
        probe_indexes = self._compaction_probe_indexes(request_data)
        compaction_probe_anchors = self._build_compaction_probe_anchors(
            request_data,
            namespace,
            probe_indexes=probe_indexes,
        )
        unsuppressed_normal_anchors = self._build_anchors(
            request_data,
            namespace,
            hints,
            provider=provider,
        )
        compaction_match = (
            self._best_match(compaction_probe_anchors, namespace, now)
            if compaction_probe_anchors
            else None
        )
        marker_probe_groups = self._compaction_marker_probe_groups(request_data)
        marker_compaction = bool(marker_probe_groups)
        compaction = self._evaluate_compaction(
            compaction_match,
            marker_compaction=marker_compaction,
            marker_probe_groups=marker_probe_groups,
            history_signatures=history_signatures,
        )
        authoritative_anchors = [
            anchor
            for anchor in unsuppressed_normal_anchors
            if self._is_authoritative_identity_anchor(anchor)
        ]
        authoritative_match = (
            self._best_match(authoritative_anchors, namespace, now)
            if authoritative_anchors
            else None
        )
        if authoritative_match and authoritative_match.is_sticky_match:
            possible_compaction = (
                compaction.possible_compaction
                and compaction.parent_session_id == authoritative_match.session_id
            )
            normal_anchors = (
                self._build_anchors(
                    request_data,
                    namespace,
                    hints,
                    provider=provider,
                    suppressed_continuity_indexes=probe_indexes,
                )
                if possible_compaction
                else unsuppressed_normal_anchors
            )
            state = self._refresh_and_bridge(
                authoritative_match.session_id,
                namespace,
                normal_anchors,
                now,
                affinity_key=self._affinity_from_anchors(normal_anchors, namespace),
                history_signatures=history_signatures,
            )
            return self._log_inference_decision(
                SessionInference(
                    session_id=state.session_id,
                    affinity_key=self._effective_affinity(state, hints, provider),
                    confidence=authoritative_match.confidence,
                    match_score=authoritative_match.score,
                    possible_compaction=possible_compaction,
                    tracking_namespace=namespace,
                ),
                action="compaction_continue" if possible_compaction else "continue",
                matched_session_id=state.session_id,
                compaction=compaction if possible_compaction else None,
                provider=provider,
                model=model,
            )

        has_authoritative_identity = bool(authoritative_anchors)
        context_binding = (
            None
            if has_authoritative_identity
            else self._find_compaction_context(
                compaction_probe_anchors,
                namespace,
                now,
            )
        )
        replay_record = (
            None
            if has_authoritative_identity
            else self._find_compaction_replay(
                compaction_probe_anchors,
                history_signatures,
                namespace,
                now,
            )
        )
        if replay_record:
            replay_anchor = self._compaction_replay_anchor(
                compaction_probe_anchors,
                history_signatures,
                namespace,
                parent_session_id=replay_record.group,
            )
            normal_anchors = self._build_anchors(
                request_data,
                namespace,
                hints,
                provider=provider,
                suppressed_continuity_indexes=probe_indexes,
            )
            context_anchors = (
                [context_binding[1]]
                if context_binding and context_binding[0].session_id == replay_record.session_id
                else []
            )
            normal_anchors = self._dedupe_anchors(
                [*normal_anchors, *context_anchors, replay_anchor]
            )
            state = self._refresh_and_bridge(
                replay_record.session_id,
                namespace,
                normal_anchors,
                now,
                affinity_key=self._affinity_from_anchors(normal_anchors, namespace),
                history_signatures=history_signatures,
            )
            return self._log_inference_decision(
                SessionInference(
                    session_id=state.session_id,
                    affinity_key=self._effective_affinity(state, hints, provider),
                    confidence="strong",
                    match_score=self._STRONG_SCORE,
                    possible_compaction=True,
                    lineage_parent_session_id=replay_record.group,
                    tracking_namespace=namespace,
                ),
                action="compaction_replay",
                matched_session_id=state.session_id,
                provider=provider,
                model=model,
            )

        if context_binding:
            context_record, context_anchor = context_binding
            normal_anchors = self._build_anchors(
                request_data,
                namespace,
                hints,
                provider=provider,
                suppressed_continuity_indexes=probe_indexes,
            )
            normal_anchors = self._dedupe_anchors([*normal_anchors, context_anchor])
            state = self._refresh_and_bridge(
                context_record.session_id,
                namespace,
                normal_anchors,
                now,
                affinity_key=self._affinity_from_anchors(normal_anchors, namespace),
                history_signatures=history_signatures,
            )
            return self._log_inference_decision(
                SessionInference(
                    session_id=state.session_id,
                    affinity_key=self._effective_affinity(state, hints, provider),
                    confidence="strong",
                    match_score=self._STRONG_SCORE,
                    possible_compaction=False,
                    lineage_parent_session_id=context_record.group,
                    tracking_namespace=namespace,
                ),
                action="compaction_continue",
                matched_session_id=state.session_id,
                provider=provider,
                model=model,
            )

        possible_compaction = compaction.possible_compaction
        normal_anchors = (
            self._build_anchors(
                request_data,
                namespace,
                hints,
                provider=provider,
                suppressed_continuity_indexes=probe_indexes,
            )
            if possible_compaction
            else unsuppressed_normal_anchors
        )

        if not normal_anchors and not compaction_probe_anchors:
            return self._log_inference_decision(
                SessionInference(session_id=None, tracking_namespace=namespace),
                action="untracked",
                provider=provider,
                model=model,
            )

        match = self._best_match(normal_anchors, namespace, now) if normal_anchors else None

        # Compaction is useful lineage information but should not hard-stick the
        # new compacted context unless a genuinely strong anchor survived.
        if match and match.is_sticky_match and not (
            possible_compaction and match.strong_matches == 0
        ):
            state = self._refresh_and_bridge(
                match.session_id,
                namespace,
                normal_anchors,
                now,
                affinity_key=self._affinity_from_anchors(normal_anchors, namespace),
                history_signatures=history_signatures,
            )
            return self._log_inference_decision(
                SessionInference(
                    session_id=state.session_id,
                    affinity_key=self._effective_affinity(state, hints, provider),
                    confidence=match.confidence,
                    match_score=match.score,
                    possible_compaction=possible_compaction,
                    lineage_parent_session_id=(
                        compaction.parent_session_id
                        if possible_compaction
                        and compaction.parent_session_id != state.session_id
                        else None
                    ),
                    tracking_namespace=namespace,
                ),
                action="compaction_continue" if possible_compaction else "continue",
                matched_session_id=match.session_id,
                compaction=compaction,
                provider=provider,
                model=model,
            )

        parent_id = compaction.parent_session_id
        if possible_compaction:
            context_anchors = self._compaction_context_anchors(
                compaction_probe_anchors,
                compaction.context_probe_groups,
                namespace,
                parent_session_id=parent_id,
            )
            normal_anchors = self._dedupe_anchors(
                [
                    *normal_anchors,
                    *context_anchors,
                    self._compaction_replay_anchor(
                        compaction_probe_anchors,
                        history_signatures,
                        namespace,
                        parent_session_id=parent_id,
                    ),
                ]
            )
        session_id = str(uuid.uuid4())
        state = self._create_session(
            session_id,
            namespace,
            normal_anchors,
            now,
            affinity_key=self._affinity_from_anchors(normal_anchors, namespace),
            history_signatures=history_signatures,
        )
        return self._log_inference_decision(
            SessionInference(
                session_id=state.session_id,
                affinity_key=self._effective_affinity(state, hints, provider),
                confidence="weak" if match else "none",
                match_score=match.score if match else 0,
                possible_compaction=possible_compaction,
                lineage_parent_session_id=parent_id,
                tracking_namespace=namespace,
            ),
            action="compaction_child" if parent_id else "new",
            candidate_session_id=match.session_id if match else None,
            compaction=compaction,
            provider=provider,
            model=model,
        )

    def _log_inference_decision(
        self,
        inference: SessionInference,
        *,
        action: str,
        matched_session_id: Optional[str] = None,
        candidate_session_id: Optional[str] = None,
        compaction: Optional[_CompactionDecision] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SessionInference:
        """Emit temporary warning-level lineage diagnostics for every request."""

        state = self._sessions.get(inference.session_id) if inference.session_id else None
        origin = (
            "persisted"
            if state and state.loaded_from_persistence
            else ("memory" if state else "none")
        )
        retained_ratio = (
            f"{compaction.retained_history_ratio:.3f}"
            if compaction and compaction.retained_history_ratio is not None
            else "-"
        )
        response_events = compaction.response_group_count if compaction else 0
        marker = compaction.marker_compaction if compaction else False
        lib_logger.warning(
            "Session tracker decision: action=%s session_id=%s matched_session_id=%s "
            "candidate_session_id=%s parent_session_id=%s namespace=%s "
            "provider=%s model=%s confidence=%s score=%d origin=%s "
            "possible_compaction=%s marker=%s retained_history=%s response_events=%d",
            action,
            inference.session_id or "-",
            matched_session_id or "-",
            candidate_session_id or "-",
            inference.lineage_parent_session_id or "-",
            inference.tracking_namespace or "-",
            provider or "-",
            model or "-",
            inference.confidence,
            inference.match_score,
            origin,
            inference.possible_compaction,
            marker,
            retained_ratio,
            response_events,
        )
        return inference

    def record_response(
        self,
        session_id: Optional[str],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        scope_key: Optional[str] = None,
        tracking_namespace: Optional[str] = None,
        response: Any = None,
    ) -> None:
        """Attach response-derived anchors to an existing live session.

        The next request often includes the previous assistant response or a tool
        call emitted by it. Recording those anchors makes tracking resilient to
        gradual context pruning without needing a dedicated compaction protocol.
        """
        save_job = None
        with self._lock:
            now = time.time()
            self._prune(now)
            if session_id and response is not None and session_id in self._sessions:
                state = self._sessions[session_id]
                namespace = tracking_namespace or state.namespace or self._namespace(
                    provider, scope_key=scope_key
                )
                if namespace != state.namespace:
                    lib_logger.warning(
                        "Session tracker normalized response namespace mismatch: "
                        "session_id=%s expected_namespace=%s received_namespace=%s",
                        session_id,
                        state.namespace,
                        namespace,
                    )
                    namespace = state.namespace
                anchors = self._anchors_from_response(response, namespace)
                if anchors:
                    self._refresh_and_bridge(session_id, namespace, anchors, now)
            save_job = self._prepare_save_locked()
        self._write_save_job(save_job)

    def flush(self) -> None:
        """Force persistence of dirty state when optional disk storage is enabled."""
        with self._lock:
            save_job = self._prepare_save_locked(force=True)
        self._write_save_job(save_job)

    def _create_session(
        self,
        session_id: str,
        namespace: str,
        anchors: List[SessionAnchor],
        now: float,
        *,
        affinity_key: Optional[str],
        history_signatures: tuple[str, ...] = (),
    ) -> _SessionState:
        state = _SessionState(
            session_id=session_id,
            namespace=namespace,
            expires_at=now + self.ttl_seconds,
            affinity_key=affinity_key,
            last_seen=now,
        )
        self._sessions[session_id] = state
        self._refresh_and_bridge(
            session_id,
            namespace,
            anchors,
            now,
            affinity_key=affinity_key,
            history_signatures=history_signatures,
        )
        return state

    def _refresh_and_bridge(
        self,
        session_id: str,
        namespace: str,
        anchors: List[SessionAnchor],
        now: float,
        *,
        affinity_key: Optional[str] = None,
        history_signatures: Optional[tuple[str, ...]] = None,
    ) -> _SessionState:
        expires_at = now + self.ttl_seconds
        state = self._sessions.setdefault(
            session_id,
            _SessionState(session_id=session_id, namespace=namespace, expires_at=expires_at),
        )
        if state.namespace != namespace:
            raise ValueError(
                f"Session {session_id} belongs to {state.namespace}, not {namespace}"
            )
        state.expires_at = expires_at
        state.last_seen = now
        if affinity_key and not state.affinity_key:
            state.affinity_key = affinity_key
        if history_signatures and len(history_signatures) >= len(state.history_signatures):
            # Keep the largest observed request as the structural baseline. Equal
            # fixed-window histories advance to the newest normal conversation.
            state.history_signatures = history_signatures

        for anchor in anchors:
            existing = self._anchors.get(anchor.value)
            if existing and existing.session_id != session_id:
                # A shared prompt/chunk is not proof that a newer independent
                # session owns it. Preserve the first live owner until TTL expiry.
                continue
            strength = (
                self._strongest(anchor.strength, existing.strength)
                if existing
                else anchor.strength
            )
            source = anchor.source
            group = anchor.group
            if existing and existing.source == "response" and anchor.source != "response":
                # Responses commonly return as ordinary assistant history. Keep
                # their response-event provenance for future lineage decisions.
                source = existing.source
                group = existing.group
            state.anchors.add(anchor.value)
            self._anchors[anchor.value] = _AnchorRecord(
                session_id=session_id,
                namespace=namespace,
                strength=strength,
                source=source,
                group=group,
                expires_at=expires_at,
                last_seen=now,
            )

        self._trim_session_anchors(state)
        self._trim_global_anchors()
        self._mark_dirty()
        return state

    def _best_match(
        self,
        anchors: List[SessionAnchor],
        namespace: str,
        now: float,
    ) -> Optional[_MatchCandidate]:
        candidates: Dict[str, _MatchCandidate] = {}
        for anchor in anchors:
            record = self._anchors.get(anchor.value)
            if not record or record.expires_at <= now or record.namespace != namespace:
                continue
            candidate = candidates.setdefault(record.session_id, _MatchCandidate(record.session_id))
            candidate.last_seen = max(candidate.last_seen, record.last_seen)
            if anchor.source == "compaction_probe" and anchor.group:
                candidate.matched_probe_groups.add(anchor.group)
            # Closure belongs to the current request. A previously closed event
            # must not upgrade a later unpaired copy merely because the stored
            # anchor retained medium strength.
            strength = (
                anchor.strength
                if anchor.source == "tool_event"
                else self._strongest(anchor.strength, record.strength)
            )
            if strength == "strong":
                candidate.score += self._STRONG_SCORE
                candidate.strong_matches += 1
            elif strength == "medium":
                candidate.score += self._MEDIUM_SCORE
                candidate.medium_matches += 1
                group = anchor.group or record.group
                if group and anchor.source != "window" and record.source != "window":
                    candidate.medium_groups.add(group)
                if anchor.source == "provider" or record.source == "provider":
                    candidate.provider_matches += 1
                response_overlap = anchor.source == "response" or record.source == "response"
                if response_overlap and self._allows_response_bridge(anchor, record):
                    candidate.response_matches += 1
                    response_group = (
                        anchor.group if anchor.source == "response" else record.group
                    )
                    if response_group:
                        candidate.response_groups.add(response_group)
                    if anchor.source == "compaction_probe" and anchor.group:
                        candidate.response_probe_groups.add(anchor.group)
                if record.source == "message" and record.group:
                    candidate.request_groups.add(record.group)
            else:
                candidate.score += self._WEAK_SCORE
                candidate.weak_matches += 1

        if not candidates:
            return None
        return max(
            candidates.values(),
            key=lambda item: (
                item.score,
                item.strong_matches,
                item.medium_matches,
                len(item.medium_groups),
                len(item.response_groups),
                item.response_matches,
                item.provider_matches,
                item.last_seen,
                item.session_id,
            ),
        )

    def _coerce_hints(self, hints: Optional[Any]) -> Optional[SessionTrackingHints]:
        if not hints:
            return None
        if isinstance(hints, SessionTrackingHints):
            return hints
        if isinstance(hints, dict):
            return SessionTrackingHints(
                strong_anchors=list(hints.get("strong_anchors") or []),
                medium_anchors=list(hints.get("medium_anchors") or []),
                weak_anchors=list(hints.get("weak_anchors") or []),
                global_strong_anchors=list(hints.get("global_strong_anchors") or []),
                global_medium_anchors=list(hints.get("global_medium_anchors") or []),
                global_weak_anchors=list(hints.get("global_weak_anchors") or []),
                affinity_key=hints.get("affinity_key"),
                session_scope=hints.get("session_scope"),
            )
        return None

    def _build_anchors(
        self,
        request_data: Dict[str, Any],
        namespace: str,
        hints: Optional[Any],
        *,
        provider: Optional[str] = None,
        allow_system_continuity: bool = False,
        suppressed_continuity_indexes: Optional[set[int]] = None,
    ) -> List[SessionAnchor]:
        anchors: List[SessionAnchor] = []
        anchors.extend(self._anchors_from_provider_hints(hints, namespace, provider))
        anchors.extend(self._anchors_from_explicit_ids(request_data, namespace))

        messages = request_data.get("messages") or []
        if isinstance(messages, list) and messages:
            anchors.extend(
                self._anchors_from_messages(
                    messages,
                    namespace,
                    allow_system_continuity=allow_system_continuity,
                    suppressed_continuity_indexes=suppressed_continuity_indexes,
                )
            )

        return self._dedupe_anchors(anchors)

    def _build_compaction_probe_anchors(
        self,
        request_data: Dict[str, Any],
        namespace: str,
        *,
        probe_indexes: Optional[set[int]] = None,
    ) -> List[SessionAnchor]:
        """Build temporary anchors for compaction lineage lookup only.

        Compaction summaries often replace prior user/assistant history and may
        be sent as system, developer, or user messages. Assistant/tool history is
        deliberately excluded because replaying it is ordinary continuity. These
        anchors are compared against existing response/message anchors to identify
        a likely parent, but they are not stored on the newly-created child session.
        """
        messages = request_data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return []

        anchors: List[SessionAnchor] = []
        if probe_indexes is None:
            probe_indexes = self._compaction_probe_indexes(request_data)
        for index, message in enumerate(messages[:2]):
            if index not in probe_indexes:
                continue
            if not isinstance(message, dict):
                continue
            text = self._normalize_text(self._extract_text(message.get("content")))
            role = str(message.get("role", "")).lower()
            anchors.append(
                SessionAnchor(
                    self._scoped(namespace, f"message:{role}:{self._hash_text(text)}"),
                    "medium",
                    source="compaction_probe",
                    group=f"compaction_probe:{index}",
                )
            )
            for chunk_hash in self._content_chunk_hashes(text):
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"chunk:{chunk_hash}"),
                        "medium",
                        source="compaction_probe",
                        group=f"compaction_probe:{index}",
                    )
                )
        return self._dedupe_anchors(anchors)

    def _compaction_probe_indexes(self, request_data: Dict[str, Any]) -> set[int]:
        messages = request_data.get("messages") or []
        if not isinstance(messages, list):
            return set()
        indexes: set[int] = set()
        for index, message in enumerate(messages[:2]):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role not in self._COMPACTION_PROBE_ROLES:
                continue
            text = self._normalize_text(self._extract_text(message.get("content")))
            if self._is_compaction_probe_text(text):
                indexes.add(index)
        return indexes

    def _evaluate_compaction(
        self,
        match: Optional[_MatchCandidate],
        *,
        marker_compaction: bool,
        marker_probe_groups: set[str],
        history_signatures: tuple[str, ...],
    ) -> _CompactionDecision:
        """Require parent evidence and replacement of most known history.

        One response overlap is intentionally insufficient for unmarked input:
        side-channel classifiers often quote the latest response verbatim. A real
        unmarked summary must span distinct response events and remove more than
        half of the parent's high-water request history.
        """

        if not match:
            return _CompactionDecision()
        parent = self._sessions.get(match.session_id)
        if not parent or not parent.history_signatures:
            return _CompactionDecision()
        retained_ratio = self._retained_history_ratio(
            history_signatures,
            parent.history_signatures,
        )
        if retained_ratio >= self._COMPACTION_MAX_RETAINED_HISTORY_RATIO:
            lib_logger.debug(
                "Session tracker rejected compaction candidate for %s: "
                "retained_history=%.3f",
                match.session_id,
                retained_ratio,
            )
            return _CompactionDecision()

        response_group_count = len(match.response_groups)
        qualifies = (
            match.score > 0
            if marker_compaction
            else (
                response_group_count >= self._MIN_UNMARKED_RESPONSE_GROUPS
                and bool(match.request_groups)
            )
        )
        if not qualifies:
            lib_logger.debug(
                "Session tracker rejected compaction candidate for %s: "
                "marker=%s, response_events=%d, request_groups=%d",
                match.session_id,
                marker_compaction,
                response_group_count,
                len(match.request_groups),
            )
            return _CompactionDecision()
        marker_context_groups = match.matched_probe_groups.intersection(marker_probe_groups)
        return _CompactionDecision(
            parent_session_id=match.session_id,
            marker_compaction=marker_compaction,
            retained_history_ratio=retained_ratio,
            response_group_count=response_group_count,
            context_probe_groups=frozenset(
                (marker_context_groups or match.response_probe_groups)
                if marker_compaction
                else match.response_probe_groups
            ),
        )

    def _compaction_marker_probe_groups(self, request_data: Dict[str, Any]) -> set[str]:
        """Return early system/developer probe groups carrying explicit markers."""

        messages = request_data.get("messages") or []
        groups: set[str] = set()
        if not isinstance(messages, list):
            return groups
        for index, message in enumerate(messages[:2]):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            text = self._extract_text(message.get("content"))
            if role in {"system", "developer"} and self._has_compaction_marker(text):
                groups.add(f"compaction_probe:{index}")
        return groups

    def _compaction_replay_anchor(
        self,
        probe_anchors: List[SessionAnchor],
        history_signatures: tuple[str, ...],
        namespace: str,
        *,
        parent_session_id: Optional[str],
    ) -> SessionAnchor:
        """Build an opaque exact-replay key for a validated compacted payload."""

        digest = self._hash_json(
            {
                "probe_anchors": sorted(anchor.value for anchor in probe_anchors),
                "history_signatures": list(history_signatures),
            }
        )
        return SessionAnchor(
            self._scoped(namespace, f"compaction_replay:{digest}"),
            "strong",
            source="compaction_replay",
            group=parent_session_id,
        )

    def _compaction_context_anchor(
        self,
        probe_anchors: List[SessionAnchor],
        namespace: str,
        *,
        parent_session_id: Optional[str],
    ) -> SessionAnchor:
        """Bind later requests that retain one validated compacted base context."""

        digest = self._hash_json(sorted(anchor.value for anchor in probe_anchors))
        return SessionAnchor(
            self._scoped(namespace, f"compaction_context:{digest}"),
            "strong",
            source="compaction_context",
            group=parent_session_id,
        )

    def _compaction_context_anchors(
        self,
        probe_anchors: List[SessionAnchor],
        probe_groups: Iterable[str],
        namespace: str,
        *,
        parent_session_id: Optional[str],
    ) -> List[SessionAnchor]:
        """Build context keys only for probes that actually matched the parent."""

        anchors: List[SessionAnchor] = []
        for group in sorted(set(probe_groups)):
            grouped = [anchor for anchor in probe_anchors if anchor.group == group]
            if grouped:
                anchors.append(
                    self._compaction_context_anchor(
                        grouped,
                        namespace,
                        parent_session_id=parent_session_id,
                    )
                )
        return anchors

    @staticmethod
    def _is_authoritative_identity_anchor(anchor: SessionAnchor) -> bool:
        """Return whether an anchor must take precedence over replay bindings."""

        return anchor.strength == "strong" and anchor.source in {
            "explicit",
            "global_hint",
            "provider",
        }

    def _find_compaction_replay(
        self,
        probe_anchors: List[SessionAnchor],
        history_signatures: tuple[str, ...],
        namespace: str,
        now: float,
    ) -> Optional[_AnchorRecord]:
        if not probe_anchors:
            return None
        anchor = self._compaction_replay_anchor(
            probe_anchors,
            history_signatures,
            namespace,
            parent_session_id=None,
        )
        record = self._anchors.get(anchor.value)
        if (
            not record
            or record.source != "compaction_replay"
            or record.namespace != namespace
            or record.expires_at <= now
            or record.session_id not in self._sessions
            or not record.group
        ):
            return None
        return record

    def _find_compaction_context(
        self,
        probe_anchors: List[SessionAnchor],
        namespace: str,
        now: float,
    ) -> Optional[tuple[_AnchorRecord, SessionAnchor]]:
        probe_groups = sorted({anchor.group for anchor in probe_anchors if anchor.group})
        for probe_group in probe_groups:
            grouped = [anchor for anchor in probe_anchors if anchor.group == probe_group]
            anchor = self._compaction_context_anchor(
                grouped,
                namespace,
                parent_session_id=None,
            )
            record = self._anchors.get(anchor.value)
            if (
                record
                and record.source == "compaction_context"
                and record.namespace == namespace
                and record.expires_at > now
                and record.session_id in self._sessions
                and record.group
            ):
                return (
                    record,
                    SessionAnchor(
                        anchor.value,
                        "strong",
                        source="compaction_context",
                        group=record.group,
                    ),
                )
        return None

    def _anchors_from_provider_hints(
        self,
        hints: Optional[SessionTrackingHints],
        namespace: str,
        provider: Optional[str],
    ) -> List[SessionAnchor]:
        if not hints:
            return []
        anchors: List[SessionAnchor] = []
        # A provider may partition only its native evidence/affinity domain. This
        # never fragments the global logical session or caller isolation domain.
        native_scope = hints.session_scope or "provider"
        provider_key = self._hash_text(f"{provider or 'unknown'}:{native_scope}")
        for strength, attr in (
            ("strong", "strong_anchors"),
            ("medium", "medium_anchors"),
            ("weak", "weak_anchors"),
        ):
            for value in getattr(hints, attr, []) or []:
                value_hash = self._hash_text(str(value))
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"provider:{provider_key}:{value_hash}"),
                        strength,
                        source="provider",
                        group=f"provider:{provider_key}:{value_hash}",
                    )
                )
        for strength, attr in (
            ("strong", "global_strong_anchors"),
            ("medium", "global_medium_anchors"),
            ("weak", "global_weak_anchors"),
        ):
            for value in getattr(hints, attr, []) or []:
                value_hash = self._hash_text(str(value))
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"global:{value_hash}"),
                        strength,
                        source="global_hint",
                        group=f"global:{value_hash}",
                    )
                )
        affinity_key = getattr(hints, "affinity_key", None)
        if affinity_key:
            affinity_hash = self._hash_text(str(affinity_key))
            anchors.append(
                SessionAnchor(
                    self._scoped(
                        namespace,
                        f"provider:{provider_key}:affinity:{affinity_hash}",
                    ),
                    "strong",
                    source="provider",
                    group=f"provider:{provider_key}:affinity",
                )
            )
        return anchors

    def _anchors_from_explicit_ids(
        self,
        request_data: Dict[str, Any],
        namespace: str,
    ) -> List[SessionAnchor]:
        # Many coding clients generate these per request. Keep them weak unless a
        # provider explicitly vouches for a stable equivalent via hints.
        anchors: List[SessionAnchor] = []
        for key in (
            "session_id",
            "conversation_id",
            "conversationId",
            "thread_id",
            "threadId",
            "chat_id",
            "chatId",
        ):
            value = request_data.get(key)
            if value:
                value_hash = self._hash_text(str(value))
                strength = "strong" if key in self.trusted_explicit_fields else "weak"
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"explicit:{key}:{value_hash}"),
                        strength,
                        source="explicit",
                        group=f"explicit:{key}",
                    )
                )
        return anchors

    def _anchors_from_messages(
        self,
        messages: List[Dict[str, Any]],
        namespace: str,
        *,
        source: str = "message",
        evidence_group: Optional[str] = None,
        allow_system_continuity: bool = False,
        suppressed_continuity_indexes: Optional[set[int]] = None,
    ) -> List[SessionAnchor]:
        anchors: List[SessionAnchor] = []
        normalized_messages: List[Dict[str, Any]] = []
        tool_ids: List[str] = []
        first_user_text: Optional[str] = None
        closed_tool_calls = self._closed_tool_call_positions(messages)

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            content = message.get("content")
            text = self._extract_text(content)
            normalized: Dict[str, Any] = {"role": role, "content": self._normalize_content(content)}

            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                tool_id = str(tool_call_id)
                tool_id_hash = self._hash_text(tool_id)
                tool_ids.append(tool_id)
                normalized["tool_call_id"] = tool_id
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"tool:{tool_id_hash}"),
                        "weak",
                        source="tool",
                        group=f"tool:{tool_id_hash}",
                    )
                )

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                call_ids: List[str] = []
                for tool_call_index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = tool_call.get("id")
                    if call_id:
                        call_id = str(call_id)
                        call_id_hash = self._hash_text(call_id)
                        call_ids.append(call_id)
                        tool_ids.append(call_id)
                        anchors.append(
                            SessionAnchor(
                                self._scoped(namespace, f"tool:{call_id_hash}"),
                                "weak",
                                source="tool",
                                group=f"tool:{call_id_hash}",
                            )
                        )
                        event = self._tool_event_descriptor(tool_call)
                        if role.lower() == "assistant" and event is not None:
                            event_hash = self._hash_json(event)
                            anchors.append(
                                SessionAnchor(
                                    self._scoped(
                                        namespace,
                                        f"tool_event:{event_hash}",
                                    ),
                                    (
                                        "medium"
                                        if (index, tool_call_index) in closed_tool_calls
                                        else "weak"
                                    ),
                                    source="tool_event",
                                    group=f"tool_event:{event_hash}",
                                )
                            )
                if call_ids:
                    normalized["tool_calls"] = call_ids

            if text:
                normalized_text = self._normalize_text(text)
                if first_user_text is None and role == "user":
                    first_user_text = normalized_text
                # System/developer prompts are commonly shared by an agent
                # harness. They can describe request shape, but must not be
                # treated as continuity evidence between independent sessions.
                contributes_continuity = not (
                    source == "message"
                    and role.lower() in {"system", "developer"}
                    and not allow_system_continuity
                )
                if suppressed_continuity_indexes and index in suppressed_continuity_indexes:
                    contributes_continuity = False
                if contributes_continuity and self._is_substantial_text(normalized_text):
                    anchors.append(
                        SessionAnchor(
                            self._scoped(namespace, f"message:{role}:{self._hash_text(normalized_text)}"),
                            "medium",
                            source=source,
                            group=evidence_group or f"{source}:{index}:{role.lower()}",
                        )
                    )
                    for chunk_hash in self._content_chunk_hashes(normalized_text):
                        anchors.append(
                            SessionAnchor(
                                self._scoped(namespace, f"chunk:{chunk_hash}"),
                                "medium",
                                source=source,
                                group=evidence_group or f"{source}:{index}:{role.lower()}",
                            )
                        )

            # Positional message hashes are intentionally medium: they are useful
            # when history is unchanged, but pruning can move or remove them.
            if index < 4 or index >= max(0, len(messages) - 4):
                normalized_messages.append(normalized)

        if tool_ids:
            anchors.append(
                SessionAnchor(
                    self._scoped(namespace, "tool_group:" + self._hash_json(sorted(tool_ids))),
                    "weak",
                    source="tool",
                    group="tool_group",
                )
            )

        if normalized_messages:
            anchors.append(
                SessionAnchor(
                    self._scoped(namespace, "window:" + self._hash_json(normalized_messages)),
                    "medium",
                    source="window",
                    group=None,
                )
            )

        if first_user_text:
            anchors.append(
                SessionAnchor(
                    self._scoped(namespace, "first_user:" + self._hash_text(first_user_text)),
                    "weak",
                    source="first_user",
                    group="first_user",
                )
            )

        return anchors

    def _anchors_from_response(self, response: Any, namespace: str) -> List[SessionAnchor]:
        data = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(data, dict):
            return []
        anchors: List[SessionAnchor] = []
        response_id = data.get("id")
        if response_id:
            response_id_hash = self._hash_text(
                f"responses_previous_response_id:{response_id}"
            )
            # Responses API continuations identify their parent with
            # previous_response_id. Record the emitted response id as strong
            # response evidence so the next request can route back to the exact
            # session/credential that produced it.
            anchors.append(
                SessionAnchor(
                    self._scoped(
                        namespace,
                        f"global:{response_id_hash}",
                    ),
                    "strong",
                    source="response",
                    group="responses_previous_response_id",
                )
            )
        messages: List[Dict[str, Any]] = []
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or choice.get("delta")
            if isinstance(message, dict):
                response_message = dict(message)
                response_message.setdefault("role", "assistant")
                messages.append(response_message)
        if messages:
            response_group = "response_event:" + self._hash_json(
                [self._message_signature(message) for message in messages]
            )
            anchors.extend(
                self._anchors_from_messages(
                    messages,
                    namespace,
                    source="response",
                    evidence_group=response_group,
                )
            )
        return anchors

    @staticmethod
    def _closed_tool_call_positions(
        messages: List[Dict[str, Any]],
    ) -> set[tuple[int, int]]:
        """Pair each tool result with one earlier assistant call of the same ID."""

        pending: Dict[str, List[tuple[int, int]]] = {}
        closed: set[tuple[int, int]] = set()
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    continue
                for tool_call_index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict) or not tool_call.get("id"):
                        continue
                    pending.setdefault(str(tool_call["id"]), []).append(
                        (message_index, tool_call_index)
                    )
                continue
            if role not in {"tool", "function"} or not message.get("tool_call_id"):
                continue
            candidates = pending.get(str(message["tool_call_id"]))
            if candidates:
                closed.add(candidates.pop(0))
        return closed

    def _tool_event_descriptor(
        self,
        tool_call: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return stable call structure used by pending and closed tool events."""

        call_id = tool_call.get("id")
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name") or tool_call.get("name")
            arguments = function.get("arguments")
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments")
        if not call_id or not name:
            return None
        return {
            "id": str(call_id),
            "name": str(name),
            "arguments": self._canonical_tool_arguments(arguments),
        }

    def _canonical_tool_arguments(self, arguments: Any) -> Any:
        """Normalize JSON arguments without retaining their plaintext in anchors."""

        if not isinstance(arguments, str):
            return arguments
        stripped = arguments.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._normalize_text(stripped)

    def _affinity_from_anchors(
        self,
        anchors: List[SessionAnchor],
        namespace: str,
    ) -> Optional[str]:
        strong = sorted(
            anchor.value
            for anchor in anchors
            if anchor.strength == "strong" and anchor.source != "provider"
        )
        if strong:
            return self._scoped(namespace, "affinity:" + self._hash_json(strong[:4]))
        medium_anchors = [
            anchor
            for anchor in anchors
            if anchor.strength == "medium" and anchor.source != "provider"
        ]
        medium_groups = {
            anchor.group
            for anchor in medium_anchors
            if anchor.group and anchor.source != "window"
        }
        has_provider_or_response = any(
            anchor.source in {"provider", "response"} for anchor in medium_anchors
        )
        medium = sorted(anchor.value for anchor in medium_anchors)
        if len(medium) >= 2 and (len(medium_groups) >= 2 or has_provider_or_response):
            return self._scoped(namespace, "affinity:" + self._hash_json(medium[:8]))
        return None

    @staticmethod
    def _effective_affinity(
        state: _SessionState,
        hints: Optional[SessionTrackingHints],
        provider: Optional[str],
    ) -> Optional[str]:
        """Use provider affinity only for the provider handling this request."""

        if hints:
            if hints.affinity_key:
                return hints.affinity_key
            native = sorted(str(value) for value in hints.strong_anchors if value)
            if native:
                payload = json.dumps(
                    [provider or "unknown", hints.session_scope or "provider", native],
                    separators=(",", ":"),
                )
                return "provider-affinity:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return state.affinity_key

    def _is_compaction_probe_text(self, text: str) -> bool:
        if not text:
            return False
        # Large early messages are often generated summaries even without an
        # explicit marker. Short text needs a marker to avoid matching ordinary
        # first user prompts as lineage probes.
        return self._has_compaction_marker(text) or len(text) >= 400 or len(text.split()) >= 80

    def _has_compaction_marker(self, text: str) -> bool:
        lowered = text.lower()
        markers = (
            "summary of previous conversation",
            "summary of the previous conversation",
            "summarized conversation",
            "compressed context",
            "compacted context",
            "conversation so far",
            "previous conversation",
            "context reminder",
        )
        return any(marker in lowered for marker in markers)

    def _trim_session_anchors(self, state: _SessionState) -> None:
        if len(state.anchors) <= self.max_anchors_per_session:
            return
        sorted_anchors = sorted(
            state.anchors,
            key=lambda value: (
                self._anchor_eviction_key(self._anchors.get(value)),
                value,
            ),
        )
        for value in sorted_anchors[: len(state.anchors) - self.max_anchors_per_session]:
            state.anchors.discard(value)
            record = self._anchors.get(value)
            if record and record.session_id == state.session_id:
                del self._anchors[value]

    def _trim_global_anchors(self) -> None:
        if len(self._anchors) <= self.max_anchor_records:
            return
        overage = len(self._anchors) - self.max_anchor_records
        for value, record in sorted(
            self._anchors.items(),
            key=lambda item: (self._anchor_eviction_key(item[1]), item[0]),
        )[:overage]:
            state = self._sessions.get(record.session_id)
            if state:
                state.anchors.discard(value)
            del self._anchors[value]

    def _prune(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        changed = False
        expired_sessions = [key for key, state in self._sessions.items() if state.expires_at <= now]
        for session_id in expired_sessions:
            state = self._sessions.pop(session_id)
            changed = True
            for anchor in list(state.anchors):
                record = self._anchors.get(anchor)
                if record and record.session_id == session_id:
                    del self._anchors[anchor]
        expired_anchors = [key for key, record in self._anchors.items() if record.expires_at <= now]
        for anchor in expired_anchors:
            record = self._anchors.pop(anchor)
            changed = True
            state = self._sessions.get(record.session_id)
            if state:
                state.anchors.discard(anchor)
        if changed:
            self._mark_dirty()

    def _load(self) -> None:
        if not self.persistence_path:
            return
        try:
            if (
                self.persistence_path.exists()
                and self.persistence_path.stat().st_size > self._MAX_PERSISTED_FILE_BYTES
            ):
                lib_logger.warning("Ignoring oversized session_stickiness.json")
                return
        except OSError:
            return
        data = safe_read_json(self.persistence_path, lib_logger)
        if not isinstance(data, dict):
            return
        now = time.time()
        if data.get("schema_version") != self._PERSISTENCE_SCHEMA_VERSION:
            lib_logger.info(
                "Ignoring unsupported session_stickiness.json format; session persistence will rebuild in memory."
            )
            return
        sessions = data.get("sessions")
        anchors = data.get("anchors")
        if not isinstance(sessions, dict) or not isinstance(anchors, dict):
            lib_logger.info(
                "Ignoring malformed session_stickiness.json containers; "
                "session persistence will rebuild in memory."
            )
            return
        def persisted_last_seen(item: tuple[Any, Any]) -> float:
            payload = item[1]
            if not isinstance(payload, dict):
                return 0.0
            return self._finite_float(payload.get("last_seen")) or 0.0

        ordered_sessions = sorted(
            sessions.items(),
            key=persisted_last_seen,
            reverse=True,
        )[: self._MAX_PERSISTED_SESSIONS]
        for session_id, payload in ordered_sessions:
            if not isinstance(session_id, str) or not session_id or not isinstance(payload, dict):
                continue
            if len(session_id) > self._MAX_PERSISTED_STRING_LENGTH:
                continue
            expires_at = self._finite_float(payload.get("expires_at"))
            if expires_at is None or expires_at <= now:
                continue
            namespace = payload.get("namespace")
            if not isinstance(namespace, str) or not namespace.startswith("session-domain:"):
                continue
            if len(namespace) > self._MAX_PERSISTED_STRING_LENGTH:
                continue
            last_seen = self._finite_float(payload.get("last_seen"))
            history_payload = payload.get("history_signatures") or []
            history_signatures = (
                tuple(
                    value
                    for value in history_payload[: self._MAX_PERSISTED_HISTORY_SIGNATURES]
                    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                )
                if isinstance(history_payload, list)
                else ()
            )
            affinity_key = payload.get("affinity_key")
            if affinity_key is not None and not isinstance(affinity_key, str):
                affinity_key = None
            if isinstance(affinity_key, str) and len(affinity_key) > self._MAX_PERSISTED_STRING_LENGTH:
                affinity_key = None
            self._sessions[session_id] = _SessionState(
                session_id=session_id,
                namespace=namespace,
                expires_at=expires_at,
                affinity_key=affinity_key,
                anchors=set(),
                last_seen=last_seen if last_seen is not None else now,
                history_signatures=history_signatures,
                loaded_from_persistence=True,
            )
        for value, payload in anchors.items():
            if not isinstance(payload, dict):
                continue
            session_id = payload.get("session_id")
            state = self._sessions.get(session_id) if isinstance(session_id, str) else None
            expires_at = self._finite_float(payload.get("expires_at"))
            if not state or expires_at is None or expires_at <= now:
                continue
            namespace = payload.get("namespace")
            if namespace != state.namespace:
                continue
            if not isinstance(value, str) or not value.startswith(f"{namespace}:"):
                continue
            if len(value) > self._MAX_PERSISTED_STRING_LENGTH:
                continue
            strength = payload.get("strength")
            source = payload.get("source")
            group = payload.get("group")
            if strength not in {"weak", "medium", "strong"}:
                continue
            if source not in self._PERSISTED_ANCHOR_SOURCES:
                continue
            if group is not None and not isinstance(group, str):
                continue
            if isinstance(group, str) and len(group) > self._MAX_PERSISTED_STRING_LENGTH:
                continue
            last_seen = self._finite_float(payload.get("last_seen"))
            self._anchors[value] = _AnchorRecord(
                session_id=session_id,
                namespace=namespace,
                strength=strength,
                source=source,
                group=group,
                expires_at=min(expires_at, state.expires_at),
                last_seen=last_seen if last_seen is not None else now,
            )
            state.anchors.add(value)

        for state in self._sessions.values():
            self._trim_session_anchors(state)
        self._trim_global_anchors()

    def _prepare_save_locked(
        self,
        *,
        force: bool = False,
    ) -> Optional[tuple[ResilientStateWriter, Dict[str, Any], int]]:
        if not self.persist_to_disk or not self.persistence_path or not self._dirty:
            return None
        now = time.time()
        if not force and now - self._last_save_attempt < self.persistence_flush_interval_seconds:
            return None
        self._last_save_attempt = now
        payload = {
            "schema_version": self._PERSISTENCE_SCHEMA_VERSION,
            "sessions": {
                session_id: {
                    "namespace": state.namespace,
                    "expires_at": state.expires_at,
                    "affinity_key": state.affinity_key,
                    "last_seen": state.last_seen,
                    "history_signatures": list(state.history_signatures),
                }
                for session_id, state in self._sessions.items()
            },
            "anchors": {
                anchor: {
                    "session_id": record.session_id,
                    "namespace": record.namespace,
                    "strength": record.strength,
                    "source": record.source,
                    "group": record.group,
                    "expires_at": record.expires_at,
                    "last_seen": record.last_seen,
                }
                for anchor, record in self._anchors.items()
            },
        }
        if self._writer is None:
            self._writer = ResilientStateWriter(
                self.persistence_path,
                lib_logger,
                serializer=lambda data: json.dumps(data, indent=2, sort_keys=True),
            )
        return self._writer, payload, self._dirty_generation

    def _write_save_job(
        self,
        save_job: Optional[tuple[ResilientStateWriter, Dict[str, Any], int]],
    ) -> None:
        if save_job is None:
            return
        writer, payload, generation = save_job
        with self._save_io_lock:
            if generation < self._last_persisted_generation:
                return
            try:
                success = writer.write(payload)
            except Exception as exc:
                lib_logger.warning("Failed to persist session tracking state: %s", exc)
                success = False
            if success:
                self._last_persisted_generation = generation
        if not success:
            return
        with self._lock:
            if self._dirty_generation == generation:
                self._dirty = False

    def _mark_dirty(self) -> None:
        self._dirty_generation += 1
        self._dirty = True

    def _namespace(
        self,
        provider: Optional[str],
        *,
        scope_key: Optional[str] = None,
        trusted_isolation_key: bool = False,
    ) -> str:
        # Logical sessions cross providers and models, but never caller/credential
        # isolation domains. Provider-native evidence is qualified separately.
        allowed_scope = self._normalize_isolation_key(
            scope_key,
            provider,
            trusted=trusted_isolation_key,
        )
        return f"session-domain:{allowed_scope}"

    @staticmethod
    def _normalize_isolation_key(
        scope_key: Optional[str],
        provider: Optional[str],
        *,
        trusted: bool,
    ) -> str:
        """Accept internal domain markers only from RequestContextBuilder."""

        scope = str(scope_key or provider or "public")
        provider_key = (provider or "").strip().lower()
        if not scope_key or scope.strip().lower() in {"public", provider_key}:
            return "public"
        if trusted and (
            re.fullmatch(r"classifier:[0-9a-f]{24}", scope)
            or re.fullmatch(r"bundle:[0-9a-f]{64}", scope)
        ):
            return scope
        return "scope:" + hashlib.sha256(scope.encode("utf-8")).hexdigest()

    def _trusted_fields_from_env(self) -> List[str]:
        raw = os.getenv("TRUSTED_SESSION_ID_FIELDS", "")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _scoped(self, namespace: str, value: str) -> str:
        return f"{namespace}:{value}"

    def _strongest(self, left: str, right: str) -> str:
        order = {"weak": 0, "medium": 1, "strong": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right

    def _anchor_eviction_key(
        self,
        record: Optional[_AnchorRecord],
    ) -> tuple[int, int, float]:
        """Evict weak/ordinary evidence before strong replay/context identity."""

        if record is None:
            return (-1, 0, 0.0)
        strength_rank = {"weak": 0, "medium": 1, "strong": 2}.get(record.strength, 0)
        replay_rank = 1 if record.source in {"compaction_context", "compaction_replay"} else 0
        return (strength_rank, replay_rank, record.last_seen)

    def _allows_response_bridge(
        self,
        anchor: SessionAnchor,
        record: _AnchorRecord,
    ) -> bool:
        """Only assistant-role request history may bridge from one response.

        Compaction probes are evaluated separately and may summarize responses in
        a user/system message. Ordinary user prompts that merely quote a response
        must not become sticky to the producer session.
        """

        message_group = None
        if anchor.source == "message":
            message_group = anchor.group
        elif record.source == "message":
            message_group = record.group
        return message_group is None or message_group.endswith(":assistant")

    def _finite_float(self, value: Any) -> Optional[float]:
        """Parse persisted numeric metadata without allowing NaN or infinity."""

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def _is_substantial_text(self, text: str) -> bool:
        return len(text) >= 24 and len(text.split()) >= 4

    def _request_history_signatures(self, request_data: Dict[str, Any]) -> tuple[str, ...]:
        """Return content-free signatures for structural history comparison."""

        messages = request_data.get("messages") or []
        if not isinstance(messages, list):
            return ()
        return tuple(
            signature
            for message in messages
            if isinstance(message, dict)
            for signature in [self._message_signature(message)]
            if signature
        )

    def _message_signature(self, message: Dict[str, Any]) -> str:
        """Hash the normalized message shape without retaining raw content."""

        payload: Dict[str, Any] = {
            "role": str(message.get("role", "")).lower(),
            "content": self._normalize_content(message.get("content")),
        }
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            payload["tool_call_id"] = str(tool_call_id)
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            normalized_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                normalized_calls.append(
                    {
                        key: call.get(key)
                        for key in ("id", "type", "function", "name")
                        if call.get(key) is not None
                    }
                )
            if normalized_calls:
                payload["tool_calls"] = normalized_calls
        return self._hash_json(payload)

    def _retained_history_ratio(
        self,
        current: tuple[str, ...],
        previous: tuple[str, ...],
    ) -> float:
        """Measure how much of the parent's high-water history remains."""

        if not previous:
            return 1.0
        retained = sum((Counter(current) & Counter(previous)).values())
        return retained / len(previous)

    def _content_chunk_hashes(self, text: str) -> List[str]:
        words = text.split()
        if len(words) < 8:
            return []
        hashes: List[str] = []
        for start in range(0, max(1, len(words) - 7), 4):
            chunk = " ".join(words[start : start + 8])
            hashes.append(self._hash_text(chunk))
        # Winnowing: keep deterministic low hashes so overlapping long content
        # survives truncation/reordering without storing every chunk.
        return sorted(set(hashes))[:8]

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text).strip())
                elif isinstance(item, str):
                    parts.append(item.strip())
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            text = content.get("text")
            if text:
                return str(text).strip()
        return ""

    def _normalize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._normalize_text(content)
        if isinstance(content, list):
            normalized: List[Any] = []
            for item in content:
                if isinstance(item, dict):
                    normalized.append(
                        {
                            key: item.get(key)
                            for key in ("type", "text", "id", "name", "function")
                            if item.get(key) is not None
                        }
                    )
                else:
                    normalized.append(item)
            return normalized
        return content

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _hash_json(self, data: Any) -> str:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(self._normalize_text(text).encode("utf-8")).hexdigest()

    def _dedupe_anchors(self, anchors: Iterable[SessionAnchor]) -> List[SessionAnchor]:
        best: Dict[str, SessionAnchor] = {}
        for anchor in anchors:
            if not anchor.value:
                continue
            current = best.get(anchor.value)
            if current is None or self._strongest(anchor.strength, current.strength) == anchor.strength:
                best[anchor.value] = anchor
        return list(best.values())
