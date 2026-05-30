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
import os
import re
import threading
import time
import uuid
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
    owns credential selection. This lets providers expose stable native markers
    or provider-specific request structure while keeping sticky policy centralized.
    """

    strong_anchors: List[str] = field(default_factory=list)
    medium_anchors: List[str] = field(default_factory=list)
    weak_anchors: List[str] = field(default_factory=list)
    affinity_key: Optional[str] = None
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
        self._last_save_attempt = 0.0
        self._writer: Optional[ResilientStateWriter] = None
        self._lock = threading.RLock()
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
    ) -> SessionInference:
        """Infer live session and deterministic affinity from a request payload."""
        with self._lock:
            return self._infer_session_locked(
                request_data,
                provider=provider,
                model=model,
                scope_key=scope_key,
                hints=hints,
            )

    def _infer_session_locked(
        self,
        request_data: Dict[str, Any],
        *,
        provider: Optional[str],
        model: Optional[str],
        scope_key: Optional[str],
        hints: Optional[Any],
    ) -> SessionInference:
        now = time.time()
        self._prune(now)

        hints = self._coerce_hints(hints)
        namespace = self._namespace(
            provider,
            model,
            scope_key=scope_key,
            session_scope=hints.session_scope if hints else None,
        )
        anchors = self._build_anchors(request_data, namespace, hints)
        if not anchors:
            return SessionInference(session_id=None, tracking_namespace=namespace)

        match = self._best_match(anchors, namespace, now)
        possible_compaction = self._looks_like_compaction(request_data)

        # Compaction is useful lineage information but should not hard-stick the
        # new compacted context unless a genuinely strong anchor survived.
        if match and match.is_sticky_match and not (
            possible_compaction and match.strong_matches == 0
        ):
            state = self._refresh_and_bridge(
                match.session_id,
                namespace,
                anchors,
                now,
                affinity_key=self._affinity_from_anchors(anchors, namespace),
            )
            return SessionInference(
                session_id=state.session_id,
                affinity_key=state.affinity_key,
                confidence=match.confidence,
                match_score=match.score,
                possible_compaction=possible_compaction,
                tracking_namespace=namespace,
            )

        parent_id = match.session_id if match and possible_compaction else None
        session_id = str(uuid.uuid4())
        state = self._create_session(
            session_id,
            namespace,
            anchors,
            now,
            affinity_key=self._affinity_from_anchors(anchors, namespace),
        )
        if parent_id:
            lib_logger.info(
                "Session tracker: possible compacted descendant %s -> %s for %s",
                parent_id,
                session_id,
                namespace,
            )
        return SessionInference(
            session_id=state.session_id,
            affinity_key=state.affinity_key,
            confidence="weak" if match else "none",
            match_score=match.score if match else 0,
            possible_compaction=possible_compaction,
            lineage_parent_session_id=parent_id,
            tracking_namespace=namespace,
        )

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
        with self._lock:
            if not session_id or response is None or session_id not in self._sessions:
                return
            now = time.time()
            namespace = tracking_namespace or self._namespace(provider, model, scope_key=scope_key)
            anchors = self._anchors_from_response(response, namespace)
            if anchors:
                self._refresh_and_bridge(session_id, namespace, anchors, now)

    def flush(self) -> None:
        """Force persistence of dirty state when optional disk storage is enabled."""
        with self._lock:
            self._save(force=True)

    def _create_session(
        self,
        session_id: str,
        namespace: str,
        anchors: List[SessionAnchor],
        now: float,
        *,
        affinity_key: Optional[str],
    ) -> _SessionState:
        state = _SessionState(
            session_id=session_id,
            namespace=namespace,
            expires_at=now + self.ttl_seconds,
            affinity_key=affinity_key,
            last_seen=now,
        )
        self._sessions[session_id] = state
        self._refresh_and_bridge(session_id, namespace, anchors, now, affinity_key=affinity_key)
        return state

    def _refresh_and_bridge(
        self,
        session_id: str,
        namespace: str,
        anchors: List[SessionAnchor],
        now: float,
        *,
        affinity_key: Optional[str] = None,
    ) -> _SessionState:
        expires_at = now + self.ttl_seconds
        state = self._sessions.setdefault(
            session_id,
            _SessionState(session_id=session_id, namespace=namespace, expires_at=expires_at),
        )
        state.namespace = namespace
        state.expires_at = expires_at
        state.last_seen = now
        if affinity_key and not state.affinity_key:
            state.affinity_key = affinity_key

        for anchor in anchors:
            state.anchors.add(anchor.value)
            self._anchors[anchor.value] = _AnchorRecord(
                session_id=session_id,
                namespace=namespace,
                strength=anchor.strength,
                source=anchor.source,
                group=anchor.group,
                expires_at=expires_at,
                last_seen=now,
            )

        self._trim_session_anchors(state)
        self._trim_global_anchors()
        self._mark_dirty()
        self._save()
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
            strength = self._strongest(anchor.strength, record.strength)
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
                if anchor.source == "response" or record.source == "response":
                    candidate.response_matches += 1
            else:
                candidate.score += self._WEAK_SCORE
                candidate.weak_matches += 1

        if not candidates:
            return None
        return max(candidates.values(), key=lambda item: item.score)

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
                affinity_key=hints.get("affinity_key"),
                session_scope=hints.get("session_scope"),
            )
        return None

    def _build_anchors(
        self,
        request_data: Dict[str, Any],
        namespace: str,
        hints: Optional[Any],
    ) -> List[SessionAnchor]:
        anchors: List[SessionAnchor] = []
        anchors.extend(self._anchors_from_provider_hints(hints, namespace))
        anchors.extend(self._anchors_from_explicit_ids(request_data, namespace))

        messages = request_data.get("messages") or []
        if isinstance(messages, list) and messages:
            anchors.extend(self._anchors_from_messages(messages, namespace))

        return self._dedupe_anchors(anchors)

    def _anchors_from_provider_hints(
        self,
        hints: Optional[SessionTrackingHints],
        namespace: str,
    ) -> List[SessionAnchor]:
        if not hints:
            return []
        anchors: List[SessionAnchor] = []
        for strength, attr in (
            ("strong", "strong_anchors"),
            ("medium", "medium_anchors"),
            ("weak", "weak_anchors"),
        ):
            for value in getattr(hints, attr, []) or []:
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"provider:{value}"),
                        strength,
                        source="provider",
                        group=f"provider:{value}",
                    )
                )
        affinity_key = getattr(hints, "affinity_key", None)
        if affinity_key:
            anchors.append(
                SessionAnchor(
                    self._scoped(namespace, f"provider_affinity:{affinity_key}"),
                    "strong",
                    source="provider",
                    group="provider_affinity",
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
                strength = "strong" if key in self.trusted_explicit_fields else "weak"
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"explicit:{key}:{value}"),
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
    ) -> List[SessionAnchor]:
        anchors: List[SessionAnchor] = []
        normalized_messages: List[Dict[str, Any]] = []
        tool_ids: List[str] = []
        first_user_text: Optional[str] = None

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
                tool_ids.append(tool_id)
                normalized["tool_call_id"] = tool_id
                anchors.append(
                    SessionAnchor(
                        self._scoped(namespace, f"tool:{tool_id}"),
                        "strong",
                        source="tool",
                        group=f"tool:{tool_id}",
                    )
                )

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                call_ids: List[str] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = tool_call.get("id")
                    if call_id:
                        call_id = str(call_id)
                        call_ids.append(call_id)
                        tool_ids.append(call_id)
                        anchors.append(
                            SessionAnchor(
                                self._scoped(namespace, f"tool:{call_id}"),
                                "strong",
                                source="tool",
                                group=f"tool:{call_id}",
                            )
                        )
                if call_ids:
                    normalized["tool_calls"] = call_ids

            if text:
                normalized_text = self._normalize_text(text)
                if first_user_text is None and role == "user":
                    first_user_text = normalized_text
                if self._is_substantial_text(normalized_text):
                    anchors.append(
                        SessionAnchor(
                            self._scoped(namespace, f"message:{role}:{self._hash_text(normalized_text)}"),
                            "medium",
                            source=source,
                            group=f"{source}:{index}",
                        )
                    )
                    for chunk_hash in self._content_chunk_hashes(normalized_text):
                        anchors.append(
                            SessionAnchor(
                                self._scoped(namespace, f"chunk:{chunk_hash}"),
                                "medium",
                                source=source,
                                group=f"{source}:{index}",
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
                    "strong",
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
        messages: List[Dict[str, Any]] = []
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or choice.get("delta")
            if isinstance(message, dict):
                response_message = dict(message)
                response_message.setdefault("role", "assistant")
                messages.append(response_message)
        return self._anchors_from_messages(messages, namespace, source="response") if messages else []

    def _affinity_from_anchors(
        self,
        anchors: List[SessionAnchor],
        namespace: str,
    ) -> Optional[str]:
        strong = sorted(anchor.value for anchor in anchors if anchor.strength == "strong")
        if strong:
            return self._scoped(namespace, "affinity:" + self._hash_json(strong[:4]))
        medium_anchors = [anchor for anchor in anchors if anchor.strength == "medium"]
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

    def _looks_like_compaction(self, request_data: Dict[str, Any]) -> bool:
        messages = request_data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return False
        summary_texts: List[str] = []
        for message in messages[:2]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role not in {"system", "developer"}:
                continue
            summary_texts.append(self._extract_text(message.get("content")))
        joined = "\n".join(summary_texts)
        lowered = joined.lower()
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
            key=lambda value: self._anchors.get(value, _AnchorRecord("", "", "weak", "", None, 0, 0)).last_seen,
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
        for value, record in sorted(self._anchors.items(), key=lambda item: item[1].last_seen)[:overage]:
            state = self._sessions.get(record.session_id)
            if state:
                state.anchors.discard(value)
            del self._anchors[value]

    def _prune(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        expired_sessions = [key for key, state in self._sessions.items() if state.expires_at <= now]
        for session_id in expired_sessions:
            state = self._sessions.pop(session_id)
            for anchor in list(state.anchors):
                record = self._anchors.get(anchor)
                if record and record.session_id == session_id:
                    del self._anchors[anchor]
        expired_anchors = [key for key, record in self._anchors.items() if record.expires_at <= now]
        for anchor in expired_anchors:
            del self._anchors[anchor]

    def _load(self) -> None:
        if not self.persistence_path:
            return
        data = safe_read_json(self.persistence_path, lib_logger)
        if not isinstance(data, dict):
            return
        now = time.time()
        if "sessions" not in data and "anchors" not in data:
            lib_logger.info(
                "Ignoring legacy session_stickiness.json format; session persistence will rebuild in memory."
            )
            return
        sessions = data.get("sessions", {})
        anchors = data.get("anchors", {})
        for session_id, payload in sessions.items():
            if not isinstance(payload, dict):
                continue
            expires_at = float(payload.get("expires_at", 0.0))
            if expires_at <= now:
                continue
            self._sessions[session_id] = _SessionState(
                session_id=session_id,
                namespace=str(payload.get("namespace") or "global"),
                expires_at=expires_at,
                affinity_key=payload.get("affinity_key"),
                anchors=set(payload.get("anchors") or []),
                last_seen=float(payload.get("last_seen", now)),
            )
        for value, payload in anchors.items():
            if not isinstance(payload, dict):
                continue
            session_id = payload.get("session_id")
            expires_at = float(payload.get("expires_at", 0.0))
            if not session_id or expires_at <= now:
                continue
            namespace = str(payload.get("namespace") or "global")
            self._anchors[value] = _AnchorRecord(
                session_id=session_id,
                namespace=namespace,
                strength=str(payload.get("strength") or "medium"),
                source=str(payload.get("source") or "generic"),
                group=payload.get("group"),
                expires_at=expires_at,
                last_seen=float(payload.get("last_seen", now)),
            )
            self._sessions.setdefault(
                session_id,
                _SessionState(session_id=session_id, namespace=namespace, expires_at=expires_at),
            ).anchors.add(value)

    def _save(self, *, force: bool = False) -> None:
        if not self.persist_to_disk or not self.persistence_path or not self._dirty:
            return
        now = time.time()
        if not force and now - self._last_save_attempt < self.persistence_flush_interval_seconds:
            return
        self._last_save_attempt = now
        payload = {
            "sessions": {
                session_id: {
                    "namespace": state.namespace,
                    "expires_at": state.expires_at,
                    "affinity_key": state.affinity_key,
                    "anchors": sorted(state.anchors),
                    "last_seen": state.last_seen,
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
        if self._writer.write(payload):
            self._dirty = False

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _namespace(
        self,
        provider: Optional[str],
        model: Optional[str],
        *,
        scope_key: Optional[str] = None,
        session_scope: Optional[str] = None,
    ) -> str:
        # The resolved usage/classifier scope is part of the namespace so sticky
        # evidence never leaks between private/classifier-scoped credential pools.
        allowed_scope = scope_key or "default"
        provider_key = provider or "global"
        model_key = session_scope or model or "default"
        return f"scope:{allowed_scope}:provider:{provider_key}:model:{model_key}"

    def _trusted_fields_from_env(self) -> List[str]:
        raw = os.getenv("TRUSTED_SESSION_ID_FIELDS", "")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _scoped(self, namespace: str, value: str) -> str:
        return f"{namespace}:{value}"

    def _strongest(self, left: str, right: str) -> str:
        order = {"weak": 0, "medium": 1, "strong": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right

    def _is_substantial_text(self, text: str) -> bool:
        return len(text) >= 24 and len(text.split()) >= 4

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
