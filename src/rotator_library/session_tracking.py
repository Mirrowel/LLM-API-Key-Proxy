# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Best-effort session inference for sticky credential routing.

This stays in-memory by default. It keeps a short-lived mapping from
conversation fingerprints to an inferred session id so sequential routing
can keep using the same credential for the same conversation.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class _SessionRecord:
    session_id: str
    expires_at: float


class SessionTracker:
    """In-memory session inference with TTL-based stickiness."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        persist_to_disk: bool = False,
        persistence_path: Optional[Path] = None,
    ) -> None:
        self.ttl_seconds = max(1, ttl_seconds)
        self.persist_to_disk = persist_to_disk
        self.persistence_path = persistence_path
        self._records: Dict[str, _SessionRecord] = {}
        if self.persist_to_disk:
            self._load()

    def infer_session_id(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Infer a stable session id for a request payload, if possible."""
        now = time.time()
        self._prune(now)

        strong_fingerprints, weak_fingerprints = self._build_fingerprints(request_data)
        fingerprints = self._dedupe(strong_fingerprints + weak_fingerprints)
        if not fingerprints:
            return None

        for fingerprint in strong_fingerprints:
            record = self._records.get(fingerprint)
            if record and record.expires_at > now:
                return self._refresh_and_bridge(
                    record.session_id,
                    fingerprints,
                    now,
                )

        for fingerprint in weak_fingerprints:
            record = self._records.get(fingerprint)
            if record and record.expires_at > now:
                record.expires_at = now + self.ttl_seconds
                self._save()
                return record.session_id

        session_id = str(uuid.uuid4())
        expires_at = now + self.ttl_seconds
        for fingerprint in fingerprints:
            self._records[fingerprint] = _SessionRecord(
                session_id=session_id,
                expires_at=expires_at,
            )
        self._save()
        return session_id

    def _refresh_and_bridge(
        self,
        session_id: str,
        fingerprints: List[str],
        now: float,
    ) -> str:
        expires_at = now + self.ttl_seconds
        for fingerprint in fingerprints:
            self._records[fingerprint] = _SessionRecord(
                session_id=session_id,
                expires_at=expires_at,
            )
        self._save()
        return session_id

    def _build_fingerprints(self, request_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        strong_fingerprints: List[str] = []
        weak_fingerprints: List[str] = []

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
                strong_fingerprints.append(f"explicit:{key}:{value}")

        messages = request_data.get("messages") or []
        if isinstance(messages, list) and messages:
            strong, weak = self._fingerprints_from_messages(messages)
            strong_fingerprints.extend(strong)
            weak_fingerprints.extend(weak)

        return self._dedupe(strong_fingerprints), self._dedupe(weak_fingerprints)

    def _fingerprints_from_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        strong_fingerprints: List[str] = []
        weak_fingerprints: List[str] = []

        tool_ids: List[str] = []
        normalized_messages: List[Dict[str, Any]] = []
        first_user_text: Optional[str] = None

        for message in messages:
            role = str(message.get("role", ""))
            normalized: Dict[str, Any] = {"role": role}

            content = message.get("content")
            normalized["content"] = self._normalize_content(content)

            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                tool_ids.append(str(tool_call_id))
                normalized["tool_call_id"] = str(tool_call_id)

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                call_ids: List[str] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = tool_call.get("id")
                    if call_id:
                        call_ids.append(str(call_id))
                        tool_ids.append(str(call_id))
                if call_ids:
                    normalized["tool_calls"] = call_ids

            normalized_messages.append(normalized)

            if first_user_text is None and role == "user":
                first_user_text = self._extract_text(content)

        if tool_ids:
            strong_fingerprints.append("tool_ids:" + self._hash_json(tool_ids))

        if normalized_messages:
            strong_fingerprints.append("conversation:" + self._hash_json(normalized_messages[:6]))

        if first_user_text:
            weak_fingerprints.append("first_user:" + self._hash_text(first_user_text))

        return strong_fingerprints, weak_fingerprints

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return str(text).strip()
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        return text
            return ""
        if isinstance(content, dict):
            text = content.get("text")
            if text:
                return str(text).strip()
        return ""

    def _normalize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content.strip()
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

    def _hash_json(self, data: Any) -> str:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _prune(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        expired = [key for key, record in self._records.items() if record.expires_at <= now]
        for key in expired:
            del self._records[key]

    def _load(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
        except Exception:
            return
        now = time.time()
        for fingerprint, payload in data.items():
            session_id = payload.get("session_id")
            expires_at = float(payload.get("expires_at", 0.0))
            if session_id and expires_at > now:
                self._records[fingerprint] = _SessionRecord(session_id=session_id, expires_at=expires_at)

    def _save(self) -> None:
        if not self.persist_to_disk or not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                fingerprint: {
                    "session_id": record.session_id,
                    "expires_at": record.expires_at,
                }
                for fingerprint, record in self._records.items()
            }
            self.persistence_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            return
