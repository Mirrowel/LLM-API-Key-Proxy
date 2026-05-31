# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Extraction and injection engine for field-cache rules."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .paths import FieldCachePathError, extract_path, inject_path, parse_path
from .store import FieldCacheStore, InMemoryFieldCacheStore
from .types import FieldCacheContext, FieldCacheRule


@dataclass
class FieldCacheOperation:
    """Summary of one field-cache rule application."""

    rule_name: str
    cache_key: Optional[str]
    matched: int = 0
    changed: bool = False
    hit: bool = False
    skipped: bool = False
    reason: Optional[str] = None
    sample_values: list[Any] = field(default_factory=list)


def _safe_scope_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return digest


def build_cache_key(rule: FieldCacheRule, context: FieldCacheContext) -> Optional[str]:
    """Build a scoped cache key or return None when required scope is absent."""

    parts = [f"rule={rule.name}"]
    for scope in rule.scope:
        value = context.value_for_scope(scope)
        if value is None or value == "":
            if scope == "session" and not rule.allow_missing_session:
                return None
            value = "_none"
        if scope in {"provider", "model"}:
            safe_value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
        else:
            safe_value = _safe_scope_value(value)
        parts.append(f"{scope}={safe_value}")
    return "|".join(parts)


class FieldCacheEngine:
    """Apply field-cache extraction and injection rules.

    The engine preserves provider protocol state; it is not session tracking.
    It defaults to copying payloads before injection so providers can opt into
    mutation explicitly. Turn and tool-call modes skip safely when the requested
    context cannot be inferred rather than silently falling back to `last`.
    """

    def __init__(self, rules: Iterable[FieldCacheRule], store: Optional[FieldCacheStore] = None) -> None:
        self.rules = tuple(rules)
        self.store = store or InMemoryFieldCacheStore()
        self._validate_rules()

    def _validate_rules(self) -> None:
        names: set[str] = set()
        for rule in self.rules:
            if rule.name in names:
                raise ValueError(f"Duplicate field-cache rule name: {rule.name}")
            names.add(rule.name)
            parse_path(rule.path)
            if rule.inject:
                parse_path(rule.inject.path)
            if rule.mode == "per_tool_call" and not rule.metadata.get("tool_call_id_path"):
                raise ValueError("per_tool_call field-cache mode requires metadata.tool_call_id_path")

    async def extract(
        self,
        source: str,
        payload: Any,
        context: FieldCacheContext,
        *,
        transaction_logger: Optional[Any] = None,
    ) -> list[FieldCacheOperation]:
        operations: list[FieldCacheOperation] = []
        rules = self._rules_for_source(source)
        self._trace_summary(transaction_logger, "field_cache_extraction_start", payload, source=source, target=None, rules=rules, operations=operations)
        for rule in rules:
            operation = FieldCacheOperation(rule_name=rule.name, cache_key=build_cache_key(rule, context))
            self._trace(transaction_logger, "before_field_cache_extraction", payload, rule, operation, source=source)
            if not operation.cache_key:
                operation.skipped = True
                operation.reason = "missing_required_scope"
                operations.append(operation)
                self._trace(transaction_logger, "after_field_cache_extraction", payload, rule, operation, source=source)
                continue
            try:
                values = extract_path(payload, rule.path)
                operation.matched = len(values)
                operation.sample_values = _sample_values(values)
                if values or rule.mode in {"last_user_turn", "last_assistant_turn", "per_tool_call"}:
                    operation.changed = await self._store_values(rule, operation.cache_key, values, payload, operation)
            except Exception as exc:
                self._log_error(transaction_logger, "field_cache_extract", exc, payload, rule)
                raise
            operations.append(operation)
            self._trace(transaction_logger, "after_field_cache_extraction", payload, rule, operation, source=source)
        self._trace_summary(transaction_logger, "field_cache_extraction_complete", payload, source=source, target=None, rules=rules, operations=operations)
        return operations

    async def inject(
        self,
        target: str,
        payload: Any,
        context: FieldCacheContext,
        *,
        transaction_logger: Optional[Any] = None,
        mutate: bool = False,
    ) -> tuple[Any, list[FieldCacheOperation]]:
        updated = payload if mutate else deepcopy(payload)
        operations: list[FieldCacheOperation] = []
        rules = self._rules_for_injection(target)
        self._trace_summary(transaction_logger, "field_cache_injection_start", updated, source=None, target=target, rules=rules, operations=operations)
        for rule in rules:
            operation = FieldCacheOperation(rule_name=rule.name, cache_key=build_cache_key(rule, context))
            if not rule.inject:
                continue
            self._trace(transaction_logger, "before_field_cache_injection", updated, rule, operation, target=target)
            if not operation.cache_key:
                operation.skipped = True
                operation.reason = "missing_required_scope"
                operations.append(operation)
                self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                continue
            try:
                cached = await self.store.get(operation.cache_key)
                if cached is None:
                    operation.reason = "cache_miss"
                    operations.append(operation)
                    self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                    continue
                operation.hit = True
                value = self._injection_value(rule, cached, updated, context, operation)
                if operation.skipped:
                    operations.append(operation)
                    self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                    continue
                operation.changed = inject_path(
                    updated,
                    rule.inject.path,
                    value,
                    when_missing_only=rule.inject.when_missing_only,
                    insert=rule.inject.insert,
                )
                operation.sample_values = _sample_values(value if isinstance(value, list) else [value])
            except Exception as exc:
                self._log_error(transaction_logger, "field_cache_inject", exc, updated, rule)
                raise
            operations.append(operation)
            self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
        self._trace_summary(transaction_logger, "field_cache_injection_complete", updated, source=None, target=target, rules=rules, operations=operations)
        return updated, operations

    def _rules_for_source(self, source: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.source == source]

    def _rules_for_injection(self, target: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.inject and rule.inject.target == target]

    async def _store_values(self, rule: FieldCacheRule, cache_key: str, values: list[Any], payload: Any, operation: FieldCacheOperation) -> bool:
        if rule.mode == "all":
            await self._store_append(cache_key, values, ttl_seconds=rule.ttl_seconds)
            return True
        if rule.mode == "last":
            await self._store_set(cache_key, _wrap_cached_value(values[-1]), ttl_seconds=rule.ttl_seconds)
            return True
        if rule.mode in {"last_user_turn", "last_assistant_turn"}:
            role = "user" if rule.mode == "last_user_turn" else "assistant"
            turn_values = _turn_values(rule, payload, role)
            if not turn_values:
                operation.skipped = True
                operation.reason = "turn_context_not_found"
                return False
            operation.matched = len(turn_values)
            operation.sample_values = _sample_values(turn_values)
            await self._store_set(cache_key, _wrap_cached_value(turn_values[-1]), ttl_seconds=rule.ttl_seconds)
            return True
        if rule.mode == "per_tool_call":
            stored = _tool_call_values(rule, payload, values)
            if not stored:
                operation.skipped = True
                operation.reason = "tool_call_id_not_found"
                return False
            operation.matched = len(stored)
            operation.sample_values = _sample_values(list(stored.values()))
            await self._store_set(cache_key, stored, ttl_seconds=rule.ttl_seconds)
            return True
        raise ValueError(f"Unsupported field-cache mode: {rule.mode}")

    async def _store_set(self, cache_key: str, value: Any, *, ttl_seconds: Optional[int]) -> None:
        try:
            await self.store.set(cache_key, value, ttl_seconds=ttl_seconds)
        except TypeError:
            # Preserve compatibility with simple injected stores that implement
            # the original set(key, value) shape. TTL is best-effort there.
            await self.store.set(cache_key, value)

    async def _store_append(self, cache_key: str, values: list[Any], *, ttl_seconds: Optional[int]) -> None:
        try:
            await self.store.append(cache_key, values, ttl_seconds=ttl_seconds)
        except TypeError:
            await self.store.append(cache_key, values)

    def _injection_value(self, rule: FieldCacheRule, cached: Any, payload: Any, context: FieldCacheContext, operation: FieldCacheOperation) -> Any:
        """Select the cached value to inject for the rule's mode.

        Per-tool-call maps require a current tool-call ID so the engine never
        injects an arbitrary provider signature into the wrong tool result.
        """

        if rule.mode == "per_tool_call":
            if not isinstance(cached, dict):
                operation.skipped = True
                operation.reason = "invalid_tool_call_cache"
                return None
            ids = _injection_tool_ids(rule, payload, context)
            if not ids:
                operation.skipped = True
                operation.reason = "tool_call_id_not_found"
                return None
            matches = [_unwrap_cached_value(cached[str(tool_id)]) for tool_id in ids if str(tool_id) in cached]
            if not matches:
                operation.skipped = True
                operation.reason = "tool_call_cache_miss"
                return None
            if rule.inject and rule.inject.as_list:
                return matches
            if len(matches) == 1:
                return matches[0]
            operation.skipped = True
            operation.reason = "ambiguous_tool_call_values"
            return None
        if rule.mode == "all":
            return cached if isinstance(cached, list) else [cached]
        if rule.inject and rule.inject.as_list:
            unwrapped = _unwrap_cached_value(cached)
            return unwrapped if isinstance(unwrapped, list) else [unwrapped]
        return _unwrap_cached_value(cached)

    def _trace(
        self,
        transaction_logger: Optional[Any],
        pass_name: str,
        payload: Any,
        rule: FieldCacheRule,
        operation: FieldCacheOperation,
        **extra_metadata: Any,
    ) -> None:
        if not transaction_logger:
            return
        transaction_logger.log_transform_pass(
            pass_name,
            _payload_shape(payload),
            direction=_trace_direction(pass_name, rule.source, extra_metadata),
            stage="adapter",
            metadata={
                "rule_name": rule.name,
                "source": rule.source,
                "path": rule.path,
                "mode": rule.mode,
                "scope": list(rule.scope),
                "cache_key": operation.cache_key,
                "matched": operation.matched,
                "changed": operation.changed,
                "hit": operation.hit,
                "skipped": operation.skipped,
                "reason": operation.reason,
                # Cached fields can include provider signatures or session keys.
                # Trace only shape/count metadata; keep raw samples out of logs.
                "sample_value_count": len(operation.sample_values),
                "sample_value_types": [type(value).__name__ for value in operation.sample_values[:3]],
                **extra_metadata,
            },
            snapshot=rule.source != "stream_event",
        )

    def _trace_summary(
        self,
        transaction_logger: Optional[Any],
        pass_name: str,
        payload: Any,
        *,
        source: Optional[str],
        target: Optional[str],
        rules: list[FieldCacheRule],
        operations: list[FieldCacheOperation],
    ) -> None:
        """Record cache-pass boundaries even when no individual rule matches."""

        if not transaction_logger:
            return
        transaction_logger.log_transform_pass(
            pass_name,
            _payload_shape(payload),
            direction="request" if target or source == "request" else "response" if source == "response" else "stream" if source == "stream_event" else "metadata",
            stage="adapter",
            metadata={
                "source": source,
                "target": target,
                "rule_count": len(rules),
                "operation_count": len(operations),
                "matched_count": sum(operation.matched for operation in operations),
                "changed_count": sum(1 for operation in operations if operation.changed),
                "hit_count": sum(1 for operation in operations if operation.hit),
                "skipped_count": sum(1 for operation in operations if operation.skipped),
            },
            snapshot=(source != "stream_event"),
        )

    def _log_error(self, transaction_logger: Optional[Any], pass_name: str, error: BaseException, payload: Any, rule: FieldCacheRule) -> None:
        if not transaction_logger:
            return
        transaction_logger.log_transform_error(
            pass_name,
            error,
            payload=_payload_shape(payload),
            stage="adapter",
            metadata={"rule_name": rule.name, "path": rule.path, "mode": rule.mode},
        )


def _last_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def _wrap_cached_value(value: Any) -> dict[str, Any]:
    """Wrap one extracted value so list-valued fields stay intact on injection."""

    return {"__field_cache_value__": True, "value": deepcopy(value)}


def _unwrap_cached_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__field_cache_value__") is True:
        return deepcopy(value.get("value"))
    return _last_value(value)


def _turn_values(rule: FieldCacheRule, payload: Any, role: str) -> list[Any]:
    """Return values from the latest turn matching `role`.

    Rules can provide explicit turn paths for provider-specific payloads. The
    default handles the common `messages[*]` shape used by OpenAI-compatible and
    Responses-like requests.
    """

    container_path = rule.metadata.get("turn_container_path", "messages")
    role_path = rule.metadata.get("turn_role_path", "role")
    value_path = rule.metadata.get("turn_value_path") or _message_relative_path(rule.path, container_path)
    if not value_path:
        return []
    turns = extract_path(payload, str(container_path))
    if len(turns) == 1 and isinstance(turns[0], list):
        turns = turns[0]
    latest: list[Any] = []
    for turn in turns:
        roles = extract_path(turn, str(role_path))
        if roles and str(roles[0]) == role:
            values = extract_path(turn, str(value_path))
            if values:
                latest = values
    return latest


def _message_relative_path(path: str, container_path: str) -> Optional[str]:
    prefixes = (f"{container_path}.*.", f"{container_path}[-1].")
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return None


def _tool_call_values(rule: FieldCacheRule, payload: Any, values: list[Any]) -> dict[str, Any]:
    """Correlate cached values to provider tool-call IDs."""

    container_path = rule.metadata.get("tool_container_path")
    tool_id_path = rule.metadata.get("tool_call_id_path")
    tool_value_path = rule.metadata.get("tool_value_path")
    stored: dict[str, Any] = {}
    if container_path and tool_value_path:
        containers = extract_path(payload, str(container_path))
        if len(containers) == 1 and isinstance(containers[0], list):
            containers = containers[0]
        for container in containers:
            tool_ids = extract_path(container, str(tool_id_path)) if tool_id_path else []
            tool_values = extract_path(container, str(tool_value_path))
            if tool_ids and tool_values:
                stored[str(tool_ids[0])] = _wrap_cached_value(tool_values[-1])
        return stored
    for value in values:
        tool_ids = extract_path(value, str(tool_id_path)) if tool_id_path else []
        if tool_ids:
            stored[str(tool_ids[0])] = _wrap_cached_value(value)
    return stored


def _injection_tool_ids(rule: FieldCacheRule, payload: Any, context: FieldCacheContext) -> list[str]:
    configured = context.metadata.get("tool_call_id")
    if configured:
        return [str(configured)]
    inject_path_value = rule.metadata.get("inject_tool_call_id_path")
    if inject_path_value:
        return [str(value) for value in extract_path(payload, str(inject_path_value))]
    return []


def _trace_direction(pass_name: str, source: str, metadata: dict[str, Any]) -> str:
    if "injection" in pass_name:
        target = metadata.get("target")
        if target in {"stream_event", "unified_stream_event"}:
            return "stream"
        if target in {"request", "unified_request", "metadata"}:
            return "request"
        return "response"
    if source in {"stream_event", "unified_stream_event"}:
        return "stream"
    if source in {"request", "unified_request"}:
        return "request"
    return "response"


def _sample_values(values: list[Any], *, max_items: int = 3, max_text: int = 500) -> list[Any]:
    samples: list[Any] = []
    for value in values[:max_items]:
        if isinstance(value, str) and len(value) > max_text:
            samples.append(f"{value[:max_text]}...<truncated {len(value) - max_text} chars>")
        else:
            samples.append(value)
    return samples


def _payload_shape(payload: Any) -> dict[str, Any]:
    """Return non-sensitive payload shape metadata for cache traces.

    Cache rules often target provider signatures, session IDs, and other opaque
    state. Logging full payloads would expose exactly the fields the cache is
    designed to preserve, so field-cache traces record structure only.
    """

    if isinstance(payload, dict):
        return {"payload_type": "dict", "keys": sorted(str(key) for key in payload.keys())[:20]}
    if isinstance(payload, list):
        return {"payload_type": "list", "length": len(payload)}
    return {"payload_type": type(payload).__name__}
