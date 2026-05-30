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

    The engine is isolated from request execution in Phase 3. It defaults to
    copying payloads before injection so tests and future providers can reason
    about mutations explicitly.
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
        for rule in self._rules_for_source(source):
            operation = FieldCacheOperation(rule_name=rule.name, cache_key=build_cache_key(rule, context))
            if not operation.cache_key:
                operation.skipped = True
                operation.reason = "missing_required_scope"
                operations.append(operation)
                self._trace(transaction_logger, "after_field_cache_extraction", payload, rule, operation, source=source)
                continue
            try:
                values = extract_path(payload, rule.path)
                operation.matched = len(values)
                operation.sample_values = values[:3]
                if values:
                    await self._store_values(rule, operation.cache_key, values)
                    operation.changed = True
            except Exception as exc:
                self._log_error(transaction_logger, "field_cache_extract", exc, payload, rule)
                raise
            operations.append(operation)
            self._trace(transaction_logger, "after_field_cache_extraction", payload, rule, operation, source=source)
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
        for rule in self._rules_for_injection(target):
            operation = FieldCacheOperation(rule_name=rule.name, cache_key=build_cache_key(rule, context))
            if not rule.inject:
                continue
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
                value = cached if rule.inject.as_list or rule.mode == "all" else _last_value(cached)
                operation.changed = inject_path(
                    updated,
                    rule.inject.path,
                    value,
                    when_missing_only=rule.inject.when_missing_only,
                )
                operation.sample_values = value if isinstance(value, list) else [value]
            except Exception as exc:
                self._log_error(transaction_logger, "field_cache_inject", exc, updated, rule)
                raise
            operations.append(operation)
            self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
        return updated, operations

    def _rules_for_source(self, source: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.source == source]

    def _rules_for_injection(self, target: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.inject and rule.inject.target == target]

    async def _store_values(self, rule: FieldCacheRule, cache_key: str, values: list[Any]) -> None:
        if rule.mode == "all":
            await self.store.append(cache_key, values)
            return
        if rule.mode in {"last", "last_user_turn", "last_assistant_turn"}:
            await self.store.set(cache_key, values[-1])
            return
        if rule.mode == "per_tool_call":
            tool_path = rule.metadata.get("tool_call_id_path")
            stored = {}
            for value in values:
                tool_ids = extract_path(value, tool_path) if tool_path else []
                if tool_ids:
                    stored[str(tool_ids[0])] = value
            await self.store.set(cache_key, stored)
            return
        raise ValueError(f"Unsupported field-cache mode: {rule.mode}")

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
            payload,
            direction="stream" if rule.source == "stream_event" else "request" if "injection" in pass_name else "response",
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
                "sample_values": operation.sample_values[:3],
                **extra_metadata,
            },
            snapshot=rule.source != "stream_event",
        )

    def _log_error(self, transaction_logger: Optional[Any], pass_name: str, error: BaseException, payload: Any, rule: FieldCacheRule) -> None:
        if not transaction_logger:
            return
        transaction_logger.log_transform_error(
            pass_name,
            error,
            payload=payload,
            stage="adapter",
            metadata={"rule_name": rule.name, "path": rule.path, "mode": rule.mode},
        )


def _last_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[-1] if value else None
    return value
