# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Extraction and injection engine for field-cache rules."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .paths import FieldCachePathError, extract_path, inject_path, parse_path
from .store import (
    FieldCacheStore,
    InMemoryFieldCacheStore,
    _bounded_set_value,
)
from .types import FieldCacheContext, FieldCacheRule


_LOGGER = logging.getLogger("rotator_library.field_cache")


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
    # Transport profile (W-PROF provenance): rules are identity-normalized
    # to the bare provider, but WHICH profile served the request stays
    # visible on every operation and trace.
    profile: Optional[str] = None


def _safe_scope_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_model(provider: Optional[str], model: str) -> str:
    """Canonicalize the model scope dimension to the stripped form.

    Provider and model are separate scope dimensions, so a prefixed model
    (``openai/gpt-4``) and its stripped form (``gpt-4``) are the SAME cache
    identity. Strip the provider prefix only when the first ``/`` segment
    equals the provider; nested model ids (``openrouter/meta/llama``) keep
    their slashes.
    """

    if not provider or not model:
        return model
    prefix = f"{provider}/"
    if model.startswith(prefix) and len(model) > len(prefix):
        return model[len(prefix):]
    return model


def _unsupported_store_keyword(error: TypeError) -> bool:
    message = str(error).lower()
    return "unexpected keyword" in message or "keyword argument" in message


def _shared_cache_signature(rule: FieldCacheRule) -> tuple[Any, ...]:
    """Return settings that must agree for source-specific cache counterparts."""

    injection = rule.inject
    behavior_metadata = tuple(
        (key, repr(rule.metadata.get(key)))
        for key in (
            "provider_continuation",
            "tool_call_id_path",
            "tool_container_path",
            "tool_value_path",
            "inject_tool_call_id_path",
            "turn_container_path",
            "turn_role_path",
            "turn_value_path",
        )
    )
    return (
        rule.mode,
        rule.scope,
        rule.ttl_seconds,
        rule.allow_missing_session,
        rule.max_values,
        rule.max_bytes,
        injection.target if injection else None,
        injection.path if injection else None,
        injection.when_missing_only if injection else None,
        injection.insert if injection else None,
        injection.as_list if injection else None,
        behavior_metadata,
    )


def build_cache_key(rule: FieldCacheRule, context: FieldCacheContext) -> Optional[str]:
    """Build a scoped cache key or return None when required scope is absent.

    D11: provider+model are the required identity; credential and session
    are optional refinements — a missing optional dimension participates as
    ``_none`` instead of disabling the rule. Provider/model absence still
    disables (identity is non-negotiable). G3 floor hardening: the
    provider+model terms are ALWAYS part of the key even when a raw plugin
    rule declares a narrower scope — a ``scope=("session",)`` rule must
    never share one key across providers (that reopens the cross-provider
    leak the compatibility classes exist to prevent).
    """

    parts = [f"rule={_safe_scope_value(rule.cache_key or rule.name)}"]
    provider_value = context.value_for_scope("provider")
    model_value = context.value_for_scope("model")
    if not provider_value or not model_value:
        return None
    parts.append(f"provider={_safe_scope_value(provider_value)}")
    parts.append(f"model={_safe_scope_value(_canonical_model(provider_value, model_value))}")
    floor = {"rule", "provider", "model"}
    for scope in rule.scope:
        if scope in floor:
            continue
        value = context.value_for_scope(scope)
        if value is None or value == "":
            if scope == "session":
                # Continuation state binds strict session ALWAYS — the
                # lenient allow_missing_session flag never widens it
                # (continuation under session=_none would leak across
                # conversations). Ordinary rules pool at _none per D11:
                # session is an optional refinement, never a disabler.
                if rule.metadata.get("provider_continuation") is True:
                    return None
            if scope == "classifier":
                # The classifier is the multi-user isolation seed (D17):
                # a rule scoped to it never pools across unknown classifiers.
                return None
            value = "_none"
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
        shared_keys: dict[str, tuple[Any, ...]] = {}
        for rule in self.rules:
            if rule.name in names:
                raise ValueError(f"Duplicate field-cache rule name: {rule.name}")
            names.add(rule.name)
            parse_path(rule.path)
            if rule.inject:
                parse_path(rule.inject.path)
            if rule.cache_key:
                signature = _shared_cache_signature(rule)
                previous = shared_keys.get(rule.cache_key)
                if previous is not None and previous != signature:
                    raise ValueError(
                        f"Field-cache rules sharing cache_key {rule.cache_key!r} must use identical mode, scope, TTL, injection, and correlation behavior"
                    )
                shared_keys[rule.cache_key] = signature

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
            operation = FieldCacheOperation(
                rule_name=rule.name,
                cache_key=build_cache_key(rule, context),
                profile=(context.metadata or {}).get("execution_profile")
                if isinstance(getattr(context, "metadata", None), dict)
                else None,
            )
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
                if values or rule.metadata.get("turn_value_path") or rule.metadata.get("turn_container_path"):
                    operation.changed = await self._store_values(rule, operation.cache_key, values, payload, operation)
            except Exception as exc:
                self._log_error(transaction_logger, "field_cache_extract", exc, payload, rule)
                if rule.critical:
                    raise
                self._contain_rule_error(rule, operation, exc)
                operations.append(operation)
                self._trace(transaction_logger, "after_field_cache_extraction", payload, rule, operation, source=source)
                continue
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
            operation = FieldCacheOperation(
                rule_name=rule.name,
                cache_key=build_cache_key(rule, context),
                profile=(context.metadata or {}).get("execution_profile")
                if isinstance(getattr(context, "metadata", None), dict)
                else None,
            )
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
                provenance_note: Optional[dict[str, Any]] = None
                if cached is None:
                    # D12 portable inheritance: on a miss, walk declared
                    # compatibility-group siblings (bound fields never walk).
                    sibling_hit = await self._lookup_compatible_sibling(rule, context)
                    if sibling_hit is not None:
                        cached, provenance_note = sibling_hit
                if cached is None:
                    operation.reason = "cache_miss"
                    operations.append(operation)
                    self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                    continue
                operation.hit = True
                if provenance_note is None:
                    provenance_note = {}
                # Optional-dimension transparency applies to direct AND
                # inherited hits (an inheritance through an unknown
                # credential/session is still visible).
                none_dimensions = [
                    scope
                    for scope in rule.scope
                    if scope in ("credential", "session")
                    and (context.value_for_scope(scope) in (None, ""))
                ]
                if none_dimensions:
                    provenance_note["none_dimensions"] = none_dimensions
                occurrence_plan = _occurrence_injection_plan(rule, updated)
                if occurrence_plan is not None:
                    changed, samples = self._apply_occurrence_injection(
                        rule,
                        cached,
                        updated,
                        context,
                        operation,
                        *occurrence_plan,
                    )
                    operation.changed = changed
                    operation.sample_values = samples
                else:
                    value = self._injection_value(rule, cached, updated, context, operation)
                    if operation.skipped:
                        operations.append(operation)
                        self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                        continue
                    transform_name = rule.metadata.get("transform")
                    if transform_name:
                        from ..protocols.transforms import apply_transform

                        if isinstance(value, list):
                            value = [apply_transform(str(transform_name), item) for item in value]
                            value = [item for item in value if item is not None]
                        else:
                            value = apply_transform(str(transform_name), value)
                            if value is None:
                                operation.reason = "transform_produced_nothing"
                                operations.append(operation)
                                self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                                continue
                    if provenance_note and provenance_note.get("inherited_from"):
                        inherited = "inherited_from_compatible_model"
                        none_dimensions = provenance_note.get("none_dimensions")
                        operation.reason = (
                            inherited + ";optional_scope_none:" + "+".join(none_dimensions)
                            if none_dimensions
                            else inherited
                        )
                    elif provenance_note and provenance_note.get("none_dimensions"):
                        operation.reason = "optional_scope_none:" + "+".join(provenance_note["none_dimensions"])
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
                if rule.critical:
                    raise
                self._contain_rule_error(rule, operation, exc)
                operations.append(operation)
                self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
                continue
            operations.append(operation)
            self._trace(transaction_logger, "after_field_cache_injection", updated, rule, operation, target=target)
        self._trace_summary(transaction_logger, "field_cache_injection_complete", updated, source=None, target=target, rules=rules, operations=operations)
        return updated, operations

    def _rules_for_source(self, source: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.source == source]

    async def _lookup_compatible_sibling(
        self,
        rule: FieldCacheRule,
        context: FieldCacheContext,
    ) -> Optional[tuple[Any, dict[str, Any]]]:
        """Portable-field inheritance across compatibility-group siblings.

        Builds sibling cache keys with the same scope dimensions; the first
        hit returns (value, provenance). Bound rules and rules without a
        model in scope never inherit (identity match only).
        """

        if rule.metadata.get("compatibility") != "portable":
            return None
        if not context.provider or not context.model or "model" not in rule.scope:
            return None
        from .compat import ModelRef, get_compatibility_registry

        registry = get_compatibility_registry()
        target_ref = ModelRef(provider=context.provider, model=context.model)
        for sibling in registry.siblings(target_ref, field_class="portable"):
            sibling_context = FieldCacheContext(
                provider=sibling.provider,
                model=sibling.model,
                credential_id=context.credential_id,
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                classifier=context.classifier,
                metadata=dict(context.metadata),
            )
            sibling_key = build_cache_key(rule, sibling_context)
            if not sibling_key:
                continue
            cached = await self.store.get(sibling_key)
            if cached is not None:
                return cached, {"inherited_from": sibling.key}
        return None

    def _rules_for_injection(self, target: str) -> list[FieldCacheRule]:
        return [rule for rule in self.rules if rule.enabled and rule.inject and rule.inject.target == target]

    async def _store_values(self, rule: FieldCacheRule, cache_key: str, values: list[Any], payload: Any, operation: FieldCacheOperation) -> bool:
        """Store every extracted occurrence under its correlation keys.

        Correlation keys: the occurrence's tool-call ids (primary) and the
        sha256 of its exact content bytes (secondary). Entries merge with the
        existing map; recently written keys move to the end so dict order is
        recency order (the mode selects values at injection time, never here).
        """

        entries = _extraction_entries(rule, payload, values)
        if not entries:
            if rule.metadata.get("turn_container_path") or rule.metadata.get("turn_value_path"):
                operation.skipped = True
                operation.reason = "turn_context_not_found"
            return False
        operation.matched = len(entries)
        current = await self.store.get(cache_key)
        merged: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
        for key, wrapped in entries:
            merged.pop(key, None)
            merged[key] = wrapped
        operation.sample_values = _sample_values([_unwrap_cached_value(value) for value in merged.values()])
        await self._store_set(
            cache_key,
            merged,
            ttl_seconds=rule.ttl_seconds,
            max_values=rule.max_values,
            max_bytes=rule.max_bytes,
            trim_collections=True,
        )
        return True

    async def _store_set(
        self,
        cache_key: str,
        value: Any,
        *,
        ttl_seconds: Optional[int],
        max_values: Optional[int],
        max_bytes: Optional[int],
        trim_collections: bool,
    ) -> None:
        bounded = _bounded_set_value(
            value,
            max_values=max_values,
            max_bytes=max_bytes,
            trim_collections=trim_collections,
        )
        try:
            await self.store.set(cache_key, bounded, ttl_seconds=ttl_seconds)
        except TypeError as exc:
            if not _unsupported_store_keyword(exc):
                raise
            # Preserve compatibility with simple injected stores that implement
            # the original set(key, value) shape. TTL is best-effort there.
            await self.store.set(cache_key, bounded)

    def _apply_occurrence_injection(
        self,
        rule: FieldCacheRule,
        cached: Any,
        payload: Any,
        context: FieldCacheContext,
        operation: FieldCacheOperation,
        container_path: str,
        role_path: str,
        content_path: str,
        relative_path: str,
    ) -> tuple[bool, list[Any]]:
        """Inject one correlated value per in-scope occurrence.

        Occurrences are assistant (or role-less) messages inside the mode's
        scoped turn regions. Each occurrence correlates by its tool-call ids
        first, then its content sha, else the rule's placeholder (with a loud
        warning); unresolvable occurrences are skipped with a traced reason
        and never raise.
        """

        entries = cached if isinstance(cached, dict) else {}
        items = _container_items(payload, container_path)
        regions = _turn_region_indexes(items, role_path, content_path)
        scoped = _scoped_region_ids(regions, rule)
        if scoped is None:
            operation.skipped = True
            operation.reason = "turn_context_not_found"
            return False, []
        tool_id_path = rule.metadata.get("tool_call_id_path")
        transform_name = rule.metadata.get("transform")
        changed = False
        samples: list[Any] = []
        warned = False
        matched = 0
        skipped_occurrences = 0
        for index, item in enumerate(items):
            if regions[index] not in scoped or not isinstance(item, dict):
                continue
            if _message_role(item, role_path) not in ("", "assistant"):
                continue
            message_obj = _message_object(item, role_path) or item
            keys = [str(value) for value in extract_path(message_obj, str(tool_id_path))] if tool_id_path else []
            content_sha = _message_content_sha(message_obj, content_path)
            matches = [_unwrap_cached_value(entries[key]) for key in keys if key in entries]
            value: Any = _MISSING
            if matches:
                if rule.inject and rule.inject.as_list:
                    value = matches
                elif len(matches) == 1:
                    value = matches[0]
                else:
                    skipped_occurrences += 1
                    operation.reason = "occurrence_skipped:ambiguous_tool_call_values"
                    continue
            elif content_sha in entries:
                value = _unwrap_cached_value(entries[content_sha])
            if value is _MISSING:
                if rule.placeholder is None:
                    skipped_occurrences += 1
                    operation.reason = "occurrence_skipped:no_correlated_value"
                    continue
                value = rule.placeholder
                if not warned:
                    _LOGGER.warning(
                        "Field-cache rule %r for %s/%s injected placeholder at occurrence %d: no correlated cached value",
                        rule.name,
                        context.provider,
                        context.model,
                        matched,
                    )
                    warned = True
            if transform_name:
                from ..protocols.transforms import apply_transform

                value = apply_transform(str(transform_name), value)
                if value is None:
                    skipped_occurrences += 1
                    operation.reason = "occurrence_skipped:transform_produced_nothing"
                    continue
            occurrence_path = f"{container_path}.{index}.{relative_path}"
            try:
                occurrence_changed = inject_path(
                    payload,
                    occurrence_path,
                    deepcopy(value),
                    when_missing_only=rule.inject.when_missing_only if rule.inject else False,
                    insert=False,
                )
            except FieldCachePathError:
                skipped_occurrences += 1
                operation.reason = "occurrence_skipped:path_unresolvable"
                continue
            matched += 1
            changed = changed or occurrence_changed
            samples.append(value)
        operation.matched = matched
        if matched == 0 and skipped_occurrences:
            operation.skipped = True
            operation.reason = "occurrence_skipped:no_correlated_value" if rule.placeholder is None else operation.reason
        return changed, _sample_values(samples)

    def _injection_value(self, rule: FieldCacheRule, cached: Any, payload: Any, context: FieldCacheContext, operation: FieldCacheOperation) -> Any:
        """Select the scalar cached value for a non-occurrence injection path.

        Tool-id driven lookups (context tool_call_id or inject_tool_call_id_path)
        keep the per-occurrence contract: an arbitrary provider signature never
        lands on the wrong tool result. Without ids, dict recency order drives
        the mode: turn = latest value, turns = last turn_count values, all =
        every value (as a list, mirroring append semantics).
        """

        injection = rule.inject
        ids = _injection_tool_ids(rule, payload, context)
        if ids:
            if not isinstance(cached, dict):
                operation.skipped = True
                operation.reason = "invalid_tool_call_cache"
                return None
            matches = [_unwrap_cached_value(cached[str(tool_id)]) for tool_id in ids if str(tool_id) in cached]
            if not matches:
                operation.skipped = True
                operation.reason = "tool_call_cache_miss"
                return None
            if injection and injection.as_list:
                return matches
            if len(matches) == 1:
                return matches[0]
            operation.skipped = True
            operation.reason = "ambiguous_tool_call_values"
            return None
        if rule.metadata.get("tool_call_id_path") or rule.metadata.get("inject_tool_call_id_path"):
            operation.skipped = True
            operation.reason = "tool_call_id_not_found"
            return None
        if not isinstance(cached, dict):
            if rule.mode == "all":
                return cached if isinstance(cached, list) else [cached]
            if injection and injection.as_list:
                unwrapped = _unwrap_cached_value(cached)
                return unwrapped if isinstance(unwrapped, list) else [unwrapped]
            return _unwrap_cached_value(cached)
        values = list(cached.values())
        if not values:
            operation.skipped = True
            operation.reason = "tool_call_cache_miss"
            return None
        if rule.mode == "turn":
            unwrapped = _unwrap_cached_value(values[-1])
            if injection and injection.as_list:
                return unwrapped if isinstance(unwrapped, list) else [unwrapped]
            return unwrapped
        count = len(values) if rule.mode == "all" else max(1, min(rule.turn_count, len(values)))
        return [_unwrap_cached_value(value) for value in values[-count:]]

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
                "profile": operation.profile,
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
            direction=_summary_direction(source, target),
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

    def _contain_rule_error(self, rule: FieldCacheRule, operation: FieldCacheOperation, error: BaseException) -> None:
        """Absorb one rule failure: loud warning + skip-this-rule semantics.

        A field-cache rule error must never fail the request. ``critical=True``
        rules opt back into fail-closed behavior (handled by the caller, which
        re-raises before invoking this helper).
        """

        operation.skipped = True
        operation.reason = f"rule_error:{type(error).__name__}"
        _LOGGER.warning(
            "Field-cache rule %r failed (%s); skipping rule and continuing request",
            rule.name,
            error,
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


_MISSING = object()

_AUTO_TURN_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("messages", "role", "content"),
    ("contents", "role", "parts"),
    ("input", "role", "content"),
    ("choices", "message.role", "content"),
)


def _resolve_turn_shape(rule: FieldCacheRule, payload: Any) -> Optional[tuple[str, str, str]]:
    """Return (container_path, role_path, content_path) for the payload shape.

    Declared metadata wins; otherwise the common conversational containers are
    auto-detected so the openai_chat, anthropic, gemini, and responses shapes
    work without per-rule boilerplate.
    """

    declared = rule.metadata.get("turn_container_path")
    if declared:
        return (
            str(declared),
            str(rule.metadata.get("turn_role_path", "role")),
            str(rule.metadata.get("turn_content_path", "content")),
        )
    if isinstance(payload, dict):
        for container, role_path, content_path in _AUTO_TURN_SHAPES:
            if isinstance(payload.get(container), list):
                return (container, role_path, content_path)
    return None


def _container_items(payload: Any, container_path: str) -> list[Any]:
    items = extract_path(payload, container_path)
    if len(items) == 1 and isinstance(items[0], list):
        return items[0]
    return items


def _message_role(item: Any, role_path: str) -> str:
    roles = extract_path(item, role_path)
    if not roles:
        return ""
    return str(roles[0]).strip().lower()


def _message_object(item: Any, role_path: str) -> Optional[dict[str, Any]]:
    segments = role_path.split(".")
    if len(segments) == 1:
        return item if isinstance(item, dict) else None
    current: Any = item
    for segment in segments[:-1]:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current if isinstance(current, dict) else None


def _is_tool_result_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    block_type = str(block.get("type") or "")
    if block_type in {"tool_result", "function_call_output"}:
        return True
    return "tool_result" in block or "functionResponse" in block


def _message_starts_turn(item: dict[str, Any], role_path: str, content_path: str, message_obj: Optional[dict[str, Any]]) -> bool:
    """A turn region starts at a user message with real user-authored content.

    Tool-result-only user messages (anthropic tool_result blocks, gemini
    functionResponse parts, responses function_call_output items) stay in the
    current region; openai_chat tool results ride separate role="tool"
    messages and never match the user check.
    """

    if _message_role(item, role_path) != "user":
        return False
    if str(item.get("type") or "") == "function_call_output":
        return False
    content = message_obj.get(content_path) if isinstance(message_obj, dict) else None
    if isinstance(content, list) and content and all(_is_tool_result_block(block) for block in content):
        return False
    return True


def _turn_region_indexes(items: list[Any], role_path: str, content_path: str) -> list[int]:
    regions: list[int] = []
    current = 0
    for index, item in enumerate(items):
        message_obj = _message_object(item, role_path) if isinstance(item, dict) else None
        if index > 0 and isinstance(item, dict) and _message_starts_turn(item, role_path, content_path, message_obj):
            current += 1
        regions.append(current)
    return regions


def _scoped_region_ids(regions: list[int], rule: FieldCacheRule) -> Optional[set[int]]:
    if not regions:
        return None
    distinct = sorted(set(regions))
    if rule.mode == "turn":
        return {distinct[-1]}
    if rule.mode == "turns":
        return set(distinct[-max(1, rule.turn_count) :])
    return set(distinct)


def _content_sha(content: Any) -> str:
    dumped = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _normalized_content(content: Any) -> Any:
    """Reduce block-list content to its text so shas correlate across shapes.

    The same completion is a plain string on the chat wire and a list of text
    blocks in serialized unified payloads; both reduce to the joined text.
    """

    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)]
        if parts:
            return "".join(parts)
    return content


def _message_content_sha(message_obj: Any, content_path: str) -> str:
    content = message_obj.get(content_path) if isinstance(message_obj, dict) else None
    return _content_sha(_normalized_content(content))


def _occurrence_keys(rule: FieldCacheRule, message_obj: Any, content_path: str) -> list[str]:
    keys: list[str] = []
    tool_id_path = rule.metadata.get("tool_call_id_path")
    if tool_id_path and isinstance(message_obj, dict):
        keys = [str(value) for value in extract_path(message_obj, str(tool_id_path))]
    keys.append(_message_content_sha(message_obj, content_path))
    return keys


def _extraction_entries(rule: FieldCacheRule, payload: Any, values: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """Correlation entries for every occurrence in the extraction payload.

    With a resolvable turn shape, occurrences are container items carrying the
    watched value path (the message object supplies tool-call ids and the
    content bytes for the sha). Otherwise plain path matches correlate by
    value-local tool-call ids or their own content sha.
    """

    entries: list[tuple[str, dict[str, Any]]] = []
    shape = _resolve_turn_shape(rule, payload)
    declared = bool(rule.metadata.get("turn_container_path") or rule.metadata.get("turn_value_path"))
    if shape is not None:
        container_path, role_path, content_path = shape
        value_path = rule.metadata.get("turn_value_path") or _extraction_value_path(rule.path, container_path)
        if value_path:
            for item in _container_items(payload, container_path):
                if not isinstance(item, dict):
                    continue
                message_obj = _message_object(item, role_path) or item
                item_values = extract_path(item, str(value_path))
                if not item_values:
                    continue
                keys = _occurrence_keys(rule, message_obj, content_path)
                for position, value in enumerate(item_values):
                    if position == len(item_values) - 1:
                        for key in keys:
                            entries.append((key, _wrap_cached_value(value)))
                    else:
                        entries.append((_content_sha(value), _wrap_cached_value(value)))
            if entries or declared:
                return entries
    tool_id_path = rule.metadata.get("tool_call_id_path")
    tool_container_path = rule.metadata.get("tool_container_path")
    tool_value_path = rule.metadata.get("tool_value_path")
    if tool_container_path and tool_value_path:
        for container in _container_items(payload, str(tool_container_path)):
            if not isinstance(container, dict):
                continue
            container_ids = extract_path(container, str(tool_id_path)) if tool_id_path else []
            container_values = extract_path(container, str(tool_value_path))
            if container_ids and container_values:
                for container_id in container_ids:
                    entries.append((str(container_id), _wrap_cached_value(container_values[-1])))
        if entries:
            return entries
    for value in values:
        keys = [str(tool_id) for tool_id in extract_path(value, str(tool_id_path))] if tool_id_path else []
        if not keys:
            keys = [_content_sha(value)]
        for key in keys:
            entries.append((key, _wrap_cached_value(value)))
    return entries


def _occurrence_injection_plan(rule: FieldCacheRule, payload: Any) -> Optional[tuple[str, str, str, str]]:
    """Per-occurrence injection plan when the inject path is turn-relative."""

    if not rule.inject:
        return None
    shape = _resolve_turn_shape(rule, payload)
    if shape is None:
        return None
    container_path, role_path, content_path = shape
    relative = _message_relative_path(rule.inject.path, container_path)
    if relative is None:
        return None
    return (container_path, role_path, content_path, relative)


def _message_relative_path(path: str, container_path: str) -> Optional[str]:
    prefixes = (f"{container_path}.*.", f"{container_path}[-1].")
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return None


def _extraction_value_path(rule_path: str, container_path: str) -> Optional[str]:
    """Container-relative value path for extraction, accepting item indexes.

    Injection keeps the strict wildcard/tail forms only (a declared
    ``messages.0.x`` targets message 0 exactly); extraction derives the
    watched field from any item addressing form, including ``choices[0]``.
    """

    relative = _message_relative_path(rule_path, container_path)
    if relative is not None:
        return relative
    match = re.match(re.escape(container_path) + r"\.(?:\*|\[-?\d+\]|\d+)\.(.+)$", rule_path)
    if match:
        return match.group(1)
    return None


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


def _summary_direction(source: Optional[str], target: Optional[str]) -> str:
    if target is not None:
        if target in {"stream_event", "unified_stream_event"}:
            return "stream"
        if target in {"request", "unified_request", "metadata"}:
            return "request"
        return "response"
    if source in {"stream_event", "unified_stream_event"}:
        return "stream"
    if source in {"request", "unified_request"}:
        return "request"
    if source in {"response", "unified_response"}:
        return "response"
    return "metadata"


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
