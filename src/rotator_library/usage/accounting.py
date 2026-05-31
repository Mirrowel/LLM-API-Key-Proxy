# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Normalized usage accounting across provider protocols.

The existing `UsageManager` remains the source of truth for persistence,
windows, and credential selection. This module only converts provider-specific
usage payloads into the numeric buckets that `CredentialContext.mark_success()`
already accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..protocols import serialize_value


@dataclass(frozen=True)
class UsageRecord:
    """Provider-neutral token usage buckets.

    `completion_tokens` excludes `reasoning_tokens` when providers report hidden
    reasoning separately. This preserves the current no-double-count behavior and
    gives Phase 9+ cost logic clear input/output/reasoning buckets.
    """

    input_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw_total_tokens: int = 0
    request_count: int = 1
    source: str = "unknown"
    provider: Optional[str] = None
    model: Optional[str] = None
    provider_reported_cost: Optional[float] = None
    cost_currency: str = "USD"
    cost_source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_tokens(self) -> int:
        return self.completion_tokens + self.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.completion_tokens
            + self.reasoning_tokens
        )

    @property
    def prompt_tokens_for_mark_success(self) -> int:
        """Return non-cache-read prompt tokens for existing usage storage."""

        return self.input_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary for transform traces and tests."""

        return {
            "input_tokens": self.input_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "raw_total_tokens": self.raw_total_tokens,
            "request_count": self.request_count,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "provider_reported_cost": self.provider_reported_cost,
            "cost_currency": self.cost_currency,
            "cost_source": self.cost_source,
            "metadata": serialize_value(self.metadata),
        }


def extract_usage_record(
    response_or_usage: Any,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    source: str = "response",
) -> UsageRecord:
    """Extract normalized usage from a response object or usage payload."""

    usage = _unwrap_usage(response_or_usage)
    if usage is None:
        return UsageRecord(provider=provider, model=model, source=source)
    data = _as_dict(usage)
    if not data:
        return UsageRecord(provider=provider, model=model, source=source)

    if "usageMetadata" in data and isinstance(data["usageMetadata"], dict):
        data = data["usageMetadata"]

    if _looks_like_gemini(data):
        return _from_gemini_usage(data, provider=provider, model=model, source=source)
    if _looks_like_anthropic(data):
        return _from_anthropic_usage(data, provider=provider, model=model, source=source)
    return _from_openai_like_usage(data, provider=provider, model=model, source=source)


def _unwrap_usage(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        if "usage" in value:
            usage = _as_dict(value.get("usage"))
            return _merge_top_level_cost_fields(usage, value)
        return value
    data = _as_dict(value)
    if "usage" in data:
        usage = _as_dict(data.get("usage"))
        return _merge_top_level_cost_fields(usage, data)
    return getattr(value, "usage", value)


def _merge_top_level_cost_fields(usage: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Copy sibling cost metadata into a nested usage payload."""

    merged = dict(usage)
    for key in ("cost", "total_cost", "estimated_cost", "cost_details", "provider_reported_cost", "request_cost_usd", "currency", "costMetadata"):
        if key in response and key not in merged:
            merged[key] = response[key]
    return merged


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "usage",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cache_read_tokens",
        "cache_creation_tokens",
        "input_tokens",
        "output_tokens",
        "input_tokens_details",
        "output_tokens_details",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "thinking_tokens",
        "cost",
        "cost_details",
        "costMetadata",
        "provider_reported_cost",
        "request_cost_usd",
        "estimated_cost",
        "total_cost",
        "currency",
    ):
        if hasattr(value, key):
            result[key] = getattr(value, key)
    return result


def _from_openai_like_usage(data: dict[str, Any], *, provider: Optional[str], model: Optional[str], source: str) -> UsageRecord:
    prompt_tokens = _int(data.get("prompt_tokens", data.get("input_tokens", 0)))
    completion_tokens = _int(data.get("completion_tokens", data.get("output_tokens", 0)))
    raw_total = _int(data.get("total_tokens", data.get("raw_total_tokens", 0)))

    prompt_details = _as_dict(data.get("prompt_tokens_details") or data.get("input_tokens_details") or {})
    completion_details = _as_dict(data.get("completion_tokens_details") or data.get("output_tokens_details") or {})
    cache_read = _int(
        data.get(
            "cache_read_tokens",
            data.get("cached_tokens", prompt_details.get("cached_tokens", 0)),
        )
    )
    cache_write = _int(
        data.get(
            "cache_creation_tokens",
            data.get("cache_write_tokens", prompt_details.get("cache_creation_tokens", 0)),
        )
    )
    reasoning = _int(
        data.get(
            "reasoning_tokens",
            completion_details.get("reasoning_tokens", completion_details.get("thinking_tokens", 0)),
        )
    )
    if reasoning and completion_tokens >= reasoning:
        completion_tokens -= reasoning
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    cost = _extract_cost(data)
    return UsageRecord(
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw_total_tokens=raw_total,
        source=source,
        provider=provider,
        model=model,
        provider_reported_cost=cost[0],
        cost_currency=cost[1],
        cost_source=cost[2],
        metadata={"shape": "openai_like"},
    )


def _from_anthropic_usage(data: dict[str, Any], *, provider: Optional[str], model: Optional[str], source: str) -> UsageRecord:
    cache_read = _int(data.get("cache_read_input_tokens", data.get("cache_read_tokens", 0)))
    cache_write = _int(data.get("cache_creation_input_tokens", data.get("cache_creation_tokens", 0)))
    input_tokens = max(0, _int(data.get("input_tokens", 0)) - cache_read - cache_write)
    output_tokens = _int(data.get("output_tokens", data.get("completion_tokens", 0)))
    reasoning = _int(data.get("reasoning_tokens", data.get("thinking_tokens", 0)))
    if reasoning and output_tokens >= reasoning:
        output_tokens -= reasoning
    cost = _extract_cost(data)
    return UsageRecord(
        input_tokens=input_tokens,
        completion_tokens=output_tokens,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw_total_tokens=_int(data.get("total_tokens", 0)),
        source=source,
        provider=provider,
        model=model,
        provider_reported_cost=cost[0],
        cost_currency=cost[1],
        cost_source=cost[2],
        metadata={"shape": "anthropic"},
    )


def _from_gemini_usage(data: dict[str, Any], *, provider: Optional[str], model: Optional[str], source: str) -> UsageRecord:
    cache_read = _int(data.get("cachedContentTokenCount", data.get("cache_read_tokens", 0)))
    prompt_tokens = _int(data.get("promptTokenCount", data.get("prompt_tokens", 0)))
    reasoning = _int(data.get("thoughtsTokenCount", data.get("reasoning_tokens", 0)))
    completion = _int(data.get("candidatesTokenCount", data.get("completion_tokens", 0)))
    if reasoning and completion >= reasoning:
        completion -= reasoning
    cost = _extract_cost(data)
    return UsageRecord(
        input_tokens=max(0, prompt_tokens - cache_read),
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        raw_total_tokens=_int(data.get("totalTokenCount", data.get("total_tokens", 0))),
        source=source,
        provider=provider,
        model=model,
        provider_reported_cost=cost[0],
        cost_currency=cost[1],
        cost_source=cost[2],
        metadata={"shape": "gemini"},
    )


def _extract_cost(data: dict[str, Any]) -> tuple[Optional[float], str, Optional[str]]:
    """Extract actual provider-reported cost without guessing prices.

    Advisory pricing belongs in `usage.costs`. This helper only preserves cost
    values explicitly reported by a provider or protocol adapter.
    """

    cost_payload = _as_dict(data.get("cost") or data.get("cost_details") or data.get("costMetadata") or {})
    raw_cost = (
        cost_payload.get("provider_reported_cost", cost_payload.get("request_cost_usd"))
        if cost_payload
        else None
    )
    breakdown_used = False
    if raw_cost is None and cost_payload:
        raw_cost = cost_payload.get("total_cost", cost_payload.get("total"))
    if raw_cost is None and cost_payload:
        raw_cost = cost_payload.get("cost")
    if raw_cost is None and cost_payload:
        breakdown_total = _sum_cost_breakdown(cost_payload)
        if breakdown_total is not None:
            raw_cost = breakdown_total
            breakdown_used = True
    if raw_cost is None:
        raw_cost = data.get("provider_reported_cost", data.get("request_cost_usd", data.get("total_cost", data.get("cost", data.get("estimated_cost")))))
    cost_value = _float_or_none(raw_cost)
    currency = str(cost_payload.get("currency") or data.get("currency") or "USD")
    source = cost_payload.get("source") or ("provider_reported_breakdown" if breakdown_used else ("provider_reported" if cost_value is not None else None))
    return cost_value, currency, str(source) if source else None


def _sum_cost_breakdown(payload: dict[str, Any]) -> Optional[float]:
    """Sum structured provider cost buckets when no total is reported."""

    total = 0.0
    found = False
    for key in (
        "input_cost",
        "prompt_cost",
        "cache_read_cost",
        "cache_write_cost",
        "cached_input_cost",
        "cache_write_input_cost",
        "output_cost",
        "completion_cost",
        "completions_cost",
        "reasoning_cost",
        "thinking_cost",
        "upstream_inference_cost",
        "upstream_inference_prompt_cost",
        "upstream_inference_completions_cost",
        "upstream_inference_input_cost",
        "upstream_inference_output_cost",
        "image_input_cost",
        "image_output_cost",
        "audio_input_cost",
        "audio_output_cost",
        "data_storage_cost",
        "estimated_cost",
        "request_cost",
        "web_search_cost",
        "search_cost",
    ):
        value = _float_or_none(payload.get(key))
        if value is not None:
            total += value
            found = True
    ticks = _float_or_none(payload.get("cost_in_usd_ticks"))
    if ticks is not None:
        total += ticks / 10_000_000_000
        found = True
    return total if found else None


def _looks_like_gemini(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "cachedContentTokenCount"))


def _looks_like_anthropic(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("cache_creation_input_tokens", "cache_read_input_tokens")) and "input_tokens" in data


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None
