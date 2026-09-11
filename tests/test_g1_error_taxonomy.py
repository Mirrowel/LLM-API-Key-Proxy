"""G1 grounded error-taxonomy pins (docs/experimental/error-reference.md)."""

from __future__ import annotations

import json

import httpx
import pytest

from rotator_library.core.errors import (
    StructuredAPIResponseError,
    structured_api_response_error,
)
from rotator_library.error_handler import (
    _parse_duration_string,
    classify_error,
    get_retry_after,
    should_rotate_on_error,
)


def _http_error(status: int, body: dict | str, headers: dict | None = None) -> httpx.HTTPStatusError:
    content = json.dumps(body) if isinstance(body, dict) else body
    request = httpx.Request("POST", "https://provider.example/v1/generate")
    response = httpx.Response(
        status,
        text=content,
        request=request,
        headers=headers or {},
    )
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


# --- quota masquerading at wrong statuses ---


def test_gemini_quota_at_400_classifies_quota() -> None:
    """Gemini-compat wraps RESOURCE_EXHAUSTED in HTTP 400 — must rotate, not stop."""
    error = _http_error(
        400,
        {"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}},
    )
    classified = classify_error(error)
    assert classified.error_type == "quota_exceeded"
    assert should_rotate_on_error(classified) is True


def test_quota_markers_in_5xx_body_classify_quota() -> None:
    error = _http_error(
        503,
        {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
    )
    assert classify_error(error).error_type == "quota_exceeded"


def test_anthropic_spend_cap_429_signature_classifies_quota() -> None:
    error = _http_error(
        429,
        {"type": "error", "error": {"type": "rate_limit_error", "message": "Spend cap reached", "details": {"error_code": "enforced_spend_limit_reached"}}},
    )
    assert classify_error(error).error_type == "quota_exceeded"


def test_structured_quota_details_carried_through() -> None:
    error = _http_error(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaValue": "10"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18.4s"},
        ]}},
    )
    classified = classify_error(error)
    assert classified.error_type == "quota_exceeded"
    assert classified.quota_id == "GenerateRequestsPerMinutePerProjectPerModel"
    assert classified.retry_after == 18


# --- substring sniffing is dead: no bare-token matching ---


def test_generate_no_longer_mangles_into_rate_limit() -> None:
    """'gene-RATE' must never classify a 400 as rate/quota."""
    error = _http_error(400, {"error": {"type": "invalid_request_error", "message": "Failed to generate content: tool schema invalid", "param": "tools", "code": None}})
    assert classify_error(error).error_type == "invalid_request"


def test_structured_normalization_rejects_substring_descriptors() -> None:
    err = structured_api_response_error(
        {"error": {"type": "invalid_request_error", "message": "Request could not be generated accurately", "code": None}}
    )
    assert err is not None
    assert err.error_type == "invalid_request"


# --- status-ladder vocabulary additions ---


def test_status_ladder_new_members() -> None:
    assert classify_error(_http_error(404, {"error": {"message": "model not found"}})).error_type == "not_found"
    assert classify_error(_http_error(409, {"error": {"message": "conflict"}})).error_type == "conflict"
    assert classify_error(_http_error(413, {"error": {"message": "too large"}})).error_type == "request_too_large"
    assert classify_error(_http_error(529, {"error": {"message": "Overloaded"}})).error_type == "server_error"


def test_not_found_and_too_large_never_rotate_credentials() -> None:
    assert should_rotate_on_error(classify_error(_http_error(404, {"error": {"message": "nope"}}))) is False
    assert should_rotate_on_error(classify_error(_http_error(413, {"error": {"message": "big"}}))) is False


def test_wire_vocabulary_round_trips_internal_names() -> None:
    assert classify_error({"error": {"type": "authentication"}}).error_type == "authentication"
    assert classify_error({"error": {"type": "billing_error"}}).error_type == "quota_exceeded"
    assert classify_error({"error": {"type": "overloaded_error"}}).error_type == "server_error"
    assert classify_error({"error": {"status": "UNAUTHENTICATED"}}).error_type == "authentication"
    assert classify_error({"error": {"code": "rate_limit_exceeded"}}).error_type == "rate_limit"


# --- retry-signal units (one parser, four formats) ---


def test_duration_parser_understands_all_four_unit_families() -> None:
    assert _parse_duration_string("6m0s") == 360
    assert _parse_duration_string("23h18m29.144s") == 83909
    assert _parse_duration_string("290.979975ms") == 1
    assert _parse_duration_string("156h14m36.752463453s") == 562476
    assert _parse_duration_string("2d5h") == 190800


def test_retry_after_header_accepts_http_date() -> None:
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(seconds=90)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    error = _http_error(429, {"error": {"message": "rate"}}, headers={"Retry-After": future})
    wait = get_retry_after(error)
    assert wait is not None
    assert 80 <= wait <= 95


def test_retry_after_go_duration_header() -> None:
    error = _http_error(429, {"error": {"message": "rate"}}, headers={"x-ratelimit-reset-tokens": "6m0s"})
    assert get_retry_after(error) == 360


def test_bare_text_429_quota_body_still_classifies_quota() -> None:
    error = _http_error(429, "Quota exceeded for this project")
    assert classify_error(error).error_type == "quota_exceeded"


# --- protocol envelopes: official vocabulary per dialect ---


def test_envelope_matrix_official_vocabulary() -> None:
    cases = [
        ("invalid_request", None, "openai_chat"),
        ("rate_limit", None, "openai_chat"),
        ("quota_exceeded", None, "openai_chat"),
    ]
    for error_type, status, protocol in cases:
        structured = StructuredAPIResponseError("boom", error_type=error_type, status_code=status)
        payload = structured.to_protocol_payload(protocol)
        assert payload["error"]["type"] in {
            "invalid_request_error", "rate_limit_error", "insufficient_quota", "server_error",
            "authentication_error", "permission_error", "not_found_error",
        }, payload

    chat = StructuredAPIResponseError("bad", error_type="invalid_request").to_protocol_payload("openai_chat")
    assert chat["error"]["type"] == "invalid_request_error"
    assert chat["error"]["param"] is None


def test_anthropic_envelope_status_aware_timeout_and_overload() -> None:
    timeout_case = StructuredAPIResponseError("slow", error_type="server_error", status_code=504)
    assert timeout_case.to_protocol_payload("anthropic_messages")["error"]["type"] == "timeout_error"

    overloaded = StructuredAPIResponseError("busy", error_type="server_error", status_code=529)
    assert overloaded.to_protocol_payload("anthropic_messages")["error"]["type"] == "overloaded_error"

    billing = StructuredAPIResponseError("pay", error_type="quota_exceeded", status_code=402)
    assert billing.to_protocol_payload("anthropic_messages")["error"]["type"] == "rate_limit_error"


def test_gemini_envelope_uses_google_rpc_status_strings() -> None:
    payload = StructuredAPIResponseError("no", error_type="not_found").to_protocol_payload("gemini")
    assert payload["error"]["status"] == "NOT_FOUND"
    assert payload["error"]["code"] == 404
