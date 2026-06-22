"""Unit tests for Umans request-based quota tracker (no network)."""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from rotator_library.providers.umans_provider import UmansProvider
from rotator_library.providers.utilities.umans_quota_tracker import (
    UmansQuotaTracker,
    _detect_plan,
    _get_credential_identifier,
    _parse_iso_to_unix,
    _parse_usage_response,
    _resolve_request_limit,
)


class _TrackerHost(UmansQuotaTracker):
    """Minimal host for exercising the mixin."""

    provider_env_name = "umans"

    def __init__(self):
        self._init_quota_tracker()
        self.model_quota_groups = {
            "5h-requests": ["_requests_5h"],
            "concurrency": ["_concurrent"],
        }


SAMPLE_CODE_PRO = {
    "limits": {
        "requests": {
            "limit": 200,
            "hard_cap": 400,
            "window_seconds": 18000,
        },
        "concurrency": {
            "limit": 3,
            "hard_cap": 6,
        },
    },
    "usage": {
        "requests_in_window": 12,
        "remaining_requests": 188,
        "weighted_requests_in_window": 9,
        "weighted_remaining_requests": 191,
        "concurrent_sessions": 0,
        "tokens_in": 394143,
        "tokens_out": 15297,
        "tokens_cached": 358144,
    },
    "window": {
        "started_at": "2026-06-22T00:41:43Z",
        "resets_at": "2026-06-22T05:41:43Z",
    },
    "throttled": False,
}


SAMPLE_MAX = {
    "limits": {
        "requests": {
            "limit": 0,
            "hard_cap": 0,
            "window_seconds": 0,
        },
        "concurrency": {
            "limit": 4,
            "hard_cap": 4,
        },
    },
    "usage": {
        "requests_in_window": 0,
        "remaining_requests": 0,
        "weighted_requests_in_window": 0,
        "weighted_remaining_requests": 0,
        "concurrent_sessions": 1,
        "tokens_in": 1000,
        "tokens_out": 200,
        "tokens_cached": 0,
    },
    "window": {
        "started_at": None,
        "resets_at": None,
    },
    "throttled": False,
}


def test_get_credential_identifier_env_path():
    assert _get_credential_identifier("env://umans/1") == "env://umans/1"


def test_get_credential_identifier_masks_long_key():
    key = "umans_" + "a" * 40
    assert _get_credential_identifier(key) == f"{key[:4]}...{key[-4:]}"


def test_get_credential_identifier_short_key_unmasked():
    assert _get_credential_identifier("abcd") == "abcd"


def test_parse_iso_to_unix_z():
    ts = _parse_iso_to_unix("2026-06-22T05:41:43Z")
    assert ts is not None
    # 2026-06-22 05:41:43 UTC is after the current test run epoch
    assert ts > time.time()


def test_detect_plan_code_pro_inferred():
    plan, has_limit, conc = _detect_plan(SAMPLE_CODE_PRO)
    assert plan == "code_pro"
    assert has_limit is True
    assert conc == 3


def test_detect_plan_code_pro_explicit():
    data = {**SAMPLE_CODE_PRO, "plan": "code_pro"}
    plan, has_limit, conc = _detect_plan(data)
    assert plan == "code_pro"
    assert has_limit is True
    assert conc == 3


def test_detect_plan_max():
    plan, has_limit, conc = _detect_plan(SAMPLE_MAX)
    assert plan == "max"
    assert has_limit is False
    assert conc == 4


def test_resolve_request_limit_code_pro():
    assert _resolve_request_limit(200, "code_pro") == (200, True)


def test_resolve_request_limit_code_pro_no_limit():
    assert _resolve_request_limit(0, "code_pro") == (0, False)


def test_resolve_request_limit_max():
    assert _resolve_request_limit(0, "max") == (0, False)


def test_resolve_request_limit_max_ignores_positive_env():
    with patch.dict(os.environ, {"UMANS_QUOTA_LIMIT": "500"}):
        assert _resolve_request_limit(0, "max") == (0, False)


def test_resolve_request_limit_code_pro_env_override():
    with patch.dict(os.environ, {"UMANS_QUOTA_LIMIT": "150"}):
        assert _resolve_request_limit(200, "code_pro") == (150, True)


def test_parse_usage_response_code_pro():
    snapshot = _parse_usage_response(SAMPLE_CODE_PRO, "my-api-key", "my-id")
    assert snapshot.status == "success"
    assert snapshot.plan == "code_pro"
    assert snapshot.has_request_limit is True
    assert snapshot.requests_limit == 200
    assert snapshot.requests_hard_cap == 400
    assert snapshot.requests_used == 12
    assert snapshot.requests_remaining == 188
    assert snapshot.weighted_used == 9
    assert snapshot.weighted_remaining == 191
    assert snapshot.concurrency_limit == 3
    assert snapshot.concurrency_hard_cap == 6
    assert snapshot.concurrent_sessions == 0
    assert snapshot.window_seconds == 18000
    assert snapshot.window_reset_ts is not None


def test_parse_usage_response_max():
    snapshot = _parse_usage_response(SAMPLE_MAX, "my-api-key", "my-id")
    assert snapshot.status == "success"
    assert snapshot.plan == "max"
    assert snapshot.has_request_limit is False
    assert snapshot.requests_limit == 0
    assert snapshot.concurrency_limit == 4
    assert snapshot.window_reset_ts is None


def test_parse_usage_response_code_pro_env_override():
    with patch.dict(os.environ, {"UMANS_QUOTA_LIMIT": "150"}):
        snapshot = _parse_usage_response(SAMPLE_CODE_PRO, "my-api-key", "my-id")
    assert snapshot.requests_limit == 150
    assert snapshot.has_request_limit is True


def test_store_baselines_to_usage_manager():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        snapshot = _parse_usage_response(SAMPLE_CODE_PRO, "my-api-key", "my-id")
        results = {"my-api-key": snapshot}
        stored = await host._store_baselines_to_usage_manager(
            results, usage_manager, force=True
        )
        assert stored == 2

        calls = usage_manager.update_quota_baseline.await_args_list
        groups = [c.kwargs.get("quota_group") for c in calls]
        assert "5h-requests" in groups
        assert "concurrency" in groups

        for call in calls:
            assert call.kwargs.get("apply_exhaustion") is False
            assert call.kwargs.get("force") is True

        req_call = next(c for c in calls if c.kwargs.get("quota_group") == "5h-requests")
        assert req_call.kwargs["model"] == "umans/_requests_5h"
        assert req_call.kwargs["quota_max_requests"] == 200
        assert req_call.kwargs["quota_used"] == 12

        conc_call = next(
            c for c in calls if c.kwargs.get("quota_group") == "concurrency"
        )
        assert conc_call.kwargs["model"] == "umans/_concurrent"
        assert conc_call.kwargs["quota_max_requests"] == 3
        assert conc_call.kwargs["quota_used"] == 0

    asyncio.run(_run())


def test_store_baselines_skips_request_group_for_max_plan():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        snapshot = _parse_usage_response(SAMPLE_MAX, "my-api-key", "my-id")
        results = {"my-api-key": snapshot}
        stored = await host._store_baselines_to_usage_manager(
            results, usage_manager, force=True
        )
        assert stored == 1
        calls = usage_manager.update_quota_baseline.await_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["quota_group"] == "concurrency"

    asyncio.run(_run())


def test_fetch_initial_baselines_mixed():
    async def _run():
        host = _TrackerHost()
        ok_snapshot = _parse_usage_response(SAMPLE_CODE_PRO, "key-ok", "key-ok")
        err_snapshot = _parse_usage_response({}, "key-err", "key-err")
        err_snapshot.status = "error"
        err_snapshot.error = "HTTP 503"

        with patch.object(
            host,
            "_fetch_usage_for_credential",
            side_effect=[ok_snapshot, err_snapshot],
        ):
            results = await host.fetch_initial_baselines(["key-ok", "key-err"])

        assert results["key-ok"].status == "success"
        assert results["key-err"].status == "error"

    asyncio.run(_run())


def test_provider_get_model_quota_group():
    provider = object.__new__(UmansProvider)
    assert provider.get_model_quota_group("umans/kimi-k2.7") == "5h-requests"
    assert provider.get_model_quota_group("umans/_requests_5h") == "5h-requests"
    assert provider.get_model_quota_group("umans/_concurrent") == "concurrency"
    assert provider.get_model_quota_group("_concurrent") == "concurrency"


def test_provider_get_models_success():
    async def _run():
        provider = object.__new__(UmansProvider)
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [
                {"id": "umans-kimi-k2.7"},
                {"id": "umans-glm-5.2"},
            ]
        }
        fake_client = MagicMock()
        fake_client.get = AsyncMock(return_value=fake_response)

        models = await provider.get_models("test-key", fake_client)
        assert models == ["umans/umans-kimi-k2.7", "umans/umans-glm-5.2"]
        fake_client.get.assert_called_once()
        call_kwargs = fake_client.get.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    asyncio.run(_run())


def test_provider_parse_quota_error_rate_limit():
    error = httpx.HTTPStatusError(
        "rate limit",
        request=MagicMock(),
        response=MagicMock(status_code=429, text="rate limit exceeded"),
    )
    parsed = UmansProvider.parse_quota_error(error)
    assert parsed is not None
    assert parsed["reason"] == "RATE_LIMITED"


def test_provider_parse_quota_error_not_quota():
    error = httpx.HTTPStatusError(
        "not found",
        request=MagicMock(),
        response=MagicMock(status_code=404, text="not found"),
    )
    parsed = UmansProvider.parse_quota_error(error)
    assert parsed is None


def test_provider_get_credential_concurrency_limit_from_cache():
    provider = object.__new__(UmansProvider)
    provider._quota_cache = {
        "key-1": _parse_usage_response(SAMPLE_MAX, "key-1", "key-1")
    }
    assert provider.get_credential_concurrency_limit("key-1") == 4
    assert provider.get_credential_concurrency_limit("missing") is None
