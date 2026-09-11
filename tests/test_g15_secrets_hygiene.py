"""G15 secrets & logging hygiene pins.

Covers: usage.json derived accessors + migration, quota-stats exfiltration,
RawIOLogger header redaction matrix, error-text URL scrubbing, ProviderLogger
path traversal containment, and ReauthCoordinator path disclosure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proxy_app.detailed_logger import RawIOLogger
from rotator_library.transaction_logger import ProviderLogger, TransactionContext
from rotator_library.transform_trace import scrub_sensitive_text
from rotator_library.usage import UsageManager
from rotator_library.usage.identity.registry import derive_accessor_id
from rotator_library.usage.persistence.storage import UsageStorage
from rotator_library.usage.types import CredentialState


RAW_KEY = "sk-SUPERSECRETKEY-0123456789"
OAUTH_ACCESSOR = "private:deadbeefdeadbeefdeadbeefdeadbeef"


# ---------------------------------------------------------------------------
# Derived accessor identifier
# ---------------------------------------------------------------------------


def test_derive_accessor_id_hides_raw_and_preserves_private() -> None:
    derived = derive_accessor_id(RAW_KEY)
    assert RAW_KEY not in derived
    assert derived.startswith("sha256:")
    assert len(derived) == len("sha256:") + 16
    # Private accessors are already derived hashes and pass through untouched.
    assert derive_accessor_id(OAUTH_ACCESSOR) == OAUTH_ACCESSOR


# ---------------------------------------------------------------------------
# RawIOLogger redaction matrix
# ---------------------------------------------------------------------------


def test_raw_io_logger_redacts_every_auth_carrier() -> None:
    headers = {
        "Authorization": "Bearer proxy-secret",
        "X-Goog-Api-Key": "AIzaREALGEMINIKEY",
        "api-key": "azure-secret",
        "x-api-key": "anthropic-secret",
        "x-custom-api-key": "gateway-secret",
        "Cookie": "session=abc",
        "Content-Type": "application/json",
    }

    redacted = RawIOLogger._redact_headers(headers)

    for sensitive in (
        "Authorization",
        "X-Goog-Api-Key",
        "api-key",
        "x-api-key",
        "x-custom-api-key",
        "Cookie",
    ):
        assert redacted[sensitive] == "<redacted>", sensitive
    assert redacted["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Error text scrubbing: bare query-string key material
# ---------------------------------------------------------------------------


def test_scrub_sensitive_text_redacts_bare_url_key_params() -> None:
    text = (
        "upstream auth failed: "
        "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaREALKEY&alt=sse "
        "plus apikey=ABCDEF and token=XYZ123"
    )

    scrubbed = scrub_sensitive_text(text)

    for secret in ("AIzaREALKEY", "ABCDEF", "XYZ123"):
        assert secret not in scrubbed, secret
    assert "[REDACTED]" in scrubbed


# ---------------------------------------------------------------------------
# usage.json persistence + migration
# ---------------------------------------------------------------------------


async def test_usage_storage_persists_derived_accessors(tmp_path: Path) -> None:
    usage_file = tmp_path / "usage.json"
    storage = UsageStorage(usage_file)
    state = CredentialState(
        stable_id="stable-one",
        provider="openai",
        accessor=RAW_KEY,
    )

    assert await storage.save({"stable-one": state}, force=True)

    data = json.loads(usage_file.read_text(encoding="utf-8"))
    serialized = json.dumps(data)
    assert RAW_KEY not in serialized
    assert data["credentials"]["stable-one"]["accessor"] == derive_accessor_id(RAW_KEY)
    assert data["accessor_index"][derive_accessor_id(RAW_KEY)] == "stable-one"


async def test_usage_storage_migrates_raw_accessors_on_load(tmp_path: Path) -> None:
    usage_file = tmp_path / "usage.json"
    legacy = {
        "schema_version": 2,
        "credentials": {
            "stable-one": {
                "provider": "openai",
                "accessor": RAW_KEY,
                "private": False,
                "priority": 1,
            }
        },
        "accessor_index": {RAW_KEY: "stable-one"},
        "fair_cycle_global": {},
    }
    usage_file.write_text(json.dumps(legacy), encoding="utf-8")

    storage = UsageStorage(usage_file)
    states, _, loaded = await storage.load()

    assert loaded is True
    assert states["stable-one"].accessor == derive_accessor_id(RAW_KEY)

    rewritten = json.loads(usage_file.read_text(encoding="utf-8"))
    rewritten_text = json.dumps(rewritten)
    assert RAW_KEY not in rewritten_text
    assert rewritten["schema_version"] == 3
    assert rewritten["credentials"]["stable-one"]["accessor"] == derive_accessor_id(
        RAW_KEY
    )
    assert rewritten["accessor_index"] == {derive_accessor_id(RAW_KEY): "stable-one"}


async def test_quota_stats_never_returns_key_material(tmp_path: Path) -> None:
    usage_file = tmp_path / "usage.json"
    manager = UsageManager(provider="openai", file_path=usage_file)
    try:
        await manager.initialize([RAW_KEY])
        stats = await manager.get_stats_for_endpoint()
        payload = json.dumps(stats)

        assert RAW_KEY not in payload
        credential = next(iter(stats["credentials"].values()))
        assert credential["full_path"] == derive_accessor_id(RAW_KEY)
        assert RAW_KEY not in credential["full_path"]
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------
# ProviderLogger containment + error scrubbing
# ---------------------------------------------------------------------------


def _provider_logger(tmp_path: Path) -> ProviderLogger:
    context = TransactionContext(
        log_dir=tmp_path / "txn",
        request_id="req-1",
        enabled=True,
        provider="openai",
        model="gpt-test",
    )
    return ProviderLogger(context)


def test_provider_logger_rejects_path_traversal(tmp_path: Path) -> None:
    logger = _provider_logger(tmp_path)

    logger.log_extra("../../escaped.log", "boom")
    assert not (tmp_path / "escaped.log").exists()
    assert not (tmp_path / "txn" / "escaped.log").exists()

    logger.log_extra(str(tmp_path / "absolute.log"), "boom")
    assert not (tmp_path / "absolute.log").exists()

    logger.log_extra("safe.log", "ok")
    assert (tmp_path / "txn" / "provider" / "safe.log").read_text(
        encoding="utf-8"
    ) == "ok"


def test_provider_logger_error_log_scrubs_url_keys(tmp_path: Path) -> None:
    logger = _provider_logger(tmp_path)

    logger.log_error("upstream failed https://api.example/v1?key=REALKEY&alt=sse")

    content = (tmp_path / "txn" / "provider" / "error.log").read_text(
        encoding="utf-8"
    )
    assert "REALKEY" not in content
    assert "[REDACTED]" in content


# ---------------------------------------------------------------------------
# Reauth status disclosure
# ---------------------------------------------------------------------------


def test_reauth_status_exposes_basenames_only() -> None:
    from rotator_library.utils.reauth_coordinator import get_reauth_coordinator

    coordinator = get_reauth_coordinator()
    previous_current = coordinator._current_reauth
    previous_pending = dict(coordinator._pending_reauths)
    try:
        coordinator._current_reauth = str(
            Path("C:/secrets/creds/oauth_1.json")
        )
        coordinator._pending_reauths = {
            str(Path("C:/secrets/creds/oauth_2.json")): 0.0
        }

        status = coordinator.get_status()

        assert status["current_reauth"] == "oauth_1.json"
        assert status["pending_credentials"] == ["oauth_2.json"]
        assert "secrets" not in json.dumps(status)
    finally:
        coordinator._current_reauth = previous_current
        coordinator._pending_reauths = previous_pending


# ---------------------------------------------------------------------------
# CORS: expose the capability header, no wildcard+credentials combo
# ---------------------------------------------------------------------------


def test_cors_exposes_session_domain_without_wildcard_credentials() -> None:
    from fastapi.testclient import TestClient

    from proxy_app import main as proxy_main

    client = TestClient(proxy_main.app)
    response = client.get("/", headers={"Origin": "https://app.example"})

    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") is None
    assert "X-Proxy-Session-Domain" in response.headers.get(
        "access-control-expose-headers", ""
    )
