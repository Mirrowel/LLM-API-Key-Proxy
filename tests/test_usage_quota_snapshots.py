from __future__ import annotations

from rotator_library.usage.quota import build_quota_snapshots
from rotator_library.usage.types import CredentialState, WindowStats


def _state() -> CredentialState:
    state = CredentialState(stable_id="credential-secret", provider="openai", accessor="credential-secret")
    model_stats = state.get_model_stats("gpt-test")
    model_stats.windows["daily"] = WindowStats(name="daily", request_count=3, limit=10, reset_at=123.0)
    group_stats = state.get_group_stats("chat")
    group_stats.windows["daily"] = WindowStats(name="daily", request_count=5, limit=20, reset_at=456.0)
    return state


def test_build_quota_snapshots_for_model_window() -> None:
    snapshots = build_quota_snapshots(provider="openai", states={"credential-secret": _state()}, model="gpt-test")

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.provider == "openai"
    assert snapshot.model == "gpt-test"
    assert snapshot.quota_group is None
    assert snapshot.used == 3
    assert snapshot.remaining == 7
    assert snapshot.credential_id != "credential-secret"


def test_build_quota_snapshots_for_group_window_without_credentials() -> None:
    snapshots = build_quota_snapshots(
        provider="openai",
        states={"credential-secret": _state()},
        model="gpt-test",
        quota_group="chat",
        include_credentials=False,
    )

    group_snapshot = [snapshot for snapshot in snapshots if snapshot.source == "group"][0]
    assert group_snapshot.quota_group == "chat"
    assert group_snapshot.used == 5
    assert group_snapshot.remaining == 15
    assert group_snapshot.credential_id is None


def test_build_quota_snapshots_missing_windows_returns_empty() -> None:
    assert build_quota_snapshots(provider="openai", states={"credential-secret": _state()}, model="missing") == []
