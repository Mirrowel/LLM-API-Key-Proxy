# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Read-only quota snapshot helpers built from existing usage state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..error_handler import mask_credential
from ..protocols import serialize_value
from .types import CredentialState, WindowStats


@dataclass(frozen=True)
class QuotaSnapshot:
    """Client-safe view of one usage window.

    Snapshots are reporting-only. They intentionally do not participate in limit
    checks, selection, or persistence so existing quota behavior remains owned by
    `UsageManager`, `TrackingEngine`, and `LimitEngine`.
    """

    provider: str
    model: Optional[str]
    quota_group: Optional[str]
    credential_id: Optional[str]
    window_name: str
    limit: Optional[int]
    used: int
    remaining: Optional[int]
    reset_at: Optional[float]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "quota_group": self.quota_group,
            "credential_id": self.credential_id,
            "window_name": self.window_name,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "source": self.source,
            "metadata": serialize_value(self.metadata),
        }


def build_quota_snapshots(
    *,
    provider: str,
    states: Mapping[str, CredentialState],
    model: Optional[str] = None,
    quota_group: Optional[str] = None,
    include_credentials: bool = True,
) -> list[QuotaSnapshot]:
    """Build read-only request/token quota snapshots from credential states.

    The current usage state stores request/token windows, not a reliable
    provider-cost ledger. Snapshots therefore avoid inventing cost totals; cost
    reporting can be added later only if the underlying state owns that data.
    """

    snapshots: list[QuotaSnapshot] = []
    for stable_id, state in states.items():
        credential_id = mask_credential(stable_id) if include_credentials else None
        if model:
            model_stats = state.get_model_stats(model, create=False)
            if model_stats:
                snapshots.extend(
                    _snapshots_for_windows(
                        provider=provider,
                        model=model,
                        quota_group=None,
                        credential_id=credential_id,
                        windows=model_stats.windows,
                        source="model",
                    )
                )
        if quota_group:
            group_stats = state.get_group_stats(quota_group, create=False)
            if group_stats:
                snapshots.extend(
                    _snapshots_for_windows(
                        provider=provider,
                        model=model,
                        quota_group=quota_group,
                        credential_id=credential_id,
                        windows=group_stats.windows,
                        source="group",
                    )
                )
    return snapshots


def _snapshots_for_windows(
    *,
    provider: str,
    model: Optional[str],
    quota_group: Optional[str],
    credential_id: Optional[str],
    windows: Mapping[str, WindowStats],
    source: str,
) -> list[QuotaSnapshot]:
    return [
        QuotaSnapshot(
            provider=provider,
            model=model,
            quota_group=quota_group,
            credential_id=credential_id,
            window_name=window.name,
            limit=window.limit,
            used=window.request_count,
            remaining=window.remaining,
            reset_at=window.reset_at,
            source=source,
            metadata={"scope": "request_token_window"},
        )
        for window in windows.values()
    ]
