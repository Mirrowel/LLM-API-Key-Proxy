# SPDX-License-Identifier: LGPL-3.0-only

"""
xAI CLI proxy billing quota tracking mixin.

Fetches subscription billing usage from GET {XAI_CLI_PROXY_BASE}/billing
using Grok CLI OAuth session headers (not api.x.ai API-key limits).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

from ..openai_oauth_base import _parse_jwt_claims

if TYPE_CHECKING:
    from ...usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

DEFAULT_QUOTA_REFRESH_INTERVAL = 300
BILLING_FETCH_CONCURRENCY = 4

XAI_CLI_PROXY_BASE_DEFAULT = "https://cli-chat-proxy.grok.com/v1"
XAI_CLI_VERSION_DEFAULT = os.getenv("XAI_CLI_VERSION", "0.1.202")


def _get_credential_identifier(credential_path: str) -> str:
    if credential_path.startswith("env://"):
        return credential_path
    return Path(credential_path).name


def _billing_val(node: Any) -> Any:
    if isinstance(node, dict) and "val" in node:
        return node.get("val")
    return node


def _parse_period_end_ts(period_end: Optional[str]) -> Optional[float]:
    if not period_end:
        return None
    try:
        dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _first_billing_val(data: dict, *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return _billing_val(data[key])
    return None


def _as_billing_int(v: Any) -> Optional[int]:
    """Parse billing numeric fields (ints, floats, numeric strings)."""
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return int(round(float(v)))
        if isinstance(v, str) and v.strip():
            return int(round(float(v.strip())))
    except (TypeError, ValueError):
        return None
    return None


def _normalize_billing_root(data: dict) -> dict:
    """Unwrap common envelopes (CLI proxy /billing uses top-level `config`)."""
    for key in ("config", "billing", "result", "data", "payload"):
        inner = data.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    return data


def parse_billing_payload(data: dict) -> Dict[str, Any]:
    """Parse billing JSON (flat, {val:}, billingCycle, usage.totalUsed)."""
    root = _normalize_billing_root(data if isinstance(data, dict) else {})

    monthly_limit = _first_billing_val(
        root, "monthlyLimit", "monthly_limit", "monthly_limit_usd"
    )
    used = _first_billing_val(root, "used", "used_amount", "monthly_used")
    on_demand_cap = _first_billing_val(
        root, "onDemandCap", "on_demand_cap", "on_demand_balance"
    )

    usage = root.get("usage")
    if isinstance(usage, dict):
        if used is None:
            used = _first_billing_val(
                usage, "totalUsed", "total_used", "used", "monthly_used"
            )

    cycle = root.get("billingCycle") or root.get("billing_cycle")
    if isinstance(cycle, dict):
        period_start = _first_billing_val(
            cycle, "billingPeriodStart", "billing_period_start", "period_start"
        )
        period_end = _first_billing_val(
            cycle, "billingPeriodEnd", "billing_period_end", "period_end"
        )
    else:
        period_start = _first_billing_val(
            root, "billingPeriodStart", "billing_period_start", "period_start"
        )
        period_end = _first_billing_val(
            root, "billingPeriodEnd", "billing_period_end", "period_end"
        )

    tier = _first_billing_val(root, "tier", "subscription_tier")

    def _as_int(v: Any) -> Optional[int]:
        return _as_billing_int(v)

    period_end_str = str(period_end) if period_end is not None else None
    period_start_str = str(period_start) if period_start is not None else None

    return {
        "monthly_limit": _as_int(monthly_limit),
        "used": _as_int(used),
        "on_demand_cap": _as_int(on_demand_cap),
        "period_start": period_start_str,
        "period_end": period_end_str,
        "period_end_ts": _parse_period_end_ts(period_end_str),
        "tier": _as_int(tier),
    }


@dataclass
class XAiBillingSnapshot:
    credential_path: str
    identifier: str
    monthly_limit: Optional[int]
    used: Optional[int]
    on_demand_cap: Optional[int]
    period_start: Optional[str]
    period_end: Optional[str]
    period_end_ts: Optional[float]
    tier: Optional[int]
    fetched_at: float
    status: str
    error: Optional[str]


class XAiQuotaTracker:
    """
    Mixin for xAI OAuth credentials: CLI proxy /billing quota baselines.

    Usage:
        class XAiProvider(XAiAuthBase, XAiQuotaTracker, ProviderInterface):
            ...
    """

    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, XAiBillingSnapshot]
    _quota_refresh_interval: int
    _usage_manager: Optional["UsageManager"]
    _initial_baselines_fetched: bool

    def _init_quota_tracker(self) -> None:
        self._quota_cache = {}
        self._quota_refresh_interval = DEFAULT_QUOTA_REFRESH_INTERVAL
        self._usage_manager = None
        self._initial_baselines_fetched = False

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        self._usage_manager = usage_manager

    def _resolve_cli_proxy_base(self) -> str:
        return getattr(self, "cli_proxy_base", None) or os.getenv(
            "XAI_CLI_PROXY_BASE", XAI_CLI_PROXY_BASE_DEFAULT
        )

    def _resolve_cli_version(self) -> str:
        return getattr(self, "_cli_version", None) or os.getenv(
            "XAI_CLI_VERSION", XAI_CLI_VERSION_DEFAULT
        )

    async def _resolve_user_id(self, credential_path: str) -> str:
        creds = await self._load_credentials(credential_path)
        token = creds.get("access_token", "")
        if token:
            claims = _parse_jwt_claims(token)
            for key in ("principal_id", "sub", "user_id"):
                if claims.get(key):
                    return str(claims[key])
        account_id = creds.get("account_id")
        if account_id:
            return str(account_id)
        meta = creds.get("_proxy_metadata") or {}
        if meta.get("account_id"):
            return str(meta["account_id"])
        raise ValueError("Could not resolve x-userid for billing request")

    def _build_billing_headers(self, bearer_token: str, user_id: str) -> Dict[str, str]:
        ver = self._resolve_cli_version()
        return {
            "Authorization": f"Bearer {bearer_token}",
            "x-xai-token-auth": "xai-grok-cli",
            "x-userid": user_id,
            "x-grok-client-version": ver,
            "User-Agent": f"grok-pager/{ver} grok-shell/{ver}",
            "Accept": "application/json",
        }

    async def _fetch_billing_for_credential(
        self, credential_path: str
    ) -> XAiBillingSnapshot:
        identifier = _get_credential_identifier(credential_path)
        if credential_path.startswith("env://"):
            return XAiBillingSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                monthly_limit=None,
                used=None,
                on_demand_cap=None,
                period_start=None,
                period_end=None,
                period_end_ts=None,
                tier=None,
                fetched_at=time.time(),
                status="error",
                error="Billing fetch skipped for env credentials (OAuth files only)",
            )

        try:
            auth_headers = await self.get_auth_header(credential_path)
            auth = auth_headers.get("Authorization", "")
            token = auth.replace("Bearer ", "").strip()
            if not token:
                raise ValueError("Empty OAuth access token")

            user_id = await self._resolve_user_id(credential_path)
            headers = self._build_billing_headers(token, user_id)
            base = self._resolve_cli_proxy_base().rstrip("/")
            url = f"{base}/billing"

            proxy_kwargs = {}
            if hasattr(self, "_build_proxy_client_kwargs"):
                proxy_kwargs = self._build_proxy_client_kwargs(credential_path)

            async with httpx.AsyncClient(timeout=30.0, **proxy_kwargs) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

            parsed = parse_billing_payload(data if isinstance(data, dict) else {})
            if parsed.get("monthly_limit") is None:
                root = data if isinstance(data, dict) else {}
                inner = _normalize_billing_root(root)
                lib_logger.warning(
                    "x-ai: billing HTTP 200 but monthly_limit unset; "
                    f"root_keys={list(root.keys())[:12]} inner_keys={list(inner.keys())[:12]}"
                )
            snapshot = XAiBillingSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                monthly_limit=parsed["monthly_limit"],
                used=parsed["used"],
                on_demand_cap=parsed["on_demand_cap"],
                period_start=parsed["period_start"],
                period_end=parsed["period_end"],
                period_end_ts=parsed["period_end_ts"],
                tier=parsed["tier"],
                fetched_at=time.time(),
                status="success",
                error=None,
            )
            self._quota_cache[credential_path] = snapshot
            return snapshot

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            lib_logger.warning(f"xAI billing fetch failed for {identifier}: {error_msg}")
            return XAiBillingSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                monthly_limit=None,
                used=None,
                on_demand_cap=None,
                period_start=None,
                period_end=None,
                period_end_ts=None,
                tier=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )
        except Exception as e:
            error_msg = str(e)
            lib_logger.warning(f"xAI billing fetch failed for {identifier}: {error_msg}")
            return XAiBillingSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                monthly_limit=None,
                used=None,
                on_demand_cap=None,
                period_start=None,
                period_end=None,
                period_end_ts=None,
                tier=None,
                fetched_at=time.time(),
                status="error",
                error=error_msg,
            )

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
        force: bool = False,
        is_initial_fetch: bool = False,
    ) -> int:
        stored_count = 0
        provider_prefix = getattr(self, "provider_env_name", "x-ai")

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            monthly_limit = quota_data.get("monthly_limit")
            used = quota_data.get("used") or 0
            period_end_ts = quota_data.get("period_end_ts")
            on_demand_cap = quota_data.get("on_demand_cap")

            if monthly_limit is not None and monthly_limit >= 0:
                exhausted = used >= monthly_limit
                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_billing_monthly",
                        quota_max_requests=monthly_limit,
                        quota_reset_ts=period_end_ts,
                        quota_used=used,
                        quota_group="monthly-limit",
                        force=force,
                        apply_exhaustion=exhausted and is_initial_fetch,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store xAI monthly baseline for {cred_path}: {e}"
                    )

            if on_demand_cap is not None and on_demand_cap > 0:
                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_billing_ondemand",
                        quota_max_requests=on_demand_cap,
                        quota_reset_ts=period_end_ts,
                        quota_used=0,
                        quota_group="on-demand($)",
                        force=force,
                        apply_exhaustion=False,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store xAI on-demand baseline for {cred_path}: {e}"
                    )

        return stored_count

    async def fetch_initial_baselines(
        self, credential_paths: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        oauth_paths = [p for p in credential_paths if not p.startswith("env://")]
        if not oauth_paths:
            return {}

        lib_logger.info(
            f"x-ai: Fetching billing baselines for {len(oauth_paths)} OAuth credentials..."
        )

        results: Dict[str, Dict[str, Any]] = {}
        semaphore = asyncio.Semaphore(BILLING_FETCH_CONCURRENCY)

        async def fetch_one(cred_path: str):
            async with semaphore:
                snapshot = await self._fetch_billing_for_credential(cred_path)
                return cred_path, snapshot

        tasks = [fetch_one(c) for c in oauth_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in fetch_results:
            if isinstance(item, Exception):
                lib_logger.warning(f"xAI billing baseline error: {item}")
                continue
            cred_path, snapshot = item
            if snapshot.status == "success":
                results[cred_path] = {
                    "status": "success",
                    "error": None,
                    "monthly_limit": snapshot.monthly_limit,
                    "used": snapshot.used,
                    "on_demand_cap": snapshot.on_demand_cap,
                    "period_end_ts": snapshot.period_end_ts,
                    "tier": snapshot.tier,
                    "fetched_at": snapshot.fetched_at,
                }
            else:
                results[cred_path] = {
                    "status": "error",
                    "error": snapshot.error or "Unknown error",
                }

        success_count = sum(1 for v in results.values() if v.get("status") == "success")
        lib_logger.info(
            f"x-ai: Fetched {success_count}/{len(oauth_paths)} billing baselines"
        )
        return results

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "interval": self._quota_refresh_interval,
            "name": "xai_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        oauth_creds = [c for c in credentials if not c.startswith("env://")]
        if not oauth_creds:
            return

        self._usage_manager = usage_manager
        quota_results = await self.fetch_initial_baselines(oauth_creds)
        is_initial = not self._initial_baselines_fetched
        stored = await self._store_baselines_to_usage_manager(
            quota_results,
            usage_manager,
            force=True,
            is_initial_fetch=is_initial,
        )
        if stored > 0:
            self._initial_baselines_fetched = True
        elif any(r.get("status") == "success" for r in quota_results.values()):
            lib_logger.warning(
                "x-ai: billing fetch succeeded but no quota baselines were stored "
                "(check monthly_limit/on_demand_cap parsing)"
            )