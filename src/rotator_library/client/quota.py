# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Quota/statistics operations for RotatingClient."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class QuotaService:
    """Aggregate usage stats and force-refresh provider quota baselines."""

    def __init__(
        self,
        *,
        usage_managers: Dict[str, Any],
        all_credentials: Dict[str, List[str]],
        provider_plugins: Dict[str, Any],
        safe_scope_name: Callable[[str], str],
        get_provider_instance: Callable[[str], Optional[Any]],
    ):
        self._usage_managers = usage_managers
        self._all_credentials = all_credentials
        self._provider_plugins = provider_plugins
        self._safe_scope_name = safe_scope_name
        self._get_provider_instance = get_provider_instance

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        providers = {}

        classifier_prefix = (
            f"classifier:{self._safe_scope_name(classifier)}:"
            if classifier is not None
            else None
        )

        for manager_key, manager in self._usage_managers.items():
            if classifier_prefix and not manager_key.startswith(classifier_prefix):
                continue
            if classifier is None and manager_key.startswith("classifier:"):
                continue

            provider_name = manager.provider
            if provider_filter and provider_name != provider_filter:
                continue

            stats = await manager.get_stats_for_endpoint()
            if classifier is not None:
                stats["classifier"] = classifier

            if stats.get("total_requests", 0) == 0:
                continue

            providers[manager_key if classifier is not None else provider_name] = stats

        summary = {
            "total_providers": len(providers),
            "total_credentials": 0,
            "active_credentials": 0,
            "exhausted_credentials": 0,
            "total_requests": 0,
            "tokens": {
                "input_cached": 0,
                "input_uncached": 0,
                "input_cache_pct": 0,
                "output": 0,
            },
            "approx_total_cost": None,
        }

        for prov in providers.values():
            summary["total_credentials"] += prov.get("credential_count", 0)
            summary["active_credentials"] += prov.get("active_count", 0)
            summary["exhausted_credentials"] += prov.get("exhausted_count", 0)
            summary["total_requests"] += prov.get("total_requests", 0)
            tokens = prov.get("tokens", {})
            summary["tokens"]["input_cached"] += tokens.get("input_cached", 0)
            summary["tokens"]["input_uncached"] += tokens.get("input_uncached", 0)
            summary["tokens"]["output"] += tokens.get("output", 0)

        total_input = (
            summary["tokens"]["input_cached"] + summary["tokens"]["input_uncached"]
        )
        summary["tokens"]["input_cache_pct"] = (
            round(summary["tokens"]["input_cached"] / total_input * 100, 1)
            if total_input > 0
            else 0
        )

        approx_total_cost = 0.0
        has_cost = False
        for prov in providers.values():
            cost = prov.get("approx_cost")
            if cost:
                approx_total_cost += cost
                has_cost = True
        summary["approx_total_cost"] = approx_total_cost if has_cost else None

        return {
            "providers": providers,
            "summary": summary,
            "data_source": "cache",
            "timestamp": time.time(),
        }

    async def reload_usage_from_disk(self) -> None:
        for manager in self._usage_managers.values():
            await manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "action": "force_refresh",
            "scope": "credential"
            if credential
            else ("provider" if provider else "all"),
            "provider": provider,
            "credential": credential,
            "credentials_refreshed": 0,
            "success_count": 0,
            "failed_count": 0,
            "duration_ms": 0,
            "errors": [],
        }

        start_time = time.time()

        if provider:
            providers_to_refresh = (
                [provider] if provider in self._all_credentials else []
            )
        else:
            providers_to_refresh = list(self._all_credentials.keys())

        for prov in providers_to_refresh:
            provider_class = self._provider_plugins.get(prov)
            if not provider_class:
                continue

            provider_instance = self._get_provider_instance(prov)
            if not provider_instance:
                continue

            if hasattr(provider_instance, "fetch_initial_baselines"):
                if credential:
                    creds_to_refresh = []
                    for cred_path in self._all_credentials.get(prov, []):
                        if cred_path.endswith(credential) or cred_path == credential:
                            creds_to_refresh.append(cred_path)
                            break
                else:
                    creds_to_refresh = self._all_credentials.get(prov, [])

                if not creds_to_refresh:
                    continue

                try:
                    quota_results = await provider_instance.fetch_initial_baselines(
                        creds_to_refresh
                    )

                    usage_manager = self._usage_managers.get(prov)
                    if usage_manager and hasattr(
                        provider_instance, "_store_baselines_to_usage_manager"
                    ):
                        stored = await provider_instance._store_baselines_to_usage_manager(
                            quota_results,
                            usage_manager,
                            force=True,
                            is_initial_fetch=True,
                        )
                        result["success_count"] += stored

                    result["credentials_refreshed"] += len(creds_to_refresh)

                    for cred_path, data in quota_results.items():
                        if data.get("status") != "success":
                            result["failed_count"] += 1
                            result["errors"].append(
                                f"{Path(cred_path).name}: {data.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    lib_logger.error(f"Failed to refresh quota for {prov}: {e}")
                    result["errors"].append(f"{prov}: {str(e)}")
                    result["failed_count"] += len(creds_to_refresh)

        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result
