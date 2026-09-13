# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import httpx
import logging
import os
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.chutes_quota_tracker import ChutesQuotaTracker

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
    """Chutes — OpenAI-compatible gateway over TEE-hosted open models.

    Quota is dollar-based at the credential level: a 402 means the account
    balance or plan allowance is gone (never retried), 429 is throttling.
    Model ids keep their vendor prefix and -TEE suffix; routing syntax
    (``default``, comma lists, ``:latency``/``:throughput``) never appears
    in listings. The ``chutes`` wire adapter owns parameter hygiene and
    the dual reasoning-field spellings (vLLM ``reasoning`` vs SGLang
    ``reasoning_content``).
    """

    protocol_name = "openai_chat"
    native_streaming_supported = True
    default_api_base = "https://llm.chutes.ai/v1"
    adapter_names = ("chutes",)
    quota_api_base = "https://api.chutes.ai"
    model_quota_groups = {
        "chutes_global": ["_quota"],
    }
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=86400,
            mode="per_model",
            description="Chutes daily quota",
            field_name="daily",
        )
    }

    def __init__(self, *args, **kwargs):
        """Initialize ChutesProvider with quota tracking."""
        super().__init__(*args, api_base=self.quota_api_base, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            os.environ.get("CHUTES_QUOTA_REFRESH_INTERVAL", "300")
        )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        return "chutes_global"

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota usage refresh.

        Returns:
            Background job configuration for quota refresh
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "chutes_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def refresh_single_credential(
            api_key: str, client: httpx.AsyncClient
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        # Update quota cache
                        self._quota_cache[api_key] = usage_data

                        # Calculate values for usage manager
                        remaining_fraction = usage_data.get("remaining_fraction", 0.0)
                        quota = usage_data.get("quota", 0)
                        reset_ts = usage_data.get("reset_at")

                        # Store baseline in usage manager
                        # Since Chutes uses credential-level quota, we use a virtual model name
                        quota_used = (
                            int((1.0 - remaining_fraction) * quota) if quota > 0 else 0
                        )
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "chutes/_quota",  # Virtual model for credential-level tracking
                            quota_max_requests=quota,
                            quota_reset_ts=reset_ts,
                            quota_used=quota_used,
                        )

                        lib_logger.debug(
                            f"Updated Chutes quota baseline for credential: "
                            f"{usage_data['remaining']:.0f}/{quota} remaining "
                            f"({remaining_fraction * 100:.0f}%)"
                        )

                except Exception as e:
                    lib_logger.warning(f"Failed to refresh Chutes quota usage: {e}")

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    def _models_url(self) -> str:
        return f"{self.get_provider_api_base()}/models"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(self._models_url())
            response.raise_for_status()
            models = []
            for model in response.json().get("data", []):
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("id") or "")
                if not model_id or model_id.startswith("default") or "," in model_id:
                    continue
                models.append(f"chutes/{model_id}")
            return models
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Chutes models: {e}")
            return []
