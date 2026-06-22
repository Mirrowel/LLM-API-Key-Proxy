# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Umans Provider

Provider for Umans (https://umans.ai) via api.code.umans.ai.
OpenAI-compatible API with request-based sliding-window quota tracking.

Environment variables:
    UMANS_API_KEY_1=<api_key>     # primary API key
    UMANS_API_KEY=<api_key>       # single-key shorthand
    UMANS_API_BASE=https://api.code.umans.ai  # optional override
    UMANS_QUOTA_REFRESH_INTERVAL=300  # optional, seconds
    UMANS_QUOTA_LIMIT=200         # optional override for code_pro request limit
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.umans_quota_tracker import UmansQuotaTracker

lib_logger = logging.getLogger("rotator_library")


class UmansProvider(UmansQuotaTracker, ProviderInterface):
    """
    Provider implementation for the Umans API.

    Tracks a request-based 5-hour sliding window for code_pro plan keys and
    concurrency usage for all plans. The proxy does not block rotation on the
    soft request limit until the burst ceiling behavior is confirmed; API 429s
    are handled by the standard error cooldown path.
    """

    # LiteLLM has no Umans pricing; quota is request-based from the API.
    skip_cost_calculation = True

    # Provider config for env var lookups (e.g. QUOTA_GROUPS_UMANS_*)
    provider_env_name = "umans"

    # Two quota groups for the TUI:
    #   5h-requests  — code_pro request window (display-only)
    #   concurrency  — concurrent sessions (display-only)
    model_quota_groups = {
        "5h-requests": ["_requests_5h"],
        "concurrency": ["_concurrent"],
    }

    # Concurrency is pushed for internal routing/display but does not trigger
    # rotation exhaustion on its own.
    hidden_quota_groups = frozenset({"concurrency"})

    # 5-hour sliding window, shared across all models for a credential.
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=18000,  # 5 hours
            mode="credential",
            description="Umans 5-hour request window",
            field_name="5h",
        )
    }

    # Safe default: most restrictive plan (code_pro) concurrency limit.
    # max-plan credentials get a per-credential override once detected.
    default_max_concurrent_per_key = 3

    def __init__(self, *args, **kwargs):
        """Initialize UmansProvider with request-based quota tracking."""
        super().__init__(*args, **kwargs)
        self._init_quota_tracker()

    # =====================================================================
    # CONCURRENCY
    # =====================================================================

    def get_credential_concurrency_limit(self, credential: str) -> Optional[int]:
        """
        Return per-credential concurrency limit from the quota snapshot.

        This lets the rotation system use 4 concurrent sessions for max-plan
        keys while keeping 3 as the safe class default.
        """
        snapshot = self._quota_cache.get(credential)
        if snapshot and snapshot.concurrency_limit > 0:
            return snapshot.concurrency_limit
        return None  # Fall back to class default (3)

    # =====================================================================
    # QUOTA GROUPING
    # =====================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Umans shares one request pool across all models per credential.

        Returns the 5h-requests group for any known Umans model; the
        concurrency synthetic model also belongs to its group.
        """
        clean_model = model.split("/")[-1] if "/" in model else model
        if clean_model in self.model_quota_groups.get("concurrency", []):
            return "concurrency"
        return "5h-requests"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """Return all synthetic models belonging to a quota group."""
        return list(self.model_quota_groups.get(group, []))

    def get_quota_groups(self) -> List[str]:
        """Return the list of quota groups for this provider."""
        return ["5h-requests", "concurrency"]

    # =====================================================================
    # MODEL DISCOVERY
    # =====================================================================

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Umans /v1/models endpoint.

        Args:
            api_key: Umans API key
            client: HTTP client for connection reuse

        Returns:
            List of model names prefixed with 'umans/'
        """
        try:
            base = os.getenv("UMANS_API_BASE", "https://api.code.umans.ai").rstrip("/")
            response = await client.get(
                f"{base}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                if model_id:
                    models.append(f"umans/{model_id}")

            if models:
                lib_logger.info(f"Discovered {len(models)} Umans models")
            return models
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Umans models: {e}")
            return []

    # =====================================================================
    # ERROR PARSING
    # =====================================================================

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Umans-specific quota and rate-limit errors.

        The proxy relies on the API to enforce the actual request limit; a 429
        response puts the credential on cooldown through the standard error
        handler path.
        """
        body = error_body
        if not body:
            response = getattr(error, "response", None)
            if response is not None:
                body = getattr(response, "text", None)
            if not body:
                err_body = getattr(error, "body", None)
                body = str(err_body) if err_body is not None else None
            if not body:
                body = str(error)

        body_lower = body.lower() if body else ""

        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)

        if status_code == 429 or "rate limit" in body_lower or "too many requests" in body_lower:
            return {
                "retry_after": None,
                "reason": "RATE_LIMITED",
            }

        if status_code == 403 or "forbidden" in body_lower:
            return {
                "retry_after": None,
                "reason": "FORBIDDEN",
            }

        return None
