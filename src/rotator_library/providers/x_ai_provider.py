# SPDX-License-Identifier: LGPL-3.0-only

# src/rotator_library/providers/xai_provider.py
"""
xAI Grok Provider

Provider for xAI Grok models via OAuth2 authentication (SuperGrok / X Premium+).
Routes requests through LiteLLM's built-in xAI support (`xai/` prefix).

Two API endpoints are supported:
  - Standard API: https://api.x.ai/v1 (public models like grok-4.3)
  - CLI Proxy:    https://cli-chat-proxy.grok.com/v1 (agentic models like
                  grok-composer-2.5-fast and grok-build with 512K context)

Models are discovered from both endpoints.  CLI-proxy-only models are routed
through the CLI proxy with the required ``User-Agent: grok/<version>`` header;
all other models go through the standard API.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, List, Optional, Union

import httpx
import litellm
import openai

from .provider_interface import ProviderInterface
from .x_ai_auth_base import XAiAuthBase
from .utilities.x_ai_quota_tracker import XAiQuotaTracker
from ..model_definitions import ModelDefinitions
from ..error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

XAI_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
XAI_CLI_PROXY_BASE = os.getenv(
    "XAI_CLI_PROXY_BASE", "https://cli-chat-proxy.grok.com/v1"
)

# Minimum CLI version the proxy accepts (426 Upgrade Required otherwise)
XAI_CLI_VERSION = os.getenv("XAI_CLI_VERSION", "0.1.202")

# Params accepted by litellm.acompletion for xAI (OpenAI-compatible)
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "extra_headers",
    "extra_body",
    "api_key",
    "api_base",
    "custom_llm_provider",
    "client",
}


class XAiProvider(XAiAuthBase, XAiQuotaTracker, ProviderInterface):
    """
    Provider for xAI Grok models using OAuth2 credentials.

    Authentication:
      - OAuth credentials stored as JSON files (via XaiAuthBase PKCE/Device flow)
      - Access token injected as Bearer auth for the OpenAI-compatible API

    Model routing:
      - Standard API models use LiteLLM's `xai/` prefix
      - CLI proxy models route through cli-chat-proxy.grok.com with version header
      - Model discovery from both endpoints, merged and deduplicated
    """

    provider_env_name = "x-ai"

    model_quota_groups = {
        "monthly-limit": ["_billing_monthly"],
        "on-demand($)": ["_billing_ondemand"],
    }

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Map chat models to monthly billing pool; virtual buckets for display."""
        clean = model.split("/")[-1] if "/" in model else model
        if clean == "_billing_ondemand":
            return "on-demand($)"
        if clean == "_billing_monthly":
            return "monthly-limit"
        return "monthly-limit"

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()
        self.api_base = XAI_API_BASE
        self.cli_proxy_base = XAI_CLI_PROXY_BASE
        self._cli_version = XAI_CLI_VERSION
        self.model_definitions = ModelDefinitions()
        # Models that are only available on the CLI proxy (not on api.x.ai)
        self._cli_proxy_models: set = set()
        # Context window metadata from CLI proxy discovery
        # Maps bare model id -> context_window (e.g. {"grok-build": 512000})
        self._cli_proxy_metadata: dict = {}
        lib_logger.debug(
            f"XAiProvider initialized: base={self.api_base}, "
            f"cli_proxy={self.cli_proxy_base}"
        )

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return the list of available xAI models.

        Discovery order:
        1. Static override from environment / model_definitions
        2. Live fetch from both xAI standard API and CLI proxy
        3. Hardcoded fallback
        """
        # 1. Check static model definitions first
        static_models = self.model_definitions.get_all_provider_models("x-ai")
        if static_models:
            return static_models

        # Resolve OAuth credential to token (needed for both endpoints)
        try:
            auth_header = await self.get_auth_header(api_key)
            token = auth_header.get("Authorization", "").replace("Bearer ", "")
        except Exception as e:
            lib_logger.warning(f"Failed to resolve xAI OAuth token for model discovery: {e}")
            return ["x-ai/grok-3", "x-ai/grok-3-mini"]

        standard_ids: set = set()
        cli_proxy_ids: set = set()

        # 2a. Fetch from standard API (api.x.ai)
        try:
            response = await client.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            standard_ids = {
                m["id"] for m in data.get("data", []) if m.get("id")
            }
            if standard_ids:
                lib_logger.info(
                    f"Discovered {len(standard_ids)} models from xAI standard API"
                )
        except Exception as e:
            lib_logger.warning(f"Failed to fetch xAI standard API models: {e}")

        # 2b. Fetch from CLI proxy (cli-chat-proxy.grok.com)
        try:
            response = await client.get(
                f"{self.cli_proxy_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            for m in data.get("data", []):
                mid = m.get("id")
                if not mid:
                    continue
                cli_proxy_ids.add(mid)
                # Capture context_window metadata from CLI proxy
                ctx = m.get("context_window")
                if ctx:
                    self._cli_proxy_metadata[mid] = int(ctx)
            if cli_proxy_ids:
                lib_logger.info(
                    f"Discovered {len(cli_proxy_ids)} models from xAI CLI proxy: "
                    f"{', '.join(sorted(cli_proxy_ids))}"
                )
        except Exception as e:
            lib_logger.warning(f"Failed to fetch xAI CLI proxy models: {e}")

        # Determine CLI-proxy-only models.
        # Some models (like grok-composer-2.5-fast) are listed by the CLI
        # proxy but also work on the standard API as "hidden" models (not in
        # /v1/models).  We only treat a model as truly CLI-proxy-only if it
        # has a name-stem collision with a standard API model that has a
        # version suffix (e.g. CLI "grok-build" vs standard "grok-build-0.1"),
        # indicating the CLI proxy exposes a versionless alias.
        cli_only_candidates = cli_proxy_ids - standard_ids
        truly_cli_only: set = set()
        for mid in cli_only_candidates:
            # Check if standard API has a versioned variant (e.g. mid-0.1)
            has_versioned_sibling = any(
                sid.startswith(f"{mid}-") for sid in standard_ids
            )
            if has_versioned_sibling:
                truly_cli_only.add(mid)
        self._cli_proxy_models = truly_cli_only
        if self._cli_proxy_models:
            lib_logger.info(
                f"CLI-proxy-only models: {', '.join(sorted(self._cli_proxy_models))}"
            )

        # Merge: all unique model IDs, prefixed with x-ai/
        # Exclude CLI-proxy aliases (e.g. grok-build) since they're just
        # versionless aliases for standard API models (e.g. grok-build-0.1).
        all_ids = (standard_ids | cli_proxy_ids) - truly_cli_only
        if all_ids:
            return sorted(f"x-ai/{mid}" for mid in all_ids)

        # 3. Graceful fallback
        return ["x-ai/grok-3", "x-ai/grok-3-mini"]

    def get_model_context_overrides(self) -> dict:
        """
        Return context window overrides for xAI models discovered from
        the CLI proxy that don't have catalog metadata.

        Returns:
            Dict mapping full model ID (e.g. "x-ai/grok-build") to
            context_window size in tokens.
        """
        return {
            f"x-ai/{mid}": ctx
            for mid, ctx in self._cli_proxy_metadata.items()
        }

    def has_custom_logic(self) -> bool:
        """
        xAI requires custom logic to inject OAuth bearer token.

        The standard LiteLLM flow sets api_key = credential_path (file path),
        which won't work for OAuth providers. We override acompletion to
        resolve the credential file into an actual token.
        """
        return True

    def _get_cli_proxy_headers(self) -> dict:
        """Return extra headers required by the CLI chat proxy."""
        ver = XAI_CLI_VERSION
        return {
            "User-Agent": f"grok/{ver}",
            "x-xai-token-auth": "xai-grok-cli",
            "x-grok-client-version": ver,
        }

    def _is_cli_proxy_model(self, model_bare: str) -> bool:
        """Check if a model should be routed through the CLI proxy."""
        return model_bare in self._cli_proxy_models

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Make a chat completion request to xAI via LiteLLM.

        Resolves the OAuth credential file path into a bearer token.
        Routes CLI-proxy-only models through cli-chat-proxy.grok.com with
        the required version header; all other models go through api.x.ai.
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        # Resolve OAuth credential to access token
        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token from credential: "
                f"{mask_credential(credential)}"
            )

        # Select endpoint based on model
        use_cli_proxy = self._is_cli_proxy_model(model_bare)
        api_base = self.cli_proxy_base if use_cli_proxy else self.api_base

        # Route through LiteLLM as xai/model
        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "xai"

        # Inject CLI proxy headers if needed
        if use_cli_proxy:
            extra_headers = self._get_cli_proxy_headers()
            existing_headers = kwargs.get("extra_headers") or {}
            kwargs["extra_headers"] = {**existing_headers, **extra_headers}
            lib_logger.debug(
                f"xai: routing {model_bare} through CLI proxy with version header"
            )

        # Set up async OpenAI client for LiteLLM
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=api_base,
            http_client=client,
        )

        # Strip unsupported params
        unsupported = set(kwargs.keys()) - SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(
                f"xai: stripping unsupported params for {model}: {unsupported}"
            )
            kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        return await litellm.acompletion(**kwargs)

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        """
        Make an embedding request to xAI via LiteLLM.
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        # Resolve OAuth credential to access token
        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token for embedding: "
                f"{mask_credential(credential)}"
            )

        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "xai"

        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=self.api_base,
            http_client=client,
        )

        return await litellm.aembedding(**kwargs)
