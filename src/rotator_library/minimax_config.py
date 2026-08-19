# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""MiniMax endpoint and model configuration."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional


lib_logger = logging.getLogger("rotator_library")


GLOBAL_EN = "global_en"
CN_ZH = "cn_zh"
OPENAI_PROTOCOL = "openai"
ANTHROPIC_PROTOCOL = "anthropic"

MINIMAX_ENDPOINTS: Dict[str, Dict[str, str]] = {
    GLOBAL_EN: {
        OPENAI_PROTOCOL: "https://api.minimax.io/v1",
        ANTHROPIC_PROTOCOL: "https://api.minimax.io/anthropic",
    },
    CN_ZH: {
        OPENAI_PROTOCOL: "https://api.minimaxi.com/v1",
        ANTHROPIC_PROTOCOL: "https://api.minimaxi.com/anthropic",
    },
}

MINIMAX_MODEL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "pricing_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
        "interleaved": True,
    },
    "MiniMax-M2.7": {
        "context_window": 204_800,
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        "input_modalities": ["text"],
        "thinking": ["always_on"],
    },
}

MINIMAX_DEFAULT_MODELS = tuple(MINIMAX_MODEL_DEFINITIONS)

_REGION_ENV_VARS = {
    GLOBAL_EN: {
        OPENAI_PROTOCOL: "MINIMAX_GLOBAL_OPENAI_BASE_URL",
        ANTHROPIC_PROTOCOL: "MINIMAX_GLOBAL_ANTHROPIC_BASE_URL",
    },
    CN_ZH: {
        OPENAI_PROTOCOL: "MINIMAX_CN_OPENAI_BASE_URL",
        ANTHROPIC_PROTOCOL: "MINIMAX_CN_ANTHROPIC_BASE_URL",
    },
}


def get_minimax_region() -> str:
    """Return the configured endpoint region, defaulting to the global service."""
    region = os.getenv("MINIMAX_API_REGION", GLOBAL_EN).strip().lower()
    return region if region in MINIMAX_ENDPOINTS else GLOBAL_EN


def get_minimax_protocol() -> str:
    """Return the configured upstream protocol."""
    protocol = os.getenv("MINIMAX_API_PROTOCOL", OPENAI_PROTOCOL).strip().lower()
    return (
        protocol
        if protocol in (OPENAI_PROTOCOL, ANTHROPIC_PROTOCOL)
        else OPENAI_PROTOCOL
    )


def get_minimax_endpoint(
    region: Optional[str] = None,
    protocol: Optional[str] = None,
) -> str:
    """Resolve a user-configured or default MiniMax endpoint."""
    selected_region = region or get_minimax_region()
    selected_protocol = protocol or get_minimax_protocol()

    if selected_region not in MINIMAX_ENDPOINTS:
        selected_region = GLOBAL_EN
    if selected_protocol not in (OPENAI_PROTOCOL, ANTHROPIC_PROTOCOL):
        selected_protocol = OPENAI_PROTOCOL

    env_var = _REGION_ENV_VARS[selected_region][selected_protocol]
    override = os.getenv(env_var, "").strip()
    if not override:
        if selected_protocol == OPENAI_PROTOCOL:
            override = os.getenv("MINIMAX_API_BASE", "").strip()

    endpoint = override or MINIMAX_ENDPOINTS[selected_region][selected_protocol]
    endpoint = endpoint.rstrip("/")
    if selected_protocol == ANTHROPIC_PROTOCOL and not endpoint.endswith("/anthropic"):
        lib_logger.warning(
            "Invalid MiniMax Anthropic base URL; using the selected default endpoint"
        )
        endpoint = MINIMAX_ENDPOINTS[selected_region][selected_protocol]
    return endpoint
