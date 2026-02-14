# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any, Set, Optional

from .token_calculator import adjust_max_tokens_in_payload


# Kilocode/OpenRouter free models often have limited parameter support
# These parameters are commonly unsupported and cause 400 errors
KILOCODE_UNSUPPORTED_PARAMS: Set[str] = {
    "stream_options",  # Not supported by many free models
    "frequency_penalty",  # Often unsupported
    "presence_penalty",  # Often unsupported
    "top_p",  # Sometimes unsupported
    "top_k",  # Sometimes unsupported
    "stop",  # Can cause issues with some models
    "n",  # Number of completions - often unsupported
    "logprobs",  # Often unsupported
    "top_logprobs",  # Often unsupported
    "user",  # User identifier - often ignored but can cause issues
    "seed",  # Not supported by all models
    "response_format",  # Only supported by some models
    "reasoning_effort",  # OpenAI-specific, not supported by Kilocode/Novita free models
}


def sanitize_request_payload(
    payload: Dict[str, Any],
    model: str,
    registry: Optional[Any] = None,
    auto_adjust_max_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Sanitizes and adjusts the request payload based on the model.

    This function:
    1. Removes unsupported parameters for specific providers
    2. Automatically adjusts max_tokens to prevent context window overflow

    Args:
        payload: The request payload dictionary
        model: The model identifier (e.g., "openai/gpt-4o")
        registry: Optional ModelRegistry instance for context window lookup
        auto_adjust_max_tokens: Whether to auto-adjust max_tokens (default: True)

    Returns:
        Sanitized payload dictionary
    """
    normalized_model = model.strip().lower() if isinstance(model, str) else ""

    if "dimensions" in payload and not normalized_model.startswith(
        "openai/text-embedding-3"
    ):
        del payload["dimensions"]

    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if normalized_model not in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]:
            del payload["thinking"]

    # Kilocode provider - remove unsupported parameters for free models
    # Free models through Kilocode/OpenRouter often reject extra parameters
    if normalized_model.startswith("kilocode/"):
        model_without_prefix = normalized_model.split("/", 1)[1] if "/" in normalized_model else normalized_model

        # Free models have stricter parameter requirements
        if ":free" in model_without_prefix or model_without_prefix.startswith("z-ai/"):
            for param in KILOCODE_UNSUPPORTED_PARAMS:
                if param in payload:
                    del payload[param]

    # Auto-adjust max_tokens to prevent context window overflow
    if auto_adjust_max_tokens:
        payload = adjust_max_tokens_in_payload(payload, model, registry)

    return payload
