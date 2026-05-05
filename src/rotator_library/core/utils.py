# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Shared utility functions for the rotator library.
"""

import logging
from typing import Any, Dict

lib_logger = logging.getLogger("rotator_library")


def normalize_usage_for_response(usage: Any, model: str = "") -> None:
    """
    Normalize usage fields so completion_tokens always includes reasoning_tokens.

    Some providers (e.g. Mistral) report reasoning_tokens as a separate count
    that is NOT included in completion_tokens ("exclusive reasoning"). The
    standard OpenAI convention expects reasoning_tokens to be a subset of
    completion_tokens ("inclusive reasoning"). Downstream consumers compute
    text_output = completion_tokens - reasoning_tokens, which produces
    negative values when the exclusive convention is used.

    This function detects the exclusive case (reasoning_tokens > completion_tokens)
    and normalizes to the inclusive convention by adding reasoning_tokens to
    completion_tokens and recalculating total_tokens. The internal tracking
    pipeline is unaffected — only the user-facing response is modified.

    Args:
        usage: Usage dict from a streamed chunk, or a pydantic/model usage
               object from a non-streamed response. Modified in-place.
        model: Model name for logging context.
    """
    if not usage:
        return

    is_dict = isinstance(usage, dict)

    reasoning = 0
    if is_dict:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = details.get("reasoning_tokens") or 0
    else:
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            if isinstance(details, dict):
                reasoning = details.get("reasoning_tokens", 0) or 0
            else:
                reasoning = getattr(details, "reasoning_tokens", 0) or 0

    if reasoning <= 0:
        return

    if is_dict:
        completion = usage.get("completion_tokens") or 0
    else:
        completion = getattr(usage, "completion_tokens", 0) or 0

    if completion >= reasoning:
        return

    new_completion = completion + reasoning
    if is_dict:
        prompt = usage.get("prompt_tokens") or 0
        usage["completion_tokens"] = new_completion
        usage["total_tokens"] = prompt + new_completion
    else:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        usage.completion_tokens = new_completion
        usage.total_tokens = prompt + new_completion

    lib_logger.warning(
        f"Provider usage does not follow inclusive reasoning convention "
        f"(completion_tokens={completion} < reasoning_tokens={reasoning}). "
        f"Auto-normalizing for response. model={model}"
    )
