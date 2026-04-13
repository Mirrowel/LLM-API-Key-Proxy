# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and "embedding" not in model:
        del payload["dimensions"]

    # Models that support the thinking parameter
    _supports_thinking = (
        model.startswith("anthropic/") or "claude-" in model
        or any(p in model for p in ("gemini-2.0-", "gemini-2.5-"))
    )

    # Strip top-level thinking key for models that don't support it
    if "thinking" in payload and not _supports_thinking:
        del payload["thinking"]

    # Strip extra_body.thinking for models that don't support it
    extra = payload.get("extra_body")
    if isinstance(extra, dict) and "thinking" in extra and not _supports_thinking:
        del extra["thinking"]
        if not extra:
            del payload["extra_body"]

    return payload
