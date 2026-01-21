# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and not model.startswith("openai/text-embedding-3"):
        del payload["dimensions"]
        
    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if model not in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]:
            del payload["thinking"]
            
    return payload
