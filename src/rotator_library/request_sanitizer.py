# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.

    The legacy hardcoded gemini-2.5 thinking-budget list was deleted (G8):
    thinking controls are owned by the declared capability rows + the effort
    chain, never by a model-name literal in the client request path.
    """
    if "dimensions" in payload and not model.startswith("openai/text-embedding-3"):
        del payload["dimensions"]

    return payload
