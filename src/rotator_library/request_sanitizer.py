# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.

    The legacy hardcoded gemini-2.5 thinking-budget list was deleted (G8):
    thinking controls are owned by the declared capability rows + the effort
    chain, never by a model-name literal in the client request path.

    The dimensions prefix hack died with it (G9): whether a model accepts
    ``dimensions`` is per-model capability data — the validation layer
    checks the value's SHAPE (positive int), the model database seam will
    own per-model legality, and a provider that rejects the word answers
    its own honest 400 instead of the control silently vanishing here.
    """

    return payload
