# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Cohere wire adapters (compatibility face)."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")


class CohereAdapter(PayloadAdapter):
    """Cohere's compatibility face accepts ``reasoning_effort`` as
    ``none | high`` only — thinking off or on, nothing between.

    The canonical effort vocabulary narrows here: ``none`` (and its
    aliases) stay ``none``; every other level becomes ``high``. The edit
    is recorded through the adapter-chain trace like every wire change.
    """

    name = "cohere"
    aliases = ("cohere_effort",)
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict) or "reasoning_effort" not in payload:
            return payload
        effort = payload.get("reasoning_effort")
        if effort is None:
            updated = deepcopy(payload)
            updated.pop("reasoning_effort", None)
            return updated
        if str(effort).strip().lower() in ("none", "off", "disabled"):
            updated = deepcopy(payload)
            updated["reasoning_effort"] = "none"
            return updated
        updated = deepcopy(payload)
        if effort != "high":
            logger.info(
                "cohere narrows reasoning_effort %r to 'high' (compat face accepts none|high only)",
                effort,
            )
        updated["reasoning_effort"] = "high"
        return updated
