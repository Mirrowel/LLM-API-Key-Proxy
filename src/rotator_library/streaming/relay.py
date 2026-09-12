# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G4 conditional re-serialization vocabulary (leaf module).

Shared by the native executor (production) and the operational stream
pipeline (consumption) without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..protocols.types import UnifiedStreamEvent


@dataclass
class StreamRepairState:
    """Shared stream repair evidence (G4).

    Owned by the client executor, fed by every parse source (ChatWire
    adapter, native executor), consulted by the pipeline tail. This is the
    adapter-to-tail handoff that lets a bare-EOF stream still be repaired
    instead of losing its finish reason.
    """

    tools_seen: bool = False
    #: provider's own final-frame reason, most recent observation
    last_provider_reason: Optional[str] = None
    #: per-choice held reasons (chat intermediate finish frames)
    held_reasons: Dict[int, str] = field(default_factory=dict)
    #: any hook/adapter edited an event → relay must disengage (operator
    #: ruling: intercept-and-edit always possible; relay is conditional on
    #: NO modification, never on protocol equality alone)
    edited_by_hook: bool = False

    def repaired_reason(self, provider_reason: Optional[str]) -> Optional[str]:
        """Ruling ladder: provider's own reason > held > tools-seen > stop."""
        if provider_reason:
            return provider_reason
        if self.last_provider_reason:
            return self.last_provider_reason
        if self.held_reasons:
            return self.held_reasons.get(0) or next(iter(self.held_reasons.values()))
        if self.tools_seen:
            return "tool_calls"
        return "stop"


@dataclass
class RelayStreamItem:
    """One transport frame carrying parsed events and (optionally) raw wire text.

    G4 conditional re-serialization: when the pipeline decides a frame is
    relay-able (same protocol, no edits, no repair needed) it forwards
    ``raw`` bytes to the client untouched while observing ``events`` on the
    side. Sources without byte access (LiteLLM objects, custom plugins)
    emit bare events and always take the formatter path.
    """

    events: List[UnifiedStreamEvent] = field(default_factory=list)
    raw: Optional[str] = None
    event_name: Optional[str] = None
    is_comment: bool = False
