# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Built-in base adapters for common provider payload quirks.

Audit (G8 final): this module keeps only the providers' ESCAPE HATCHES —
the adapters that express something the declaration surfaces genuinely
cannot — plus the no-op placeholder:

- ``noop`` — an explicit identity stage (declarations that need a
  placeholder entry, tests, operator JSON configs).
- ``model_override`` — outbound model replacement from config/metadata.
- ``suppress_developer_role`` — developer-role conversion/removal.
- ``antigravity_envelope`` — the internal request envelope wrapper.

The former ``field_rename`` and ``reasoning_content`` adapters were
retired: they had no live provider users, and their behavior is
declarable (request-path renames ride ``model_rules``); provider-specific
response spelling fixes live in the provider's own adapter.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from .base import AdapterContext, PayloadAdapter


class NoOpAdapter(PayloadAdapter):
    """Adapter that intentionally leaves payloads unchanged.

    Escape-hatch role: an EXPLICIT identity stage — a declaration
    placeholder (JSON adapter_names / hook chains) that documents intent
    instead of leaving a gap, and the test vehicle for chain mechanics.
    """

    name = "noop"
    aliases = ("none", "passthrough")


class ModelOverrideAdapter(PayloadAdapter):
    """Replace the outbound model field from adapter config.

    Escape-hatch role: the outbound model id is provider-owned wire data
    that no generic declaration can rewrite (aliases pinned by the
    operator). Config shape:
    `{ "model": "provider/native-model-name" }`
    """

    name = "model_override"
    aliases = ("override_model",)
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        config = context.config_for(self.name)
        override = config.get("model") or context.metadata.get("model_override")
        if not override or not isinstance(payload, dict):
            return payload
        updated = deepcopy(payload)
        updated["model"] = override
        return updated


class SuppressDeveloperRoleAdapter(PayloadAdapter):
    """Convert or remove developer-role messages for providers that reject them.

    Escape-hatch role: role-vocabulary surgery is provider-specific and
    configurable (which role to fold to, or to drop), beyond what a flat
    strip/rename rule can express. Config shape:
    `{ "mode": "system" | "user" | "drop" }`
    """

    name = "suppress_developer_role"
    aliases = ("developer_role",)
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return payload
        mode = context.config_for(self.name).get("mode", "system")
        if mode not in {"system", "user", "drop"}:
            raise ValueError("suppress_developer_role mode must be system, user, or drop")
        updated = deepcopy(payload)
        messages = []
        for message in updated.get("messages", []):
            if not isinstance(message, dict) or message.get("role") != "developer":
                messages.append(message)
                continue
            if mode == "drop":
                continue
            converted = dict(message)
            converted["role"] = mode
            messages.append(converted)
        updated["messages"] = messages
        return updated


class AntigravityEnvelopeAdapter(PayloadAdapter):
    """Wrap Gemini payloads in the Antigravity internal request envelope.

    Escape-hatch role: an entire WIRE ENVELOPE (request wrapping, volatile
    request ids, client-emulation fields) — a provider-specific protocol
    shape that cannot be expressed as parameter rules.

    The active provider restores only stable envelope fields. Device profiles,
    fingerprints, and other volatile client-emulation fields are intentionally
    not generated here until they are verified against current service behavior.

    Ordering convention: envelope adapters must be declared LAST in
    ``adapter_names`` — they wrap the final payload, so content-level
    adapters (model aliases, renames, role fixes) must run before the
    envelope to land inside it; content edits after an envelope would
    write siblings of ``request`` that never reach the provider.
    """

    name = "antigravity_envelope"
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        if _looks_like_antigravity_envelope(payload):
            return payload
        config = context.config_for(self.name)
        model = payload.get("model") or context.model
        request_payload = {key: deepcopy(value) for key, value in payload.items() if key != "model"}
        envelope = {
            "model": model,
            "request": request_payload,
            "requestType": config.get("request_type", "CHAT_COMPLETION"),
            "requestId": str(uuid4()),
            "userAgent": config.get("user_agent"),
        }
        project = config.get("project")
        if project:
            envelope["project"] = project
        return {key: value for key, value in envelope.items() if value is not None}


def _looks_like_antigravity_envelope(payload: dict[str, Any]) -> bool:
    """Return whether a payload already has the controlled envelope shape."""

    return "request" in payload and "requestType" in payload and "requestId" in payload
