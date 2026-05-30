# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Built-in base adapters for common provider payload quirks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter


class NoOpAdapter(PayloadAdapter):
    """Adapter that intentionally leaves payloads unchanged."""

    name = "noop"
    aliases = ("none", "passthrough")


class ModelOverrideAdapter(PayloadAdapter):
    """Replace the outbound model field from adapter config.

    Config shape:
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

    Config shape:
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


class ReasoningContentAdapter(PayloadAdapter):
    """Normalize common reasoning fields on assistant messages.

    This base adapter copies `reasoning`, `reasoning_content`, or configured
    aliases into the configured output field. It deliberately does not delete
    source fields; provider-specific subclasses can choose stricter behavior.
    """

    name = "reasoning_content"
    aliases = ("reasoning_rewrite",)
    supported_stages = ("response",)

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        config = context.config_for(self.name)
        output_field = config.get("output_field", "reasoning_content")
        source_fields = tuple(config.get("source_fields", ("reasoning_content", "reasoning")))
        updated = deepcopy(payload)
        for choice in updated.get("choices", []) if isinstance(updated.get("choices"), list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            if output_field in message:
                continue
            for source_field in source_fields:
                if source_field in message:
                    message[output_field] = message[source_field]
                    break
        return updated
