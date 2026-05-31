# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Built-in base adapters for common provider payload quirks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from ..field_cache.paths import extract_path, inject_path
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


class FieldRenameAdapter(PayloadAdapter):
    """Copy configured values between paths on raw payloads.

    Config shape:
    `{ "rules": [{ "source_path": "a.b", "target_path": "c.d", "stage": "request", "move": false }] }`

    This adapter is conservative by design: it copies the last matched value by
    default, delegates target ambiguity checks to `inject_path`, and only removes
    the source for simple dotted-key paths when `move=true`.
    """

    name = "field_rename"
    aliases = ("field_copy",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        return self._transform_stage(payload, context, "request")

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return self._transform_stage(payload, context, "response")

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return self._transform_stage(payload, context, "stream_event")

    def _transform_stage(self, payload: Any, context: AdapterContext, stage: str) -> Any:
        if not isinstance(payload, dict):
            return payload
        updated = deepcopy(payload)
        for rule in context.config_for(self.name).get("rules", []):
            if rule.get("stage", stage) != stage:
                continue
            values = extract_path(updated, rule["source_path"])
            if not values:
                continue
            value = values if rule.get("as_list") else values[-1]
            inject_path(
                updated,
                rule["target_path"],
                value,
                when_missing_only=bool(rule.get("when_missing_only", False)),
            )
            if rule.get("move"):
                _delete_simple_path(updated, rule["source_path"])
        return updated


class AntigravityEnvelopeAdapter(PayloadAdapter):
    """Wrap Gemini payloads in the Antigravity internal request envelope.

    The active provider restores only stable envelope fields. Device profiles,
    fingerprints, and other volatile client-emulation fields are intentionally
    not generated here until they are verified against current service behavior.
    """

    name = "antigravity_envelope"
    supported_stages = ("request",)

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict) or "request" in payload:
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


def _delete_simple_path(payload: dict[str, Any], path: str) -> None:
    """Delete a simple dotted dict path after a conservative move operation."""

    parts = path.split(".")
    if any("[" in part or part == "*" for part in parts):
        return
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(part)
    if isinstance(current, dict):
        current.pop(parts[-1], None)
