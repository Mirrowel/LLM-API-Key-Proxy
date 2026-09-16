# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI-compatible embeddings protocol adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Mapping, Optional

from .base import ProtocolAdapter
from .canonical import add_conversion_warning, is_same_protocol
from .operation import OPERATION_EMBEDDINGS
from .types import ProtocolContext, UnifiedRequest, UnifiedResponse, Usage
from .validation import validate_embeddings_request

_REQUEST_CORE_FIELDS = {"model", "input", "encoding_format", "dimensions", "user", "operation"}
_REQUEST_OPTION_FIELDS = {"encoding_format", "dimensions", "user"}

# Canonical controls that have no openai /embeddings representation. They
# arrive from the gemini adapter (task_type/title/output_dimensionality maps
# onto dimensions), and dropping them silently would change the meaning of a
# retrieval/document embedding request — so the drop is disclosed.
_FOREIGN_EMBEDDING_CONTROLS = {"task_type", "title"}


class OpenAIEmbeddingsProtocol(ProtocolAdapter):
    """Adapter for `/v1/embeddings` style request and response payloads.

    The adapter intentionally treats embedding vectors as opaque data entries.
    That keeps it usable for OpenAI-compatible providers with additional index,
    metadata, or sparse-vector fields without narrowing the schema too early.

    G10 Phase B disclosure: unknown option keys are passed through verbatim
    (round-trip fidelity) but recorded as ``unsupported_optional_control``
    request warnings; :meth:`format_response` carries them under the private
    ``_proxy_warnings`` transport key for the finalizer to drain and strip.
    G9: the adapter is the wire face for embeddings on every openai-family
    target (the native executor selects it when the requested operation is
    embeddings), so a same-protocol openai client rides the raw fast path
    byte-for-byte and a cross-protocol source converts through the
    canonical input/controls below.
    """

    name: ClassVar[str] = "openai_embeddings"
    aliases: ClassVar[tuple[str, ...]] = ("embeddings", "openai_embedding")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_EMBEDDINGS,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)
    # G11: embeddings live ON the openai wire (the same /v1 base, the same
    # bearer auth, the same envelope conventions) — the family is openai_chat
    # so profile matching treats an embeddings client as the chat face of a
    # multi-face provider instead of a conversion, and same-wire checks agree.
    base_family: ClassVar[str] = "openai_chat"

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        warnings: list = []
        unknown_options = sorted(key for key in request if key not in _REQUEST_CORE_FIELDS)
        if unknown_options:
            # Unknown option keys are passed through verbatim (same-protocol
            # round-trip fidelity), but provider support is not guaranteed —
            # disclosed rather than silent.
            from .canonical import record_conversion_warning

            record_conversion_warning(
                warnings,
                code="unsupported_optional_control",
                message=(
                    "embeddings option key(s) outside the canonical surface pass through "
                    f"verbatim (provider support not guaranteed): {', '.join(unknown_options)}"
                ),
                field="options",
                source_protocol=self.name,
                target_protocol=self.name,
            )
        return UnifiedRequest(
            operation=OPERATION_EMBEDDINGS,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            input=deepcopy(request.get("input")),
            generation_params={k: deepcopy(request[k]) for k in _REQUEST_OPTION_FIELDS if k in request},
            raw=deepcopy(raw_request),
            warnings=warnings,
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Build the openai /embeddings wire from canonical state (G9).

        Canonical input (string, list of strings, or token arrays) rides
        verbatim — embeddings inputs have no uniform per-item controls, so
        no shape rewriting is needed. The canonical controls map:
        ``dimensions`` (native), ``output_dimensionality`` (gemini source ->
        the same openai control), ``encoding_format``/``user`` pass through;
        gemini-only retrieval controls are dropped with a recorded warning.
        """

        validate_embeddings_request(unified_request, self.name, context, capabilities=capabilities)
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "input": deepcopy(unified_request.input),
        }
        params = deepcopy(unified_request.generation_params)
        dimensions = params.pop("dimensions", None)
        if dimensions is None:
            # Gemini's outputDimensionality is the same control on the
            # openai wire.
            dimensions = params.pop("output_dimensionality", None)
        if dimensions is not None:
            payload["dimensions"] = dimensions
        for key in ("encoding_format", "user"):
            if key in params:
                payload[key] = params.pop(key)
        # Preserve gemini-bound retrieval controls ON their home protocol
        # (source replay) but never leak them onto a foreign wire.
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        for key in sorted(_FOREIGN_EMBEDDING_CONTROLS):
            if key in params:
                params.pop(key)
                if not preserve_source:
                    add_conversion_warning(
                        unified_request,
                        code="unsupported_optional_control",
                        message=f"embeddings control {key!r} has no openai representation; dropped",
                        field=key,
                        target_protocol=self.name,
                    )
        # Anything else is a custom option: opaque pass-through, same
        # doctrine as the request-side unknown-option disclosure.
        payload.update(params)
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        return UnifiedResponse(
            operation=OPERATION_EMBEDDINGS,
            model=response.get("model") or getattr(context, "model", None),
            data=deepcopy(response.get("data") or []),
            usage=self.extract_usage(response, context),
            source_protocol=self.name,
            raw=deepcopy(raw_response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"model", "data", "usage"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Normalize any embeddings vector source into the openai list envelope.

        Entries stay OPAQUE (opaque-entries doctrine): every source-provided
        key survives; only the openai-required ``embedding``/``index`` (and
        the conventional ``object``) are guaranteed. That lets a gemini
        ``embedding.values`` entry convert losslessly while a same-protocol
        openai entry (sparse vectors, provider metadata) replays intact.
        Usage is prompt-only and emitted in the openai spelling even when
        the source reported gemini/ollama buckets in ``raw``.
        """

        entries: list[dict[str, Any]] = []
        for position, entry in enumerate(unified_response.data or []):
            if isinstance(entry, dict):
                normalized = deepcopy(entry)
                if "embedding" not in normalized and "values" in normalized:
                    normalized["embedding"] = deepcopy(normalized.pop("values"))
            else:
                normalized = {"embedding": deepcopy(entry)}
            normalized.setdefault("object", "embedding")
            normalized.setdefault("index", position)
            entries.append(normalized)
        payload: dict[str, Any] = {"object": "list", "data": entries}
        if unified_response.model:
            payload["model"] = unified_response.model
        usage = unified_response.usage
        if usage is not None:
            raw_usage = usage.raw if isinstance(usage.raw, dict) else None
            if raw_usage is not None and ("prompt_tokens" in raw_usage or "total_tokens" in raw_usage):
                # Same-wire replay keeps provider detail (cached buckets etc.).
                payload["usage"] = deepcopy(raw_usage)
            else:
                prompt_tokens = int(usage.input_tokens or 0)
                payload["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    # Embeddings are prompt-only: no completion bucket exists.
                    "total_tokens": int(usage.total_tokens or prompt_tokens),
                }
        payload.update(deepcopy(unified_response.extra))
        # Private transport key (G10 Phase B): request option-drop warnings
        # ride the response to the finalizer, which sinks them into the
        # change log and strips the key before the client sees it.
        if unified_response.warnings:
            payload["_proxy_warnings"] = list(unified_response.warnings)
        return payload

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        """Prompt-only usage (G9): embeddings have no completion bucket."""

        if isinstance(raw_or_unified, UnifiedResponse):
            return raw_or_unified.usage
        usage = raw_or_unified.get("usage") if isinstance(raw_or_unified, dict) else None
        if not isinstance(usage, dict):
            return None
        # prompt_tokens is the openai spelling; input_tokens covers compat
        # providers that report the chat-style names; a bare total on a
        # prompt-only call IS the prompt count.
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("total_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
            raw=deepcopy(usage),
        )
