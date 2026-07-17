"""Resolve client output protocol choices without coupling providers to routes."""

from __future__ import annotations

from typing import Any, Optional

from ..protocols import get_protocol
from ..core.errors import protocol_error_payload


OUTPUT_PROTOCOL_HEADER = "X-Proxy-Output-Protocol"


def request_output_protocol(request: Optional[Any]) -> Optional[str]:
    """Return the canonical output protocol requested at the HTTP boundary."""

    headers = getattr(request, "headers", None)
    if not headers:
        return None
    value = headers.get(OUTPUT_PROTOCOL_HEADER) or headers.get(OUTPUT_PROTOCOL_HEADER.lower())
    if not value:
        return None
    return canonical_protocol_name(str(value))


def canonical_protocol_name(value: str) -> str:
    """Resolve protocol aliases and reject unknown output formats early."""

    try:
        return get_protocol(str(value).strip().lower()).name
    except KeyError as exc:
        raise ValueError(f"Unsupported output protocol: {value}") from exc


def require_same_protocol_stream(input_protocol: str, output_protocol: str) -> None:
    """Validate that both stream protocols use canonical generative events."""

    source = canonical_protocol_name(input_protocol)
    target = canonical_protocol_name(output_protocol)
    supported = {"openai_chat", "anthropic_messages", "responses", "gemini"}
    if source not in supported or target not in supported:
        raise ValueError(
            f"Streaming conversion from {source} to {target} is not supported"
        )


def resolve_client_output_protocol(
    client: Any,
    payload: dict[str, Any],
    *,
    input_protocol: str,
    request: Optional[Any] = None,
) -> str:
    """Resolve output through RotatingClient or a compatible minimal facade."""

    resolver = getattr(client, "resolve_output_protocol", None)
    if callable(resolver):
        return resolver(payload, input_protocol=input_protocol, request=request)
    return request_output_protocol(request) or canonical_protocol_name(input_protocol)


def format_client_protocol_error(
    client: Any,
    payload: dict[str, Any],
    *,
    input_protocol: str,
    request: Any,
    error: BaseException | str,
    error_type: str,
    status_code: int,
) -> tuple[int, dict[str, Any]]:
    """Format a proxy-side failure using the selected client protocol."""

    try:
        output_protocol = resolve_client_output_protocol(
            client,
            payload,
            input_protocol=input_protocol,
            request=request,
        )
    except ValueError as selection_error:
        output_protocol = canonical_protocol_name(input_protocol)
        error = selection_error
        error_type = "invalid_request"
        status_code = 400
    return protocol_error_payload(
        error,
        output_protocol,
        error_type=error_type,
        status_code=status_code,
    )
