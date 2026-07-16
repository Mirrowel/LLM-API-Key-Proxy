"""Resolve client output protocol choices without coupling providers to routes."""

from __future__ import annotations

from typing import Any, Optional

from ..protocols import get_protocol


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
    """Reject cross-protocol streams until canonical event formatting is active."""

    source = canonical_protocol_name(input_protocol)
    target = canonical_protocol_name(output_protocol)
    if source != target:
        raise ValueError(
            f"Streaming conversion from {source} to {target} is not enabled yet"
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
