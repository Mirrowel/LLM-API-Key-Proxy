"""Client protocol helpers for proxy-side errors.

The client response protocol always equals the client request protocol (plan
Decision 1). This module no longer performs independent output-protocol
selection; it only canonicalizes protocol names and formats proxy-side
failures in the client's own protocol.
"""

from __future__ import annotations

from typing import Any

from ..protocols import get_protocol
from ..core.errors import protocol_error_payload


def canonical_protocol_name(value: str) -> str:
    """Resolve protocol aliases and reject unknown formats early."""

    try:
        return get_protocol(str(value).strip().lower()).name
    except KeyError as exc:
        raise ValueError(f"Unsupported protocol: {value}") from exc


def format_client_protocol_error(
    *,
    input_protocol: str,
    error: BaseException | str,
    error_type: str,
    status_code: int,
) -> tuple[int, dict[str, Any]]:
    """Format a proxy-side failure in the client's own protocol."""

    return protocol_error_payload(
        error,
        canonical_protocol_name(input_protocol),
        error_type=error_type,
        status_code=status_code,
    )
