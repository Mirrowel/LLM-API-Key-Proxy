"""Startup display helpers that avoid leaking secrets to the console."""

from __future__ import annotations


def mask_secret_for_display(value: str) -> str:
    """Return a startup-safe representation of a configured secret.

    The proxy only needs to show that a key is configured and let the
    operator eyeball-match it against a client's config. Printing the full
    value makes terminal scrollback a credential leak, and revealing the
    length narrows a brute-force space, so the shape is fixed: first and
    last three characters joined by an ellipsis, nothing else.
    """

    if len(value) < 8:
        return "Set (redacted)"
    return f"{value[:3]}…{value[-3:]}"
