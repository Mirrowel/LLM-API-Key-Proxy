"""Startup display helpers that avoid leaking secrets to the console."""

from __future__ import annotations


def mask_secret_for_display(value: str) -> str:
    """Return a startup-safe representation of a configured secret.

    The proxy only needs to show that a key is configured. Printing the full
    value makes terminal scrollback a credential leak, so this helper exposes at
    most short edge fragments and the length for operator sanity checks.
    """

    if len(value) <= 8:
        return f"Set (redacted, {len(value)} chars)"
    return f"Set ({value[:4]}...{value[-4:]}, {len(value)} chars)"
