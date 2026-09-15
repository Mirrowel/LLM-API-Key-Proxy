# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""Proxy API key policy — one home for the default-key rules.

The well-known default key is an intentional convenience for local
development: it is accepted ONLY when the server binds a localhost
interface. Binding anything else with the default key triggers the
interactive policy prompt (enter / generate / skip) or, when no
interactive terminal exists, blocks startup outright.

Every entry path converges here: the TUI launcher falls through into
main.py's module flow, direct and Docker launches run the same module
code, and manual `uvicorn proxy_app.main:app` imports execute it at
import time. The launcher and the credential tool call the same
primitives instead of owning their own copies.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from rotator_library.core.constants import DEFAULT_PROXY_API_KEY as _DEFAULT_KEY

DEFAULT_PROXY_API_KEY = _DEFAULT_KEY

_LOCALHOST_BINDS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def is_localhost_bind(host: str | None) -> bool:
    """True only for loopback binds — 0.0.0.0/empty is NOT localhost."""

    return str(host or "").strip().lower() in _LOCALHOST_BINDS


def generate_proxy_api_key() -> str:
    """A fresh URL-safe key (32 chars of entropy, no prefix)."""

    return secrets.token_urlsafe(24)


def resolve_env_file() -> Path:
    """The .env file the running process loaded (frozen-aware)."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / ".env"
    return Path.cwd() / ".env"


def save_proxy_api_key(key: str) -> Path:
    """Persist the key to the .env file (creating it if needed)."""

    from dotenv import set_key

    env_file = resolve_env_file()
    set_key(str(env_file), "PROXY_API_KEY", key)
    return env_file


def _is_default_key() -> bool:
    return os.getenv("PROXY_API_KEY") == DEFAULT_PROXY_API_KEY


def _interactive_stream_available() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def uvicorn_bind_host() -> str | None:
    """The bind host when an ASGI server (uvicorn) is launching this module.

    Returns None for any non-serving import (tests, tooling) so the policy
    never fires at collection time. uvicorn's own default is localhost.
    """

    argv0 = Path(sys.argv[0]).name.lower() if sys.argv and sys.argv[0] else ""
    if "uvicorn" not in argv0:
        return None
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]
    return "127.0.0.1"


def _pause_for_acknowledgement() -> None:
    try:
        input("\nPress Enter to continue...")
    except (EOFError, KeyboardInterrupt):
        print()


def _prompt_for_new_key() -> str:
    while True:
        candidate = input("Enter new PROXY_API_KEY: ").strip()
        if candidate:
            return candidate
        print("The key cannot be empty.")


def _run_key_policy_prompt(host: str) -> str | None:
    """Offer enter/generate/skip; returns the newly adopted key or None.

    A stdin that EOFs mid-prompt (closed pipe, detached service, or a
    misreported isatty) falls through to the hard block — an answerable
    terminal always provides input.
    """

    try:
        return _prompt_interaction(host)
    except EOFError:
        _print_block_message(host)
        raise SystemExit(1) from None


def _prompt_interaction(host: str) -> str | None:
    print()
    print("!" * 70)
    print("SECURITY WARNING")
    print(f"The proxy API key is the public default and the server binds {host},")
    print("which is NOT a localhost address. Anyone who can reach this port can")
    print("use your provider credentials and quota.")
    print()
    print("  [1] Enter your own key")
    print("  [2] Auto-generate a strong key")
    print("  [3] Skip - keep the default for this launch (asked again next time)")
    print()
    while True:
        choice = input("Choose [1/2/3]: ").strip()
        if choice in {"1", "2", "3"}:
            break
        print("Please answer 1, 2, or 3.")

    if choice == "3":
        print("Keeping the default key for this launch.")
        return None

    key = _prompt_for_new_key() if choice == "1" else generate_proxy_api_key()
    env_file = save_proxy_api_key(key)
    os.environ["PROXY_API_KEY"] = key
    print()
    print(f"Saved to {env_file}")
    print(f"New PROXY_API_KEY: {key}")
    print("Configure your clients with this key (Authorization: Bearer <key>).")
    _pause_for_acknowledgement()
    return key


def _print_block_message(host: str) -> None:
    # A container/pipe cannot make an informed choice - refuse to serve
    # the well-known key to the network.
    print()
    print("!" * 70)
    print("REFUSING TO START (insecure configuration)")
    print("PROXY_API_KEY is set to the public default and the server would bind")
    print(f"{host}, which is not localhost. There is no terminal to ask for a")
    print("replacement key, so the proxy will not serve.")
    print()
    print("Fix one of the following and start again:")
    print("  1. Set PROXY_API_KEY to a strong secret in your environment or .env")
    print("  2. Bind a localhost interface instead (e.g. --host 127.0.0.1)")
    print("!" * 70)


def enforce_proxy_key_policy(host: str | None) -> str | None:
    """Apply the default-key policy for this launch.

    Returns the newly adopted key when one was generated or entered
    (already saved and set in the environment), or None when nothing
    changed (custom key, localhost bind, or an explicit skip).
    """

    if not _is_default_key() or is_localhost_bind(host):
        return None

    if not _interactive_stream_available():
        _print_block_message(str(host))
        raise SystemExit(1)

    return _run_key_policy_prompt(str(host))
