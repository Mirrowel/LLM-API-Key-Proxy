#!/usr/bin/env python3
"""
Two-step Anthropic OAuth credential setup.

Step 1 (no args): Generate auth URL + save verifier
  python scripts/setup_anthropic_cred.py

Step 2 (with code): Exchange code for tokens
  python scripts/setup_anthropic_cred.py "CODE_FROM_BROWSER"
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
from pathlib import Path
from rotator_library.providers.anthropic_auth_base import (
    _generate_pkce, _build_authorize_url, AnthropicAuthBase
)

STATE_FILE = Path(__file__).parent / ".anthropic_pkce_state.json"
OAUTH_DIR = Path(__file__).parent / ".." / "oauth_creds"

async def exchange_code(auth_code: str):
    if not STATE_FILE.exists():
        print("Error: PKCE state file not found. Please run Step 1 first.")
        sys.exit(1)
    state = json.loads(STATE_FILE.read_text())
    verifier = state["verifier"]

    auth = AnthropicAuthBase()
    tokens = await auth._exchange_code(auth_code.strip(), verifier)

    import time
    creds = {
        **tokens,
        "email": "anthropic-oauth-user",
        "_proxy_metadata": {
            "email": "anthropic-oauth-user",
            "last_check_timestamp": time.time(),
            "credential_type": "oauth",
        },
    }

    oauth_dir = OAUTH_DIR.resolve()
    oauth_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(oauth_dir.glob("anthropic_oauth_*.json"))
    next_num = len(existing) + 1
    file_path = oauth_dir / f"anthropic_oauth_{next_num}.json"

    file_path.write_text(json.dumps(creds, indent=2))
    os.chmod(file_path, 0o600)
    STATE_FILE.unlink(missing_ok=True)

    print(f"Credential saved to: {file_path}")
    print(f"Access token prefix: {tokens['access_token'][:20]}...")

def step1():
    verifier, challenge = _generate_pkce()
    url = _build_authorize_url(verifier, challenge)
    STATE_FILE.write_text(json.dumps({"verifier": verifier, "challenge": challenge}))
    print("Open this URL in your browser, authorize, then copy the code:\n")
    print(url)
    print(f"\nThen run: python scripts/setup_anthropic_cred.py \"PASTE_CODE_HERE\"")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(exchange_code(sys.argv[1]))
    else:
        step1()
