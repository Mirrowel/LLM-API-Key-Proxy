# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional


lib_logger = logging.getLogger("rotator_library")


KIRO_REFRESH_URL_TEMPLATE = "https://prod.{region}.auth.desktop.kiro.dev/refreshToken"
AWS_SSO_OIDC_URL_TEMPLATE = "https://oidc.{region}.amazonaws.com/token"
KIRO_API_HOST_TEMPLATE = "https://q.{region}.amazonaws.com"
KIRO_Q_HOST_TEMPLATE = "https://q.{region}.amazonaws.com"

TOKEN_REFRESH_THRESHOLD = int(os.getenv("KIRO_TOKEN_REFRESH_THRESHOLD", "600"))
FIRST_TOKEN_TIMEOUT = float(os.getenv("KIRO_FIRST_TOKEN_TIMEOUT", "20"))
FIRST_TOKEN_MAX_RETRIES = int(os.getenv("KIRO_FIRST_TOKEN_MAX_RETRIES", "3"))
MAX_RETRIES = int(os.getenv("KIRO_MAX_RETRIES", "3"))
BASE_RETRY_DELAY = float(os.getenv("KIRO_BASE_RETRY_DELAY", "1"))
STREAMING_READ_TIMEOUT = float(os.getenv("KIRO_STREAMING_READ_TIMEOUT", "300"))

TOOL_DESCRIPTION_MAX_LENGTH = int(os.getenv("KIRO_TOOL_DESCRIPTION_MAX_LENGTH", "400"))

FAKE_REASONING_ENABLED = os.getenv("KIRO_FAKE_REASONING_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)
FAKE_REASONING_MAX_TOKENS = int(os.getenv("KIRO_FAKE_REASONING_MAX_TOKENS", "1024"))
FAKE_REASONING_HANDLING = os.getenv(
    "KIRO_FAKE_REASONING_HANDLING", "as_reasoning_content"
)
FAKE_REASONING_OPEN_TAGS = [
    "<thinking>",
    "<think>",
    "<reasoning>",
]
FAKE_REASONING_INITIAL_BUFFER_SIZE = int(
    os.getenv("KIRO_FAKE_REASONING_INITIAL_BUFFER_SIZE", "200")
)


def get_kiro_refresh_url(region: str) -> str:
    return KIRO_REFRESH_URL_TEMPLATE.format(region=region)


def get_aws_sso_oidc_url(region: str) -> str:
    return AWS_SSO_OIDC_URL_TEMPLATE.format(region=region)


def get_kiro_api_host(region: str) -> str:
    return KIRO_API_HOST_TEMPLATE.format(region=region)


def get_kiro_q_host(region: str) -> str:
    return KIRO_Q_HOST_TEMPLATE.format(region=region)


def get_machine_fingerprint() -> str:
    try:
        import socket
        import getpass

        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-gateway"
        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as exc:
        lib_logger.warning(f"Failed to get machine fingerprint: {exc}")
        return hashlib.sha256(b"default-kiro-gateway").hexdigest()


def get_kiro_headers(auth_manager: Any, token: str) -> Dict[str, str]:
    fingerprint = getattr(auth_manager, "fingerprint", "unknown")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": (
            "aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js "
            "md/nodejs#22.21.1 api/codewhispererstreaming#1.0.27 "
            f"m/E KiroIDE-0.7.45-{fingerprint}"
        ),
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-0.7.45-{fingerprint}",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": "vibe",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }


def generate_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id(messages: Optional[List[Dict[str, Any]]] = None) -> str:
    if not messages:
        return str(uuid.uuid4())

    if len(messages) <= 3:
        key_messages = messages
    else:
        key_messages = messages[:3] + [messages[-1]]

    simplified_messages = []
    for msg in key_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            content_str = content[:100]
        elif isinstance(content, list):
            content_str = json.dumps(content, sort_keys=True)[:100]
        else:
            content_str = str(content)[:100]

        simplified_messages.append({"role": role, "content": content_str})

    content_json = json.dumps(simplified_messages, sort_keys=True)
    hash_digest = hashlib.sha256(content_json.encode()).hexdigest()
    return hash_digest[:16]


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"
