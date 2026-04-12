# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Shared test fixtures and utilities.

All fixtures use synthetic credentials and mock HTTP — zero cost, zero network.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# =============================================================================
# Synthetic Credential Fixtures
# =============================================================================

FAKE_API_KEY = "sk-fake-test-key-0000000000000000"
FAKE_API_KEY_2 = "sk-fake-test-key-1111111111111111"
FAKE_API_KEY_3 = "sk-fake-test-key-2222222222222222"

FAKE_OAUTH_TOKEN = {
    "access_token": "fake-access-token-12345",
    "refresh_token": "fake-refresh-token-12345",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "fake-client-id.apps.googleusercontent.com",
    "client_secret": "fake-client-secret",
    "expiry_date": "2099-12-31T23:59:59.000000Z",
    "_proxy_metadata": {
        "email": "test@example.com",
        "last_check_timestamp": 1700000000.0,
    },
}


@pytest.fixture
def fake_api_keys():
    """Synthetic API key dict matching the format main.py discovers."""
    return {
        "openai": [FAKE_API_KEY, FAKE_API_KEY_2],
        "anthropic": [FAKE_API_KEY],
        "groq": [FAKE_API_KEY_3],
    }


@pytest.fixture
def temp_oauth_dir(tmp_path):
    """Temporary directory with synthetic OAuth credential files."""
    oauth_dir = tmp_path / "oauth_creds"
    oauth_dir.mkdir()

    # Gemini CLI credential
    gemini_cred = oauth_dir / "gemini_cli_oauth_1.json"
    gemini_data = dict(FAKE_OAUTH_TOKEN)
    gemini_data["_proxy_metadata"]["email"] = "gemini-test@example.com"
    gemini_data["project_id"] = "fake-project"
    gemini_cred.write_text(json.dumps(gemini_data))

    # Copilot credential
    copilot_cred = oauth_dir / "copilot_oauth_1.json"
    copilot_data = {
        "access_token": "ghu_fake_copilot_token",
        "refresh_token": "fake-copilot-refresh",
        "token_uri": "https://github.com/login/oauth/access_token",
        "client_id": "fake-copilot-client-id",
        "client_secret": "fake-copilot-client-secret",
        "expiry_date": "2099-12-31T23:59:59.000000Z",
        "_proxy_metadata": {
            "login": "testuser",
            "last_check_timestamp": 1700000000.0,
        },
    }
    copilot_cred.write_text(json.dumps(copilot_data))

    return oauth_dir


@pytest.fixture
def temp_usage_dir(tmp_path):
    """Temporary directory for usage tracking files."""
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    return usage_dir


@pytest.fixture
def temp_env(tmp_path, fake_api_keys, temp_oauth_dir, temp_usage_dir):
    """
    Minimal environment dict for proxy initialization.
    No real keys, no real endpoints.
    """
    env = {
        "PROXY_API_KEY": "test-proxy-key",
        "SKIP_OAUTH_INIT_CHECK": "true",
        "GLOBAL_TIMEOUT": "5",
        # Provide fake API keys
        **{f"{k.upper()}_API_KEY": v[0] for k, v in fake_api_keys.items()},
        # Disable background jobs
        "ANTIGRAVITY_QUOTA_REFRESH_INTERVAL": "0",
        "GEMINI_CLI_QUOTA_REFRESH_INTERVAL": "0",
    }
    return env


# =============================================================================
# Mock HTTP Client
# =============================================================================


class MockResponse:
    """Minimal mock for httpx.Response used in tests."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict] = None,
        text_data: str = "",
        headers: Optional[Dict] = None,
    ):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text_data
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}: {self.text}")


class MockAsyncClient:
    """
    Mock httpx.AsyncClient that never makes real network requests.

    All responses are controlled via `set_response()`.
    """

    def __init__(self):
        self._responses: List[MockResponse] = []
        self._call_log: List[Dict] = []

    def set_response(self, response: MockResponse):
        self._responses.append(response)

    def set_response_sequence(self, responses: List[MockResponse]):
        self._responses.extend(responses)

    async def get(self, url: str, **kwargs) -> MockResponse:
        self._call_log.append({"method": "GET", "url": url, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(status_code=200, json_data={"data": []})

    async def post(self, url: str, **kwargs) -> MockResponse:
        self._call_log.append({"method": "POST", "url": url, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(status_code=200, json_data={})

    async def send(self, request: Any, **kwargs) -> MockResponse:
        self._call_log.append({"method": "SEND", "request": request, **kwargs})
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(status_code=200, json_data={})

    async def aclose(self):
        pass


@pytest.fixture
def mock_http_client():
    """Provide a mock HTTP client that captures calls but never hits the network."""
    return MockAsyncClient()


# =============================================================================
# Anthropic Format Fixtures
# =============================================================================


@pytest.fixture
def anthropic_simple_request():
    """A minimal valid Anthropic Messages API request."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ],
    }


@pytest.fixture
def anthropic_tool_request():
    """Anthropic request with tools (tests tool translation)."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    }


@pytest.fixture
def anthropic_thinking_request():
    """Anthropic request with thinking enabled (tests thinking translation)."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16000,
        "messages": [
            {"role": "user", "content": "Solve this problem step by step"}
        ],
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    }


@pytest.fixture
def anthropic_multiturn_request():
    """Anthropic request with tool use in conversation history."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user wants weather info. I should use the tool.",
                        "signature": "fake-signature-abc123",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "New York, NY"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "72°F, sunny",
                    }
                ],
            },
        ],
    }


@pytest.fixture
def openai_simple_response():
    """A minimal valid OpenAI Chat Completions response."""
    return {
        "id": "chatcmpl-fake123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def openai_tool_response():
    """OpenAI response with tool calls."""
    return {
        "id": "chatcmpl-fake456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fake123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York, NY"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35,
        },
    }


@pytest.fixture
def openai_thinking_response():
    """OpenAI response with reasoning_content (thinking)."""
    return {
        "id": "chatcmpl-fake789",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4-5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step... The user asked about the meaning of life.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 100,
            "total_tokens": 115,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }


# =============================================================================
# Streaming Fixtures
# =============================================================================


@pytest.fixture
def openai_streaming_chunks():
    """Sequence of OpenAI SSE streaming chunks."""
    return [
        {
            "id": "chatcmpl-stream1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "!"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    ]


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
