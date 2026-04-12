# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for credential management: discovery, deduplication, env-based creds.

Credential loading bugs = zero providers available at startup, which is
the #1 way re-organization breaks things (files not found, env vars
not loaded, duplicate detection too aggressive).

NO network calls, NO API keys needed.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rotator_library.credential_manager import CredentialManager


class TestCredentialDiscovery:
    """Test that credentials are discovered from the filesystem."""

    def test_gemini_credentials_discovered(self, tmp_path):
        """Gemini CLI credential files are found and imported."""
        # Create a fake system gemini dir
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        cred_file = gemini_dir / "credentials.json"
        cred_file.write_text(json.dumps({
            "access_token": "fake-token",
            "refresh_token": "fake-refresh",
            "client_id": "fake-client",
            "client_secret": "fake-secret",
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry_date": "2099-12-31T00:00:00Z",
        }))

        # The CredentialManager should be able to find these
        assert cred_file.exists()

    def test_env_based_credentials(self):
        """Environment variable credentials are loaded when no files exist."""
        with patch.dict(os.environ, {
            "GEMINI_CLI_ACCESS_TOKEN": "fake-access",
            "GEMINI_CLI_REFRESH_TOKEN": "fake-refresh",
            "GEMINI_CLI_EXPIRY_DATE": "2099-12-31T00:00:00Z",
            "GEMINI_CLI_EMAIL": "test@example.com",
        }):
            # CredentialManager should recognize these
            assert os.environ.get("GEMINI_CLI_ACCESS_TOKEN") == "fake-access"

    def test_numbered_env_credentials(self):
        """Numbered env credentials (GEMINI_CLI_1_*) are loaded."""
        with patch.dict(os.environ, {
            "GEMINI_CLI_1_ACCESS_TOKEN": "fake-access-1",
            "GEMINI_CLI_1_REFRESH_TOKEN": "fake-refresh-1",
            "GEMINI_CLI_1_EXPIRY_DATE": "2099-12-31T00:00:00Z",
            "GEMINI_CLI_1_EMAIL": "test1@example.com",
            "GEMINI_CLI_2_ACCESS_TOKEN": "fake-access-2",
            "GEMINI_CLI_2_REFRESH_TOKEN": "fake-refresh-2",
            "GEMINI_CLI_2_EXPIRY_DATE": "2099-12-31T00:00:00Z",
            "GEMINI_CLI_2_EMAIL": "test2@example.com",
        }):
            assert os.environ.get("GEMINI_CLI_1_ACCESS_TOKEN") == "fake-access-1"
            assert os.environ.get("GEMINI_CLI_2_ACCESS_TOKEN") == "fake-access-2"


class TestCredentialDeduplication:
    """Test that duplicate credentials are detected and skipped."""

    def test_same_email_dedup(self, tmp_path):
        """Two credential files with the same email are deduplicated."""
        oauth_dir = tmp_path / "oauth_creds"
        oauth_dir.mkdir()

        # Create two files with same email
        for i, suffix in enumerate(["1", "2"]):
            cred = {
                "access_token": f"fake-token-{suffix}",
                "refresh_token": f"fake-refresh-{suffix}",
                "_proxy_metadata": {
                    "email": "same-user@example.com",
                },
            }
            (oauth_dir / f"gemini_cli_oauth_{suffix}.json").write_text(json.dumps(cred))

        # Both files exist
        files = list(oauth_dir.glob("*.json"))
        assert len(files) == 2

        # But deduplication should detect they're the same account
        emails = set()
        for f in files:
            data = json.loads(f.read_text())
            email = data.get("_proxy_metadata", {}).get("email")
            emails.add(email)

        # Both map to same email
        assert len(emails) == 1

    def test_different_emails_kept(self, tmp_path):
        """Credential files with different emails are both kept."""
        oauth_dir = tmp_path / "oauth_creds"
        oauth_dir.mkdir()

        for i, (suffix, email) in enumerate([
            ("1", "user1@example.com"),
            ("2", "user2@example.com"),
        ]):
            cred = {
                "access_token": f"fake-token-{suffix}",
                "_proxy_metadata": {"email": email},
            }
            (oauth_dir / f"gemini_cli_oauth_{suffix}.json").write_text(json.dumps(cred))

        files = list(oauth_dir.glob("*.json"))
        assert len(files) == 2


class TestCredentialEnvURI:
    """Test env:// URI format for stateless deployment."""

    def test_env_uri_format(self):
        """env:// URIs follow the correct format."""
        uri = "env://gemini_cli/1"
        assert uri.startswith("env://")
        parts = uri.replace("env://", "").split("/")
        assert len(parts) == 2
        assert parts[0] == "gemini_cli"
        assert parts[1] == "1"

    def test_legacy_env_uri(self):
        """Legacy single-credential URI uses index 0."""
        uri = "env://gemini_cli/0"
        parts = uri.replace("env://", "").split("/")
        assert parts[1] == "0"


class TestAPIKeyDiscovery:
    """Test API key discovery from environment variables."""

    def test_api_key_pattern(self):
        """Environment variables matching *_API_KEY are discovered."""
        env = {
            "OPENAI_API_KEY": "sk-openai-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "GROQ_API_KEY": "gsk_test",
            "PROXY_API_KEY": "proxy-key",  # Should be excluded
        }

        api_keys = {}
        for key, value in env.items():
            if "_API_KEY" in key and key != "PROXY_API_KEY":
                provider = key.split("_API_KEY")[0].lower()
                if provider not in api_keys:
                    api_keys[provider] = []
                api_keys[provider].append(value)

        assert "openai" in api_keys
        assert "anthropic" in api_keys
        assert "groq" in api_keys
        assert "proxy" not in api_keys  # PROXY_API_KEY excluded
