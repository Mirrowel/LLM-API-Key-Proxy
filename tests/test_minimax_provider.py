"""
Unit and integration tests for the MiniMax provider.

Unit tests run with no external dependencies (network calls are mocked).
Integration tests (marked with @pytest.mark.integration) require a real
MINIMAX_API_KEY environment variable and make live API calls.
"""

import pytest
import pytest_asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_models_response(model_ids):
    """Build a fake /v1/models JSON payload."""
    return {"data": [{"id": m} for m in model_ids]}


# ---------------------------------------------------------------------------
# Unit tests – get_models
# ---------------------------------------------------------------------------


class TestGetModels:
    """Tests for MiniMaxProvider.get_models()."""

    @pytest.mark.asyncio
    async def test_returns_models_from_api(self):
        """Dynamic model list from the API is returned with the minimax/ prefix."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        fake_response = MagicMock()
        fake_response.json.return_value = _make_models_response(
            ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
        )
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)

        models = await provider.get_models("test-key", mock_client)

        assert "minimax/MiniMax-M2.7" in models
        assert "minimax/MiniMax-M2.7-highspeed" in models
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_request_error(self):
        """Static fallback list is returned when the API is unreachable."""
        import httpx
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_STATIC_MODELS,
        )

        provider = MiniMaxProvider()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.RequestError("connection refused")
        )

        models = await provider.get_models("test-key", mock_client)

        expected = {f"minimax/{m}" for m in _MINIMAX_STATIC_MODELS}
        assert set(models) == expected

    @pytest.mark.asyncio
    async def test_fallback_on_empty_data(self):
        """Static fallback list is returned when the API returns an empty data array."""
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_STATIC_MODELS,
        )

        provider = MiniMaxProvider()
        fake_response = MagicMock()
        fake_response.json.return_value = {"data": []}
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)

        models = await provider.get_models("test-key", mock_client)

        expected = {f"minimax/{m}" for m in _MINIMAX_STATIC_MODELS}
        assert set(models) == expected

    @pytest.mark.asyncio
    async def test_fallback_on_unexpected_exception(self):
        """Static fallback list is returned on unexpected errors."""
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_STATIC_MODELS,
        )

        provider = MiniMaxProvider()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("unexpected"))

        models = await provider.get_models("test-key", mock_client)

        expected = {f"minimax/{m}" for m in _MINIMAX_STATIC_MODELS}
        assert set(models) == expected

    @pytest.mark.asyncio
    async def test_all_static_models_have_minimax_prefix(self):
        """Every static fallback model must start with 'minimax/'."""
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_STATIC_MODELS,
        )

        provider = MiniMaxProvider()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("fail"))

        models = await provider.get_models("test-key", mock_client)

        assert all(m.startswith("minimax/") for m in models)
        assert len(models) == len(_MINIMAX_STATIC_MODELS)

    @pytest.mark.asyncio
    async def test_api_key_sent_as_bearer_token(self):
        """The Authorization header must use Bearer + the supplied API key."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        fake_response = MagicMock()
        fake_response.json.return_value = _make_models_response(["MiniMax-M2.7"])
        fake_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)

        await provider.get_models("sk-abc123", mock_client)

        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs.args[1]
        assert headers.get("Authorization") == "Bearer sk-abc123"


# ---------------------------------------------------------------------------
# Unit tests – temperature clamping
# ---------------------------------------------------------------------------


class TestTemperatureClamping:
    """Tests for MiniMaxProvider temperature clamping in acompletion()."""

    def test_has_custom_logic_returns_true(self):
        """has_custom_logic() must return True so the proxy delegates to acompletion."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        assert MiniMaxProvider().has_custom_logic() is True

    @pytest.mark.asyncio
    async def test_temperature_zero_is_clamped(self):
        """temperature=0 must be replaced with _MINIMAX_TEMPERATURE_MIN."""
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_TEMPERATURE_MIN,
        )

        provider = MiniMaxProvider()
        captured = {}

        async def fake_acompletion(**kw):
            captured.update(kw)
            return MagicMock()

        with patch("litellm.acompletion", new=fake_acompletion):
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0,
            )

        assert captured["temperature"] == _MINIMAX_TEMPERATURE_MIN

    @pytest.mark.asyncio
    async def test_negative_temperature_is_clamped(self):
        """Negative temperature values are also clamped."""
        from rotator_library.providers.minimax_provider import (
            MiniMaxProvider,
            _MINIMAX_TEMPERATURE_MIN,
        )

        provider = MiniMaxProvider()
        captured = {}

        async def fake_acompletion(**kw):
            captured.update(kw)
            return MagicMock()

        with patch("litellm.acompletion", new=fake_acompletion):
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
                temperature=-0.5,
            )

        assert captured["temperature"] == _MINIMAX_TEMPERATURE_MIN

    @pytest.mark.asyncio
    async def test_positive_temperature_is_not_changed(self):
        """Positive temperature values pass through unchanged."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        captured = {}

        async def fake_acompletion(**kw):
            captured.update(kw)
            return MagicMock()

        with patch("litellm.acompletion", new=fake_acompletion):
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
            )

        assert captured["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_missing_temperature_is_not_injected(self):
        """When no temperature key is present the default behaviour is preserved."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        captured = {}

        async def fake_acompletion(**kw):
            captured.update(kw)
            return MagicMock()

        with patch("litellm.acompletion", new=fake_acompletion):
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert "temperature" not in captured

    @pytest.mark.asyncio
    async def test_internal_proxy_keys_stripped(self):
        """credential_identifier and transaction_context must not reach LiteLLM."""
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        captured = {}

        async def fake_acompletion(**kw):
            captured.update(kw)
            return MagicMock()

        with patch("litellm.acompletion", new=fake_acompletion):
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
                credential_identifier="sk-secret",
                transaction_context={"id": "tx-1"},
            )

        assert "credential_identifier" not in captured
        assert "transaction_context" not in captured


# ---------------------------------------------------------------------------
# Unit tests – provider_config integration
# ---------------------------------------------------------------------------


class TestProviderConfig:
    """Verify that MiniMax is properly wired into the provider configuration."""

    def test_minimax_in_known_providers(self):
        """minimax must appear in the pre-computed KNOWN_PROVIDERS set."""
        from rotator_library.provider_config import KNOWN_PROVIDERS

        assert "minimax" in KNOWN_PROVIDERS

    def test_minimax_in_litellm_providers(self):
        """minimax must have a UI configuration entry."""
        from rotator_library.provider_config import LITELLM_PROVIDERS

        assert "minimax" in LITELLM_PROVIDERS
        cfg = LITELLM_PROVIDERS["minimax"]
        assert cfg.get("category") == "popular"

    def test_minimax_in_scraped_providers(self):
        """minimax must have scraped provider data with the correct API base."""
        from rotator_library.litellm_providers import SCRAPED_PROVIDERS

        assert "minimax" in SCRAPED_PROVIDERS
        entry = SCRAPED_PROVIDERS["minimax"]
        assert entry["api_base_url"] == "https://api.minimax.io/v1"
        assert "MINIMAX_API_KEY" in entry["api_key_env_vars"]

    def test_minimax_plugin_registered(self):
        """MiniMaxProvider must be auto-discovered and registered as a plugin."""
        from rotator_library.providers import PROVIDER_PLUGINS
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        assert "minimax" in PROVIDER_PLUGINS
        assert PROVIDER_PLUGINS["minimax"] is MiniMaxProvider

    def test_minimax_api_base_override_env_var(self):
        """extra_vars for minimax must include MINIMAX_API_BASE."""
        from rotator_library.provider_config import LITELLM_PROVIDERS

        extra_vars = LITELLM_PROVIDERS["minimax"].get("extra_vars", [])
        extra_var_names = [v[0] for v in extra_vars]
        assert "MINIMAX_API_BASE" in extra_var_names


# ---------------------------------------------------------------------------
# Unit tests – provider_urls
# ---------------------------------------------------------------------------


class TestProviderUrls:
    """Verify that MiniMax is listed in the provider URL map."""

    def test_minimax_in_provider_url_map(self):
        """MiniMax must have a hardcoded API base URL in PROVIDER_URL_MAP."""
        from proxy_app.provider_urls import PROVIDER_URL_MAP

        assert "minimax" in PROVIDER_URL_MAP
        assert PROVIDER_URL_MAP["minimax"] == "https://api.minimax.io/v1"


# ---------------------------------------------------------------------------
# Integration tests – live API calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMiniMaxIntegration:
    """
    Live integration tests.

    These tests are skipped automatically when MINIMAX_API_KEY is not set.
    Run them with:

        MINIMAX_API_KEY=your_key pytest -m integration tests/test_minimax_provider.py
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            pytest.skip("MINIMAX_API_KEY not set – skipping integration tests")
        self.api_key = api_key

    @pytest.mark.asyncio
    async def test_live_model_list(self):
        """The live API should return at least one model."""
        import httpx
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()
        async with httpx.AsyncClient(timeout=10) as client:
            models = await provider.get_models(self.api_key, client)

        assert len(models) > 0
        assert all(m.startswith("minimax/") for m in models)

    @pytest.mark.asyncio
    async def test_live_chat_completion(self):
        """A basic completion request should succeed end-to-end."""
        import litellm

        response = await litellm.acompletion(
            model="minimax/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Reply with the single word: hello"}],
            api_key=self.api_key,
            api_base="https://api.minimax.io/v1",
            temperature=0.5,
        )

        assert response.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_live_temperature_clamping(self):
        """Sending temperature=0 through the provider should not raise an error."""
        import httpx
        from rotator_library.providers.minimax_provider import MiniMaxProvider

        provider = MiniMaxProvider()

        with patch("litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = MagicMock()
            await provider.acompletion(
                MagicMock(),
                model="minimax/MiniMax-M2.7",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0,
            )
            called_kwargs = mock_acompletion.call_args.kwargs
            assert called_kwargs["temperature"] > 0
