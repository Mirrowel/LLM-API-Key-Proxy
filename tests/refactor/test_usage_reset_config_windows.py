import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.rotating_client import RotatingClient
from rotator_library.providers.provider_interface import ProviderInterface


class ResetConfigProvider(ProviderInterface):
    provider_env_name = "resetcfg"

    def get_usage_reset_config(self, credential: str):
        return {
            "mode": "credential",
            "window_seconds": 7200,
            "field_name": "models",
        }

    async def get_models(self, api_key, client):
        return []


def test_usage_reset_config_overrides_windows():
    original_plugins = RotatingClient.__init__.__globals__["PROVIDER_PLUGINS"]
    previous = original_plugins.get("resetcfg")
    original_plugins["resetcfg"] = ResetConfigProvider
    try:
        with patch(
            "rotator_library.client.rotating_client.CredentialManager.discover_and_prepare",
            return_value={},
        ):
            client = RotatingClient(
                api_keys={"resetcfg": ["key-1"]},
                configure_logging=False,
            )
        manager = client.usage_managers["resetcfg"]

        windows = manager.config.windows
        assert len(windows) == 1
        assert windows[0].duration_seconds == 7200
        assert windows[0].applies_to == "credential"
        assert windows[0].is_primary is True
        import asyncio

        asyncio.run(client.close())
    finally:
        if previous is None:
            original_plugins.pop("resetcfg", None)
        else:
            original_plugins["resetcfg"] = previous
