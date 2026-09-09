import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.transforms import ProviderTransforms


class ThinkingProvider:
    def handle_thinking_parameter(self, kwargs, model):
        kwargs["thinking_handled"] = True


def test_gemini_safety_injection_removed():
    # Safety defaults/fill were deliberately REMOVED (the proxy never
    # fabricates safety settings the client did not declare — W4 D4):
    # empty payloads stay empty, explicit settings stay untouched.
    transforms = ProviderTransforms(provider_plugins={})
    result = transforms.apply_sync("gemini", "gemini-1.5-pro", {})
    assert "safety_settings" not in result

    payload = {"safety_settings": {"harassment": "OFF"}}
    result = transforms.apply_sync("gemini", "gemini-1.5-pro", payload)
    assert result["safety_settings"] == {"harassment": "OFF"}


def test_thinking_transforms_for_gemini_and_nvidia():
    transforms = ProviderTransforms(
        provider_plugins={"gemini": ThinkingProvider, "nvidia_nim": ThinkingProvider}
    )

    payload = {}
    result = transforms.apply_sync("gemini", "gemini-1.5-pro", payload)
    assert result["thinking_handled"] is True

    payload = {}
    result = transforms.apply_sync("nvidia_nim", "nvidia/test", payload)
    assert result["thinking_handled"] is True
