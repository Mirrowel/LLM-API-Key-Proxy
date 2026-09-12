import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.transforms import ProviderTransforms


def test_gemma_system_message_conversion():
    transforms = ProviderTransforms(provider_plugins={})
    payload = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]
    }
    result = transforms.apply_sync("gemma", "gemma-3-test", payload)
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "user"]
