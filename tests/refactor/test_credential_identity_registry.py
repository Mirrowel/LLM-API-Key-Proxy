import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.identity.registry import CredentialRegistry


def test_registry_uses_email_for_oauth_and_hash_for_api_keys():
    registry = CredentialRegistry()
    data = {"_proxy_metadata": {"email": "user@example.com"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        oauth_path = f.name

    oauth_id = registry.get_stable_id(oauth_path, "provider")
    assert oauth_id == "user@example.com"

    api_id = registry.get_stable_id("secret-key-123", "provider")
    assert api_id != "secret-key-123"
    assert len(api_id) == 12
