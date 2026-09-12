import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.models import ModelResolver


def test_model_resolver_whitelist_overrides_blacklist():
    resolver = ModelResolver(
        provider_plugins={},
        ignore_models={"dummy": ["blocked-*", "dummy/blocked-explicit"]},
        whitelist_models={"dummy": ["allowed-*", "dummy/blocked-explicit"]},
    )

    assert resolver.is_model_allowed("dummy/allowed-1", "dummy") is True
    assert resolver.is_model_allowed("dummy/blocked-explicit", "dummy") is True
    assert resolver.is_model_allowed("dummy/blocked-test", "dummy") is False
    assert resolver.is_model_allowed("dummy/other", "dummy") is True
