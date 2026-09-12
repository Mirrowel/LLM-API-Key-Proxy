import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.models import ModelResolver


class FakeDefinitions:
    def get_model_id(self, provider, name):
        if provider == "map" and name == "alias":
            return "real"
        return None


class MappedProvider:
    model_definitions = FakeDefinitions()


def test_model_resolver_maps_alias_to_id():
    resolver = ModelResolver(
        provider_plugins={"map": MappedProvider},
        model_definitions=FakeDefinitions(),
    )

    assert resolver.resolve_model_id("map/alias", "map") == "map/real"
