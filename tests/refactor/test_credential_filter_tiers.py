import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.filters import CredentialFilter


class TieredProvider:
    def get_model_tier_requirement(self, model):
        return 2

    def get_credential_priority(self, cred):
        return {"c1": 1, "c2": 2, "c3": 3}.get(cred)

    def get_credential_tier_name(self, cred):
        return {"c1": "pro", "c2": "std", "c3": "free"}.get(cred)


def test_filter_by_tier_and_priority_groups():
    filterer = CredentialFilter({"tiered": TieredProvider})
    result = filterer.filter_by_tier(["c1", "c2", "c3", "c4"], "model", "tiered")

    assert result.compatible == ["c1", "c2"]
    assert result.incompatible == ["c3"]
    assert result.unknown == ["c4"]

    grouped = filterer.group_by_priority(result.all_usable, result.priorities)
    assert grouped[1] == ["c1"]
    assert grouped[2] == ["c2"]
