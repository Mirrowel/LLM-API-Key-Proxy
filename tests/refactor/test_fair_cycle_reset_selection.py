import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.config import ProviderUsageConfig
from rotator_library.usage.limits.engine import LimitEngine
from rotator_library.usage.selection.engine import SelectionEngine
from rotator_library.usage.tracking.windows import WindowManager
from rotator_library.usage.types import CredentialState, FairCycleState, RotationMode


def test_selection_resets_fair_cycle_when_all_exhausted():
    config = ProviderUsageConfig(rotation_mode=RotationMode.SEQUENTIAL)
    config.fair_cycle.enabled = True

    windows = WindowManager(config.windows or [])
    limits = LimitEngine(config, windows)
    selector = SelectionEngine(config, limits, windows)

    state_a = CredentialState(stable_id="a", provider="p", accessor="a")
    state_b = CredentialState(stable_id="b", provider="p", accessor="b")
    state_a.fair_cycle["model-x"] = FairCycleState(
        model_or_group="model-x", exhausted=True
    )
    state_b.fair_cycle["model-x"] = FairCycleState(
        model_or_group="model-x", exhausted=True
    )

    states = {"a": state_a, "b": state_b}
    selected = selector.select(
        provider="p",
        model="model-x",
        states=states,
        quota_group=None,
        deadline=0.0,
    )

    assert selected in {"a", "b"}
    assert state_a.fair_cycle["model-x"].exhausted is False
    assert state_b.fair_cycle["model-x"].exhausted is False
