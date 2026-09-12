import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.config import load_provider_usage_config
from rotator_library.usage.types import RotationMode


def test_fair_cycle_enabled_by_default_for_sequential():
    with patch.dict(os.environ, {"ROTATION_MODE_MOCK": "sequential"}, clear=False):
        config = load_provider_usage_config("mock", {})

    assert config.rotation_mode == RotationMode.SEQUENTIAL
    assert config.fair_cycle.enabled is True
