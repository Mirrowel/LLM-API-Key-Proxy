import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.config import (
    CustomCapConfig,
    ProviderUsageConfig,
    get_default_windows,
    CooldownMode,
)
from rotator_library.usage.limits.engine import LimitEngine
from rotator_library.usage.tracking.windows import WindowManager
from rotator_library.usage.types import (
    CredentialState,
    ModelStats,
    WindowStats,
    LimitResult,
)


def test_custom_cap_blocks_with_cooldown():
    config = ProviderUsageConfig(
        windows=get_default_windows(),
        custom_caps=[
            CustomCapConfig(
                tier_key="1",
                model_or_group="model-x",
                max_requests=2,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            )
        ],
    )

    windows = WindowManager(config.windows)
    engine = LimitEngine(config, windows)

    state = CredentialState(stable_id="id", provider="p", accessor="key")
    state.priority = 1
    usage = ModelStats()
    usage.windows["daily"] = WindowStats(
        name="daily",
        request_count=2,
        started_at=100.0,
        reset_at=9999999999.0,
        limit=10,
    )
    state.model_usage["model-x"] = usage

    result = engine.check_all(state, "model-x", None)
    assert result.allowed is False
    assert result.result == LimitResult.BLOCKED_CUSTOM_CAP
    assert result.blocked_until is not None
