from __future__ import annotations

import asyncio

import pytest

from rotator_library.cooldown_manager import CooldownManager


@pytest.mark.asyncio
async def test_start_cooldown_extends_but_does_not_shorten() -> None:
    manager = CooldownManager()

    await manager.start_cooldown("provider", 30)
    initial = await manager.get_remaining_cooldown("provider")
    await asyncio.sleep(0.01)
    await manager.start_cooldown("provider", 1)
    after_shorter = await manager.get_remaining_cooldown("provider")
    await manager.start_cooldown("provider", 60)
    after_longer = await manager.get_remaining_cooldown("provider")

    assert after_shorter > 25
    assert after_shorter <= initial
    assert after_longer > after_shorter
