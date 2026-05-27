# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Balanced rotation strategy.

Distributes load evenly across credentials using weighted random selection.
"""

import random
import logging
from typing import Dict, List, Optional

from ...types import CredentialState, SelectionContext, RotationMode

lib_logger = logging.getLogger("rotator_library")


class BalancedStrategy:
    """
    Balanced credential rotation strategy.

    Uses weighted random selection where less-used credentials have
    higher probability of being selected. The tolerance parameter
    controls how much randomness is introduced.

    Weight formula: weight = (max_usage - credential_usage) + tolerance + 1
    """

    def __init__(self, tolerance: float = 3.0):
        """
        Initialize balanced strategy.

        Args:
            tolerance: Controls randomness of selection.
                - 0.0: Deterministic, least-used always selected
                - 2.0-4.0: Recommended, balanced randomness
                - 5.0+: High randomness
        """
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "balanced"

    @property
    def mode(self) -> RotationMode:
        return RotationMode.BALANCED

    def select(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
    ) -> Optional[str]:
        """
        Select a credential using weighted random selection.

        Args:
            context: Selection context with candidates and usage info
            states: Dict of stable_id -> CredentialState

        Returns:
            Selected stable_id, or None if no candidates
        """
        if not context.candidates:
            return None

        if len(context.candidates) == 1:
            return context.candidates[0]

        # Group by priority for tiered selection
        priority_groups = self._group_by_priority(
            context.candidates, context.priorities
        )

        # Try each priority tier in order
        for priority in sorted(priority_groups.keys()):
            candidates = priority_groups[priority]
            if not candidates:
                continue

            # Calculate weights for this tier
            weights = self._calculate_weights(candidates, context.usage_counts)

            # Weighted random selection
            selected = self._weighted_random_choice(candidates, weights)
            if selected:
                return selected

        # Fallback: first candidate
        return context.candidates[0]

    def _group_by_priority(
        self,
        candidates: List[str],
        priorities: Dict[str, int],
    ) -> Dict[int, List[str]]:
        """Group candidates by priority tier."""
        groups: Dict[int, List[str]] = {}
        for stable_id in candidates:
            priority = priorities.get(stable_id, 999)
            groups.setdefault(priority, []).append(stable_id)
        return groups

    def _calculate_weights(
        self,
        candidates: List[str],
        usage_counts: Dict[str, int],
    ) -> List[float]:
        """
        Calculate selection weights for candidates.

        Weight formula: weight = (max_usage - credential_usage) + tolerance + 1
        """
        if not candidates:
            return []

        # Get usage counts
        usages = [usage_counts.get(stable_id, 0) for stable_id in candidates]
        max_usage = max(usages) if usages else 0

        # Calculate weights
        weights = []
        for usage in usages:
            weight = (max_usage - usage) + self.tolerance + 1
            weights.append(max(weight, 0.1))  # Ensure minimum weight

        return weights

    def _weighted_random_choice(
        self,
        candidates: List[str],
        weights: List[float],
    ) -> Optional[str]:
        """Select a candidate using weighted random choice."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Normalize weights
        total = sum(weights)
        if total <= 0:
            return random.choice(candidates)

        # Weighted selection
        r = random.uniform(0, total)
        cumulative = 0
        for candidate, weight in zip(candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return candidate

        return candidates[-1]
