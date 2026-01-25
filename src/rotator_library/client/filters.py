# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Credential filtering by tier compatibility and priority.

Extracts the tier filtering logic that was duplicated in client.py
at lines 1242-1315 and 2004-2076.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.types import FilterResult

lib_logger = logging.getLogger("rotator_library")


class CredentialFilter:
    """
    Filter and group credentials by tier compatibility and priority.

    This class extracts the credential filtering logic that was previously
    duplicated in both _execute_with_retry and _streaming_acompletion_with_retry.
    """

    def __init__(
        self,
        provider_plugins: Dict[str, Any],
        provider_instances: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CredentialFilter.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes/instances
            provider_instances: Shared dict for caching provider instances.
                If None, creates a new dict (not recommended - leads to duplicate instances).
        """
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = (
            provider_instances if provider_instances is not None else {}
        )

    def _get_plugin_instance(self, provider: str) -> Optional[Any]:
        """
        Get or create a plugin instance for a provider.

        Args:
            provider: Provider name

        Returns:
            Plugin instance or None if not found
        """
        if provider not in self._plugin_instances:
            plugin_class = self._plugins.get(provider)
            if plugin_class:
                # Check if it's a class or already an instance
                if isinstance(plugin_class, type):
                    lib_logger.debug(
                        f"[CredentialFilter] CREATING NEW INSTANCE for {provider}"
                    )
                    self._plugin_instances[provider] = plugin_class()
                else:
                    self._plugin_instances[provider] = plugin_class
            else:
                return None
        return self._plugin_instances[provider]

    def filter_by_tier(
        self,
        credentials: List[str],
        model: str,
        provider: str,
    ) -> FilterResult:
        """
        Filter credentials by tier compatibility for a model.

        Args:
            credentials: List of credential identifiers
            model: Model being requested
            provider: Provider name

        Returns:
            FilterResult with categorized credentials
        """
        plugin = self._get_plugin_instance(provider)

        # Get tier requirement for model
        required_tier = None
        if plugin and hasattr(plugin, "get_model_tier_requirement"):
            required_tier = plugin.get_model_tier_requirement(model)

        compatible: List[str] = []
        unknown: List[str] = []
        incompatible: List[str] = []
        priorities: Dict[str, int] = {}
        tier_names: Dict[str, str] = {}

        for cred in credentials:
            # Get priority and tier name
            priority = None
            tier_name = None

            if plugin:
                if hasattr(plugin, "get_credential_priority"):
                    priority = plugin.get_credential_priority(cred)
                if hasattr(plugin, "get_credential_tier_name"):
                    tier_name = plugin.get_credential_tier_name(cred)

            if priority is not None:
                priorities[cred] = priority
            if tier_name:
                tier_names[cred] = tier_name

            # Categorize by tier compatibility
            if required_tier is None:
                # No tier requirement - all compatible
                compatible.append(cred)
            elif priority is None:
                # Unknown priority - keep as candidate
                unknown.append(cred)
            elif priority <= required_tier:
                # Known compatible (lower priority number = higher tier)
                compatible.append(cred)
            else:
                # Known incompatible
                incompatible.append(cred)

        # Log if all credentials are incompatible
        if incompatible and not compatible and not unknown:
            lib_logger.warning(
                f"Model {model} requires tier <= {required_tier}, "
                f"but all {len(incompatible)} credentials are incompatible"
            )

        return FilterResult(
            compatible=compatible,
            unknown=unknown,
            incompatible=incompatible,
            priorities=priorities,
            tier_names=tier_names,
        )

    def group_by_priority(
        self,
        credentials: List[str],
        priorities: Dict[str, int],
    ) -> Dict[int, List[str]]:
        """
        Group credentials by priority level.

        Args:
            credentials: List of credential identifiers
            priorities: Dict mapping credentials to priority levels

        Returns:
            Dict mapping priority levels to credential lists, sorted by priority
        """
        groups: Dict[int, List[str]] = {}

        for cred in credentials:
            priority = priorities.get(cred, 999)
            if priority not in groups:
                groups[priority] = []
            groups[priority].append(cred)

        # Return sorted by priority (lower = higher priority)
        return dict(sorted(groups.items()))

    def get_highest_priority_credentials(
        self,
        credentials: List[str],
        priorities: Dict[str, int],
    ) -> List[str]:
        """
        Get credentials with the highest priority (lowest priority number).

        Args:
            credentials: List of credential identifiers
            priorities: Dict mapping credentials to priority levels

        Returns:
            List of credentials with the highest priority
        """
        if not credentials:
            return []

        groups = self.group_by_priority(credentials, priorities)
        if not groups:
            return credentials

        # Get the lowest priority number (highest priority)
        highest_priority = min(groups.keys())
        return groups[highest_priority]
