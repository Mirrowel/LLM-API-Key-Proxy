# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Credential identity registry.

Provides stable identifiers for credentials that persist across
file path changes (for OAuth) and hide sensitive data (for API keys).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from ...core.types import CredentialInfo

lib_logger = logging.getLogger("rotator_library")


class CredentialRegistry:
    """
    Manages stable identifiers for credentials.

    Stable IDs are:
    - For OAuth credentials: The email address from _proxy_metadata.email
    - For API keys: SHA-256 hash of the key (truncated for readability)

    This ensures usage data persists even when:
    - OAuth credential files are moved/renamed
    - API keys are passed in different orders
    """

    def __init__(self):
        # Cache: accessor -> CredentialInfo
        self._cache: Dict[str, CredentialInfo] = {}
        # Reverse index: stable_id -> accessor
        self._id_to_accessor: Dict[str, str] = {}

    def get_stable_id(self, accessor: str, provider: str) -> str:
        """
        Get or create a stable ID for a credential accessor.

        Args:
            accessor: The credential accessor (file path or API key)
            provider: Provider name

        Returns:
            Stable identifier string
        """
        # Check cache first
        if accessor in self._cache:
            return self._cache[accessor].stable_id

        # Determine if OAuth or API key
        if self._is_oauth_path(accessor):
            stable_id = self._get_oauth_stable_id(accessor)
        else:
            stable_id = self._get_api_key_stable_id(accessor)

        # Cache the result
        info = CredentialInfo(
            accessor=accessor,
            stable_id=stable_id,
            provider=provider,
        )
        self._cache[accessor] = info
        self._id_to_accessor[stable_id] = accessor

        return stable_id

    def get_info(self, accessor: str, provider: str) -> CredentialInfo:
        """
        Get complete credential info for an accessor.

        Args:
            accessor: The credential accessor
            provider: Provider name

        Returns:
            CredentialInfo with stable_id and metadata
        """
        # Ensure stable ID is computed
        self.get_stable_id(accessor, provider)
        return self._cache[accessor]

    def get_accessor(self, stable_id: str) -> Optional[str]:
        """
        Get the current accessor for a stable ID.

        Args:
            stable_id: The stable identifier

        Returns:
            Current accessor string, or None if not found
        """
        return self._id_to_accessor.get(stable_id)

    def update_accessor(self, stable_id: str, new_accessor: str) -> None:
        """
        Update the accessor for a stable ID.

        Used when an OAuth credential file is moved/renamed.

        Args:
            stable_id: The stable identifier
            new_accessor: New accessor path
        """
        old_accessor = self._id_to_accessor.get(stable_id)
        if old_accessor and old_accessor in self._cache:
            info = self._cache.pop(old_accessor)
            info.accessor = new_accessor
            self._cache[new_accessor] = info
        self._id_to_accessor[stable_id] = new_accessor

    def update_metadata(
        self,
        accessor: str,
        provider: str,
        tier: Optional[str] = None,
        priority: Optional[int] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Update metadata for a credential.

        Args:
            accessor: The credential accessor
            provider: Provider name
            tier: Tier name (e.g., "standard-tier")
            priority: Priority level (lower = higher priority)
            display_name: Human-readable name
        """
        info = self.get_info(accessor, provider)
        if tier is not None:
            info.tier = tier
        if priority is not None:
            info.priority = priority
        if display_name is not None:
            info.display_name = display_name

    def get_all_accessors(self) -> Set[str]:
        """Get all registered accessors."""
        return set(self._cache.keys())

    def get_all_stable_ids(self) -> Set[str]:
        """Get all registered stable IDs."""
        return set(self._id_to_accessor.keys())

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        self._id_to_accessor.clear()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _is_oauth_path(self, accessor: str) -> bool:
        """
        Check if accessor is an OAuth credential file path.

        OAuth paths typically end with .json and exist on disk.
        API keys are typically raw strings.
        """
        # Simple heuristic: if it looks like a file path with .json, it's OAuth
        if accessor.endswith(".json"):
            return True
        # If it contains path separators, it's likely a file path
        if "/" in accessor or "\\" in accessor:
            return True
        return False

    def _get_oauth_stable_id(self, accessor: str) -> str:
        """
        Get stable ID for an OAuth credential.

        Reads the email from _proxy_metadata.email in the credential file.
        Falls back to file hash if email not found.
        """
        try:
            path = Path(accessor)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Try to get email from _proxy_metadata
                metadata = data.get("_proxy_metadata", {})
                email = metadata.get("email")
                if email:
                    return email

                # Fallback: try common OAuth fields
                for field in ["email", "client_email", "account"]:
                    if field in data:
                        return data[field]

                # Last resort: hash the file content
                lib_logger.debug(
                    f"No email found in OAuth credential {accessor}, using content hash"
                )
                return self._hash_content(json.dumps(data, sort_keys=True))

        except Exception as e:
            lib_logger.warning(f"Failed to read OAuth credential {accessor}: {e}")

        # Fallback: hash the path
        return self._hash_content(accessor)

    def _get_api_key_stable_id(self, accessor: str) -> str:
        """
        Get stable ID for an API key.

        Uses truncated SHA-256 hash to hide the actual key.
        """
        return self._hash_content(accessor)

    def _hash_content(self, content: str) -> str:
        """
        Create a stable hash of content.

        Uses first 12 characters of SHA-256 for readability.
        """
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize registry state for persistence.

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "accessor_index": dict(self._id_to_accessor),
            "credentials": {
                accessor: {
                    "stable_id": info.stable_id,
                    "provider": info.provider,
                    "tier": info.tier,
                    "priority": info.priority,
                    "display_name": info.display_name,
                }
                for accessor, info in self._cache.items()
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore registry state from persistence.

        Args:
            data: Dictionary from to_dict()
        """
        self._id_to_accessor = dict(data.get("accessor_index", {}))

        for accessor, cred_data in data.get("credentials", {}).items():
            info = CredentialInfo(
                accessor=accessor,
                stable_id=cred_data["stable_id"],
                provider=cred_data["provider"],
                tier=cred_data.get("tier"),
                priority=cred_data.get("priority", 999),
                display_name=cred_data.get("display_name"),
            )
            self._cache[accessor] = info
