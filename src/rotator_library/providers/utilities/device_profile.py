# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/utilities/device_profile.py
"""
Device profile generation, binding, and storage for Gemini-based providers.

This module provides hardware ID simulation.
Each credential can have a unique device profile bound to it,
with version history tracking for audit purposes.

Device profiles contain 4 identifiers:
- machine_id: auth0|user_{random_hex(32)}
- mac_machine_id: UUID v4 format (custom builder)
- dev_device_id: Standard UUID v4
- sqm_id: {UUID} uppercase in braces

Storage: Per-credential profiles are stored in cache/device_profiles/{email_hash}.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.paths import get_cache_dir

lib_logger = logging.getLogger("rotator_library")

# Cache subdirectory for device profiles
DEVICE_PROFILES_SUBDIR = "device_profiles"


@dataclass
class DeviceProfile:
    """
    Device profile containing 4 hardware identifiers.

    Matches the DeviceProfile struct in device.rs:
    - machine_id: auth0|user_{random_hex(32)}
    - mac_machine_id: UUID v4 format
    - dev_device_id: Standard UUID v4
    - sqm_id: {UUID} uppercase in braces
    """

    machine_id: str
    mac_machine_id: str
    dev_device_id: str
    sqm_id: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DeviceProfile":
        """Create from dictionary."""
        return cls(
            machine_id=data["machine_id"],
            mac_machine_id=data["mac_machine_id"],
            dev_device_id=data["dev_device_id"],
            sqm_id=data["sqm_id"],
        )


@dataclass
class DeviceProfileVersion:
    """
    Versioned device profile with metadata for history tracking.

    Matches DeviceProfileVersion struct in account.rs.
    """

    id: str  # Random UUID v4 for this version
    created_at: int  # Unix timestamp
    label: str  # e.g., "auto_generated", "capture", "generate"
    profile: DeviceProfile
    is_current: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "label": self.label,
            "profile": self.profile.to_dict(),
            "is_current": self.is_current,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceProfileVersion":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            label=data["label"],
            profile=DeviceProfile.from_dict(data["profile"]),
            is_current=data.get("is_current", False),
        )


@dataclass
class CredentialDeviceData:
    """
    Complete device data for a credential, including current profile and history.
    """

    email: str
    current_profile: Optional[DeviceProfile] = None
    device_history: List[DeviceProfileVersion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "email": self.email,
            "current_profile": (
                self.current_profile.to_dict() if self.current_profile else None
            ),
            "device_history": [v.to_dict() for v in self.device_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredentialDeviceData":
        """Create from dictionary."""
        current = data.get("current_profile")
        return cls(
            email=data["email"],
            current_profile=DeviceProfile.from_dict(current) if current else None,
            device_history=[
                DeviceProfileVersion.from_dict(v)
                for v in data.get("device_history", [])
            ],
        )


# =============================================================================
# ID GENERATION FUNCTIONS (matching device.rs)
# =============================================================================


def random_hex(length: int) -> str:
    """
    Generate a random lowercase alphanumeric string.

    Matches rand::distributions::Alphanumeric + to_lowercase()

    Args:
        length: Number of characters to generate

    Returns:
        Random alphanumeric string (lowercase)
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def new_standard_machine_id() -> str:
    """
    Generate a UUID v4 format string with custom builder.

    Matches new_standard_machine_id() in device.rs:
    Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    where x is random hex [0-f] and y is random hex [8-b]

    Returns:
        UUID v4 format string
    """

    # Generate random hex characters
    def rand_hex(n: int) -> str:
        return "".join(random.choice("0123456789abcdef") for _ in range(n))

    # y must be in range 8-b (UUID v4 variant bits)
    y = random.choice("89ab")

    return f"{rand_hex(8)}-{rand_hex(4)}-4{rand_hex(3)}-{y}{rand_hex(3)}-{rand_hex(12)}"


def generate_profile() -> DeviceProfile:
    """
    Generate a new random device profile.

    Matches generate_profile() in device.rs:
    - machine_id: auth0|user_{random_hex(32)}
    - mac_machine_id: Custom UUID v4 format
    - dev_device_id: Standard UUID v4
    - sqm_id: {UUID} uppercase in braces

    Returns:
        New DeviceProfile with random identifiers
    """
    dev_device_id = str(uuid.uuid4())
    sqm_uuid = str(uuid.uuid4()).upper()

    return DeviceProfile(
        machine_id=f"auth0|user_{random_hex(32)}",
        mac_machine_id=new_standard_machine_id(),
        dev_device_id=dev_device_id,
        sqm_id=f"{{{sqm_uuid}}}",
    )


# =============================================================================
# STORAGE AND RETRIEVAL
# =============================================================================


def _get_email_hash(email: str) -> str:
    """Get a safe filename hash for an email address."""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def _get_profile_path(email: str) -> Path:
    """Get the path to the device profile file for an email."""
    cache_dir = get_cache_dir(subdir=DEVICE_PROFILES_SUBDIR)
    return cache_dir / f"{_get_email_hash(email)}.json"


def load_credential_device_data(email: str) -> Optional[CredentialDeviceData]:
    """
    Load device data for a credential from disk.

    Args:
        email: Email address of the credential

    Returns:
        CredentialDeviceData if found, None otherwise
    """
    profile_path = _get_profile_path(email)
    if not profile_path.exists():
        return None

    try:
        with open(profile_path, "r") as f:
            data = json.load(f)
        return CredentialDeviceData.from_dict(data)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        lib_logger.warning(f"Failed to load device profile for {email}: {e}")
        return None


def save_credential_device_data(data: CredentialDeviceData) -> bool:
    """
    Save device data for a credential to disk.

    Args:
        data: CredentialDeviceData to save

    Returns:
        True if saved successfully, False otherwise
    """
    profile_path = _get_profile_path(data.email)

    try:
        # Ensure directory exists
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = profile_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data.to_dict(), f, indent=2)

        # Atomic rename
        temp_path.replace(profile_path)

        lib_logger.debug(f"Saved device profile for {data.email}")
        return True
    except Exception as e:
        lib_logger.error(f"Failed to save device profile for {data.email}: {e}")
        return False


# =============================================================================
# HIGH-LEVEL API
# =============================================================================


def get_or_create_device_profile(
    email: str, auto_generate: bool = True
) -> Optional[DeviceProfile]:
    """
    Get the current device profile for a credential, optionally creating one.

    Args:
        email: Email address of the credential
        auto_generate: If True and no profile exists, generate one

    Returns:
        DeviceProfile if available, None otherwise
    """
    data = load_credential_device_data(email)

    if data and data.current_profile:
        return data.current_profile

    if not auto_generate:
        return None

    # Generate new profile
    return bind_new_device_profile(email, label="auto_generated")


def bind_new_device_profile(
    email: str, label: str = "generate", profile: Optional[DeviceProfile] = None
) -> DeviceProfile:
    """
    Bind a new device profile to a credential.

    Creates a new profile (or uses provided one), marks it as current,
    and adds it to the version history.

    Args:
        email: Email address of the credential
        label: Label for this version (e.g., "auto_generated", "generate", "capture")
        profile: Optional profile to bind. If None, generates a new one.

    Returns:
        The bound DeviceProfile
    """
    # Load existing data or create new
    data = load_credential_device_data(email)
    if not data:
        data = CredentialDeviceData(email=email)

    # Generate profile if not provided
    if profile is None:
        profile = generate_profile()

    # Mark all existing versions as not current
    for version in data.device_history:
        version.is_current = False

    # Create new version
    version = DeviceProfileVersion(
        id=str(uuid.uuid4()),
        created_at=int(time.time()),
        label=label,
        profile=profile,
        is_current=True,
    )

    # Update data
    data.current_profile = profile
    data.device_history.append(version)

    # Save
    save_credential_device_data(data)

    lib_logger.info(
        f"Bound new device profile for {email} (label={label}, "
        f"machine_id={profile.machine_id[:20]}...)"
    )

    return profile


def get_device_history(email: str) -> List[DeviceProfileVersion]:
    """
    Get the device profile version history for a credential.

    Args:
        email: Email address of the credential

    Returns:
        List of DeviceProfileVersion entries
    """
    data = load_credential_device_data(email)
    return data.device_history if data else []


def build_client_metadata(
    profile: Optional[DeviceProfile] = None,
    ide_type: str = "ANTIGRAVITY",
    platform: str = "WINDOWS_AMD64",
    plugin_type: str = "GEMINI",
) -> Dict[str, Any]:
    """
    Build Client-Metadata dict with device profile information.

    Args:
        profile: Optional DeviceProfile to include. If None, uses UNSPECIFIED values.
        ide_type: IDE type identifier
        platform: Platform identifier
        plugin_type: Plugin type identifier

    Returns:
        Client metadata dictionary
    """
    metadata = {
        "ideType": ide_type if profile else "IDE_UNSPECIFIED",
        "platform": platform if profile else "PLATFORM_UNSPECIFIED",
        "pluginType": plugin_type,
    }

    if profile:
        # Add device identifiers matching client headers
        metadata["machineId"] = profile.machine_id
        metadata["macMachineId"] = profile.mac_machine_id
        metadata["devDeviceId"] = profile.dev_device_id
        metadata["sqmId"] = profile.sqm_id

    return metadata


def build_client_metadata_header(
    profile: Optional[DeviceProfile] = None, **kwargs
) -> str:
    """
    Build Client-Metadata header value as JSON string.

    Args:
        profile: Optional DeviceProfile to include
        **kwargs: Additional arguments passed to build_client_metadata

    Returns:
        JSON string for Client-Metadata header
    """
    return json.dumps(build_client_metadata(profile, **kwargs))
