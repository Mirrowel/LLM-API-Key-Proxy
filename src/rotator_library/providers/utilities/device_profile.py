# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/utilities/device_profile.py
"""
Device fingerprint generation, binding, and storage for Antigravity provider.

This module provides complete device fingerprinting for rate-limit mitigation.

Each credential gets a unique, persistent fingerprint that includes:
- User-Agent: antigravity/{FIXED_VERSION} {platform}/{arch}
- X-Goog-Api-Client: randomized SDK client string
- X-Goog-QuotaUser: device-{random_hex}
- X-Client-Device-Id: UUID v4
- Client-Metadata: JSON with IDE/platform/OS info + legacy hardware IDs

Fingerprints are stored per-credential in cache/device_profiles/{email_hash}.json
with version history for audit purposes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.paths import get_cache_dir

lib_logger = logging.getLogger("rotator_library")

# Cache subdirectory for device profiles
DEVICE_PROFILES_SUBDIR = "device_profiles"

# =============================================================================
# FINGERPRINT CONSTANTS
# =============================================================================

# Fixed version - does NOT randomize per user request
ANTIGRAVITY_VERSION = "1.15.8"

# Platform configurations with OS versions
PLATFORMS = {
    "win32": {
        "name": "WINDOWS",
        "os_versions": [
            "10.0.19041",
            "10.0.19042",
            "10.0.19043",
            "10.0.22000",
            "10.0.22621",
            "10.0.22631",
        ],
    },
    "darwin": {
        "name": "MACOS",
        "os_versions": ["10.15.7", "11.6.8", "12.6.3", "13.5.2", "14.2.1", "14.5"],
    },
    "linux": {
        "name": "LINUX",
        "os_versions": ["5.15.0", "5.19.0", "6.1.0", "6.2.0", "6.5.0", "6.6.0"],
    },
}

# Architecture options
ARCHITECTURES = ["x64", "arm64"]

# SDK client strings (randomized per credential)
SDK_CLIENTS = [
    "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "google-cloud-sdk vscode/1.96.0",
    "google-cloud-sdk vscode/1.95.0",
    #"google-cloud-sdk jetbrains/2024.3",
    #"google-cloud-sdk intellij/2024.1",
    #"google-cloud-sdk android-studio/2024.1",
]

# IDE types for Client-Metadata
IDE_TYPES = [
    "VSCODE",
    #"INTELLIJ",
    #"ANDROID_STUDIO",
    #"CLOUD_SHELL_EDITOR",
]


# =============================================================================
# LEGACY DEVICE PROFILE (kept for backward compatibility)
# =============================================================================


@dataclass
class DeviceProfile:
    """
    Legacy device profile containing 4 hardware identifiers.

    Kept for backward compatibility with existing stored profiles.
    New code should use DeviceFingerprint instead.
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


# =============================================================================
# DEVICE FINGERPRINT (new complete implementation)
# =============================================================================


@dataclass
class DeviceFingerprint:
    """
    Complete device fingerprint for a credential.

    Contains all necessary hardware identifiers and metadata for API authentication.
    """

    # === HTTP Header Fields ===
    user_agent: str  # "antigravity/1.15.8 win32/x64"
    api_client: str  # "google-cloud-sdk vscode/1.96.0"
    quota_user: str  # "device-a1b2c3d4e5f6"
    device_id: str  # UUID v4 for X-Client-Device-Id header

    # === Client-Metadata Fields ===
    ide_type: str  # "VSCODE", "INTELLIJ", etc.
    platform: str  # "WINDOWS", "MACOS", "LINUX"
    platform_raw: str  # "win32", "darwin", "linux" (for UA)
    arch: str  # "x64", "arm64"
    os_version: str  # "10.0.22631", "14.5", etc.
    sqm_id: str  # "{UUID}" uppercase in braces
    plugin_type: str  # "GEMINI" (always)

    # === Legacy Hardware IDs (kept for compatibility) ===
    machine_id: str  # "auth0|user_{hex}"
    mac_machine_id: str  # Custom UUID v4
    dev_device_id: str  # Standard UUID v4

    # === Metadata ===
    created_at: int  # Unix timestamp
    session_token: str  # 16-byte hex

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceFingerprint":
        """Create from dictionary."""
        return cls(
            user_agent=data["user_agent"],
            api_client=data["api_client"],
            quota_user=data["quota_user"],
            device_id=data["device_id"],
            ide_type=data["ide_type"],
            platform=data["platform"],
            platform_raw=data["platform_raw"],
            arch=data["arch"],
            os_version=data["os_version"],
            sqm_id=data["sqm_id"],
            plugin_type=data["plugin_type"],
            machine_id=data["machine_id"],
            mac_machine_id=data["mac_machine_id"],
            dev_device_id=data["dev_device_id"],
            created_at=data["created_at"],
            session_token=data["session_token"],
        )

    def to_legacy_profile(self) -> DeviceProfile:
        """Convert to legacy DeviceProfile for backward compatibility."""
        return DeviceProfile(
            machine_id=self.machine_id,
            mac_machine_id=self.mac_machine_id,
            dev_device_id=self.dev_device_id,
            sqm_id=self.sqm_id,
        )


@dataclass
class DeviceFingerprintVersion:
    """
    Versioned device fingerprint with metadata for history tracking.
    """

    id: str  # Random UUID v4 for this version
    created_at: int  # Unix timestamp
    label: str  # e.g., "auto_generated", "upgraded", "regenerated"
    fingerprint: DeviceFingerprint
    is_current: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "label": self.label,
            "fingerprint": self.fingerprint.to_dict(),
            "is_current": self.is_current,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceFingerprintVersion":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            label=data["label"],
            fingerprint=DeviceFingerprint.from_dict(data["fingerprint"]),
            is_current=data.get("is_current", False),
        )


@dataclass
class CredentialDeviceData:
    """
    Complete device data for a credential, including current fingerprint and history.
    """

    email: str
    current_fingerprint: Optional[DeviceFingerprint] = None
    fingerprint_history: List[DeviceFingerprintVersion] = field(default_factory=list)

    # Legacy fields for migration
    current_profile: Optional[DeviceProfile] = None
    device_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "email": self.email,
            "current_fingerprint": (
                self.current_fingerprint.to_dict() if self.current_fingerprint else None
            ),
            "fingerprint_history": [v.to_dict() for v in self.fingerprint_history],
            # Legacy fields (kept for backward compat, but prefer new fields)
            "current_profile": (
                self.current_profile.to_dict() if self.current_profile else None
            ),
            "device_history": self.device_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredentialDeviceData":
        """Create from dictionary with silent upgrade support."""
        current_fp = data.get("current_fingerprint")
        fp_history = data.get("fingerprint_history", [])

        # Legacy profile data
        current_profile_data = data.get("current_profile")
        device_history = data.get("device_history", [])

        return cls(
            email=data["email"],
            current_fingerprint=(
                DeviceFingerprint.from_dict(current_fp) if current_fp else None
            ),
            fingerprint_history=[
                DeviceFingerprintVersion.from_dict(v) for v in fp_history
            ],
            current_profile=(
                DeviceProfile.from_dict(current_profile_data)
                if current_profile_data
                else None
            ),
            device_history=device_history,
        )


# =============================================================================
# ID GENERATION FUNCTIONS
# =============================================================================


def random_hex(length: int) -> str:
    """
    Generate a random lowercase alphanumeric string.

    Args:
        length: Number of characters to generate

    Returns:
        Random alphanumeric string (lowercase)
    """
    import string

    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def new_standard_machine_id() -> str:
    """
    Generate a UUID v4 format string with custom builder.

    Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    where x is random hex [0-f] and y is random hex [8-b]

    Returns:
        UUID v4 format string
    """

    def rand_hex(n: int) -> str:
        return "".join(random.choice("0123456789abcdef") for _ in range(n))

    # y must be in range 8-b (UUID v4 variant bits)
    y = random.choice("89ab")

    return f"{rand_hex(8)}-{rand_hex(4)}-4{rand_hex(3)}-{y}{rand_hex(3)}-{rand_hex(12)}"


def generate_device_fingerprint() -> DeviceFingerprint:
    """
    Generate a complete device fingerprint.

    Returns:
        New DeviceFingerprint with all fields populated
    """
    # Pick platform (raw + name + os_version together for consistency)
    platform_raw = random.choice(list(PLATFORMS.keys()))  # "win32"
    platform_info = PLATFORMS[platform_raw]
    platform_name = platform_info["name"]  # "WINDOWS"
    os_version = random.choice(platform_info["os_versions"])

    # Pick arch
    arch = random.choice(ARCHITECTURES)  # "x64"

    # Build user agent with FIXED version
    user_agent = f"antigravity/{ANTIGRAVITY_VERSION} {platform_raw}/{arch}"

    # Other randomized fields (picked once, persisted)
    api_client = random.choice(SDK_CLIENTS)
    quota_user = f"device-{secrets.token_hex(8)}"
    device_id = str(uuid.uuid4())
    ide_type = random.choice(IDE_TYPES)
    sqm_id = f"{{{str(uuid.uuid4()).upper()}}}"
    session_token = secrets.token_hex(16)

    # Legacy hardware IDs (for compatibility)
    machine_id = f"auth0|user_{random_hex(32)}"
    mac_machine_id = new_standard_machine_id()
    dev_device_id = str(uuid.uuid4())

    return DeviceFingerprint(
        user_agent=user_agent,
        api_client=api_client,
        quota_user=quota_user,
        device_id=device_id,
        ide_type=ide_type,
        platform=platform_name,
        platform_raw=platform_raw,
        arch=arch,
        os_version=os_version,
        sqm_id=sqm_id,
        plugin_type="GEMINI",
        machine_id=machine_id,
        mac_machine_id=mac_machine_id,
        dev_device_id=dev_device_id,
        created_at=int(time.time()),
        session_token=session_token,
    )


def upgrade_legacy_profile(profile: DeviceProfile) -> DeviceFingerprint:
    """
    Upgrade a legacy DeviceProfile to a full DeviceFingerprint.

    Preserves the legacy hardware IDs and generates the missing fields.

    Args:
        profile: Legacy DeviceProfile to upgrade

    Returns:
        New DeviceFingerprint with legacy IDs preserved
    """
    # Pick platform (raw + name + os_version together for consistency)
    platform_raw = random.choice(list(PLATFORMS.keys()))
    platform_info = PLATFORMS[platform_raw]
    platform_name = platform_info["name"]
    os_version = random.choice(platform_info["os_versions"])

    # Pick arch
    arch = random.choice(ARCHITECTURES)

    # Build user agent with FIXED version
    user_agent = f"antigravity/{ANTIGRAVITY_VERSION} {platform_raw}/{arch}"

    # Generate missing fields
    api_client = random.choice(SDK_CLIENTS)
    quota_user = f"device-{secrets.token_hex(8)}"
    device_id = str(uuid.uuid4())
    ide_type = random.choice(IDE_TYPES)
    session_token = secrets.token_hex(16)

    return DeviceFingerprint(
        user_agent=user_agent,
        api_client=api_client,
        quota_user=quota_user,
        device_id=device_id,
        ide_type=ide_type,
        platform=platform_name,
        platform_raw=platform_raw,
        arch=arch,
        os_version=os_version,
        sqm_id=profile.sqm_id,  # Preserve
        plugin_type="GEMINI",
        machine_id=profile.machine_id,  # Preserve
        mac_machine_id=profile.mac_machine_id,  # Preserve
        dev_device_id=profile.dev_device_id,  # Preserve
        created_at=int(time.time()),
        session_token=session_token,
    )


# =============================================================================
# HEADER BUILDER
# =============================================================================


def build_fingerprint_headers(fp: DeviceFingerprint) -> Dict[str, str]:
    """
    Build all 5 HTTP headers from a fingerprint.

    Args:
        fp: DeviceFingerprint to build headers from

    Returns:
        Dict with User-Agent, X-Goog-Api-Client, Client-Metadata,
        X-Goog-QuotaUser, X-Client-Device-Id
    """
    client_metadata = {
        "ideType": fp.ide_type,
        "platform": fp.platform,
        "pluginType": fp.plugin_type,
        "osVersion": fp.os_version,
        "arch": fp.arch,
        "sqmId": fp.sqm_id,
    }

    return {
        "User-Agent": fp.user_agent,
        "X-Goog-Api-Client": fp.api_client,
        "Client-Metadata": json.dumps(client_metadata),
        "X-Goog-QuotaUser": fp.quota_user,
        "X-Client-Device-Id": fp.device_id,
    }


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

        lib_logger.debug(f"Saved device fingerprint for {data.email}")
        return True
    except Exception as e:
        lib_logger.error(f"Failed to save device fingerprint for {data.email}: {e}")
        return False


# =============================================================================
# HIGH-LEVEL API
# =============================================================================


def get_or_create_fingerprint(
    email: str, auto_generate: bool = True
) -> Optional[DeviceFingerprint]:
    """
    Get the current device fingerprint for a credential, optionally creating one.

    Handles silent upgrade from legacy DeviceProfile to DeviceFingerprint.

    Args:
        email: Email address of the credential
        auto_generate: If True and no fingerprint exists, generate one

    Returns:
        DeviceFingerprint if available, None otherwise
    """
    data = load_credential_device_data(email)

    # Check for existing fingerprint
    if data and data.current_fingerprint:
        return data.current_fingerprint

    # Check for legacy profile and upgrade (silent upgrade)
    if data and data.current_profile:
        lib_logger.info(f"Upgrading legacy device profile to fingerprint for {email}")
        fingerprint = upgrade_legacy_profile(data.current_profile)
        _save_fingerprint(data, fingerprint, label="upgraded")
        return fingerprint

    if not auto_generate:
        return None

    # Generate new fingerprint
    return bind_new_fingerprint(email, label="auto_generated")


def bind_new_fingerprint(
    email: str,
    label: str = "auto_generated",
    fingerprint: Optional[DeviceFingerprint] = None,
) -> DeviceFingerprint:
    """
    Bind a new device fingerprint to a credential.

    Creates a new fingerprint (or uses provided one), marks it as current,
    and adds it to the version history.

    Args:
        email: Email address of the credential
        label: Label for this version (e.g., "auto_generated", "regenerated")
        fingerprint: Optional fingerprint to bind. If None, generates a new one.

    Returns:
        The bound DeviceFingerprint
    """
    # Load existing data or create new
    data = load_credential_device_data(email)
    if not data:
        data = CredentialDeviceData(email=email)

    # Generate fingerprint if not provided
    if fingerprint is None:
        fingerprint = generate_device_fingerprint()

    _save_fingerprint(data, fingerprint, label)

    lib_logger.info(
        f"Bound new device fingerprint for {email} (label={label}, "
        f"ua={fingerprint.user_agent})"
    )

    return fingerprint


def _save_fingerprint(
    data: CredentialDeviceData, fingerprint: DeviceFingerprint, label: str
) -> None:
    """
    Internal helper to save a fingerprint to credential data.
    """
    # Mark all existing versions as not current
    for version in data.fingerprint_history:
        version.is_current = False

    # Create new version
    version = DeviceFingerprintVersion(
        id=str(uuid.uuid4()),
        created_at=int(time.time()),
        label=label,
        fingerprint=fingerprint,
        is_current=True,
    )

    # Update data
    data.current_fingerprint = fingerprint
    data.fingerprint_history.append(version)

    # Also update legacy profile for backward compatibility
    data.current_profile = fingerprint.to_legacy_profile()

    # Save
    save_credential_device_data(data)


def get_fingerprint_history(email: str) -> List[DeviceFingerprintVersion]:
    """
    Get the device fingerprint version history for a credential.

    Args:
        email: Email address of the credential

    Returns:
        List of DeviceFingerprintVersion entries
    """
    data = load_credential_device_data(email)
    return data.fingerprint_history if data else []


def regenerate_fingerprint(email: str) -> DeviceFingerprint:
    """
    Regenerate the device fingerprint for a credential.

    Call this to get a fresh identity (e.g., after rate limiting).

    Args:
        email: Email address of the credential

    Returns:
        New DeviceFingerprint
    """
    return bind_new_fingerprint(email, label="regenerated")


# =============================================================================
# LEGACY API (kept for backward compatibility)
# =============================================================================


def get_or_create_device_profile(
    email: str, auto_generate: bool = True
) -> Optional[DeviceProfile]:
    """
    Get the current device profile for a credential.

    DEPRECATED: Use get_or_create_fingerprint() instead.

    Args:
        email: Email address of the credential
        auto_generate: If True and no profile exists, generate one

    Returns:
        DeviceProfile if available, None otherwise
    """
    fingerprint = get_or_create_fingerprint(email, auto_generate)
    if fingerprint:
        return fingerprint.to_legacy_profile()
    return None


def generate_profile() -> DeviceProfile:
    """
    Generate a new random device profile.

    DEPRECATED: Use generate_device_fingerprint() instead.

    Returns:
        New DeviceProfile with random identifiers
    """
    fingerprint = generate_device_fingerprint()
    return fingerprint.to_legacy_profile()


def build_client_metadata(
    profile: Optional[DeviceProfile] = None,
    ide_type: str = "ANTIGRAVITY",
    platform: str = "WINDOWS_AMD64",
    plugin_type: str = "GEMINI",
) -> Dict[str, Any]:
    """
    Build Client-Metadata dict with device profile information.

    DEPRECATED: Use build_fingerprint_headers() instead.

    Args:
        profile: Optional DeviceProfile to include.
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

    DEPRECATED: Use build_fingerprint_headers() instead.

    Args:
        profile: Optional DeviceProfile to include
        **kwargs: Additional arguments passed to build_client_metadata

    Returns:
        JSON string for Client-Metadata header
    """
    return json.dumps(build_client_metadata(profile, **kwargs))
