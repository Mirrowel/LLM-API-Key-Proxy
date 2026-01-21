# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/__init__.py

from .headless_detection import is_headless_environment
from .paths import (
    get_default_root,
    get_logs_dir,
    get_cache_dir,
    get_oauth_dir,
    get_data_file,
)
from .reauth_coordinator import get_reauth_coordinator, ReauthCoordinator
from .resilient_io import (
    BufferedWriteRegistry,
    ResilientStateWriter,
    safe_write_json,
    safe_log_write,
    safe_mkdir,
)
from .suppress_litellm_warnings import suppress_litellm_serialization_warnings

__all__ = [
    "is_headless_environment",
    "get_default_root",
    "get_logs_dir",
    "get_cache_dir",
    "get_oauth_dir",
    "get_data_file",
    "get_reauth_coordinator",
    "ReauthCoordinator",
    "BufferedWriteRegistry",
    "ResilientStateWriter",
    "safe_write_json",
    "safe_log_write",
    "safe_mkdir",
    "suppress_litellm_serialization_warnings",
]
