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
]
