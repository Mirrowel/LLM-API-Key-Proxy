# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/paths.py
"""
Centralized path management for the rotator library.

Supports two runtime modes:
1. PyInstaller EXE -> files in the directory containing the executable
2. Script/Library  -> files in the current working directory (overridable)

Library users can override by passing `data_dir` to RotatingClient.
"""

import sys
from pathlib import Path
from typing import Optional, Union


def get_default_root() -> Path:
    """
    Get the default root directory for data files.

    - EXE mode (PyInstaller): directory containing the executable
    - Otherwise: current working directory

    Returns:
        Path to the root directory
    """
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle - use executable's directory
        return Path(sys.executable).parent
    # Running as script or library - use current working directory
    return Path.cwd()


def get_logs_dir(root: Optional[Union[Path, str]] = None) -> Path:
    """
    Get the logs directory, creating it if needed.

    Args:
        root: Optional root directory. If None, uses get_default_root().

    Returns:
        Path to the logs directory
    """
    base = Path(root) if root else get_default_root()
    logs_dir = base / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_cache_dir(
    root: Optional[Union[Path, str]] = None, subdir: Optional[str] = None
) -> Path:
    """
    Get the cache directory, optionally with a subdirectory.

    Args:
        root: Optional root directory. If None, uses get_default_root().
        subdir: Optional subdirectory name (e.g., "gemini_cli", "antigravity")

    Returns:
        Path to the cache directory (or subdirectory)
    """
    base = Path(root) if root else get_default_root()
    cache_dir = base / "cache"
    if subdir:
        cache_dir = cache_dir / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_oauth_dir(root: Optional[Union[Path, str]] = None) -> Path:
    """
    Get the OAuth credentials directory, creating it if needed.

    Args:
        root: Optional root directory. If None, uses get_default_root().

    Returns:
        Path to the oauth_creds directory
    """
    base = Path(root) if root else get_default_root()
    oauth_dir = base / "oauth_creds"
    oauth_dir.mkdir(exist_ok=True)
    return oauth_dir


def get_data_file(filename: str, root: Optional[Union[Path, str]] = None) -> Path:
    """
    Get the path to a data file in the root directory.

    Args:
        filename: Name of the file (e.g., "key_usage.json", ".env")
        root: Optional root directory. If None, uses get_default_root().

    Returns:
        Path to the file (does not create the file)
    """
    base = Path(root) if root else get_default_root()
    return base / filename
