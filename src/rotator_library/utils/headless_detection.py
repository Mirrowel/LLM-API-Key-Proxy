# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/headless_detection.py

import os
import sys
import logging

lib_logger = logging.getLogger("rotator_library")

# Import console for user-visible output
try:
    from rich.console import Console

    console = Console()
except ImportError:
    console = None


def is_headless_environment() -> bool:
    """
    Detects if the current environment is headless (no GUI available).

    Returns:
        True if headless environment is detected, False otherwise

    Detection logic:
    - Linux/Unix: Check DISPLAY environment variable
    - SSH detection: Check SSH_CONNECTION or SSH_CLIENT
    - CI environments: Check common CI environment variables
    - Windows: Check SESSIONNAME for service/headless indicators
    """
    headless_indicators = []

    # Check DISPLAY for Linux GUI availability (skip on Windows and macOS)
    # NOTE: DISPLAY is an X11 (X Window System) variable used on Linux.
    # macOS uses its native Quartz windowing system, NOT X11, so DISPLAY is
    # typically unset on macOS even with a full GUI. Only check DISPLAY on Linux.
    if os.name != "nt" and sys.platform != "darwin":  # Linux only
        display = os.getenv("DISPLAY")
        if display is None or display.strip() == "":
            headless_indicators.append("No DISPLAY variable (Linux headless)")

    # Check for SSH connection
    if os.getenv("SSH_CONNECTION") or os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        headless_indicators.append("SSH connection detected")

    # Check for CI environments
    ci_vars = [
        "CI",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_URL",  # Jenkins
        "CIRCLECI",  # CircleCI
        "TRAVIS",  # Travis CI
        "BUILDKITE",  # Buildkite
        "DRONE",  # Drone CI
        "TEAMCITY_VERSION",  # TeamCity
        "TF_BUILD",  # Azure Pipelines
        "CODEBUILD_BUILD_ID",  # AWS CodeBuild
    ]
    for var in ci_vars:
        if os.getenv(var):
            headless_indicators.append(f"CI environment detected ({var})")
            break

    # Check Windows session type
    if os.name == "nt":  # Windows
        session_name = os.getenv("SESSIONNAME", "").lower()
        if session_name in ["services", "rdp-tcp"]:
            headless_indicators.append(f"Windows headless session ({session_name})")

    # Detect Docker/container environment
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        headless_indicators.append("Container environment detected")

    # Determine if headless
    is_headless = len(headless_indicators) > 0

    if is_headless:
        # Log to logger
        lib_logger.info(
            f"Headless environment detected: {'; '.join(headless_indicators)}"
        )

        # Print to console for user visibility
        if console:
            console.print(
                f"[yellow]ℹ Headless environment detected:[/yellow] {'; '.join(headless_indicators)}"
            )
            console.print(
                "[yellow]→ Browser will NOT open automatically. Please use the URL below.[/yellow]\n"
            )
    else:
        # Only log to debug, no console output
        lib_logger.debug(
            "GUI environment detected, browser auto-open will be attempted"
        )

    return is_headless
