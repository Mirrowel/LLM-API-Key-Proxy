# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
GitHub Copilot plan/model mapping.

Scrapes the GitHub Copilot plans documentation to build a mapping of
which models are available under which plan tiers. This is used at
proxy startup to filter the model list based on each credential's SKU.

The mapping is cached to disk for 24 hours to avoid hitting GitHub docs
on every startup.

Source: https://docs.github.com/en/copilot/get-started/plans
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from ..utils.paths import get_oauth_dir
from typing import Dict, List, Optional, Set

import httpx

lib_logger = logging.getLogger("rotator_library")

# Cache file location (next to credential files)
_CACHE_DIR = get_oauth_dir()
_CACHE_FILE = _CACHE_DIR / ".copilot_plan_cache.json"
_CACHE_TTL = 24 * 60 * 60  # 24 hours

# GitHub Copilot plans page
_PLANS_URL = "https://docs.github.com/en/copilot/get-started/plans"

# Plan columns in the docs table (left to right)
PLAN_COLUMNS = ["free", "student", "pro", "pro_plus", "business", "enterprise"]

# SKU from /copilot_internal/v2/token → plan tier mapping
# The token response has a "sku" field; map it to our plan column names.
SKU_TO_PLAN = {
    "free_educational_quota": "student",  # GitHub Education accounts
    "free": "free",
    "monthly": "pro",  # Standard Copilot Pro
    "pro": "pro",
    "pro_plus": "pro_plus",
    "business": "business",
    "enterprise": "enterprise",
}


def _scrape_plan_table(html: str) -> Dict[str, Set[str]]:
    """
    Parse the GitHub docs HTML to extract model→plans mapping.

    Returns dict like: {"gpt-5-mini": {"free", "student", "pro", ...}}
    """
    # Find the "Available models in chat" section
    match = re.search(
        r"Available models in chat(.*?)(?=<h[23]|Inline suggestions)",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        lib_logger.warning("Could not find 'Available models in chat' section in docs")
        return {}

    section = match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", section, re.DOTALL)

    result = {}
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        if not cells:
            continue

        # First cell is the model name
        model_name = re.sub(r"<[^>]+>", "", cells[0]).strip()

        # Skip header rows
        if model_name in PLAN_COLUMNS or model_name == "":
            continue

        # Remaining cells map to plan columns
        accessible_plans = set()
        for i, cell in enumerate(cells[1:], 0):
            if i >= len(PLAN_COLUMNS):
                break
            # Check for checkmark vs X
            has_check = bool(
                re.search(r"octicon-check|Color-fg-success|✓", cell)
            )
            no_check = bool(
                re.search(r"octicon-x|Color-fg-danger|✗", cell)
            )
            if has_check and not no_check:
                accessible_plans.add(PLAN_COLUMNS[i])

        if accessible_plans:
            # Normalize model name to match Copilot API model IDs
            model_id = _normalize_model_name(model_name)
            result[model_id] = accessible_plans

    return result


def _normalize_model_name(name: str) -> str:
    """
    Normalize a model name from the docs table to match Copilot API model IDs.

    Docs use: "Claude Haiku 4.5", "GPT-5 mini", "GPT-5.2-Codex"
    API uses: "claude-haiku-4.5", "gpt-5-mini", "gpt-5.2-codex"
    """
    # Lowercase
    result = name.lower()
    # Remove "(fast mode)" and "(preview)" annotations
    result = re.sub(r"\s*\(.*?\)\s*", "", result)
    # Replace spaces with hyphens
    result = result.replace(" ", "-")
    # Collapse multiple hyphens
    result = re.sub(r"-+", "-", result)
    # Strip
    result = result.strip("-")
    return result


def _load_cache(allow_stale: bool = False) -> Optional[Dict[str, List[str]]]:
    """Load cached plan mapping from disk if still valid.

    Args:
        allow_stale: If True, return cache even if expired (used as fallback
                     when live fetch fails).
    """
    if not _CACHE_FILE.exists():
        return None

    try:
        with open(_CACHE_FILE, "r") as f:
            cache = json.load(f)

        cached_at = cache.get("cached_at", 0)
        age = time.time() - cached_at

        if age > _CACHE_TTL and not allow_stale:
            lib_logger.debug("Copilot plan cache expired")
            return None

        if age > _CACHE_TTL:
            lib_logger.info(
                f"Using stale copilot plan cache as fallback "
                f"({len(cache.get('models', {}))} models, age: {int(age)}s)"
            )

        # Convert lists back to sets
        mapping = {}
        for model_id, plans in cache.get("models", {}).items():
            mapping[model_id] = set(plans)

        if age <= _CACHE_TTL:
            lib_logger.info(
                f"Loaded copilot plan mapping from cache "
                f"({len(mapping)} models, age: {int(age)}s)"
            )
        return mapping

    except Exception as e:
        lib_logger.debug(f"Failed to load plan cache: {e}")
        return None


def _save_cache(mapping: Dict[str, Set[str]]) -> None:
    """Save plan mapping to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {
            "cached_at": time.time(),
            "models": {
                model_id: sorted(plans)
                for model_id, plans in mapping.items()
            },
        }
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        lib_logger.debug(f"Saved copilot plan mapping to cache ({len(mapping)} models)")
    except Exception as e:
        lib_logger.warning(f"Failed to save plan cache: {e}")


# Lazy-initialized lock to prevent concurrent fetches on startup.
# Created inside an event loop to avoid Python 3.10+ deprecation warnings
# about locks created outside a running loop.
_fetch_lock: Optional[asyncio.Lock] = None


def _get_fetch_lock() -> asyncio.Lock:
    """Return the module-level fetch lock, creating it lazily."""
    global _fetch_lock
    if _fetch_lock is None:
        _fetch_lock = asyncio.Lock()
    return _fetch_lock


async def fetch_plan_mapping() -> Dict[str, Set[str]]:
    """
    Fetch the model→plan mapping, using cache if available.

    Guarded by an asyncio lock so that concurrent callers (e.g. simultaneous
    get_models() requests at startup) don't fire N parallel HTTP requests.

    Returns dict like: {"gpt-5-mini": {"free", "student", "pro", "pro_plus", "business", "enterprise"}}
    """
    # Fast path — no lock needed if cache is already valid
    cached = _load_cache()
    if cached is not None:
        return cached

    # Slow path — acquire lock, then re-check cache (another caller may have
    # fetched while we waited)
    async with _get_fetch_lock():
        cached = _load_cache()
        if cached is not None:
            return cached

        # Fetch from GitHub docs
        lib_logger.info("Fetching Copilot plan/model mapping from GitHub docs...")

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(_PLANS_URL)
                response.raise_for_status()

            mapping = _scrape_plan_table(response.text)

            if not mapping:
                lib_logger.warning(
                    "Failed to extract model/plan mapping from docs. "
                    "Model filtering by SKU will be unavailable."
                )
                return {}

            lib_logger.info(
                f"Fetched copilot plan mapping: {len(mapping)} models across "
                f"{len(PLAN_COLUMNS)} plan tiers"
            )

            # Cache for next startup
            _save_cache(mapping)

            return mapping

        except Exception as e:
            lib_logger.warning(
                f"Failed to fetch Copilot plan mapping: {e}. "
                "Model filtering by SKU will be unavailable."
            )
            # Try loading stale cache as fallback
            stale = _load_cache(allow_stale=True)
            if stale is not None:
                return stale
            return {}


def get_plan_for_sku(sku: str) -> Optional[str]:
    """
    Map a Copilot SKU (from /copilot_internal/v2/token) to a plan tier name.

    Args:
        sku: The SKU string like "free_educational_quota", "monthly", etc.

    Returns:
        Plan tier name like "student", "pro", etc. or None if unknown.
    """
    return SKU_TO_PLAN.get(sku)


def filter_models_for_plan(
    models: List[str],
    plan_mapping: Dict[str, Set[str]],
    plan: Optional[str],
) -> List[str]:
    """
    Filter a model list to only include models accessible under the given plan.

    If plan is None (unknown SKU), all models are returned (permissive fallback).
    If plan_mapping is empty (scrape failed), all models are returned.

    For multiple credentials with different plans, the union of all accessible
    models should be computed by the caller.

    Args:
        models: List of model IDs (e.g., "gpt-5-mini", "claude-sonnet-4")
        plan_mapping: Model→plans mapping from fetch_plan_mapping()
        plan: Plan tier name (e.g., "student", "pro")

    Returns:
        Filtered list of model IDs
    """
    if not plan_mapping or plan is None:
        return models

    filtered = []
    for model_id in models:
        accessible_plans = plan_mapping.get(model_id)
        if accessible_plans is None:
            # Model not in mapping — might be new, include it optimistically
            filtered.append(model_id)
        elif plan in accessible_plans:
            filtered.append(model_id)
        # else: model not available under this plan, exclude it

    excluded = set(models) - set(filtered)
    if excluded:
        lib_logger.info(
            f"Filtered out {len(excluded)} models not available under "
            f"plan '{plan}': {sorted(excluded)}"
        )

    return filtered
