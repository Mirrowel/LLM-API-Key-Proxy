# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Startup orchestration helpers for the proxy shell.

The OAuth bootstrap lives here so main.py stays a thin route surface.
The three-pass flow (pre-scan dedup by stored email, parallel
initialization, sequential post-init dedup) is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from rotator_library.providers import PROVIDER_PLUGINS


async def bootstrap_oauth_credentials(
    oauth_credentials: dict[str, list[str]],
    *,
    skip: bool,
) -> dict[str, list[str]]:
    """Validate and deduplicate discovered OAuth credentials."""

    if skip or not oauth_credentials:
        return oauth_credentials

    logging.info("Starting OAuth credential validation and deduplication...")
    processed_emails: dict[str, dict[str, str]] = {}  # email -> {provider: path}
    credentials_to_initialize: dict[str, list[str]] = {}

    # --- Pass 1: Pre-initialization Scan & Deduplication ---
    for provider, paths in oauth_credentials.items():
        credentials_to_initialize.setdefault(provider, [])
        for path in paths:
            # Env-based credentials (virtual paths) have no metadata files.
            if path.startswith("env://"):
                credentials_to_initialize[provider].append(path)
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                metadata = data.get("_proxy_metadata", {})
                email = metadata.get("email")
                if email:
                    processed_emails.setdefault(email, {})
                    if provider in processed_emails[email]:
                        original_path = processed_emails[email][provider]
                        logging.warning(
                            f"Duplicate for '{email}' on '{provider}' found in pre-scan: "
                            f"'{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                        )
                        continue
                    processed_emails[email][provider] = path
                credentials_to_initialize[provider].append(path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Could not pre-read metadata from '{path}': {e}. Will process during initialization."
                )
                credentials_to_initialize[provider].append(path)

    # --- Pass 2: Parallel Initialization of Filtered Credentials ---
    async def process_credential(provider: str, path: str, provider_instance):
        try:
            await provider_instance.initialize_token(path)
            if not hasattr(provider_instance, "get_user_info"):
                return (provider, path, None, None)
            user_info = await provider_instance.get_user_info(path)
            return (provider, path, user_info.get("email"), None)
        except Exception as e:
            logging.error(
                f"Failed to process OAuth token for {provider} at '{path}': {e}"
            )
            return (provider, path, None, e)

    tasks = []
    for provider, paths in credentials_to_initialize.items():
        if not paths:
            continue
        provider_plugin_class = PROVIDER_PLUGINS.get(provider)
        if not provider_plugin_class:
            continue
        provider_instance = provider_plugin_class()
        for path in paths:
            tasks.append(process_credential(provider, path, provider_instance))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # --- Pass 3: Sequential Deduplication and Final Assembly ---
    final_oauth_credentials: dict[str, list[str]] = {}
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Credential processing raised exception: {result}")
            continue

        provider, path, email, error = result
        if error:
            continue

        if email is None:
            final_oauth_credentials.setdefault(provider, []).append(path)
            continue

        if not email:
            logging.warning(
                f"Could not retrieve email for '{path}'. Treating as unique."
            )
            final_oauth_credentials.setdefault(provider, []).append(path)
            continue

        processed_emails.setdefault(email, {})
        if (
            provider in processed_emails[email]
            and processed_emails[email][provider] != path
        ):
            original_path = processed_emails[email][provider]
            logging.warning(
                f"Duplicate for '{email}' on '{provider}' found post-init: "
                f"'{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
            )
            continue

        processed_emails[email][provider] = path
        final_oauth_credentials.setdefault(provider, []).append(path)

        # Update metadata (env-based credentials have no files).
        if not path.startswith("env://"):
            try:
                with open(path, "r+") as f:
                    data = json.load(f)
                    metadata = data.get("_proxy_metadata", {})
                    metadata["email"] = email
                    metadata["last_check_timestamp"] = time.time()
                    data["_proxy_metadata"] = metadata
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
            except Exception as e:
                logging.error(f"Failed to update metadata for '{path}': {e}")

    logging.info("OAuth credential processing complete.")
    return final_oauth_credentials
