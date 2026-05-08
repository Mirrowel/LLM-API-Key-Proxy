# SPDX-License-Identifier: LGPL-3.0-only

"""
HTTP/SOCKS5 forward proxy configuration for outbound LLM API traffic.

Allows routing upstream requests through proxy servers with per-provider,
per-credential, rotational, or global default configurations.

Configuration is loaded from environment variables and/or a JSON config file.

Environment variable patterns:
    PROXY_URL_DEFAULT               - Global fallback proxy URL
    PROXY_URL_<PROVIDER>            - Per-provider proxy (e.g. PROXY_URL_ANTHROPIC)
    PROXY_URL_CREDENTIAL_<ID>       - Per-credential proxy (keyed by stable_id slug)
    PROXY_ROTATION_POOL             - Comma-separated proxy URLs for rotation
    PROXY_ROTATION_STRATEGY         - "round_robin" (default) or "random"
    PROXY_ROTATION_SCOPE            - "global" (default), "provider", or "credential"
    PROXY_CONFIG_PATH               - Path to proxy_config.json file

Resolution priority (highest to lowest):
    1. Per-credential  (PROXY_URL_CREDENTIAL_*)
    2. Per-provider    (PROXY_URL_*)
    3. Rotation pool   (PROXY_ROTATION_POOL)
    4. Global default  (PROXY_URL_DEFAULT)
    5. Direct connection (no proxy)

Supported proxy schemes:
    http, https     - Standard HTTP proxies (CONNECT tunnelling for TLS)
    socks5          - SOCKS5 with *local* DNS resolution (client resolves
                      the upstream hostname before connecting to the proxy)
    socks5h         - SOCKS5 with *remote* DNS resolution (the proxy server
                      resolves hostnames).  **Prefer socks5h** in almost all
                      cases — it avoids DNS leaks and works correctly when
                      the proxy is on a different network or inside a
                      container that cannot resolve upstream API hostnames.
    socks4          - SOCKS4 (rarely needed)

    SOCKS5 proxy support requires the 'socksio' package (pip install socksio).

Identifying stable IDs for per-credential proxy configuration:
    Stable IDs uniquely identify each credential across restarts. They are:
    - OAuth credentials: the email address (e.g. "user@gmail.com") or login
      (e.g. "github-username"), visible in the quota-stats API and TUI
    - API keys: a truncated SHA-256 hash (first 12 hex chars), also visible
      in the quota-stats API response under the "stable_id" field

    To find your credential stable IDs:
    1. API endpoint: GET /v1/quota-stats - each credential entry includes
       "stable_id" and "accessor_masked" fields
    2. Usage JSON files: check usage/usage_<provider>.json under
       "credentials" -> look for "stable_id" values
    3. Proxy logs: credential acquisition logs show masked credential IDs

    For env var keys, convert the stable_id to uppercase and replace
    non-alphanumeric characters with underscores:
        user@gmail.com  -> PROXY_URL_CREDENTIAL_USER_GMAIL_COM
        abc123def456    -> PROXY_URL_CREDENTIAL_ABC123DEF456
        myuser::org-123 -> PROXY_URL_CREDENTIAL_MYUSER__ORG_123
"""

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

lib_logger = logging.getLogger("rotator_library")


@dataclass(frozen=True)
class ProxySpec:
    """A single proxy endpoint with optional overrides."""

    url: str

    def __post_init__(self):
        if not self.url:
            raise ValueError("ProxySpec url must not be empty")
        scheme = self.url.split("://")[0].lower() if "://" in self.url else ""
        valid = {"http", "https", "socks5", "socks5h", "socks4"}
        if scheme not in valid:
            raise ValueError(
                f"Unsupported proxy scheme '{scheme}' in '{self.url}'. "
                f"Must be one of: {', '.join(sorted(valid))}"
            )
        if scheme == "socks5":
            lib_logger.warning(
                f"Proxy '{self.url}' uses socks5:// (local DNS).  Consider "
                f"socks5h:// instead for remote DNS resolution — this avoids "
                f"failures when the client cannot resolve upstream hostnames."
            )


def _slugify_stable_id(stable_id: str) -> str:
    """Convert a stable_id to an env-var-safe slug (uppercase, _ for specials)."""
    return re.sub(r"[^A-Z0-9]", "_", stable_id.upper())


@dataclass
class ProxyConfig:
    """
    Complete proxy routing configuration.

    Loaded once at startup and shared across the application.
    """

    default: Optional[ProxySpec] = None

    provider_proxies: Dict[str, ProxySpec] = field(default_factory=dict)

    credential_proxies: Dict[str, ProxySpec] = field(default_factory=dict)

    rotation_pool: List[ProxySpec] = field(default_factory=list)
    rotation_strategy: str = "round_robin"
    rotation_scope: str = "global"

    # Internal counter for round-robin (keyed by scope discriminator)
    _rr_counters: Dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def has_any_proxy(self) -> bool:
        return bool(
            self.default
            or self.provider_proxies
            or self.credential_proxies
            or self.rotation_pool
        )

    def resolve(
        self,
        provider: str,
        credential: str,
        stable_id: str,
    ) -> Optional[ProxySpec]:
        """
        Resolve the proxy to use for a given request.

        Priority: credential > provider > rotation pool > default.
        """
        # 1. Per-credential (match by stable_id, case-insensitive)
        sid_lower = stable_id.lower()
        for key, spec in self.credential_proxies.items():
            if key.lower() == sid_lower:
                return spec

        # 2. Per-provider
        spec = self.provider_proxies.get(provider)
        if spec:
            return spec

        # 3. Rotation pool
        if self.rotation_pool:
            return self._pick_from_pool(provider, stable_id)

        # 4. Global default
        return self.default

    def resolve_for_provider(self, provider: str) -> Optional[ProxySpec]:
        """Resolve proxy using only provider-level or global config (no credential)."""
        spec = self.provider_proxies.get(provider)
        if spec:
            return spec
        if self.rotation_pool:
            return self._pick_from_pool(provider, "")
        return self.default

    def _pick_from_pool(self, provider: str, stable_id: str) -> ProxySpec:
        if self.rotation_strategy == "random":
            return random.choice(self.rotation_pool)

        # Round-robin keyed by scope
        if self.rotation_scope == "provider":
            key = provider
        elif self.rotation_scope == "credential":
            key = f"{provider}:{stable_id}"
        else:
            key = "_global_"

        idx = self._rr_counters.get(key, 0)
        spec = self.rotation_pool[idx % len(self.rotation_pool)]
        self._rr_counters[key] = idx + 1
        return spec


def load_proxy_config(
    env: Optional[Dict[str, str]] = None,
    config_path: Optional[str] = None,
) -> ProxyConfig:
    """
    Load proxy configuration from environment variables and optional JSON file.

    JSON file values serve as defaults; env vars always win.
    """
    env = env if env is not None else os.environ
    config = ProxyConfig()

    # --- Load JSON config file first (lowest priority) ---
    json_path = config_path or env.get("PROXY_CONFIG_PATH")
    if json_path:
        path = Path(json_path)
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _apply_json_config(config, data)
                lib_logger.info(f"Loaded proxy config from {path}")
            except Exception as exc:
                lib_logger.error(f"Failed to load proxy config from {path}: {exc}")
        else:
            lib_logger.warning(f"PROXY_CONFIG_PATH set but file not found: {path}")

    # --- Environment variables (override JSON) ---

    # Global default
    default_url = env.get("PROXY_URL_DEFAULT")
    if default_url:
        config.default = ProxySpec(url=default_url)

    # Per-provider: PROXY_URL_<PROVIDER>
    _known_suffixes = {"DEFAULT", "CREDENTIAL"}
    for key, value in env.items():
        if not key.startswith("PROXY_URL_"):
            continue
        suffix = key[len("PROXY_URL_"):]
        # Skip non-provider keys
        if not suffix or suffix.startswith("CREDENTIAL_"):
            continue
        if suffix in _known_suffixes:
            continue
        provider = suffix.lower()
        config.provider_proxies[provider] = ProxySpec(url=value)

    # Per-credential: PROXY_URL_CREDENTIAL_<SLUG>
    for key, value in env.items():
        if not key.startswith("PROXY_URL_CREDENTIAL_"):
            continue
        slug = key[len("PROXY_URL_CREDENTIAL_"):]
        if slug:
            config.credential_proxies[slug] = ProxySpec(url=value)

    # Rotation pool
    pool_raw = env.get("PROXY_ROTATION_POOL")
    if pool_raw:
        config.rotation_pool = [
            ProxySpec(url=u.strip())
            for u in pool_raw.split(",")
            if u.strip()
        ]

    strategy = env.get("PROXY_ROTATION_STRATEGY", "").lower()
    if strategy in ("round_robin", "random"):
        config.rotation_strategy = strategy

    scope = env.get("PROXY_ROTATION_SCOPE", "").lower()
    if scope in ("global", "provider", "credential"):
        config.rotation_scope = scope

    return config


def _apply_json_config(config: ProxyConfig, data: Dict[str, Any]) -> None:
    """Apply values from parsed JSON config to a ProxyConfig."""
    if "default" in data and data["default"]:
        config.default = ProxySpec(url=data["default"])

    for provider, url in data.get("providers", {}).items():
        config.provider_proxies[provider.lower()] = ProxySpec(url=url)

    for cred_id, url in data.get("credentials", {}).items():
        config.credential_proxies[cred_id] = ProxySpec(url=url)

    rotation = data.get("rotation", {})
    if "pool" in rotation:
        config.rotation_pool = [ProxySpec(url=u) for u in rotation["pool"] if u]
    if "strategy" in rotation:
        config.rotation_strategy = rotation["strategy"]
    if "scope" in rotation:
        config.rotation_scope = rotation["scope"]


class ProxiedClientPool:
    """
    Manages httpx.AsyncClient instances, one per distinct proxy URL.

    The no-proxy (direct) case is keyed by None. Clients are created
    lazily on first use and all closed together on shutdown.
    """

    def __init__(self, proxy_config: ProxyConfig):
        self.config = proxy_config
        self._clients: Dict[Optional[str], httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()

    async def get_client(
        self,
        provider: str,
        credential: str,
        stable_id: str,
    ) -> httpx.AsyncClient:
        """Get or create an httpx client for the resolved proxy."""
        spec = self.config.resolve(provider, credential, stable_id)
        proxy_url = spec.url if spec else None

        if proxy_url in self._clients:
            client = self._clients[proxy_url]
            if not client.is_closed:
                return client

        async with self._lock:
            # Double-check after acquiring lock
            if proxy_url in self._clients:
                client = self._clients[proxy_url]
                if not client.is_closed:
                    return client

            client = self._create_client(proxy_url)
            self._clients[proxy_url] = client

            if proxy_url:
                lib_logger.info(
                    f"Created proxied httpx client for {proxy_url} "
                    f"(pool size: {len(self._clients)})"
                )

            return client

    async def get_client_for_provider(self, provider: str) -> httpx.AsyncClient:
        """Get a client using only provider-level proxy resolution."""
        spec = self.config.resolve_for_provider(provider)
        proxy_url = spec.url if spec else None

        if proxy_url in self._clients:
            client = self._clients[proxy_url]
            if not client.is_closed:
                return client

        async with self._lock:
            if proxy_url in self._clients:
                client = self._clients[proxy_url]
                if not client.is_closed:
                    return client

            client = self._create_client(proxy_url)
            self._clients[proxy_url] = client
            return client

    @staticmethod
    def _create_client(proxy_url: Optional[str]) -> httpx.AsyncClient:
        kwargs: Dict[str, Any] = {
            "timeout": httpx.Timeout(300.0, connect=10.0),
            "follow_redirects": True,
        }
        if proxy_url:
            kwargs["proxy"] = proxy_url
        return httpx.AsyncClient(**kwargs)

    async def close_all(self) -> None:
        """Close all managed httpx clients."""
        for proxy_url, client in self._clients.items():
            try:
                await client.aclose()
            except Exception as exc:
                lib_logger.debug(f"Error closing client for proxy {proxy_url}: {exc}")
        self._clients.clear()
