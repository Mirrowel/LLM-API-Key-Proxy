"""Admin API for proxy configuration and credential management."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from rotator_library.utils.paths import get_data_file

_credential_lock = asyncio.Lock()

router = APIRouter(prefix="/v1/admin", tags=["admin-config"])


def _read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)

logger = logging.getLogger(__name__)

# Matches a .env KEY (exported or not), capturing the key name.
# Handles: KEY=..., export KEY=...,  KEY =...
_ENV_KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _env_path() -> Path:
    return get_data_file(".env")


def _inplace_set_key(dotenv_path: str, key: str, value: str) -> None:
    """Write-in-place replacement for dotenv.set_key.

    python-dotenv's set_key uses os.replace() under the hood, which fails
    with EBUSY when the .env file is a Docker bind-mount. This helper
    reads, modifies, and writes back in-place (truncate mode) instead.
    """
    path = Path(dotenv_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines(keepends=True)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    new_line = f'{key}="{escaped}"\n'

    found = False
    for i, line in enumerate(lines):
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) == key:
            lines[i] = new_line
            found = True
            break

    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _inplace_unset_key(dotenv_path: str, key: str) -> None:
    """Write-in-place replacement for dotenv.unset_key.

    Same motivation as _inplace_set_key — avoids os.replace().
    """
    path = Path(dotenv_path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for line in lines:
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) == key:
            continue
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _oauth_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    d = base / "oauth_creds"
    d.mkdir(exist_ok=True)
    return d


def _get_env_vars() -> dict[str, str]:
    """Read all env vars from the .env file."""
    from dotenv import dotenv_values
    vals = dotenv_values(_env_path())
    return {k: v for k, v in vals.items() if v is not None}


def _mask_key(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


@router.get("/config")
async def get_config():
    env_vars = _get_env_vars()
    oauth_dir = _oauth_dir()

    try:
        from proxy_app.provider_urls import PROVIDER_URL_MAP
    except ImportError:
        PROVIDER_URL_MAP = {}

    providers: dict = {}
    custom_providers: dict = {}
    concurrency: dict = {}
    rotation_modes: dict = {}
    model_filters: dict = {}
    latest_aliases: dict = {}
    strip_suffixes: list = []

    for key, value in env_vars.items():
        if key == "PROXY_API_KEY":
            continue

        api_key_match = re.match(r"^(.+?)_API_KEY(?:_\d+)?$", key)
        if api_key_match and not key.startswith("PROXY_"):
            provider_name = api_key_match.group(1).lower()
            if provider_name not in providers:
                providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": False}
            providers[provider_name]["api_key_count"] += 1

        elif key.endswith("_API_BASE"):
            provider_name = key.replace("_API_BASE", "").lower()
            if provider_name not in PROVIDER_URL_MAP:
                custom_providers[provider_name] = value
            if provider_name not in providers:
                providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": True}
            else:
                providers[provider_name]["has_custom_base"] = True

        elif key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
            provider_name = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
            if provider_name not in concurrency:
                concurrency[provider_name] = {"max": -1, "optimal": -1}
            try:
                concurrency[provider_name]["max"] = int(value)
            except ValueError:
                pass

        elif key.startswith("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_"):
            provider_name = key.replace("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
            if provider_name not in concurrency:
                concurrency[provider_name] = {"max": -1, "optimal": -1}
            try:
                concurrency[provider_name]["optimal"] = int(value)
            except ValueError:
                pass

        elif key.startswith("ROTATION_MODE_"):
            provider_name = key.replace("ROTATION_MODE_", "").lower()
            rotation_modes[provider_name] = value

        elif key.startswith("IGNORE_MODELS_"):
            provider_name = key.replace("IGNORE_MODELS_", "").lower()
            if provider_name not in model_filters:
                model_filters[provider_name] = {"ignore": [], "whitelist": []}
            model_filters[provider_name]["ignore"] = [p.strip() for p in value.split(",") if p.strip()]

        elif key.startswith("WHITELIST_MODELS_"):
            provider_name = key.replace("WHITELIST_MODELS_", "").lower()
            if provider_name not in model_filters:
                model_filters[provider_name] = {"ignore": [], "whitelist": []}
            model_filters[provider_name]["whitelist"] = [p.strip() for p in value.split(",") if p.strip()]

        elif key.startswith("MODEL_LATEST_") and key != "MODEL_LATEST_STRIP_SUFFIXES":
            alias_name = key.replace("MODEL_LATEST_", "").lower()
            latest_aliases[alias_name] = value

        elif key == "MODEL_LATEST_STRIP_SUFFIXES":
            strip_suffixes = [s.strip() for s in value.split(",") if s.strip()]

    # Collect PROXY_URL_* settings
    proxy_urls: dict = {}
    for key, value in env_vars.items():
        if key == "PROXY_URL_DEFAULT":
            proxy_urls["default"] = value
        elif key.startswith("PROXY_URL_CREDENTIAL_"):
            slug = key[len("PROXY_URL_CREDENTIAL_"):].lower()
            proxy_urls.setdefault("credentials", {})[slug] = value
        elif key.startswith("PROXY_URL_") and not key.startswith("PROXY_URL_CREDENTIAL_"):
            provider = key[len("PROXY_URL_"):].lower()
            proxy_urls.setdefault("providers", {})[provider] = value

    # Count OAuth credentials from files
    if oauth_dir.exists():
        for f in oauth_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and "_oauth_" in f.name:
                provider_name = f.name.split("_oauth_")[0].lower()
                if provider_name not in providers:
                    providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": False}
                providers[provider_name]["oauth_count"] += 1

    result: dict = {
        "proxy_api_key_set": bool(env_vars.get("PROXY_API_KEY")),
        "providers": providers,
        "custom_providers": custom_providers,
        "concurrency": concurrency,
        "rotation_modes": rotation_modes,
        "model_filters": model_filters,
        "latest_aliases": latest_aliases,
        "strip_suffixes": strip_suffixes,
    }
    if proxy_urls:
        result["proxy_urls"] = proxy_urls
    return result


class ConfigUpdate(BaseModel):
    changes: dict[str, Optional[str]]


_CONFIG_BLOCKED_KEYS = {"PROXY_API_KEY", "PATH", "HOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"}
_CONFIG_ALLOWED_PREFIXES = (
    "ROTATION_MODE_", "MAX_CONCURRENT_REQUESTS_PER_KEY_", "OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_",
    "IGNORE_MODELS_", "WHITELIST_MODELS_", "MODEL_LATEST_",
)


@router.patch("/config")
async def update_config(update: ConfigUpdate):
    env_file = str(_env_path())
    updated = []
    rejected = []
    for key, value in update.changes.items():
        if key in _CONFIG_BLOCKED_KEYS:
            rejected.append(key)
            continue
        if not any(key.startswith(p) for p in _CONFIG_ALLOWED_PREFIXES) and not key.endswith(("_API_BASE",)):
            rejected.append(key)
            continue
        if value is None:
            _inplace_unset_key(env_file, key)
            os.environ.pop(key, None)
        else:
            _inplace_set_key(env_file, key, value)
            os.environ[key] = value
        updated.append(key)

    load_dotenv(env_file, override=True)
    result: dict = {"updated": updated}
    if rejected:
        result["rejected"] = rejected
    return result


@router.get("/credentials")
async def get_credentials(request: Request):
    env_vars = _get_env_vars()
    oauth_dir = _oauth_dir()

    # Build a lookup of runtime credential status from the proxy's quota stats
    runtime_status: dict[str, str] = {}
    loaded_providers: set[str] = set()
    try:
        client = request.app.state.rotating_client
        loaded_providers = {p.lower() for p in client.all_credentials}
        quota_stats = await client.get_quota_stats()
        for pstats in quota_stats.get("providers", {}).values():
            for cred_data in pstats.get("credentials", {}).values():
                full_path = cred_data.get("full_path", "")
                if full_path:
                    runtime_status[Path(full_path).name] = cred_data.get("status", "unknown")
    except Exception:
        pass

    # Cross-reference ErrorTracker for credentials with token refresh errors
    errored_creds: set[str] = set()
    try:
        from rotator_library.error_tracker import get_error_tracker
        tracker = get_error_tracker()
        records, _ = tracker.get_recent_errors(limit=50)
        for rec in records:
            if rec.error_type in ("CredentialNeedsReauth", "TokenRefreshFailed"):
                cred_id = rec.credential_masked
                errored_creds.add(cred_id)
    except Exception:
        pass

    api_keys: dict[str, list] = {}
    for key, value in env_vars.items():
        api_key_match = re.match(r"^(.+?)_API_KEY(?:_\d+)?$", key)
        if api_key_match and not key.startswith("PROXY_"):
            provider_name = api_key_match.group(1).lower()
            if provider_name not in api_keys:
                api_keys[provider_name] = []
            api_keys[provider_name].append({
                "key_name": key,
                "masked_value": _mask_key(value),
                "provider": provider_name,
            })

    oauth: dict[str, list] = {}
    if oauth_dir.exists():
        for f in sorted(oauth_dir.iterdir()):
            if f.is_file() and f.suffix == ".json" and "_oauth_" in f.name:
                provider_name = f.name.split("_oauth_")[0].lower()
                if provider_name not in oauth:
                    oauth[provider_name] = []
                # Extract number from filename (e.g. codex_oauth_2.json -> 2)
                num_match = re.search(r"_oauth_(\d+)\.json$", f.name)
                cred_number = int(num_match.group(1)) if num_match else None
                info: dict = {
                    "filename": f.name,
                    "provider": provider_name,
                    "number": cred_number,
                }
                try:
                    data = await asyncio.to_thread(_read_json, f)
                    meta = data.get("_proxy_metadata", {})
                    info["email"] = meta.get("email") or meta.get("login") or data.get("email")
                    info["tier"] = meta.get("tier") or meta.get("plan_type") or meta.get("sku")
                    file_status = meta.get("status", "unknown")
                    # Runtime status takes precedence, then file metadata,
                    # then infer "active" if the provider is loaded in the proxy
                    resolved = runtime_status.get(f.name)
                    if not resolved:
                        if file_status and file_status != "unknown":
                            resolved = file_status
                        elif provider_name in loaded_providers:
                            resolved = "active"
                        else:
                            resolved = "unknown"
                    # Override to needs_reauth if ErrorTracker has recent refresh errors
                    if resolved == "active" and f.name in errored_creds:
                        resolved = "needs_reauth"
                    info["status"] = resolved
                except Exception:
                    info["status"] = runtime_status.get(f.name, "error")
                oauth[provider_name].append(info)

    return {"api_keys": api_keys, "oauth": oauth}


class AddApiKeyRequest(BaseModel):
    provider: str = Field(pattern=r"^[a-zA-Z0-9_]+$", min_length=1, max_length=50)
    key: str = Field(min_length=1, max_length=500)


@router.post("/credentials/api-key")
async def add_api_key(req: AddApiKeyRequest):
    env_file = str(_env_path())
    env_vars = _get_env_vars()

    provider_upper = req.provider.upper()
    existing = [k for k in env_vars if k.startswith(f"{provider_upper}_API_KEY")]
    if existing:
        nums = []
        for k in existing:
            suffix = k.replace(f"{provider_upper}_API_KEY", "")
            if suffix.startswith("_") and suffix[1:].isdigit():
                nums.append(int(suffix[1:]))
            elif not suffix:
                nums.append(0)
        next_num = max(nums) + 1 if nums else 1
        key_name = f"{provider_upper}_API_KEY_{next_num}"
    else:
        key_name = f"{provider_upper}_API_KEY"

    _inplace_set_key(env_file, key_name, req.key)
    os.environ[key_name] = req.key
    load_dotenv(env_file, override=True)

    return {"key_name": key_name}


@router.delete("/credentials/api-key/{provider}/{key_name}")
async def delete_api_key(provider: str, key_name: str, request: Request):
    async with _credential_lock:
        env_file = str(_env_path())
        env_vars = _get_env_vars()
        if key_name not in env_vars:
            raise HTTPException(status_code=404, detail=f"Key {key_name} not found")

        key_value = env_vars[key_name]
        _inplace_unset_key(env_file, key_name)
        os.environ.pop(key_name, None)
        load_dotenv(env_file, override=True)

        try:
            client = request.app.state.rotating_client
            provider_lower = provider.lower()
            if provider_lower in client.all_credentials:
                client.all_credentials[provider_lower] = [
                    c for c in client.all_credentials[provider_lower]
                    if c != key_value
                ]
            if provider_lower in client.api_keys:
                client.api_keys[provider_lower] = [
                    c for c in client.api_keys[provider_lower]
                    if c != key_value
                ]
        except Exception as e:
            logger.warning(f"Could not remove API key from running proxy: {e}")

    return {"deleted": key_name}


@router.delete("/credentials/oauth/{provider}/{filename}")
async def delete_oauth_credential(provider: str, filename: str, request: Request):
    async with _credential_lock:
        oauth_dir = _oauth_dir()
        target = oauth_dir / filename
        if not target.exists():
            raise HTTPException(status_code=404, detail="OAuth credential not found")
        if not target.resolve().is_relative_to(oauth_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        removed_accessor = str(target.resolve())
        target.unlink()

        removed_from_proxy = False
        try:
            client = request.app.state.rotating_client
            provider_lower = provider.lower()
            if provider_lower in client.all_credentials:
                before = len(client.all_credentials[provider_lower])
                client.all_credentials[provider_lower] = [
                    c for c in client.all_credentials[provider_lower]
                    if not c.endswith(filename)
                ]
                removed_from_proxy = len(client.all_credentials[provider_lower]) < before
            if provider_lower in client.oauth_credentials:
                client.oauth_credentials[provider_lower] = [
                    c for c in client.oauth_credentials[provider_lower]
                    if not c.endswith(filename)
                ]

            # Remove from usage manager so stale state isn't persisted on shutdown
            usage_manager = client.get_usage_manager(provider_lower)
            if usage_manager:
                await usage_manager.remove_credential(removed_accessor)
        except Exception as e:
            logger.warning(f"Could not remove credential from running proxy: {e}")

    return {"deleted": filename, "removed_from_proxy": removed_from_proxy}


class AddCustomProviderRequest(BaseModel):
    name: str
    base_url: str
    api_key: str


@router.post("/credentials/custom-provider")
async def add_custom_provider(req: AddCustomProviderRequest):
    env_file = str(_env_path())
    provider_upper = req.name.upper()

    _inplace_set_key(env_file, f"{provider_upper}_API_BASE", req.base_url)
    _inplace_set_key(env_file, f"{provider_upper}_API_KEY", req.api_key)
    os.environ[f"{provider_upper}_API_BASE"] = req.base_url
    os.environ[f"{provider_upper}_API_KEY"] = req.api_key
    load_dotenv(env_file, override=True)

    return {"provider": req.name}


@router.get("/config/model-filters/{provider}")
async def get_model_filters(provider: str):
    env_vars = _get_env_vars()
    provider_upper = provider.upper()

    ignore_key = f"IGNORE_MODELS_{provider_upper}"
    whitelist_key = f"WHITELIST_MODELS_{provider_upper}"

    ignore = [p.strip() for p in env_vars.get(ignore_key, "").split(",") if p.strip()]
    whitelist = [p.strip() for p in env_vars.get(whitelist_key, "").split(",") if p.strip()]

    return {"ignore": ignore, "whitelist": whitelist}


class ModelFilterUpdate(BaseModel):
    ignore: list[str]
    whitelist: list[str]


@router.put("/config/model-filters/{provider}")
async def update_model_filters(provider: str, filters: ModelFilterUpdate):
    env_file = str(_env_path())
    provider_upper = provider.upper()

    ignore_key = f"IGNORE_MODELS_{provider_upper}"
    whitelist_key = f"WHITELIST_MODELS_{provider_upper}"

    if filters.ignore:
        _inplace_set_key(env_file, ignore_key, ",".join(filters.ignore))
    else:
        _inplace_unset_key(env_file, ignore_key)

    if filters.whitelist:
        _inplace_set_key(env_file, whitelist_key, ",".join(filters.whitelist))
    else:
        _inplace_unset_key(env_file, whitelist_key)

    load_dotenv(env_file, override=True)
    return {"provider": provider, "updated": True}


@router.post("/reload")
async def reload_proxy():
    try:
        env_file = _env_path()
        load_dotenv(str(env_file), override=True)
        logger.info("Proxy configuration reloaded via admin API")
        return {"status": "ok", "message": "Configuration reloaded from .env"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
