# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import importlib
import pkgutil
import os
from typing import Any, Dict, Mapping, Optional, Type

from .dynamic import DynamicProvider
from .provider_interface import (
    ProviderInterface,
    auth_header_pair,
    declared_endpoint_path,
    render_endpoint_path,
)

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}


def _create_dynamic_plugin_class(
    name: str,
    config_snapshot: Any = None,
    *,
    no_auth: bool = False,
) -> Type[ProviderInterface]:
    """Create one ProviderInterface implementation bound to a config name.

    ``no_auth`` mints the provider for keyless local serving: the internal
    no-auth credential slot applies and nothing is ever sent as a Bearer
    credential on fallback or discovery.
    """

    class DynamicPlugin(DynamicProvider, ProviderInterface):
        provider_env_name = name
        if no_auth:
            default_auth_mode = "none"

        def __init__(self):
            DynamicProvider.__init__(
                self,
                name,
                config_snapshot=config_snapshot,
            )

    DynamicPlugin.__name__ = f"{''.join(part.title() for part in name.split('_'))}DynamicProvider"
    return DynamicPlugin


def validate_provider_hooks(config_snapshot: Any = None) -> None:
    """Fail startup when any declared hook name/stage cannot resolve.

    Mirrors adapter-name validation: provider class ``hooks``, JSON provider
    ``hooks``, and configured global hook names all resolve through
    ``hooks.registry.validate_declared_names``. Called after every provider
    module is imported so declarations and the global registry are complete.
    """

    from ..config.experimental import (
        _configured_hooks,
        get_global_hook_names,
        load_experimental_config,
    )
    from ..hooks.registry import validate_declared_names

    active = config_snapshot if config_snapshot is not None else load_experimental_config()
    class_hooks: list[Any] = []
    for plugin in PROVIDER_PLUGINS.values():
        class_hooks.extend(getattr(plugin, "hooks", ()) or ())
    config_hooks: list[Any] = []
    providers = active.providers if isinstance(getattr(active, "providers", None), Mapping) else {}
    for raw in providers.values():
        if isinstance(raw, Mapping) and "hooks" in raw:
            config_hooks.extend(_configured_hooks(raw.get("hooks")) or ())
    validate_declared_names(
        class_hooks=class_hooks,
        config_hooks=config_hooks,
        global_hooks=get_global_hook_names(config=active),
    )


def _validate_provider_hooks_at_startup(config_snapshot: Any = None) -> None:
    """Startup wrapper kept tolerant of import-time registry gaps.

    Provider modules register their hook classes as they import; a genuinely
    unknown name must fail. A missing dependency (partial import) must not
    crash the launcher — it will surface again when the request runs.
    """

    try:
        validate_provider_hooks(config_snapshot)
    except (KeyError, ValueError):
        raise
    except Exception as exc:  # pragma: no cover - defensive import guard
        import logging

        logging.getLogger("rotator_library").warning(
            "hook declaration startup validation skipped: %s", exc
        )


def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    Also creates dynamic plugins for custom OpenAI-compatible providers.
    """
    package_path = __path__
    package_name = __name__
    from ..config.experimental import load_experimental_config

    config_snapshot = load_experimental_config()

    # First, register file-based providers. Archive/private modules are skipped
    # so retired providers can remain in-tree without becoming accessible.
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.startswith("_"):
            continue

        # Construct the full module path
        full_module_path = f"{package_name}.{module_name}"

        # Import the module
        module = importlib.import_module(full_module_path)

        # Look for a class that inherits from ProviderInterface
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, ProviderInterface)
                and attribute is not ProviderInterface
            ):
                # Derives 'openrouter' from 'openrouter_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                import logging

                logging.getLogger("rotator_library").debug(
                    f"Registered provider: {provider_name}"
                )

    # Then, create dynamic plugins for custom OpenAI-compatible providers
    # These use the pattern: <NAME>_API_BASE where NAME is not a known LiteLLM provider
    # Known providers just get their api_base overridden via ProviderConfig

    # Import KNOWN_PROVIDERS to check against
    from ..provider_config import KNOWN_PROVIDERS

    configured = config_snapshot.providers

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip if this is a known LiteLLM provider (not a custom provider)
            if provider_name in KNOWN_PROVIDERS:
                continue

            # Skip if this provider name already exists (file-based plugin)
            if provider_name in PROVIDER_PLUGINS:
                continue

            raw_base = str(os.environ[env_var] or "").strip()
            if not raw_base:
                # An empty declaration is a configuration error, not a
                # deferred crash at first request.
                raise ValueError(
                    f"Environment variable {env_var} is set but empty — "
                    "remove it or provide the provider's base URL"
                )

            # Keyless local serving: no credential env of any shape and no
            # explicit auth declaration means the internal no-auth slot.
            has_credential_env = any(
                other.startswith(f"{provider_name.upper()}_API_KEY")
                for other in os.environ
            )
            explicit_auth = False
            raw_config = configured.get(provider_name)
            if isinstance(raw_config, dict):
                explicit_auth = bool(raw_config.get("auth_mode"))

            plugin_class = _create_dynamic_plugin_class(
                provider_name,
                config_snapshot=config_snapshot,
                no_auth=not has_credential_env and not explicit_auth,
            )
            PROVIDER_PLUGINS[provider_name] = plugin_class
            import logging

            logging.getLogger("rotator_library").debug(
                f"Registered dynamic provider: {provider_name}"
            )

    # Structured config can define custom providers without a parallel API_BASE
    # environment variable. Credentials remain in the existing secret stores.
    for raw_name, raw in configured.items():
        provider_name = str(raw_name).lower()
        if provider_name in PROVIDER_PLUGINS:
            transport_keys = {
                "api_base",
                "endpoint_paths",
                "auth_mode",
                "auth_header_name",
                "models",
                "transport_profiles",
                "profiles",
                "default_profile",
            } & set(raw if isinstance(raw, dict) else {})
            if transport_keys:
                raise ValueError(
                    f"Provider {provider_name!r} is implemented in code; custom transport keys are not applied: "
                    f"{', '.join(sorted(transport_keys))}"
                )
            continue
        if not isinstance(raw, dict):
            continue
        configured_protocol = raw.get("protocol_name")
        if configured_protocol:
            from ..protocols import get_protocol
            from ..protocols.registry import is_generative_protocol

            protocol_name = get_protocol(str(configured_protocol).strip().lower()).name
            # Registry-derived allowlist (G11) — matches the config surface;
            # sibling variants (responses_stateful, ...) qualify.
            if not is_generative_protocol(protocol_name):
                raise ValueError(
                    f"Configured custom provider {provider_name!r} requires a supported generative protocol, got {protocol_name!r}"
                )
        api_base = str(raw.get("api_base") or os.getenv(f"{provider_name.upper()}_API_BASE") or "").strip()
        if not api_base:
            raise ValueError(
                f"Configured custom provider {provider_name!r} requires providers.{provider_name}.api_base"
            )
        PROVIDER_PLUGINS[provider_name] = _create_dynamic_plugin_class(
            provider_name,
            config_snapshot=config_snapshot,
        )
        import logging

        logging.getLogger("rotator_library").debug(
            f"Registered config-defined provider: {provider_name}"
        )

    # G2 startup validation: every declared hook name/stage must resolve now,
    # never on a request. Runs after all provider modules (and their hook
    # classes) are registered.
    _validate_provider_hooks_at_startup(config_snapshot)


# Discover and register providers when the package is imported
_register_providers()
