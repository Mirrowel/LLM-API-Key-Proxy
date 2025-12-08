"""
Advanced settings configuration tool for the LLM API Key Proxy.
Provides interactive configuration for custom providers, model definitions, and concurrency limits.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel
from dotenv import set_key, unset_key

console = Console()


def clear_screen():
    """
    Cross-platform terminal clear that works robustly on both
    classic Windows conhost and modern terminals (Windows Terminal, Linux, Mac).

    Uses native OS commands instead of ANSI escape sequences:
    - Windows (conhost & Windows Terminal): cls
    - Unix-like systems (Linux, Mac): clear
    """
    os.system("cls" if os.name == "nt" else "clear")


class AdvancedSettings:
    """Manages pending changes to .env"""

    def __init__(self):
        self.env_file = Path.cwd() / ".env"
        self.pending_changes = {}  # key -> value (None means delete)
        self.load_current_settings()

    def load_current_settings(self):
        """Load current .env values into env vars"""
        from dotenv import load_dotenv

        load_dotenv(override=True)

    def set(self, key: str, value: str):
        """Stage a change"""
        self.pending_changes[key] = value

    def remove(self, key: str):
        """Stage a removal"""
        self.pending_changes[key] = None

    def save(self):
        """Write pending changes to .env"""
        for key, value in self.pending_changes.items():
            if value is None:
                # Remove key
                unset_key(str(self.env_file), key)
            else:
                # Set key
                set_key(str(self.env_file), key, value)

        self.pending_changes.clear()
        self.load_current_settings()

    def discard(self):
        """Discard pending changes"""
        self.pending_changes.clear()

    def has_pending(self) -> bool:
        """Check if there are pending changes"""
        return bool(self.pending_changes)


class CustomProviderManager:
    """Manages custom provider API bases"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_providers(self) -> Dict[str, str]:
        """Get currently configured custom providers"""
        from proxy_app.provider_urls import PROVIDER_URL_MAP

        providers = {}
        for key, value in os.environ.items():
            if key.endswith("_API_BASE"):
                provider = key.replace("_API_BASE", "").lower()
                # Only include if NOT in hardcoded map
                if provider not in PROVIDER_URL_MAP:
                    providers[provider] = value
        return providers

    def add_provider(self, name: str, api_base: str):
        """Add PROVIDER_API_BASE"""
        key = f"{name.upper()}_API_BASE"
        self.settings.set(key, api_base)

    def edit_provider(self, name: str, api_base: str):
        """Edit PROVIDER_API_BASE"""
        self.add_provider(name, api_base)

    def remove_provider(self, name: str):
        """Remove PROVIDER_API_BASE"""
        key = f"{name.upper()}_API_BASE"
        self.settings.remove(key)


class ModelDefinitionManager:
    """Manages PROVIDER_MODELS"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_provider_models(self, provider: str) -> Optional[Dict]:
        """Get currently configured models for a provider"""
        key = f"{provider.upper()}_MODELS"
        value = os.getenv(key)
        if value:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def get_all_providers_with_models(self) -> Dict[str, int]:
        """Get all providers with model definitions"""
        providers = {}
        for key, value in os.environ.items():
            if key.endswith("_MODELS"):
                provider = key.replace("_MODELS", "").lower()
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        providers[provider] = len(parsed)
                    elif isinstance(parsed, list):
                        providers[provider] = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
        return providers

    def set_models(self, provider: str, models: Dict[str, Dict[str, Any]]):
        """Set PROVIDER_MODELS"""
        key = f"{provider.upper()}_MODELS"
        value = json.dumps(models)
        self.settings.set(key, value)

    def remove_models(self, provider: str):
        """Remove PROVIDER_MODELS"""
        key = f"{provider.upper()}_MODELS"
        self.settings.remove(key)


class ConcurrencyManager:
    """Manages MAX_CONCURRENT_REQUESTS_PER_KEY_PROVIDER"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_limits(self) -> Dict[str, int]:
        """Get currently configured concurrency limits"""
        limits = {}
        for key, value in os.environ.items():
            if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
                provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
                try:
                    limits[provider] = int(value)
                except (json.JSONDecodeError, ValueError):
                    pass
        return limits

    def set_limit(self, provider: str, limit: int):
        """Set concurrency limit"""
        key = f"MAX_CONCURRENT_REQUESTS_PER_KEY_{provider.upper()}"
        self.settings.set(key, str(limit))

    def remove_limit(self, provider: str):
        """Remove concurrency limit (reset to default)"""
        key = f"MAX_CONCURRENT_REQUESTS_PER_KEY_{provider.upper()}"
        self.settings.remove(key)


class RotationModeManager:
    """Manages ROTATION_MODE_PROVIDER settings for sequential/balanced credential rotation"""

    VALID_MODES = ["balanced", "sequential"]

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_modes(self) -> Dict[str, str]:
        """Get currently configured rotation modes"""
        modes = {}
        for key, value in os.environ.items():
            if key.startswith("ROTATION_MODE_"):
                provider = key.replace("ROTATION_MODE_", "").lower()
                if value.lower() in self.VALID_MODES:
                    modes[provider] = value.lower()
        return modes

    def get_default_mode(self, provider: str) -> str:
        """Get the default rotation mode for a provider"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(provider_class, "default_rotation_mode"):
                return provider_class.default_rotation_mode
            return "balanced"
        except ImportError:
            # Fallback defaults if import fails
            if provider.lower() == "antigravity":
                return "sequential"
            return "balanced"

    def get_effective_mode(self, provider: str) -> str:
        """Get the effective rotation mode (configured or default)"""
        configured = self.get_current_modes().get(provider.lower())
        if configured:
            return configured
        return self.get_default_mode(provider)

    def set_mode(self, provider: str, mode: str):
        """Set rotation mode for a provider"""
        if mode.lower() not in self.VALID_MODES:
            raise ValueError(
                f"Invalid rotation mode: {mode}. Must be one of {self.VALID_MODES}"
            )
        key = f"ROTATION_MODE_{provider.upper()}"
        self.settings.set(key, mode.lower())

    def remove_mode(self, provider: str):
        """Remove rotation mode (reset to provider default)"""
        key = f"ROTATION_MODE_{provider.upper()}"
        self.settings.remove(key)


class PriorityMultiplierManager:
    """Manages CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N> settings"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_provider_defaults(self, provider: str) -> Dict[int, int]:
        """Get default priority multipliers from provider class"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(
                provider_class, "default_priority_multipliers"
            ):
                return dict(provider_class.default_priority_multipliers)
        except ImportError:
            pass
        return {}

    def get_sequential_fallback(self, provider: str) -> int:
        """Get sequential fallback multiplier from provider class"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(
                provider_class, "default_sequential_fallback_multiplier"
            ):
                return provider_class.default_sequential_fallback_multiplier
        except ImportError:
            pass
        return 1

    def get_current_multipliers(self) -> Dict[str, Dict[int, int]]:
        """Get currently configured priority multipliers from env vars"""
        multipliers: Dict[str, Dict[int, int]] = {}
        for key, value in os.environ.items():
            if key.startswith("CONCURRENCY_MULTIPLIER_") and "_PRIORITY_" in key:
                try:
                    # Parse: CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>
                    parts = key.split("_PRIORITY_")
                    provider = parts[0].replace("CONCURRENCY_MULTIPLIER_", "").lower()
                    remainder = parts[1]

                    # Check if mode-specific (has _SEQUENTIAL or _BALANCED suffix)
                    if "_" in remainder:
                        continue  # Skip mode-specific for now (show in separate view)

                    priority = int(remainder)
                    multiplier = int(value)

                    if provider not in multipliers:
                        multipliers[provider] = {}
                    multipliers[provider][priority] = multiplier
                except (ValueError, IndexError):
                    pass
        return multipliers

    def get_effective_multiplier(self, provider: str, priority: int) -> int:
        """Get effective multiplier (configured, provider default, or 1)"""
        # Check env var override
        current = self.get_current_multipliers()
        if provider.lower() in current:
            if priority in current[provider.lower()]:
                return current[provider.lower()][priority]

        # Check provider defaults
        defaults = self.get_provider_defaults(provider)
        if priority in defaults:
            return defaults[priority]

        # Return 1 (no multiplier)
        return 1

    def set_multiplier(self, provider: str, priority: int, multiplier: int):
        """Set priority multiplier for a provider"""
        if multiplier < 1:
            raise ValueError("Multiplier must be >= 1")
        key = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_{priority}"
        self.settings.set(key, str(multiplier))

    def remove_multiplier(self, provider: str, priority: int):
        """Remove multiplier (reset to provider default)"""
        key = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_{priority}"
        self.settings.remove(key)


# =============================================================================
# PROVIDER-SPECIFIC SETTINGS DEFINITIONS
# =============================================================================

# Antigravity provider environment variables
ANTIGRAVITY_SETTINGS = {
    "ANTIGRAVITY_SIGNATURE_CACHE_TTL": {
        "type": "int",
        "default": 3600,
        "description": "Memory cache TTL for Gemini 3 thought signatures (seconds)",
    },
    "ANTIGRAVITY_SIGNATURE_DISK_TTL": {
        "type": "int",
        "default": 86400,
        "description": "Disk cache TTL for Gemini 3 thought signatures (seconds)",
    },
    "ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES": {
        "type": "bool",
        "default": True,
        "description": "Preserve thought signatures in client responses",
    },
    "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE": {
        "type": "bool",
        "default": True,
        "description": "Enable signature caching for multi-turn conversations",
    },
    "ANTIGRAVITY_ENABLE_DYNAMIC_MODELS": {
        "type": "bool",
        "default": False,
        "description": "Enable dynamic model discovery from API",
    },
    "ANTIGRAVITY_GEMINI3_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Gemini 3 tool hallucination prevention",
    },
    "ANTIGRAVITY_CLAUDE_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Claude tool hallucination prevention",
    },
    "ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION": {
        "type": "bool",
        "default": True,
        "description": "Sanitize thinking blocks for Claude multi-turn conversations",
    },
    "ANTIGRAVITY_GEMINI3_TOOL_PREFIX": {
        "type": "str",
        "default": "gemini3_",
        "description": "Prefix added to tool names for Gemini 3 disambiguation",
    },
    "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for strict parameter hints in tool descriptions",
    },
    "ANTIGRAVITY_CLAUDE_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for Claude strict parameter hints in tool descriptions",
    },
}

# Gemini CLI provider environment variables
GEMINI_CLI_SETTINGS = {
    "GEMINI_CLI_SIGNATURE_CACHE_TTL": {
        "type": "int",
        "default": 3600,
        "description": "Memory cache TTL for thought signatures (seconds)",
    },
    "GEMINI_CLI_SIGNATURE_DISK_TTL": {
        "type": "int",
        "default": 86400,
        "description": "Disk cache TTL for thought signatures (seconds)",
    },
    "GEMINI_CLI_PRESERVE_THOUGHT_SIGNATURES": {
        "type": "bool",
        "default": True,
        "description": "Preserve thought signatures in client responses",
    },
    "GEMINI_CLI_ENABLE_SIGNATURE_CACHE": {
        "type": "bool",
        "default": True,
        "description": "Enable signature caching for multi-turn conversations",
    },
    "GEMINI_CLI_GEMINI3_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Gemini 3 tool hallucination prevention",
    },
    "GEMINI_CLI_GEMINI3_TOOL_PREFIX": {
        "type": "str",
        "default": "gemini3_",
        "description": "Prefix added to tool names for Gemini 3 disambiguation",
    },
    "GEMINI_CLI_GEMINI3_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for strict parameter hints in tool descriptions",
    },
    "GEMINI_CLI_PROJECT_ID": {
        "type": "str",
        "default": "",
        "description": "GCP Project ID for paid tier users (required for paid tiers)",
    },
}

# Map provider names to their settings definitions
PROVIDER_SETTINGS_MAP = {
    "antigravity": ANTIGRAVITY_SETTINGS,
    "gemini_cli": GEMINI_CLI_SETTINGS,
}


class ProviderSettingsManager:
    """Manages provider-specific configuration settings"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_available_providers(self) -> List[str]:
        """Get list of providers with specific settings available"""
        return list(PROVIDER_SETTINGS_MAP.keys())

    def get_provider_settings_definitions(
        self, provider: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get settings definitions for a provider"""
        return PROVIDER_SETTINGS_MAP.get(provider, {})

    def get_current_value(self, key: str, definition: Dict[str, Any]) -> Any:
        """Get current value of a setting from environment"""
        env_value = os.getenv(key)
        if env_value is None:
            return definition.get("default")

        setting_type = definition.get("type", "str")
        try:
            if setting_type == "bool":
                return env_value.lower() in ("true", "1", "yes")
            elif setting_type == "int":
                return int(env_value)
            else:
                return env_value
        except (ValueError, AttributeError):
            return definition.get("default")

    def get_all_current_values(self, provider: str) -> Dict[str, Any]:
        """Get all current values for a provider"""
        definitions = self.get_provider_settings_definitions(provider)
        values = {}
        for key, definition in definitions.items():
            values[key] = self.get_current_value(key, definition)
        return values

    def set_value(self, key: str, value: Any, definition: Dict[str, Any]):
        """Set a setting value, converting to string for .env storage"""
        setting_type = definition.get("type", "str")
        if setting_type == "bool":
            str_value = "true" if value else "false"
        else:
            str_value = str(value)
        self.settings.set(key, str_value)

    def reset_to_default(self, key: str):
        """Remove a setting to reset it to default"""
        self.settings.remove(key)

    def get_modified_settings(self, provider: str) -> Dict[str, Any]:
        """Get settings that differ from defaults"""
        definitions = self.get_provider_settings_definitions(provider)
        modified = {}
        for key, definition in definitions.items():
            current = self.get_current_value(key, definition)
            default = definition.get("default")
            if current != default:
                modified[key] = current
        return modified


class SettingsTool:
    """Main settings tool TUI"""

    def __init__(self):
        self.console = Console()
        self.settings = AdvancedSettings()
        self.provider_mgr = CustomProviderManager(self.settings)
        self.model_mgr = ModelDefinitionManager(self.settings)
        self.concurrency_mgr = ConcurrencyManager(self.settings)
        self.rotation_mgr = RotationModeManager(self.settings)
        self.priority_multiplier_mgr = PriorityMultiplierManager(self.settings)
        self.provider_settings_mgr = ProviderSettingsManager(self.settings)
        self.running = True

    def get_available_providers(self) -> List[str]:
        """Get list of providers that have credentials configured"""
        env_file = Path.cwd() / ".env"
        providers = set()

        # Scan for providers with API keys from local .env
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith("#"):
                            continue
                        if (
                            "_API_KEY" in line
                            and "PROXY_API_KEY" not in line
                            and "=" in line
                        ):
                            provider = line.split("_API_KEY")[0].strip().lower()
                            providers.add(provider)
            except (IOError, OSError):
                pass

        # Also check for OAuth providers from files
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                providers.add(provider)

        return sorted(list(providers))

    def run(self):
        """Main loop"""
        while self.running:
            self.show_main_menu()

    def show_main_menu(self):
        """Display settings categories"""
        clear_screen()

        self.console.print(
            Panel.fit(
                "[bold cyan]🔧 Advanced Settings Configuration[/bold cyan]",
                border_style="cyan",
            )
        )

        self.console.print()
        self.console.print("[bold]⚙️  Configuration Categories[/bold]")
        self.console.print()
        self.console.print("   1. 🌐 Custom Provider API Bases")
        self.console.print("   2. 📦 Provider Model Definitions")
        self.console.print("   3. ⚡ Concurrency Limits")
        self.console.print("   4. 🔄 Rotation Modes")
        self.console.print("   5. 🔬 Provider-Specific Settings")
        self.console.print("   6. 💾 Save & Exit")
        self.console.print("   7. 🚫 Exit Without Saving")

        self.console.print()
        self.console.print("━" * 70)

        if self.settings.has_pending():
            self.console.print(
                '[yellow]ℹ️  Changes are pending until you select "Save & Exit"[/yellow]'
            )
        else:
            self.console.print("[dim]ℹ️  No pending changes[/dim]")

        self.console.print()
        self.console.print(
            "[dim]⚠️  Model filters not supported - edit .env for IGNORE_MODELS_* / WHITELIST_MODELS_*[/dim]"
        )
        self.console.print()

        choice = Prompt.ask(
            "Select option",
            choices=["1", "2", "3", "4", "5", "6", "7"],
            show_choices=False,
        )

        if choice == "1":
            self.manage_custom_providers()
        elif choice == "2":
            self.manage_model_definitions()
        elif choice == "3":
            self.manage_concurrency_limits()
        elif choice == "4":
            self.manage_rotation_modes()
        elif choice == "5":
            self.manage_provider_settings()
        elif choice == "6":
            self.save_and_exit()
        elif choice == "7":
            self.exit_without_saving()

    def manage_custom_providers(self):
        """Manage custom provider API bases"""
        while True:
            clear_screen()

            providers = self.provider_mgr.get_current_providers()

            self.console.print(
                Panel.fit(
                    "[bold cyan]🌐 Custom Provider API Bases[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print("[bold]📋 Configured Custom Providers[/bold]")
            self.console.print("━" * 70)

            if providers:
                for name, base in providers.items():
                    self.console.print(f"   • {name:15} {base}")
            else:
                self.console.print("   [dim]No custom providers configured[/dim]")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            self.console.print("[bold]⚙️  Actions[/bold]")
            self.console.print()
            self.console.print("   1. ➕ Add New Custom Provider")
            self.console.print("   2. ✏️  Edit Existing Provider")
            self.console.print("   3. 🗑️  Remove Provider")
            self.console.print("   4. ↩️  Back to Settings Menu")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choice = Prompt.ask(
                "Select option", choices=["1", "2", "3", "4"], show_choices=False
            )

            if choice == "1":
                name = Prompt.ask("Provider name (e.g., 'opencode')").strip().lower()
                if name:
                    api_base = Prompt.ask("API Base URL").strip()
                    if api_base:
                        self.provider_mgr.add_provider(name, api_base)
                        self.console.print(
                            f"\n[green]✅ Custom provider '{name}' configured![/green]"
                        )
                        self.console.print(
                            f"   To use: set {name.upper()}_API_KEY in credentials"
                        )
                        input("\nPress Enter to continue...")

            elif choice == "2":
                if not providers:
                    self.console.print("\n[yellow]No providers to edit[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print("\n[bold]Select provider to edit:[/bold]")
                providers_list = list(providers.keys())
                for idx, prov in enumerate(providers_list, 1):
                    self.console.print(f"   {idx}. {prov}")

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(providers_list) + 1)],
                )
                name = providers_list[choice_idx - 1]
                current_base = providers.get(name, "")

                self.console.print(f"\nCurrent API Base: {current_base}")
                new_base = Prompt.ask(
                    "New API Base [press Enter to keep current]", default=current_base
                ).strip()

                if new_base and new_base != current_base:
                    self.provider_mgr.edit_provider(name, new_base)
                    self.console.print(
                        f"\n[green]✅ Custom provider '{name}' updated![/green]"
                    )
                else:
                    self.console.print("\n[yellow]No changes made[/yellow]")
                input("\nPress Enter to continue...")

            elif choice == "3":
                if not providers:
                    self.console.print("\n[yellow]No providers to remove[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print("\n[bold]Select provider to remove:[/bold]")
                providers_list = list(providers.keys())
                for idx, prov in enumerate(providers_list, 1):
                    self.console.print(f"   {idx}. {prov}")

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(providers_list) + 1)],
                )
                name = providers_list[choice_idx - 1]

                if Confirm.ask(f"Remove '{name}'?"):
                    self.provider_mgr.remove_provider(name)
                    self.console.print(
                        f"\n[green]✅ Provider '{name}' removed![/green]"
                    )
                    input("\nPress Enter to continue...")

            elif choice == "4":
                break

    def manage_model_definitions(self):
        """Manage provider model definitions"""
        while True:
            clear_screen()

            all_providers = self.model_mgr.get_all_providers_with_models()

            self.console.print(
                Panel.fit(
                    "[bold cyan]📦 Provider Model Definitions[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print("[bold]📋 Configured Provider Models[/bold]")
            self.console.print("━" * 70)

            if all_providers:
                for provider, count in all_providers.items():
                    self.console.print(
                        f"   • {provider:15} {count} model{'s' if count > 1 else ''}"
                    )
            else:
                self.console.print("   [dim]No model definitions configured[/dim]")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            self.console.print("[bold]⚙️  Actions[/bold]")
            self.console.print()
            self.console.print("   1. ➕ Add Models for Provider")
            self.console.print("   2. ✏️  Edit Provider Models")
            self.console.print("   3. 👁️  View Provider Models")
            self.console.print("   4. 🗑️  Remove Provider Models")
            self.console.print("   5. ↩️  Back to Settings Menu")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choice = Prompt.ask(
                "Select option", choices=["1", "2", "3", "4", "5"], show_choices=False
            )

            if choice == "1":
                self.add_model_definitions()
            elif choice == "2":
                if not all_providers:
                    self.console.print("\n[yellow]No providers to edit[/yellow]")
                    input("\nPress Enter to continue...")
                    continue
                self.edit_model_definitions(list(all_providers.keys()))
            elif choice == "3":
                if not all_providers:
                    self.console.print("\n[yellow]No providers to view[/yellow]")
                    input("\nPress Enter to continue...")
                    continue
                self.view_model_definitions(list(all_providers.keys()))
            elif choice == "4":
                if not all_providers:
                    self.console.print("\n[yellow]No providers to remove[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print(
                    "\n[bold]Select provider to remove models from:[/bold]"
                )
                providers_list = list(all_providers.keys())
                for idx, prov in enumerate(providers_list, 1):
                    self.console.print(f"   {idx}. {prov}")

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(providers_list) + 1)],
                )
                provider = providers_list[choice_idx - 1]

                if Confirm.ask(f"Remove all model definitions for '{provider}'?"):
                    self.model_mgr.remove_models(provider)
                    self.console.print(
                        f"\n[green]✅ Model definitions removed for '{provider}'![/green]"
                    )
                    input("\nPress Enter to continue...")
            elif choice == "5":
                break

    def add_model_definitions(self):
        """Add model definitions for a provider"""
        # Get available providers from credentials
        available_providers = self.get_available_providers()

        if not available_providers:
            self.console.print(
                "\n[yellow]No providers with credentials found. Please add credentials first.[/yellow]"
            )
            input("\nPress Enter to continue...")
            return

        # Show provider selection menu
        self.console.print("\n[bold]Select provider:[/bold]")
        for idx, prov in enumerate(available_providers, 1):
            self.console.print(f"   {idx}. {prov}")
        self.console.print(
            f"   {len(available_providers) + 1}. Enter custom provider name"
        )

        choice = IntPrompt.ask(
            "Select option",
            choices=[str(i) for i in range(1, len(available_providers) + 2)],
        )

        if choice == len(available_providers) + 1:
            provider = Prompt.ask("Provider name").strip().lower()
        else:
            provider = available_providers[choice - 1]

        if not provider:
            return

        self.console.print("\nHow would you like to define models?")
        self.console.print("   1. Simple list (names only)")
        self.console.print("   2. Advanced (names with IDs and options)")

        mode = Prompt.ask("Select mode", choices=["1", "2"], show_choices=False)

        models = {}

        if mode == "1":
            # Simple mode
            while True:
                name = Prompt.ask("\nModel name (or 'done' to finish)").strip()
                if name.lower() == "done":
                    break
                if name:
                    models[name] = {}
        else:
            # Advanced mode
            while True:
                name = Prompt.ask("\nModel name (or 'done' to finish)").strip()
                if name.lower() == "done":
                    break
                if name:
                    model_def = {}
                    model_id = Prompt.ask(
                        f"Model ID [press Enter to use '{name}']", default=name
                    ).strip()
                    if model_id and model_id != name:
                        model_def["id"] = model_id

                    # Optional: model options
                    if Confirm.ask(
                        "Add model options (e.g., temperature limits)?", default=False
                    ):
                        self.console.print(
                            "\nEnter options as key=value pairs (one per line, 'done' to finish):"
                        )
                        options = {}
                        while True:
                            opt = Prompt.ask("Option").strip()
                            if opt.lower() == "done":
                                break
                            if "=" in opt:
                                key, value = opt.split("=", 1)
                                value = value.strip()
                                # Try to convert to number if possible
                                try:
                                    value = float(value) if "." in value else int(value)
                                except (ValueError, TypeError):
                                    pass
                                options[key.strip()] = value
                        if options:
                            model_def["options"] = options

                    models[name] = model_def

        if models:
            self.model_mgr.set_models(provider, models)
            self.console.print(
                f"\n[green]✅ Model definitions saved for '{provider}'![/green]"
            )
        else:
            self.console.print("\n[yellow]No models added[/yellow]")

        input("\nPress Enter to continue...")

    def edit_model_definitions(self, providers: List[str]):
        """Edit existing model definitions"""
        # Show numbered list
        self.console.print("\n[bold]Select provider to edit:[/bold]")
        for idx, prov in enumerate(providers, 1):
            self.console.print(f"   {idx}. {prov}")

        choice_idx = IntPrompt.ask(
            "Select option", choices=[str(i) for i in range(1, len(providers) + 1)]
        )
        provider = providers[choice_idx - 1]

        current_models = self.model_mgr.get_current_provider_models(provider)
        if not current_models:
            self.console.print(f"\n[yellow]No models found for '{provider}'[/yellow]")
            input("\nPress Enter to continue...")
            return

        # Convert to dict if list
        if isinstance(current_models, list):
            current_models = {m: {} for m in current_models}

        while True:
            clear_screen()
            self.console.print(f"[bold]Editing models for: {provider}[/bold]\n")
            self.console.print("Current models:")
            for i, (name, definition) in enumerate(current_models.items(), 1):
                model_id = (
                    definition.get("id", name) if isinstance(definition, dict) else name
                )
                self.console.print(f"   {i}. {name} (ID: {model_id})")

            self.console.print("\nOptions:")
            self.console.print("   1. Add new model")
            self.console.print("   2. Edit existing model")
            self.console.print("   3. Remove model")
            self.console.print("   4. Done")

            choice = Prompt.ask(
                "\nSelect option", choices=["1", "2", "3", "4"], show_choices=False
            )

            if choice == "1":
                name = Prompt.ask("New model name").strip()
                if name and name not in current_models:
                    model_id = Prompt.ask("Model ID", default=name).strip()
                    current_models[name] = {"id": model_id} if model_id != name else {}

            elif choice == "2":
                # Show numbered list
                models_list = list(current_models.keys())
                self.console.print("\n[bold]Select model to edit:[/bold]")
                for idx, model_name in enumerate(models_list, 1):
                    self.console.print(f"   {idx}. {model_name}")

                model_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(models_list) + 1)],
                )
                name = models_list[model_idx - 1]

                current_def = current_models[name]
                current_id = (
                    current_def.get("id", name)
                    if isinstance(current_def, dict)
                    else name
                )

                new_id = Prompt.ask("Model ID", default=current_id).strip()
                current_models[name] = {"id": new_id} if new_id != name else {}

            elif choice == "3":
                # Show numbered list
                models_list = list(current_models.keys())
                self.console.print("\n[bold]Select model to remove:[/bold]")
                for idx, model_name in enumerate(models_list, 1):
                    self.console.print(f"   {idx}. {model_name}")

                model_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(models_list) + 1)],
                )
                name = models_list[model_idx - 1]

                if Confirm.ask(f"Remove '{name}'?"):
                    del current_models[name]

            elif choice == "4":
                break

        if current_models:
            self.model_mgr.set_models(provider, current_models)
            self.console.print(f"\n[green]✅ Models updated for '{provider}'![/green]")
        else:
            self.console.print(
                "\n[yellow]No models left - removing definition[/yellow]"
            )
            self.model_mgr.remove_models(provider)

        input("\nPress Enter to continue...")

    def view_model_definitions(self, providers: List[str]):
        """View model definitions for a provider"""
        # Show numbered list
        self.console.print("\n[bold]Select provider to view:[/bold]")
        for idx, prov in enumerate(providers, 1):
            self.console.print(f"   {idx}. {prov}")

        choice_idx = IntPrompt.ask(
            "Select option", choices=[str(i) for i in range(1, len(providers) + 1)]
        )
        provider = providers[choice_idx - 1]

        models = self.model_mgr.get_current_provider_models(provider)
        if not models:
            self.console.print(f"\n[yellow]No models found for '{provider}'[/yellow]")
            input("\nPress Enter to continue...")
            return

        clear_screen()
        self.console.print(f"[bold]Provider: {provider}[/bold]\n")
        self.console.print("[bold]📦 Configured Models:[/bold]")
        self.console.print("━" * 50)

        # Handle both dict and list formats
        if isinstance(models, dict):
            for name, definition in models.items():
                if isinstance(definition, dict):
                    model_id = definition.get("id", name)
                    self.console.print(f"   Name: {name}")
                    self.console.print(f"   ID:   {model_id}")
                    if "options" in definition:
                        self.console.print(f"   Options: {definition['options']}")
                    self.console.print()
                else:
                    self.console.print(f"   Name: {name}")
                    self.console.print()
        elif isinstance(models, list):
            for name in models:
                self.console.print(f"   Name: {name}")
                self.console.print()

        input("Press Enter to return...")

    def manage_provider_settings(self):
        """Manage provider-specific settings (Antigravity, Gemini CLI)"""
        while True:
            clear_screen()

            available_providers = self.provider_settings_mgr.get_available_providers()

            self.console.print(
                Panel.fit(
                    "[bold cyan]🔬 Provider-Specific Settings[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print(
                "[bold]📋 Available Providers with Custom Settings[/bold]"
            )
            self.console.print("━" * 70)

            for provider in available_providers:
                modified = self.provider_settings_mgr.get_modified_settings(provider)
                status = (
                    f"[yellow]{len(modified)} modified[/yellow]"
                    if modified
                    else "[dim]defaults[/dim]"
                )
                display_name = provider.replace("_", " ").title()
                self.console.print(f"   • {display_name:20} {status}")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            self.console.print("[bold]⚙️  Select Provider to Configure[/bold]")
            self.console.print()

            for idx, provider in enumerate(available_providers, 1):
                display_name = provider.replace("_", " ").title()
                self.console.print(f"   {idx}. {display_name}")
            self.console.print(
                f"   {len(available_providers) + 1}. ↩️  Back to Settings Menu"
            )

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choices = [str(i) for i in range(1, len(available_providers) + 2)]
            choice = Prompt.ask("Select option", choices=choices, show_choices=False)
            choice_idx = int(choice)

            if choice_idx == len(available_providers) + 1:
                break

            provider = available_providers[choice_idx - 1]
            self._manage_single_provider_settings(provider)

    def _manage_single_provider_settings(self, provider: str):
        """Manage settings for a single provider"""
        while True:
            clear_screen()

            display_name = provider.replace("_", " ").title()
            definitions = self.provider_settings_mgr.get_provider_settings_definitions(
                provider
            )
            current_values = self.provider_settings_mgr.get_all_current_values(provider)

            self.console.print(
                Panel.fit(
                    f"[bold cyan]🔬 {display_name} Settings[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print("[bold]📋 Current Settings[/bold]")
            self.console.print("━" * 70)

            # Display all settings with current values
            settings_list = list(definitions.keys())
            for idx, key in enumerate(settings_list, 1):
                definition = definitions[key]
                current = current_values.get(key)
                default = definition.get("default")
                setting_type = definition.get("type", "str")
                description = definition.get("description", "")

                # Format value display
                if setting_type == "bool":
                    value_display = (
                        "[green]✓ Enabled[/green]"
                        if current
                        else "[red]✗ Disabled[/red]"
                    )
                elif setting_type == "int":
                    value_display = f"[cyan]{current}[/cyan]"
                else:
                    value_display = (
                        f"[cyan]{current or '(not set)'}[/cyan]"
                        if current
                        else "[dim](not set)[/dim]"
                    )

                # Check if modified from default
                modified = current != default
                mod_marker = "[yellow]*[/yellow]" if modified else " "

                # Short key name for display (strip provider prefix)
                short_key = key.replace(f"{provider.upper()}_", "")

                self.console.print(
                    f"  {mod_marker}{idx:2}. {short_key:35} {value_display}"
                )
                self.console.print(f"       [dim]{description}[/dim]")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print("[dim]* = modified from default[/dim]")
            self.console.print()
            self.console.print("[bold]⚙️  Actions[/bold]")
            self.console.print()
            self.console.print("   E. ✏️  Edit a Setting")
            self.console.print("   R. 🔄 Reset Setting to Default")
            self.console.print("   A. 🔄 Reset All to Defaults")
            self.console.print("   B. ↩️  Back to Provider Selection")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choice = Prompt.ask(
                "Select action",
                choices=["e", "r", "a", "b", "E", "R", "A", "B"],
                show_choices=False,
            ).lower()

            if choice == "b":
                break
            elif choice == "e":
                self._edit_provider_setting(provider, settings_list, definitions)
            elif choice == "r":
                self._reset_provider_setting(provider, settings_list, definitions)
            elif choice == "a":
                self._reset_all_provider_settings(provider, settings_list)

    def _edit_provider_setting(
        self,
        provider: str,
        settings_list: List[str],
        definitions: Dict[str, Dict[str, Any]],
    ):
        """Edit a single provider setting"""
        self.console.print("\n[bold]Select setting number to edit:[/bold]")

        choices = [str(i) for i in range(1, len(settings_list) + 1)]
        choice = IntPrompt.ask("Setting number", choices=choices)
        key = settings_list[choice - 1]
        definition = definitions[key]

        current = self.provider_settings_mgr.get_current_value(key, definition)
        default = definition.get("default")
        setting_type = definition.get("type", "str")
        short_key = key.replace(f"{provider.upper()}_", "")

        self.console.print(f"\n[bold]Editing: {short_key}[/bold]")
        self.console.print(f"Current value: [cyan]{current}[/cyan]")
        self.console.print(f"Default value: [dim]{default}[/dim]")
        self.console.print(f"Type: {setting_type}")

        if setting_type == "bool":
            new_value = Confirm.ask("\nEnable this setting?", default=current)
            self.provider_settings_mgr.set_value(key, new_value, definition)
            status = "enabled" if new_value else "disabled"
            self.console.print(f"\n[green]✅ {short_key} {status}![/green]")
        elif setting_type == "int":
            new_value = IntPrompt.ask("\nNew value", default=current)
            self.provider_settings_mgr.set_value(key, new_value, definition)
            self.console.print(f"\n[green]✅ {short_key} set to {new_value}![/green]")
        else:
            new_value = Prompt.ask(
                "\nNew value", default=str(current) if current else ""
            ).strip()
            if new_value:
                self.provider_settings_mgr.set_value(key, new_value, definition)
                self.console.print(f"\n[green]✅ {short_key} updated![/green]")
            else:
                self.console.print("\n[yellow]No changes made[/yellow]")

        input("\nPress Enter to continue...")

    def _reset_provider_setting(
        self,
        provider: str,
        settings_list: List[str],
        definitions: Dict[str, Dict[str, Any]],
    ):
        """Reset a single provider setting to default"""
        self.console.print("\n[bold]Select setting number to reset:[/bold]")

        choices = [str(i) for i in range(1, len(settings_list) + 1)]
        choice = IntPrompt.ask("Setting number", choices=choices)
        key = settings_list[choice - 1]
        definition = definitions[key]

        default = definition.get("default")
        short_key = key.replace(f"{provider.upper()}_", "")

        if Confirm.ask(f"\nReset {short_key} to default ({default})?"):
            self.provider_settings_mgr.reset_to_default(key)
            self.console.print(f"\n[green]✅ {short_key} reset to default![/green]")
        else:
            self.console.print("\n[yellow]No changes made[/yellow]")

        input("\nPress Enter to continue...")

    def _reset_all_provider_settings(self, provider: str, settings_list: List[str]):
        """Reset all provider settings to defaults"""
        display_name = provider.replace("_", " ").title()

        if Confirm.ask(
            f"\n[bold red]Reset ALL {display_name} settings to defaults?[/bold red]"
        ):
            for key in settings_list:
                self.provider_settings_mgr.reset_to_default(key)
            self.console.print(
                f"\n[green]✅ All {display_name} settings reset to defaults![/green]"
            )
        else:
            self.console.print("\n[yellow]No changes made[/yellow]")

        input("\nPress Enter to continue...")

    def manage_rotation_modes(self):
        """Manage credential rotation modes (sequential vs balanced)"""
        while True:
            clear_screen()

            modes = self.rotation_mgr.get_current_modes()
            available_providers = self.get_available_providers()

            self.console.print(
                Panel.fit(
                    "[bold cyan]🔄 Credential Rotation Mode Configuration[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print("[bold]📋 Rotation Modes Explained[/bold]")
            self.console.print("━" * 70)
            self.console.print(
                "   [cyan]balanced[/cyan]   - Rotate credentials evenly across requests (default)"
            )
            self.console.print(
                "   [cyan]sequential[/cyan] - Use one credential until exhausted (429), then switch"
            )
            self.console.print()
            self.console.print("[bold]📋 Current Rotation Mode Settings[/bold]")
            self.console.print("━" * 70)

            if modes:
                for provider, mode in modes.items():
                    default_mode = self.rotation_mgr.get_default_mode(provider)
                    is_custom = mode != default_mode
                    marker = "[yellow]*[/yellow]" if is_custom else " "
                    mode_display = (
                        f"[green]{mode}[/green]"
                        if mode == "sequential"
                        else f"[blue]{mode}[/blue]"
                    )
                    self.console.print(f"  {marker}• {provider:20} {mode_display}")

            # Show providers with default modes
            providers_with_defaults = [p for p in available_providers if p not in modes]
            if providers_with_defaults:
                self.console.print()
                self.console.print("[dim]Providers using default modes:[/dim]")
                for provider in providers_with_defaults:
                    default_mode = self.rotation_mgr.get_default_mode(provider)
                    mode_display = (
                        f"[green]{default_mode}[/green]"
                        if default_mode == "sequential"
                        else f"[blue]{default_mode}[/blue]"
                    )
                    self.console.print(
                        f"   • {provider:20} {mode_display} [dim](default)[/dim]"
                    )

            self.console.print()
            self.console.print("━" * 70)
            self.console.print(
                "[dim]* = custom setting (differs from provider default)[/dim]"
            )
            self.console.print()
            self.console.print("[bold]⚙️  Actions[/bold]")
            self.console.print()
            self.console.print("   1. ➕ Set Rotation Mode for Provider")
            self.console.print("   2. 🗑️  Reset to Provider Default")
            self.console.print("   3. ⚡ Configure Priority Concurrency Multipliers")
            self.console.print("   4. ↩️  Back to Settings Menu")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choice = Prompt.ask(
                "Select option", choices=["1", "2", "3", "4"], show_choices=False
            )

            if choice == "1":
                if not available_providers:
                    self.console.print(
                        "\n[yellow]No providers with credentials found. Please add credentials first.[/yellow]"
                    )
                    input("\nPress Enter to continue...")
                    continue

                # Show provider selection menu
                self.console.print("\n[bold]Select provider:[/bold]")
                for idx, prov in enumerate(available_providers, 1):
                    current_mode = self.rotation_mgr.get_effective_mode(prov)
                    mode_display = (
                        f"[green]{current_mode}[/green]"
                        if current_mode == "sequential"
                        else f"[blue]{current_mode}[/blue]"
                    )
                    self.console.print(f"   {idx}. {prov} ({mode_display})")
                self.console.print(
                    f"   {len(available_providers) + 1}. Enter custom provider name"
                )

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(available_providers) + 2)],
                )

                if choice_idx == len(available_providers) + 1:
                    provider = Prompt.ask("Provider name").strip().lower()
                else:
                    provider = available_providers[choice_idx - 1]

                if provider:
                    current_mode = self.rotation_mgr.get_effective_mode(provider)
                    self.console.print(
                        f"\nCurrent mode for {provider}: [cyan]{current_mode}[/cyan]"
                    )
                    self.console.print("\nSelect new rotation mode:")
                    self.console.print(
                        "   1. [blue]balanced[/blue] - Rotate credentials evenly"
                    )
                    self.console.print(
                        "   2. [green]sequential[/green] - Use until exhausted"
                    )

                    mode_choice = Prompt.ask(
                        "Select mode", choices=["1", "2"], show_choices=False
                    )
                    new_mode = "balanced" if mode_choice == "1" else "sequential"

                    self.rotation_mgr.set_mode(provider, new_mode)
                    self.console.print(
                        f"\n[green]✅ Rotation mode for '{provider}' set to {new_mode}![/green]"
                    )
                    input("\nPress Enter to continue...")

            elif choice == "2":
                if not modes:
                    self.console.print(
                        "\n[yellow]No custom rotation modes to reset[/yellow]"
                    )
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print(
                    "\n[bold]Select provider to reset to default:[/bold]"
                )
                modes_list = list(modes.keys())
                for idx, prov in enumerate(modes_list, 1):
                    default_mode = self.rotation_mgr.get_default_mode(prov)
                    self.console.print(
                        f"   {idx}. {prov} (will reset to: {default_mode})"
                    )

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(modes_list) + 1)],
                )
                provider = modes_list[choice_idx - 1]
                default_mode = self.rotation_mgr.get_default_mode(provider)

                if Confirm.ask(f"Reset '{provider}' to default mode ({default_mode})?"):
                    self.rotation_mgr.remove_mode(provider)
                    self.console.print(
                        f"\n[green]✅ Rotation mode for '{provider}' reset to default ({default_mode})![/green]"
                    )
                    input("\nPress Enter to continue...")

            elif choice == "3":
                self.manage_priority_multipliers()

            elif choice == "4":
                break

    def manage_priority_multipliers(self):
        """Manage priority-based concurrency multipliers per provider"""
        clear_screen()

        current_multipliers = self.priority_multiplier_mgr.get_current_multipliers()
        available_providers = self.get_available_providers()

        self.console.print(
            Panel.fit(
                "[bold cyan]⚡ Priority Concurrency Multipliers[/bold cyan]",
                border_style="cyan",
            )
        )

        self.console.print()
        self.console.print("[bold]📋 Current Priority Multiplier Settings[/bold]")
        self.console.print("━" * 70)

        # Show all providers with their priority multipliers
        has_settings = False
        for provider in available_providers:
            defaults = self.priority_multiplier_mgr.get_provider_defaults(provider)
            overrides = current_multipliers.get(provider, {})
            seq_fallback = self.priority_multiplier_mgr.get_sequential_fallback(
                provider
            )
            rotation_mode = self.rotation_mgr.get_effective_mode(provider)

            if defaults or overrides or seq_fallback != 1:
                has_settings = True
                self.console.print(
                    f"\n   [bold]{provider}[/bold] ({rotation_mode} mode)"
                )

                # Combine and display priorities
                all_priorities = set(defaults.keys()) | set(overrides.keys())
                for priority in sorted(all_priorities):
                    default_val = defaults.get(priority, 1)
                    override_val = overrides.get(priority)

                    if override_val is not None:
                        self.console.print(
                            f"      Priority {priority}: [cyan]{override_val}x[/cyan] (override, default: {default_val}x)"
                        )
                    else:
                        self.console.print(
                            f"      Priority {priority}: {default_val}x [dim](default)[/dim]"
                        )

                # Show sequential fallback if applicable
                if rotation_mode == "sequential" and seq_fallback != 1:
                    self.console.print(
                        f"      Others (seq): {seq_fallback}x [dim](fallback)[/dim]"
                    )

        if not has_settings:
            self.console.print("   [dim]No priority multipliers configured[/dim]")

        self.console.print()
        self.console.print("[bold]ℹ️  About Priority Multipliers:[/bold]")
        self.console.print(
            "   Higher priority tiers (lower numbers) can have higher multipliers."
        )
        self.console.print("   Example: Priority 1 = 5x, Priority 2 = 3x, Others = 1x")
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("   1. ✏️  Set Priority Multiplier")
        self.console.print("   2. 🔄 Reset to Provider Default")
        self.console.print("   3. ↩️  Back")

        choice = Prompt.ask(
            "Select option", choices=["1", "2", "3"], show_choices=False
        )

        if choice == "1":
            if not available_providers:
                self.console.print("\n[yellow]No providers available[/yellow]")
                input("\nPress Enter to continue...")
                return

            # Select provider
            self.console.print("\n[bold]Select provider:[/bold]")
            for idx, prov in enumerate(available_providers, 1):
                self.console.print(f"   {idx}. {prov}")

            prov_idx = IntPrompt.ask(
                "Provider",
                choices=[str(i) for i in range(1, len(available_providers) + 1)],
            )
            provider = available_providers[prov_idx - 1]

            # Get priority level
            priority = IntPrompt.ask("Priority level (e.g., 1, 2, 3)")

            # Get current value
            current = self.priority_multiplier_mgr.get_effective_multiplier(
                provider, priority
            )
            self.console.print(
                f"\nCurrent multiplier for priority {priority}: {current}x"
            )

            multiplier = IntPrompt.ask("New multiplier (1-10)", default=current)
            if 1 <= multiplier <= 10:
                self.priority_multiplier_mgr.set_multiplier(
                    provider, priority, multiplier
                )
                self.console.print(
                    f"\n[green]✅ Priority {priority} multiplier for '{provider}' set to {multiplier}x[/green]"
                )
            else:
                self.console.print(
                    "\n[yellow]Multiplier must be between 1 and 10[/yellow]"
                )
            input("\nPress Enter to continue...")

        elif choice == "2":
            # Find providers with overrides
            providers_with_overrides = [
                p for p in available_providers if p in current_multipliers
            ]
            if not providers_with_overrides:
                self.console.print("\n[yellow]No custom multipliers to reset[/yellow]")
                input("\nPress Enter to continue...")
                return

            self.console.print("\n[bold]Select provider to reset:[/bold]")
            for idx, prov in enumerate(providers_with_overrides, 1):
                self.console.print(f"   {idx}. {prov}")

            prov_idx = IntPrompt.ask(
                "Provider",
                choices=[str(i) for i in range(1, len(providers_with_overrides) + 1)],
            )
            provider = providers_with_overrides[prov_idx - 1]

            # Get priority to reset
            overrides = current_multipliers.get(provider, {})
            if len(overrides) == 1:
                priority = list(overrides.keys())[0]
            else:
                self.console.print(f"\nOverrides for {provider}: {overrides}")
                priority = IntPrompt.ask("Priority level to reset")

            if priority in overrides:
                self.priority_multiplier_mgr.remove_multiplier(provider, priority)
                default = self.priority_multiplier_mgr.get_effective_multiplier(
                    provider, priority
                )
                self.console.print(
                    f"\n[green]✅ Reset priority {priority} for '{provider}' to default ({default}x)[/green]"
                )
            else:
                self.console.print(
                    f"\n[yellow]No override for priority {priority}[/yellow]"
                )
            input("\nPress Enter to continue...")

    def manage_concurrency_limits(self):
        """Manage concurrency limits"""
        while True:
            clear_screen()

            limits = self.concurrency_mgr.get_current_limits()

            self.console.print(
                Panel.fit(
                    "[bold cyan]⚡ Concurrency Limits Configuration[/bold cyan]",
                    border_style="cyan",
                )
            )

            self.console.print()
            self.console.print("[bold]📋 Current Concurrency Settings[/bold]")
            self.console.print("━" * 70)

            if limits:
                for provider, limit in limits.items():
                    self.console.print(f"   • {provider:15} {limit} requests/key")
                self.console.print(f"   • Default:        1 request/key (all others)")
            else:
                self.console.print("   • Default:        1 request/key (all providers)")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            self.console.print("[bold]⚙️  Actions[/bold]")
            self.console.print()
            self.console.print("   1. ➕ Add Concurrency Limit for Provider")
            self.console.print("   2. ✏️  Edit Existing Limit")
            self.console.print("   3. 🗑️  Remove Limit (reset to default)")
            self.console.print("   4. ↩️  Back to Settings Menu")

            self.console.print()
            self.console.print("━" * 70)
            self.console.print()

            choice = Prompt.ask(
                "Select option", choices=["1", "2", "3", "4"], show_choices=False
            )

            if choice == "1":
                # Get available providers
                available_providers = self.get_available_providers()

                if not available_providers:
                    self.console.print(
                        "\n[yellow]No providers with credentials found. Please add credentials first.[/yellow]"
                    )
                    input("\nPress Enter to continue...")
                    continue

                # Show provider selection menu
                self.console.print("\n[bold]Select provider:[/bold]")
                for idx, prov in enumerate(available_providers, 1):
                    self.console.print(f"   {idx}. {prov}")
                self.console.print(
                    f"   {len(available_providers) + 1}. Enter custom provider name"
                )

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(available_providers) + 2)],
                )

                if choice_idx == len(available_providers) + 1:
                    provider = Prompt.ask("Provider name").strip().lower()
                else:
                    provider = available_providers[choice_idx - 1]

                if provider:
                    limit = IntPrompt.ask(
                        "Max concurrent requests per key (1-100)", default=1
                    )
                    if 1 <= limit <= 100:
                        self.concurrency_mgr.set_limit(provider, limit)
                        self.console.print(
                            f"\n[green]✅ Concurrency limit set for '{provider}': {limit} requests/key[/green]"
                        )
                    else:
                        self.console.print(
                            "\n[red]❌ Limit must be between 1-100[/red]"
                        )
                    input("\nPress Enter to continue...")

            elif choice == "2":
                if not limits:
                    self.console.print("\n[yellow]No limits to edit[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print("\n[bold]Select provider to edit:[/bold]")
                limits_list = list(limits.keys())
                for idx, prov in enumerate(limits_list, 1):
                    self.console.print(f"   {idx}. {prov}")

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(limits_list) + 1)],
                )
                provider = limits_list[choice_idx - 1]
                current_limit = limits.get(provider, 1)

                self.console.print(f"\nCurrent limit: {current_limit} requests/key")
                new_limit = IntPrompt.ask(
                    "New limit (1-100) [press Enter to keep current]",
                    default=current_limit,
                )

                if 1 <= new_limit <= 100:
                    if new_limit != current_limit:
                        self.concurrency_mgr.set_limit(provider, new_limit)
                        self.console.print(
                            f"\n[green]✅ Concurrency limit updated for '{provider}': {new_limit} requests/key[/green]"
                        )
                    else:
                        self.console.print("\n[yellow]No changes made[/yellow]")
                else:
                    self.console.print("\n[red]Limit must be between 1-100[/red]")
                input("\nPress Enter to continue...")

            elif choice == "3":
                if not limits:
                    self.console.print("\n[yellow]No limits to remove[/yellow]")
                    input("\nPress Enter to continue...")
                    continue

                # Show numbered list
                self.console.print(
                    "\n[bold]Select provider to remove limit from:[/bold]"
                )
                limits_list = list(limits.keys())
                for idx, prov in enumerate(limits_list, 1):
                    self.console.print(f"   {idx}. {prov}")

                choice_idx = IntPrompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(limits_list) + 1)],
                )
                provider = limits_list[choice_idx - 1]

                if Confirm.ask(
                    f"Remove concurrency limit for '{provider}' (reset to default 1)?"
                ):
                    self.concurrency_mgr.remove_limit(provider)
                    self.console.print(
                        f"\n[green]✅ Limit removed for '{provider}' - using default (1 request/key)[/green]"
                    )
                    input("\nPress Enter to continue...")

            elif choice == "4":
                break

    def save_and_exit(self):
        """Save pending changes and exit"""
        if self.settings.has_pending():
            if Confirm.ask("\n[bold yellow]Save all pending changes?[/bold yellow]"):
                self.settings.save()
                self.console.print("\n[green]✅ All changes saved to .env![/green]")
                input("\nPress Enter to return to launcher...")
            else:
                self.console.print("\n[yellow]Changes not saved[/yellow]")
                input("\nPress Enter to continue...")
                return
        else:
            self.console.print("\n[dim]No changes to save[/dim]")
            input("\nPress Enter to return to launcher...")

        self.running = False

    def exit_without_saving(self):
        """Exit without saving"""
        if self.settings.has_pending():
            if Confirm.ask("\n[bold red]Discard all pending changes?[/bold red]"):
                self.settings.discard()
                self.console.print("\n[yellow]Changes discarded[/yellow]")
                input("\nPress Enter to return to launcher...")
                self.running = False
            else:
                return
        else:
            self.running = False


def run_settings_tool():
    """Entry point for settings tool"""
    tool = SettingsTool()
    tool.run()
