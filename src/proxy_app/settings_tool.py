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


class SettingsTool:
    """Main settings tool TUI"""
    
    def __init__(self):
        self.console = Console()
        self.settings = AdvancedSettings()
        self.provider_mgr = CustomProviderManager(self.settings)
        self.model_mgr = ModelDefinitionManager(self.settings)
        self.concurrency_mgr = ConcurrencyManager(self.settings)
        self.running = True
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers that have credentials configured"""
        env_file = Path.cwd() / ".env"
        providers = set()
        
        # Scan for providers with API keys from local .env
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if "_API_KEY" in line and "PROXY_API_KEY" not in line and "=" in line:
                            provider = line.split("_API_KEY")[0].strip().lower()
                            providers.add(provider)
            except (IOError, OSError):
                pass
        
        # Also check for OAuth providers from files
        oauth_dir = Path("oauth_credentials")
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
        self.console.clear()
        
        self.console.print(Panel.fit(
            "[bold cyan]🔧 Advanced Settings Configuration[/bold cyan]",
            border_style="cyan"
        ))
        
        self.console.print()
        self.console.print("[bold]⚙️  Configuration Categories[/bold]")
        self.console.print()
        self.console.print("   1. 🌐 Custom Provider API Bases")
        self.console.print("   2. 📦 Provider Model Definitions")
        self.console.print("   3. ⚡ Concurrency Limits")
        self.console.print("   4. 💾 Save & Exit")
        self.console.print("   5. 🚫 Exit Without Saving")
        
        self.console.print()
        self.console.print("━" * 70)
        
        if self.settings.has_pending():
            self.console.print("[yellow]ℹ️  Changes are pending until you select \"Save & Exit\"[/yellow]")
        else:
            self.console.print("[dim]ℹ️  No pending changes[/dim]")
        
        self.console.print()
        self.console.print("[dim]⚠️  Model filters not supported - edit .env for IGNORE_MODELS_* / WHITELIST_MODELS_*[/dim]")
        self.console.print()
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], show_choices=False)
        
        if choice == "1":
            self.manage_custom_providers()
        elif choice == "2":
            self.manage_model_definitions()
        elif choice == "3":
            self.manage_concurrency_limits()
        elif choice == "4":
            self.save_and_exit()
        elif choice == "5":
            self.exit_without_saving()
    
    def manage_custom_providers(self):
        """Manage custom provider API bases"""
        while True:
            self.console.clear()
            
            providers = self.provider_mgr.get_current_providers()
            
            self.console.print(Panel.fit(
                "[bold cyan]🌐 Custom Provider API Bases[/bold cyan]",
                border_style="cyan"
            ))
            
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
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4"], show_choices=False)
            
            if choice == "1":
                name = Prompt.ask("Provider name (e.g., 'opencode')").strip().lower()
                if name:
                    api_base = Prompt.ask("API Base URL").strip()
                    if api_base:
                        self.provider_mgr.add_provider(name, api_base)
                        self.console.print(f"\n[green]✅ Custom provider '{name}' configured![/green]")
                        self.console.print(f"   To use: set {name.upper()}_API_KEY in credentials")
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
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(providers_list) + 1)])
                name = providers_list[choice_idx - 1]
                current_base = providers.get(name, "")
                
                self.console.print(f"\nCurrent API Base: {current_base}")
                new_base = Prompt.ask("New API Base [press Enter to keep current]", default=current_base).strip()
                
                if new_base and new_base != current_base:
                    self.provider_mgr.edit_provider(name, new_base)
                    self.console.print(f"\n[green]✅ Custom provider '{name}' updated![/green]")
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
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(providers_list) + 1)])
                name = providers_list[choice_idx - 1]
                
                if Confirm.ask(f"Remove '{name}'?"):
                    self.provider_mgr.remove_provider(name)
                    self.console.print(f"\n[green]✅ Provider '{name}' removed![/green]")
                    input("\nPress Enter to continue...")
            
            elif choice == "4":
                break
    
    def manage_model_definitions(self):
        """Manage provider model definitions"""
        while True:
            self.console.clear()
            
            all_providers = self.model_mgr.get_all_providers_with_models()
            
            self.console.print(Panel.fit(
                "[bold cyan]📦 Provider Model Definitions[/bold cyan]",
                border_style="cyan"
            ))
            
            self.console.print()
            self.console.print("[bold]📋 Configured Provider Models[/bold]")
            self.console.print("━" * 70)
            
            if all_providers:
                for provider, count in all_providers.items():
                    self.console.print(f"   • {provider:15} {count} model{'s' if count > 1 else ''}")
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
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], show_choices=False)
            
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
                self.console.print("\n[bold]Select provider to remove models from:[/bold]")
                providers_list = list(all_providers.keys())
                for idx, prov in enumerate(providers_list, 1):
                    self.console.print(f"   {idx}. {prov}")
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(providers_list) + 1)])
                provider = providers_list[choice_idx - 1]
                
                if Confirm.ask(f"Remove all model definitions for '{provider}'?"):
                    self.model_mgr.remove_models(provider)
                    self.console.print(f"\n[green]✅ Model definitions removed for '{provider}'![/green]")
                    input("\nPress Enter to continue...")
            elif choice == "5":
                break
    
    def add_model_definitions(self):
        """Add model definitions for a provider"""
        # Get available providers from credentials
        available_providers = self.get_available_providers()
        
        if not available_providers:
            self.console.print("\n[yellow]No providers with credentials found. Please add credentials first.[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        # Show provider selection menu
        self.console.print("\n[bold]Select provider:[/bold]")
        for idx, prov in enumerate(available_providers, 1):
            self.console.print(f"   {idx}. {prov}")
        self.console.print(f"   {len(available_providers) + 1}. Enter custom provider name")
        
        choice = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(available_providers) + 2)])
        
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
                    model_id = Prompt.ask(f"Model ID [press Enter to use '{name}']", default=name).strip()
                    if model_id and model_id != name:
                        model_def["id"] = model_id
                    
                    # Optional: model options
                    if Confirm.ask("Add model options (e.g., temperature limits)?", default=False):
                        self.console.print("\nEnter options as key=value pairs (one per line, 'done' to finish):")
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
            self.console.print(f"\n[green]✅ Model definitions saved for '{provider}'![/green]")
        else:
            self.console.print("\n[yellow]No models added[/yellow]")
        
        input("\nPress Enter to continue...")
    
    def edit_model_definitions(self, providers: List[str]):
        """Edit existing model definitions"""
        # Show numbered list
        self.console.print("\n[bold]Select provider to edit:[/bold]")
        for idx, prov in enumerate(providers, 1):
            self.console.print(f"   {idx}. {prov}")
        
        choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(providers) + 1)])
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
            self.console.clear()
            self.console.print(f"[bold]Editing models for: {provider}[/bold]\n")
            self.console.print("Current models:")
            for i, (name, definition) in enumerate(current_models.items(), 1):
                model_id = definition.get("id", name) if isinstance(definition, dict) else name
                self.console.print(f"   {i}. {name} (ID: {model_id})")
            
            self.console.print("\nOptions:")
            self.console.print("   1. Add new model")
            self.console.print("   2. Edit existing model")
            self.console.print("   3. Remove model")
            self.console.print("   4. Done")
            
            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4"], show_choices=False)
            
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
                
                model_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(models_list) + 1)])
                name = models_list[model_idx - 1]
                
                current_def = current_models[name]
                current_id = current_def.get("id", name) if isinstance(current_def, dict) else name
                
                new_id = Prompt.ask("Model ID", default=current_id).strip()
                current_models[name] = {"id": new_id} if new_id != name else {}
            
            elif choice == "3":
                # Show numbered list
                models_list = list(current_models.keys())
                self.console.print("\n[bold]Select model to remove:[/bold]")
                for idx, model_name in enumerate(models_list, 1):
                    self.console.print(f"   {idx}. {model_name}")
                
                model_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(models_list) + 1)])
                name = models_list[model_idx - 1]
                
                if Confirm.ask(f"Remove '{name}'?"):
                    del current_models[name]
            
            elif choice == "4":
                break
        
        if current_models:
            self.model_mgr.set_models(provider, current_models)
            self.console.print(f"\n[green]✅ Models updated for '{provider}'![/green]")
        else:
            self.console.print("\n[yellow]No models left - removing definition[/yellow]")
            self.model_mgr.remove_models(provider)
        
        input("\nPress Enter to continue...")
    
    def view_model_definitions(self, providers: List[str]):
        """View model definitions for a provider"""
        # Show numbered list
        self.console.print("\n[bold]Select provider to view:[/bold]")
        for idx, prov in enumerate(providers, 1):
            self.console.print(f"   {idx}. {prov}")
        
        choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(providers) + 1)])
        provider = providers[choice_idx - 1]
        
        models = self.model_mgr.get_current_provider_models(provider)
        if not models:
            self.console.print(f"\n[yellow]No models found for '{provider}'[/yellow]")
            input("\nPress Enter to continue...")
            return
        
        self.console.clear()
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
    
    def manage_concurrency_limits(self):
        """Manage concurrency limits"""
        while True:
            self.console.clear()
            
            limits = self.concurrency_mgr.get_current_limits()
            
            self.console.print(Panel.fit(
                "[bold cyan]⚡ Concurrency Limits Configuration[/bold cyan]",
                border_style="cyan"
            ))
            
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
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4"], show_choices=False)
            
            if choice == "1":
                # Get available providers
                available_providers = self.get_available_providers()
                
                if not available_providers:
                    self.console.print("\n[yellow]No providers with credentials found. Please add credentials first.[/yellow]")
                    input("\nPress Enter to continue...")
                    continue
                
                # Show provider selection menu
                self.console.print("\n[bold]Select provider:[/bold]")
                for idx, prov in enumerate(available_providers, 1):
                    self.console.print(f"   {idx}. {prov}")
                self.console.print(f"   {len(available_providers) + 1}. Enter custom provider name")
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(available_providers) + 2)])
                
                if choice_idx == len(available_providers) + 1:
                    provider = Prompt.ask("Provider name").strip().lower()
                else:
                    provider = available_providers[choice_idx - 1]
                
                if provider:
                    limit = IntPrompt.ask("Max concurrent requests per key (1-100)", default=1)
                    if 1 <= limit <= 100:
                        self.concurrency_mgr.set_limit(provider, limit)
                        self.console.print(f"\n[green]✅ Concurrency limit set for '{provider}': {limit} requests/key[/green]")
                    else:
                        self.console.print("\n[red]❌ Limit must be between 1-100[/red]")
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
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(limits_list) + 1)])
                provider = limits_list[choice_idx - 1]
                current_limit = limits.get(provider, 1)
                
                self.console.print(f"\nCurrent limit: {current_limit} requests/key")
                new_limit = IntPrompt.ask("New limit (1-100) [press Enter to keep current]", default=current_limit)
                
                if 1 <= new_limit <= 100:
                    if new_limit != current_limit:
                        self.concurrency_mgr.set_limit(provider, new_limit)
                        self.console.print(f"\n[green]✅ Concurrency limit updated for '{provider}': {new_limit} requests/key[/green]")
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
                self.console.print("\n[bold]Select provider to remove limit from:[/bold]")
                limits_list = list(limits.keys())
                for idx, prov in enumerate(limits_list, 1):
                    self.console.print(f"   {idx}. {prov}")
                
                choice_idx = IntPrompt.ask("Select option", choices=[str(i) for i in range(1, len(limits_list) + 1)])
                provider = limits_list[choice_idx - 1]
                
                if Confirm.ask(f"Remove concurrency limit for '{provider}' (reset to default 1)?"):
                    self.concurrency_mgr.remove_limit(provider)
                    self.console.print(f"\n[green]✅ Limit removed for '{provider}' - using default (1 request/key)[/green]")
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
