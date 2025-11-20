"""
Interactive TUI launcher for the LLM API Key Proxy.
Provides a beautiful Rich-based interface for configuration and execution.
"""

import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv, set_key

console = Console()


class LauncherConfig:
    """Manages launcher_config.json (host, port, logging only)"""
    
    def __init__(self, config_path: Path = Path("launcher_config.json")):
        self.config_path = config_path
        self.defaults = {
            "host": "127.0.0.1",
            "port": 8000,
            "enable_request_logging": False
        }
        self.config = self.load()
    
    def load(self) -> dict:
        """Load config from file or create with defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in self.defaults.items():
                    if key not in config:
                        config[key] = value
                return config
            except (json.JSONDecodeError, IOError):
                return self.defaults.copy()
        return self.defaults.copy()
    
    def save(self):
        """Save current config to file."""
        import datetime
        self.config["last_updated"] = datetime.datetime.now().isoformat()
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    def update(self, **kwargs):
        """Update config values."""
        self.config.update(kwargs)
        self.save()
    
    @staticmethod
    def update_proxy_api_key(new_key: str):
        """Update PROXY_API_KEY in .env only"""
        env_file = Path.cwd() / ".env"
        set_key(str(env_file), "PROXY_API_KEY", new_key)
        load_dotenv(dotenv_path=env_file, override=True)


class SettingsDetector:
    """Detects settings from .env for display"""
    
    @staticmethod
    def _load_local_env() -> dict:
        """Load environment variables from local .env file only"""
        env_file = Path.cwd() / ".env"
        env_dict = {}
        if not env_file.exists():
            return env_dict
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key, value = key.strip(), value.strip()
                        if value and value[0] in ('"', "'") and value[-1] == value[0]:
                            value = value[1:-1]
                        env_dict[key] = value
        except (IOError, OSError):
            pass
        return env_dict

    @staticmethod
    def get_all_settings() -> dict:
        """Returns comprehensive settings overview"""
        return {
            "credentials": SettingsDetector.detect_credentials(),
            "custom_bases": SettingsDetector.detect_custom_api_bases(),
            "model_definitions": SettingsDetector.detect_model_definitions(),
            "concurrency_limits": SettingsDetector.detect_concurrency_limits(),
            "model_filters": SettingsDetector.detect_model_filters()
        }
    
    @staticmethod
    def detect_credentials() -> dict:
        """Detect API keys and OAuth credentials"""
        from pathlib import Path
        
        providers = {}
        
        # Scan for API keys
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if "_API_KEY" in key and key != "PROXY_API_KEY":
                provider = key.split("_API_KEY")[0].lower()
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["api_keys"] += 1
        
        # Scan for OAuth credentials
        oauth_dir = Path("oauth_credentials")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["oauth"] += 1
        
        # Mark custom providers (have API_BASE set)
        for provider in providers:
            if os.getenv(f"{provider.upper()}_API_BASE"):
                providers[provider]["custom"] = True
        
        return providers
    
    @staticmethod
    def detect_custom_api_bases() -> dict:
        """Detect custom API base URLs (not in hardcoded map)"""
        from proxy_app.provider_urls import PROVIDER_URL_MAP
        
        bases = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_API_BASE"):
                provider = key.replace("_API_BASE", "").lower()
                # Only include if NOT in hardcoded map
                if provider not in PROVIDER_URL_MAP:
                    bases[provider] = value
        return bases
    
    @staticmethod
    def detect_model_definitions() -> dict:
        """Detect provider model definitions"""
        models = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_MODELS"):
                provider = key.replace("_MODELS", "").lower()
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        models[provider] = len(parsed)
                    elif isinstance(parsed, list):
                        models[provider] = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
        return models
    
    @staticmethod
    def detect_concurrency_limits() -> dict:
        """Detect max concurrent requests per key"""
        limits = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
                provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
                try:
                    limits[provider] = int(value)
                except (json.JSONDecodeError, ValueError):
                    pass
        return limits
    
    @staticmethod
    def detect_model_filters() -> dict:
        """Detect active model filters (basic info only: defined or not)"""
        filters = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("IGNORE_MODELS_") or key.startswith("WHITELIST_MODELS_"):
                filter_type = "ignore" if key.startswith("IGNORE") else "whitelist"
                provider = key.replace(f"{filter_type.upper()}_MODELS_", "").lower()
                if provider not in filters:
                    filters[provider] = {"has_ignore": False, "has_whitelist": False}
                if filter_type == "ignore":
                    filters[provider]["has_ignore"] = True
                else:
                    filters[provider]["has_whitelist"] = True
        return filters


class LauncherTUI:
    """Main launcher interface"""
    
    def __init__(self):
        self.console = Console()
        self.config = LauncherConfig()
        self.running = True
        self.env_file = Path.cwd() / ".env"
    
    def needs_onboarding(self) -> bool:
        """Check if onboarding is needed"""
        return not self.env_file.exists() or not os.getenv("PROXY_API_KEY")
    
    def run(self):
        """Main TUI loop"""
        while self.running:
            self.show_main_menu()
    
    def show_main_menu(self):
        """Display main menu and handle selection"""
        self.console.clear()
        
        # Detect all settings
        settings = SettingsDetector.get_all_settings()
        credentials = settings["credentials"]
        custom_bases = settings["custom_bases"]
        
        # Check if setup is needed
        show_warning = self.needs_onboarding()
        
        # Build title
        self.console.print(Panel.fit(
            "[bold cyan]🚀 LLM API Key Proxy - Interactive Launcher[/bold cyan]",
            border_style="cyan"
        ))
        
        # Show warning if needed
        if show_warning:
            self.console.print()
            self.console.print(Panel(
                Text.from_markup(
                    "⚠️  [bold yellow]CONFIGURATION REQUIRED[/bold yellow]\n\n"
                    "The proxy cannot start because:\n"
                    "  ❌ No .env file found (or)\n"
                    "  ❌ PROXY_API_KEY is not set in .env\n\n"
                    "Why this matters:\n"
                    "  • The .env file stores your proxy's authentication key\n"
                    "  • The PROXY_API_KEY protects your proxy from unauthorized access\n"
                    "  • Without it, the proxy cannot securely start\n\n"
                    "What to do:\n"
                    "  1. Select option \"3. Manage Credentials\" to launch the credential tool\n"
                    "  2. The tool will create .env and set up PROXY_API_KEY automatically\n"
                    "  3. You can also add LLM provider credentials while you're there\n\n"
                    "⚠️  Important: While provider credentials are optional for startup,\n"
                    "   the proxy won't do anything useful without them. See README.md\n"
                    "   for supported providers and setup instructions."
                ),
                border_style="yellow",
                expand=False
            ))
        
        # Show config
        self.console.print()
        self.console.print("[bold]📋 Proxy Configuration[/bold]")
        self.console.print("━" * 70)
        self.console.print(f"   Host:                {self.config.config['host']}")
        self.console.print(f"   Port:                {self.config.config['port']}")
        self.console.print(f"   Request Logging:     {'✅ Enabled' if self.config.config['enable_request_logging'] else '❌ Disabled'}")
        self.console.print(f"   Proxy API Key:       {'✅ Set' if os.getenv('PROXY_API_KEY') else '❌ Not Set'}")
        
        # Show status summary
        self.console.print()
        self.console.print("[bold]📊 Status Summary[/bold]")
        self.console.print("━" * 70)
        provider_count = len(credentials)
        custom_count = len(custom_bases)
        has_advanced = bool(settings["model_definitions"] or settings["concurrency_limits"] or settings["model_filters"])
        
        self.console.print(f"   Providers:           {provider_count} configured")
        self.console.print(f"   Custom Providers:    {custom_count} configured")
        self.console.print(f"   Advanced Settings:   {'Active (view in menu 4)' if has_advanced else 'None'}")
        
        # Show menu
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("[bold]🎯 Main Menu[/bold]")
        self.console.print()
        
        if show_warning:
            self.console.print("   1. ▶️  Run Proxy Server")
            self.console.print("   2. ⚙️  Configure Proxy Settings")
            self.console.print("   3. 🔑 Manage Credentials            ⬅️  [bold yellow]Start here![/bold yellow]")
        else:
            self.console.print("   1. ▶️  Run Proxy Server")
            self.console.print("   2. ⚙️  Configure Proxy Settings")
            self.console.print("   3. 🔑 Manage Credentials")
        
        self.console.print("   4. 📊 View Provider & Advanced Settings")
        self.console.print("   5. 🔄 Reload Configuration")
        self.console.print("   6. 🚪 Exit")
        
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"], show_choices=False)
        
        if choice == "1":
            self.run_proxy()
        elif choice == "2":
            self.show_config_menu()
        elif choice == "3":
            self.launch_credential_tool()
        elif choice == "4":
            self.show_provider_settings_menu()
        elif choice == "5":
            load_dotenv(dotenv_path=Path.cwd() / ".env",override=True)
            self.config = LauncherConfig()  # Reload config
            self.console.print("\n[green]✅ Configuration reloaded![/green]")
        elif choice == "6":
            self.running = False
            sys.exit(0)
    
    def show_config_menu(self):
        """Display configuration sub-menu"""
        while True:
            self.console.clear()
            
            self.console.print(Panel.fit(
                "[bold cyan]⚙️  Proxy Configuration[/bold cyan]",
                border_style="cyan"
            ))
            
            self.console.print()
            self.console.print("[bold]📋 Current Settings[/bold]")
            self.console.print("━" * 70)
            self.console.print(f"   Host:                {self.config.config['host']}")
            self.console.print(f"   Port:                {self.config.config['port']}")
            self.console.print(f"   Request Logging:     {'✅ Enabled' if self.config.config['enable_request_logging'] else '❌ Disabled'}")
            self.console.print(f"   Proxy API Key:       {'✅ Set' if os.getenv('PROXY_API_KEY') else '❌ Not Set'}")
            
            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            self.console.print("[bold]⚙️  Configuration Options[/bold]")
            self.console.print()
            self.console.print("   1. 🌐 Set Host IP")
            self.console.print("   2. 🔌 Set Port")
            self.console.print("   3. 🔑 Set Proxy API Key")
            self.console.print("   4. 📝 Toggle Request Logging")
            self.console.print("   5. ↩️  Back to Main Menu")
            
            self.console.print()
            self.console.print("━" * 70)
            self.console.print()
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], show_choices=False)
            
            if choice == "1":
                new_host = Prompt.ask("Enter new host IP", default=self.config.config["host"])
                self.config.update(host=new_host)
                self.console.print(f"\n[green]✅ Host updated to: {new_host}[/green]")
            elif choice == "2":
                new_port = IntPrompt.ask("Enter new port", default=self.config.config["port"])
                if 1 <= new_port <= 65535:
                    self.config.update(port=new_port)
                    self.console.print(f"\n[green]✅ Port updated to: {new_port}[/green]")
                else:
                    self.console.print("\n[red]❌ Port must be between 1-65535[/red]")
            elif choice == "3":
                current = os.getenv("PROXY_API_KEY", "")
                new_key = Prompt.ask("Enter new Proxy API Key", default=current)
                if new_key and new_key != current:
                    LauncherConfig.update_proxy_api_key(new_key)
                    self.console.print("\n[green]✅ Proxy API Key updated successfully![/green]")
                    self.console.print("   Updated in .env file")
                else:
                    self.console.print("\n[yellow]No changes made[/yellow]")
            elif choice == "4":
                current = self.config.config["enable_request_logging"]
                self.config.update(enable_request_logging=not current)
                self.console.print(f"\n[green]✅ Request Logging {'enabled' if not current else 'disabled'}![/green]")
            elif choice == "5":
                break
    
    def show_provider_settings_menu(self):
        """Display provider/advanced settings (read-only + launch tool)"""
        self.console.clear()
        
        settings = SettingsDetector.get_all_settings()
        credentials = settings["credentials"]
        custom_bases = settings["custom_bases"]
        model_defs = settings["model_definitions"]
        concurrency = settings["concurrency_limits"]
        filters = settings["model_filters"]
        
        self.console.print(Panel.fit(
            "[bold cyan]📊 Provider & Advanced Settings[/bold cyan]",
            border_style="cyan"
        ))
        
        # Configured Providers
        self.console.print()
        self.console.print("[bold]📊 Configured Providers[/bold]")
        self.console.print("━" * 70)
        if credentials:
            for provider, info in credentials.items():
                provider_name = provider.title()
                parts = []
                if info["api_keys"] > 0:
                    parts.append(f"{info['api_keys']} API key{'s' if info['api_keys'] > 1 else ''}")
                if info["oauth"] > 0:
                    parts.append(f"{info['oauth']} OAuth credential{'s' if info['oauth'] > 1 else ''}")
                
                display = " + ".join(parts)
                if info["custom"]:
                    display += " (Custom)"
                
                self.console.print(f"   ✅ {provider_name:20} {display}")
        else:
            self.console.print("   [dim]No providers configured[/dim]")
        
        # Custom API Bases
        if custom_bases:
            self.console.print()
            self.console.print("[bold]🌐 Custom API Bases[/bold]")
            self.console.print("━" * 70)
            for provider, base in custom_bases.items():
                self.console.print(f"   • {provider:15} {base}")
        
        # Model Definitions
        if model_defs:
            self.console.print()
            self.console.print("[bold]📦 Provider Model Definitions[/bold]")
            self.console.print("━" * 70)
            for provider, count in model_defs.items():
                self.console.print(f"   • {provider:15} {count} model{'s' if count > 1 else ''} configured")
        
        # Concurrency Limits
        if concurrency:
            self.console.print()
            self.console.print("[bold]⚡ Concurrency Limits[/bold]")
            self.console.print("━" * 70)
            for provider, limit in concurrency.items():
                self.console.print(f"   • {provider:15} {limit} requests/key")
            self.console.print(f"   • Default:        1 request/key (all others)")
        
        # Model Filters (basic info only)
        if filters:
            self.console.print()
            self.console.print("[bold]🎯 Model Filters[/bold]")
            self.console.print("━" * 70)
            for provider, filter_info in filters.items():
                status_parts = []
                if filter_info["has_whitelist"]:
                    status_parts.append("Whitelist")
                if filter_info["has_ignore"]:
                    status_parts.append("Ignore list")
                status = " + ".join(status_parts) if status_parts else "None"
                self.console.print(f"   • {provider:15} ✅ {status}")
        
        # Actions
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("[bold]💡 Actions[/bold]")
        self.console.print()
        self.console.print("   1. 🔧 Launch Settings Tool      (configure advanced settings)")
        self.console.print("   2. ↩️  Back to Main Menu")
        
        self.console.print()
        self.console.print("━" * 70)
        self.console.print("[dim]ℹ️  Advanced settings are stored in .env file.\n   Use the Settings Tool to configure them interactively.[/dim]")
        self.console.print()
        self.console.print("[dim]⚠️  Note: Settings Tool supports only common configuration types.\n   For complex settings, edit .env directly.[/dim]")
        self.console.print()
        
        choice = Prompt.ask("Select option", choices=["1", "2"], show_choices=False)
        
        if choice == "1":
            self.launch_settings_tool()
        # choice == "2" returns to main menu
    
    def launch_credential_tool(self):
        """Launch credential management tool"""
        from rotator_library.credential_tool import run_credential_tool
        run_credential_tool()
        # Reload environment after credential tool
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
    
    def launch_settings_tool(self):
        """Launch settings configuration tool"""
        from proxy_app.settings_tool import run_settings_tool
        run_settings_tool()
        # Reload environment after settings tool
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
    
    def run_proxy(self):
        """Prepare and launch proxy in same window"""
        # Check if forced onboarding needed
        if self.needs_onboarding():
            self.console.clear()
            self.console.print(Panel(
                Text.from_markup(
                    "⚠️  [bold yellow]Setup Required[/bold yellow]\n\n"
                    "Cannot start without .env and PROXY_API_KEY.\n"
                    "Launching credential tool..."
                ),
                border_style="yellow"
            ))
            
            # Force credential tool
            from rotator_library.credential_tool import ensure_env_defaults, run_credential_tool
            ensure_env_defaults()
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
            run_credential_tool()
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
            
            # Check again after credential tool
            if not os.getenv("PROXY_API_KEY"):
                self.console.print("\n[red]❌ PROXY_API_KEY still not set. Cannot start proxy.[/red]")
                return
        
        # Clear console and modify sys.argv
        self.console.clear()
        self.console.print(f"\n[bold green]🚀 Starting proxy on {self.config.config['host']}:{self.config.config['port']}...[/bold green]\n")
        
        # Reconstruct sys.argv for main.py
        sys.argv = [
            "main.py",
            "--host", self.config.config["host"],
            "--port", str(self.config.config["port"])
        ]
        if self.config.config["enable_request_logging"]:
            sys.argv.append("--enable-request-logging")
        
        # Exit TUI - main.py will continue execution
        self.running = False


def run_launcher_tui():
    """Entry point for launcher TUI"""
    tui = LauncherTUI()
    tui.run()
