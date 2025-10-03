# src/rotator_library/credential_tool.py

import asyncio
import json
import re
from pathlib import Path

from .provider_factory import get_provider_auth_class, get_available_providers
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)

console = Console()

async def setup_new_credential(provider_name: str):
    """
    Interactively sets up a new OAuth credential for a given provider.
    """
    try:
        auth_class = get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        temp_creds = {}
        initialized_creds = await auth_instance.initialize_token(temp_creds)
        
        user_info = await auth_instance.get_user_info(initialized_creds)
        email = user_info.get("email")

        if not email:
            console.print(Panel(f"Could not retrieve a unique identifier for {provider_name}. Aborting.", style="bold red", title="Error"))
            return

        for cred_file in OAUTH_BASE_DIR.glob(f"{provider_name}_oauth_*.json"):
            with open(cred_file, 'r') as f:
                existing_creds = json.load(f)
            
            metadata = existing_creds.get("_proxy_metadata", {})
            if metadata.get("email") == email:
                error_text = Text.from_markup(f"An existing credential for [bold cyan]'{email}'[/bold cyan] already exists at [bold yellow]'{cred_file.name}'[/bold yellow]. Aborting.")
                console.print(Panel(error_text, style="bold red", title="Duplicate Credential"))
                return

        existing_files = list(OAUTH_BASE_DIR.glob(f"{provider_name}_oauth_*.json"))
        next_num = 1
        if existing_files:
            nums = [int(re.search(r'_(\d+)\.json$', f.name).group(1)) for f in existing_files if re.search(r'_(\d+)\.json$', f.name)]
            if nums:
                next_num = max(nums) + 1
        
        new_filename = f"{provider_name}_oauth_{next_num}.json"
        new_filepath = OAUTH_BASE_DIR / new_filename

        with open(new_filepath, 'w') as f:
            json.dump(initialized_creds, f, indent=2)

        success_text = Text.from_markup(f"Successfully created new credential at [bold yellow]'{new_filepath.name}'[/bold yellow] for user [bold cyan]'{email}'[/bold cyan].")
        console.print(Panel(success_text, style="bold green", title="Success"))

    except Exception as e:
        console.print(Panel(f"An error occurred during setup for {provider_name}: {e}", style="bold red", title="Error"))


async def main():
    """
    An interactive CLI tool to add new OAuth credentials.
    """
    console.print(Panel("[bold cyan]Interactive Credential Setup[/bold cyan]", title="--- API Key Proxy ---", expand=False))
    
    while True:
        available_providers = get_available_providers()
        
        provider_text = Text()
        for i, provider in enumerate(available_providers):
            provider_text.append(f"  {i + 1}. {provider.capitalize()}\n")
        
        console.print(Panel(provider_text, title="Available Providers", style="bold blue"))

        choice = Prompt.ask(
            Text.from_markup("[bold]Please select a provider or type [red]'q'[/red] to quit[/bold]"),
            choices=[str(i + 1) for i in range(len(available_providers))] + ["q"],
            show_choices=False
        )

        if choice.lower() == 'q':
            break
        
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(available_providers):
                provider_name = available_providers[choice_index]
                console.print(f"\nStarting setup for [bold cyan]{provider_name.capitalize()}[/bold cyan]...")
                await setup_new_credential(provider_name)
            else:
                console.print("[bold red]Invalid choice. Please try again.[/bold red]")
        except ValueError:
            console.print("[bold red]Invalid input. Please enter a number or 'q'.[/bold red]")
        
        console.print("\n" + "="*50 + "\n")

def run_credential_tool():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting setup.[/bold yellow]")