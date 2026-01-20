# src/rotator_library/credential_tool.py

import asyncio
import json
import os
import re
import time
from pathlib import Path
from dotenv import set_key, get_key

# NOTE: Heavy imports (provider_factory, PROVIDER_PLUGINS) are deferred
# to avoid 6-7 second delay before showing loading screen
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from .utils.paths import get_oauth_dir, get_data_file
from .provider_config import LITELLM_PROVIDERS, PROVIDER_CATEGORIES, PROVIDER_BLACKLIST
from .litellm_providers import (
    SCRAPED_PROVIDERS,
    get_provider_api_key_var,
    get_provider_display_name,
)


def _get_oauth_base_dir() -> Path:
    """Get the OAuth base directory (lazy, respects EXE vs script mode)."""
    oauth_dir = get_oauth_dir()
    oauth_dir.mkdir(parents=True, exist_ok=True)
    return oauth_dir


def _get_env_file() -> Path:
    """Get the .env file path (lazy, respects EXE vs script mode)."""
    return get_data_file(".env")


console = Console()

# Global variables for lazily loaded modules
_provider_factory = None
_provider_plugins = None


def _ensure_providers_loaded():
    """Lazy load provider modules only when needed"""
    global _provider_factory, _provider_plugins
    if _provider_factory is None:
        from . import provider_factory as pf
        from .providers import PROVIDER_PLUGINS as pp

        _provider_factory = pf
        _provider_plugins = pp
    return _provider_factory, _provider_plugins


# OAuth provider display names mapping (no "(OAuth)" suffix - context makes it clear)
OAUTH_FRIENDLY_NAMES = {
    "gemini_cli": "Gemini CLI",
    "qwen_code": "Qwen Code",
    "iflow": "iFlow",
    "antigravity": "Antigravity",
}


def _extract_key_number(key_name: str) -> int:
    """Extract the numeric suffix from a key name for proper sorting.

    Examples:
        GEMINI_API_KEY_1 -> 1
        GEMINI_API_KEY_10 -> 10
        GEMINI_API_KEY -> 0
    """
    match = re.search(r"_(\d+)$", key_name)
    return int(match.group(1)) if match else 0


def _normalize_tier_name(tier: str) -> str:
    """Normalize tier names for consistent display.

    Examples:
        "free-tier" -> "free"
        "FREE_TIER" -> "free"
        "PAID" -> "paid"
        "standard" -> "standard"
        None -> "unknown"
    """
    if not tier:
        return "unknown"

    # Lowercase and remove common suffixes/prefixes
    normalized = tier.lower().strip()
    normalized = normalized.replace("-tier", "").replace("_tier", "")
    normalized = normalized.replace("-", "").replace("_", "")

    return normalized


def _count_tiers(credentials: list) -> dict:
    """Count credentials by tier.

    Args:
        credentials: List of credential info dicts with optional 'tier' key

    Returns:
        Dict mapping normalized tier names to counts, e.g. {"free": 15, "paid": 2}
    """
    tier_counts = {}
    for cred in credentials:
        tier = cred.get("tier")
        if tier:
            normalized = _normalize_tier_name(tier)
            tier_counts[normalized] = tier_counts.get(normalized, 0) + 1
    return tier_counts


def _format_tier_counts(tier_counts: dict) -> str:
    """Format tier counts as a compact string.

    Examples:
        {"free": 15, "paid": 2} -> "(15 free, 2 paid)"
        {"free": 5} -> "(5 free)"
        {} -> ""
    """
    if not tier_counts:
        return ""

    # Sort by count descending, then alphabetically
    sorted_tiers = sorted(tier_counts.items(), key=lambda x: (-x[1], x[0]))
    parts = [f"{count} {tier}" for tier, count in sorted_tiers]
    return f"({', '.join(parts)})"


def _get_api_keys_from_env() -> dict:
    """
    Parse the .env file and return a dictionary of API keys grouped by provider.
    Keys are sorted numerically within each provider.

    Returns:
        Dict mapping provider names to lists of (key_name, key_value) tuples.
        Example: {"GEMINI": [("GEMINI_API_KEY_1", "abc123"), ("GEMINI_API_KEY_2", "def456")]}
    """
    api_keys = {}
    env_file = _get_env_file()

    if not env_file.is_file():
        return api_keys

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Look for lines with API_KEY pattern
                if "_API_KEY" in line and "=" in line:
                    key_name, _, key_value = line.partition("=")
                    key_name = key_name.strip()
                    key_value = key_value.strip().strip('"').strip("'")

                    # Skip PROXY_API_KEY and empty values
                    if key_name == "PROXY_API_KEY" or not key_value:
                        continue

                    # Skip placeholder values
                    if key_value.startswith("YOUR_") or key_value == "":
                        continue

                    # Extract provider name (everything before _API_KEY)
                    # Handle cases like GEMINI_API_KEY_1 -> GEMINI
                    parts = key_name.split("_API_KEY")
                    if parts:
                        provider_name = parts[0]
                        if provider_name not in api_keys:
                            api_keys[provider_name] = []
                        api_keys[provider_name].append((key_name, key_value))

        # Sort keys numerically within each provider
        for provider_name in api_keys:
            api_keys[provider_name].sort(key=lambda x: _extract_key_number(x[0]))

    except Exception as e:
        console.print(f"[bold red]Error reading .env file: {e}[/bold red]")

    return api_keys


def _delete_api_key_from_env(key_name: str) -> bool:
    """
    Delete an API key from the .env file with safety backup and comparison.

    This function creates a backup of all API keys before deletion,
    performs the deletion, and then verifies no unintended keys were lost.

    Args:
        key_name: The exact key name to delete (e.g., "GEMINI_API_KEY_2")

    Returns:
        True if deletion was successful and verified, False otherwise
    """
    env_file = _get_env_file()

    if not env_file.is_file():
        console.print("[bold red]Error: .env file not found[/bold red]")
        return False

    try:
        # Step 1: Read all lines and backup all API keys
        with open(env_file, "r") as f:
            original_lines = f.readlines()

        # Create backup of all API keys before modification
        api_keys_before = _get_api_keys_from_env()
        all_keys_before = set()
        for provider_keys in api_keys_before.values():
            for kn, kv in provider_keys:
                all_keys_before.add((kn, kv))

        # Step 2: Find and remove the target key
        new_lines = []
        key_found = False
        deleted_key_value = None

        for line in original_lines:
            stripped = line.strip()
            # Check if this line contains our target key
            if stripped.startswith(f"{key_name}="):
                key_found = True
                # Store the value being deleted for verification
                _, _, deleted_key_value = stripped.partition("=")
                deleted_key_value = deleted_key_value.strip().strip('"').strip("'")
                continue  # Skip this line (delete it)
            new_lines.append(line)

        if not key_found:
            console.print(
                f"[bold red]Error: Key '{key_name}' not found in .env file[/bold red]"
            )
            return False

        # Step 3: Write the modified content
        with open(env_file, "w") as f:
            f.writelines(new_lines)

        # Step 4: Verify the deletion - compare before and after
        api_keys_after = _get_api_keys_from_env()
        all_keys_after = set()
        for provider_keys in api_keys_after.values():
            for kn, kv in provider_keys:
                all_keys_after.add((kn, kv))

        # Check that only the intended key was removed
        expected_remaining = all_keys_before - {(key_name, deleted_key_value)}

        if all_keys_after != expected_remaining:
            # Something went wrong - restore from backup
            console.print(
                "[bold red]Error: Unexpected keys were affected during deletion![/bold red]"
            )
            console.print("[bold yellow]Restoring original file...[/bold yellow]")
            with open(env_file, "w") as f:
                f.writelines(original_lines)
            return False

        return True

    except Exception as e:
        console.print(f"[bold red]Error during API key deletion: {e}[/bold red]")
        return False


def _get_oauth_credentials_summary() -> dict:
    """
    Get a summary of all OAuth credentials for all providers.

    Returns:
        Dict mapping provider names to lists of credential info dicts.
        Example: {"gemini_cli": [{"email": "user@example.com", "tier": "free-tier", ...}, ...]}
    """
    provider_factory, _ = _ensure_providers_loaded()
    oauth_providers = ["gemini_cli", "qwen_code", "iflow", "antigravity"]
    oauth_summary = {}

    for provider_name in oauth_providers:
        try:
            auth_class = provider_factory.get_provider_auth_class(provider_name)
            auth_instance = auth_class()
            credentials = auth_instance.list_credentials(_get_oauth_base_dir())
            oauth_summary[provider_name] = credentials
        except Exception:
            oauth_summary[provider_name] = []

    return oauth_summary


def _get_all_credentials_summary() -> dict:
    """
    Get a complete summary of all credentials (API keys and OAuth).

    Returns:
        Dict with "api_keys" and "oauth" sections containing credential summaries.
    """
    return {
        "api_keys": _get_api_keys_from_env(),
        "oauth": _get_oauth_credentials_summary(),
    }


def _get_existing_custom_providers() -> list:
    """
    Scan the .env file for existing custom OpenAI-compatible providers.

    Custom providers are identified by *_API_BASE entries where the provider
    name is NOT a known LiteLLM provider.

    Returns:
        List of dicts with provider info:
        [{"name": "myserver", "api_base": "http://...", "has_key": True}, ...]
    """
    from .provider_config import KNOWN_PROVIDERS

    custom_providers = []
    env_file = _get_env_file()

    if not env_file.is_file():
        return custom_providers

    try:
        # First pass: collect all _API_BASE entries
        api_bases = {}
        api_keys = set()

        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key_name, _, value = line.partition("=")
                key_name = key_name.strip()
                value = value.strip().strip('"').strip("'")

                if key_name.endswith("_API_BASE") and value:
                    provider_name = key_name[:-9].lower()  # Remove _API_BASE
                    # Only include if NOT a known provider
                    if provider_name not in KNOWN_PROVIDERS:
                        api_bases[provider_name] = value
                elif "_API_KEY" in key_name and value:
                    # Extract provider name from API key
                    provider_prefix = key_name.split("_API_KEY")[0].lower()
                    api_keys.add(provider_prefix)

        # Build result list
        for provider_name, api_base in sorted(api_bases.items()):
            custom_providers.append(
                {
                    "name": provider_name,
                    "api_base": api_base,
                    "has_key": provider_name in api_keys,
                }
            )

    except Exception as e:
        console.print(f"[bold red]Error reading .env file: {e}[/bold red]")

    return custom_providers


def _display_custom_providers_summary():
    """
    Display a summary of existing custom OpenAI-compatible providers.
    """
    custom_providers = _get_existing_custom_providers()

    if not custom_providers:
        console.print(
            "[dim]No custom OpenAI-compatible providers configured yet.[/dim]\n"
        )
        return

    table = Table(
        title="Existing Custom Providers",
        box=None,
        padding=(0, 2),
        title_style="bold cyan",
    )
    table.add_column("Provider", style="yellow", no_wrap=True)
    table.add_column("API Base", style="dim")
    table.add_column("API Key", style="green", justify="center")

    for provider in custom_providers:
        name = provider["name"].upper()
        api_base = provider["api_base"]
        # Truncate long URLs
        if len(api_base) > 40:
            api_base = api_base[:37] + "..."
        has_key = "✓" if provider["has_key"] else "✗"
        key_style = "green" if provider["has_key"] else "red"
        table.add_row(name, api_base, Text(has_key, style=key_style))

    console.print(table)
    console.print()


def _display_credentials_summary():
    """
    Display a compact 2-column summary of all configured credentials.
    API Keys on the left, OAuth credentials on the right.
    Handles cases where only one type exists or neither.
    """
    from rich.columns import Columns

    summary = _get_all_credentials_summary()
    api_keys = summary["api_keys"]
    oauth_creds = summary["oauth"]

    # Calculate totals
    total_api_keys = sum(len(keys) for keys in api_keys.values())
    total_oauth = sum(len(creds) for creds in oauth_creds.values() if creds)

    # Handle empty case
    if total_api_keys == 0 and total_oauth == 0:
        console.print("[dim]No credentials configured yet.[/dim]\n")
        return

    # Build API Keys table (left column)
    api_table = None
    if total_api_keys > 0:
        api_table = Table(
            title="API Keys", box=None, padding=(0, 1), title_style="bold cyan"
        )
        api_table.add_column("Provider", style="yellow", no_wrap=True)
        api_table.add_column("Count", style="green", justify="right")

        for provider, keys in sorted(api_keys.items()):
            api_table.add_row(provider, str(len(keys)))

        # Add total row
        api_table.add_row("─" * 12, "─" * 5, style="dim")
        api_table.add_row("Total", str(total_api_keys), style="bold")

    # Build OAuth table (right column)
    oauth_table = None
    if total_oauth > 0:
        oauth_table = Table(
            title="OAuth Credentials", box=None, padding=(0, 1), title_style="bold cyan"
        )
        oauth_table.add_column("Provider", style="yellow", no_wrap=True)
        oauth_table.add_column("Count", style="green", justify="right")
        oauth_table.add_column("Tiers", style="dim", no_wrap=True)

        for provider, creds in sorted(oauth_creds.items()):
            if not creds:
                continue
            display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
            count = len(creds)

            # Count and format tiers for providers that have tier info
            tier_counts = _count_tiers(creds)
            tier_str = _format_tier_counts(tier_counts)

            oauth_table.add_row(display_name, str(count), tier_str)

        # Add total row
        oauth_table.add_row("─" * 12, "─" * 5, "", style="dim")
        oauth_table.add_row("Total", str(total_oauth), "", style="bold")

    # Display based on what's available
    if api_table and oauth_table:
        # Both columns - use Columns for side-by-side layout
        console.print(Columns([api_table, oauth_table], padding=(0, 4), expand=False))
    elif api_table:
        # Only API keys
        console.print(api_table)
    elif oauth_table:
        # Only OAuth
        console.print(oauth_table)

    console.print("")  # Blank line after summary


def _display_oauth_providers_summary():
    """
    Display a compact summary of OAuth providers only (used when adding OAuth credentials).
    """
    oauth_summary = _get_oauth_credentials_summary()

    total = sum(len(creds) for creds in oauth_summary.values())

    # Build compact table
    table = Table(
        title="Current OAuth Credentials",
        box=None,
        padding=(0, 1),
        title_style="bold cyan",
    )
    table.add_column("Provider", style="yellow", no_wrap=True)
    table.add_column("Count", style="green", justify="right")

    for provider, creds in sorted(oauth_summary.items()):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        table.add_row(display_name, str(len(creds)))

    if total > 0:
        table.add_row("─" * 12, "─" * 5, style="dim")
        table.add_row("Total", str(total), style="bold")

    console.print(table)
    console.print("")


def _display_provider_credentials(provider_name: str):
    """
    Display all credentials for a specific OAuth provider.

    Args:
        provider_name: The provider key (e.g., "gemini_cli", "qwen_code")
    """
    provider_factory, _ = _ensure_providers_loaded()

    try:
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = auth_instance.list_credentials(_get_oauth_base_dir())
    except Exception:
        credentials = []

    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

    if not credentials:
        console.print(f"\n[dim]No existing credentials for {display_name}[/dim]\n")
        return

    console.print(f"\n[bold cyan]Existing {display_name} Credentials:[/bold cyan]")

    table = Table(box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="yellow")
    table.add_column("Email/Identifier", style="cyan")

    # Add tier/project columns for Google OAuth providers
    if provider_name in ["gemini_cli", "antigravity"]:
        table.add_column("Tier", style="green")
        table.add_column("Project", style="dim")

    for i, cred in enumerate(credentials, 1):
        file_name = Path(cred["file_path"]).name
        email = cred.get("email", "unknown")

        if provider_name in ["gemini_cli", "antigravity"]:
            tier = cred.get("tier", "-")
            project = cred.get("project_id", "-")
            if project and len(project) > 20:
                project = project[:17] + "..."
            table.add_row(str(i), file_name, email, tier or "-", project or "-")
        else:
            table.add_row(str(i), file_name, email)

    console.print(table)
    console.print("")


async def _edit_oauth_credential_email(provider_name: str):
    """
    Edit the email field of an OAuth credential.

    Args:
        provider_name: The provider key (e.g., "qwen_code")
    """
    provider_factory, _ = _ensure_providers_loaded()

    try:
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = auth_instance.list_credentials(_get_oauth_base_dir())
    except Exception as e:
        console.print(f"[bold red]Error loading credentials: {e}[/bold red]")
        return

    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

    if not credentials:
        console.print(
            f"[bold yellow]No {display_name} credentials found.[/bold yellow]"
        )
        return

    # Display credentials for selection
    _display_provider_credentials(provider_name)

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Select credential to edit or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(credentials) + 1)] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        idx = int(choice) - 1
        cred_info = credentials[idx]
        cred_path = cred_info["file_path"]
        current_email = cred_info.get("email", "unknown")

        console.print(f"\nCurrent email: [cyan]{current_email}[/cyan]")
        new_email = Prompt.ask("Enter new email/identifier")

        if not new_email.strip():
            console.print("[bold yellow]No changes made (empty input).[/bold yellow]")
            return

        # Load and update the credential file
        with open(cred_path, "r") as f:
            creds = json.load(f)

        if "_proxy_metadata" not in creds:
            creds["_proxy_metadata"] = {}

        old_email = creds["_proxy_metadata"].get("email")
        creds["_proxy_metadata"]["email"] = new_email.strip()

        # Save the updated credentials
        with open(cred_path, "w") as f:
            json.dump(creds, f, indent=2)

        console.print(
            Panel(
                f"Email updated from [yellow]'{old_email}'[/yellow] to [green]'{new_email.strip()}'[/green]",
                style="bold green",
                title="Success",
                expand=False,
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error editing credential: {e}[/bold red]")


async def view_credentials_menu():
    """
    Menu for viewing credentials. Shows summary first, then allows drilling
    down to view detailed credentials for a specific provider.
    """
    while True:
        clear_screen("View Credentials")

        # Display summary
        _display_credentials_summary()

        # Build list of all providers with credentials
        api_keys = _get_api_keys_from_env()
        oauth_creds = _get_oauth_credentials_summary()

        all_providers = []

        # Add API key providers
        for provider in sorted(api_keys.keys()):
            count = len(api_keys[provider])
            all_providers.append(("api", provider, count))

        # Add OAuth providers with credentials
        for provider in sorted(oauth_creds.keys()):
            if oauth_creds[provider]:
                count = len(oauth_creds[provider])
                display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
                all_providers.append(("oauth", provider, count, display_name))

        if not all_providers:
            console.print("[bold yellow]No credentials configured.[/bold yellow]")
            console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()
            break

        # Display provider selection menu
        console.print(
            Panel(
                Text.from_markup("[bold]Select a provider to view details:[/bold]"),
                title="View Provider Credentials",
                style="bold blue",
            )
        )

        for i, provider_info in enumerate(all_providers, 1):
            if provider_info[0] == "api":
                _, provider, count = provider_info
                console.print(f"  {i}. [cyan]API:[/cyan] {provider} ({count} key(s))")
            else:
                _, provider, count, display_name = provider_info
                console.print(
                    f"  {i}. [cyan]OAuth:[/cyan] {display_name} ({count} credential(s))"
                )

        choice = Prompt.ask(
            Text.from_markup(
                "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[str(i) for i in range(1, len(all_providers) + 1)] + ["b"],
            show_choices=False,
        )

        if choice.lower() == "b":
            break

        try:
            idx = int(choice) - 1
            provider_info = all_providers[idx]

            if provider_info[0] == "api":
                _, provider, _ = provider_info
                await _view_api_keys_detail(provider)
            else:
                _, provider, _, _ = provider_info
                await _view_oauth_credentials_detail(provider)

        except (ValueError, IndexError):
            console.print("[bold red]Invalid choice.[/bold red]")
            await asyncio.sleep(1)


async def _view_api_keys_detail(provider_name: str):
    """Display detailed view of API keys for a specific provider."""
    clear_screen(f"View {provider_name} API Keys")

    api_keys = _get_api_keys_from_env()
    keys = api_keys.get(provider_name, [])

    if not keys:
        console.print(
            f"[bold yellow]No API keys found for {provider_name}.[/bold yellow]"
        )
        console.print("\n[dim]Press Enter to go back...[/dim]")
        input()
        return

    # Display detailed table
    table = Table(title=f"{provider_name} API Keys", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("Key Name", style="yellow")
    table.add_column("Value (masked)", style="dim")

    for i, (key_name, key_value) in enumerate(keys, 1):
        masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
        table.add_row(str(i), key_name, masked)

    console.print(table)
    console.print(f"\n[dim]Total: {len(keys)} key(s)[/dim]")
    console.print("\n[dim]Press Enter to go back...[/dim]")
    input()


async def _view_oauth_credentials_detail(provider_name: str):
    """Display detailed view of OAuth credentials for a specific provider."""
    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())
    clear_screen(f"View {display_name} Credentials")

    provider_factory, _ = _ensure_providers_loaded()

    try:
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = auth_instance.list_credentials(_get_oauth_base_dir())
    except Exception:
        credentials = []

    if not credentials:
        console.print(
            f"[bold yellow]No credentials found for {display_name}.[/bold yellow]"
        )
        console.print("\n[dim]Press Enter to go back...[/dim]")
        input()
        return

    # Display detailed table
    table = Table(title=f"{display_name} Credentials", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="yellow")
    table.add_column("Email/Identifier", style="cyan")

    # Add tier/project columns for Google OAuth providers
    if provider_name in ["gemini_cli", "antigravity"]:
        table.add_column("Tier", style="green")
        table.add_column("Project", style="dim")

    for i, cred in enumerate(credentials, 1):
        file_name = Path(cred["file_path"]).name
        email = cred.get("email", "unknown")

        if provider_name in ["gemini_cli", "antigravity"]:
            tier = _normalize_tier_name(cred.get("tier")) if cred.get("tier") else "-"
            project = cred.get("project_id", "-")
            if project and len(project) > 25:
                project = project[:22] + "..."
            table.add_row(str(i), file_name, email, tier, project or "-")
        else:
            table.add_row(str(i), file_name, email)

    console.print(table)
    console.print(f"\n[dim]Total: {len(credentials)} credential(s)[/dim]")
    console.print("\n[dim]Press Enter to go back...[/dim]")
    input()


async def manage_credentials_submenu():
    """
    Submenu for viewing and managing all credentials (API keys and OAuth).
    Allows deletion of any credential and editing email for OAuth credentials.
    """
    while True:
        clear_screen("Manage Credentials")

        # Display full summary
        _display_credentials_summary()

        console.print(
            Panel(
                Text.from_markup(
                    "[bold]Actions:[/bold]\n"
                    "1. Delete an API Key\n"
                    "2. Delete an OAuth Credential\n"
                    "3. Edit OAuth Credential Email [dim](Qwen Code recommended)[/dim]"
                ),
                title="Choose action",
                style="bold blue",
            )
        )

        action = Prompt.ask(
            Text.from_markup(
                "[bold]Select an option or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=["1", "2", "3", "b"],
            show_choices=False,
        )

        if action.lower() == "b":
            break

        if action == "1":
            # Delete API Key
            await _delete_api_key_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

        elif action == "2":
            # Delete OAuth Credential
            await _delete_oauth_credential_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

        elif action == "3":
            # Edit OAuth Credential Email
            await _edit_oauth_credential_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()


async def _delete_api_key_menu():
    """Menu for deleting an API key from the .env file."""
    clear_screen("Delete API Key")
    api_keys = _get_api_keys_from_env()

    if not api_keys:
        console.print("[bold yellow]No API keys configured.[/bold yellow]")
        return

    # Build a flat list of all keys for selection
    all_keys = []
    console.print("\n[bold cyan]Configured API Keys:[/bold cyan]")

    table = Table(box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Key Name", style="yellow")
    table.add_column("Provider", style="cyan")
    table.add_column("Value", style="dim")

    idx = 1
    for provider, keys in sorted(api_keys.items()):
        for key_name, key_value in keys:
            masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
            table.add_row(str(idx), key_name, provider, masked)
            all_keys.append((key_name, key_value, provider))
            idx += 1

    console.print(table)

    choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select API key to delete or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(all_keys) + 1)] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        idx = int(choice) - 1
        key_name, key_value, provider = all_keys[idx]

        # Confirmation prompt
        masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
        confirmed = Confirm.ask(
            f"[bold red]Delete[/bold red] [yellow]{key_name}[/yellow] ({masked})?"
        )

        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

        if _delete_api_key_from_env(key_name):
            console.print(
                Panel(
                    f"Successfully deleted [yellow]{key_name}[/yellow]",
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    f"Failed to delete [yellow]{key_name}[/yellow]",
                    style="bold red",
                    title="Error",
                    expand=False,
                )
            )

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


async def _delete_oauth_credential_menu():
    """Menu for deleting an OAuth credential file."""
    clear_screen("Delete OAuth Credential")
    oauth_summary = _get_oauth_credentials_summary()

    # Check if there are any credentials
    total = sum(len(creds) for creds in oauth_summary.values())
    if total == 0:
        console.print("[bold yellow]No OAuth credentials configured.[/bold yellow]")
        return

    # First, select provider
    console.print("\n[bold cyan]Select OAuth Provider:[/bold cyan]")

    providers_with_creds = [(p, c) for p, c in oauth_summary.items() if c]
    for i, (provider, creds) in enumerate(providers_with_creds, 1):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        console.print(f"  {i}. {display_name} ({len(creds)} credential(s))")

    provider_choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(providers_with_creds) + 1)] + ["b"],
        show_choices=False,
    )

    if provider_choice.lower() == "b":
        return

    try:
        provider_idx = int(provider_choice) - 1
        provider_name, credentials = providers_with_creds[provider_idx]
        display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

        # Now select credential
        _display_provider_credentials(provider_name)

        cred_choice = Prompt.ask(
            Text.from_markup(
                "[bold]Select credential to delete or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[str(i) for i in range(1, len(credentials) + 1)] + ["b"],
            show_choices=False,
        )

        if cred_choice.lower() == "b":
            return

        cred_idx = int(cred_choice) - 1
        cred_info = credentials[cred_idx]
        cred_path = cred_info["file_path"]
        email = cred_info.get("email", "unknown")

        # Confirmation prompt
        confirmed = Confirm.ask(
            f"[bold red]Delete[/bold red] credential for [cyan]{email}[/cyan] from {display_name}?"
        )

        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

        # Use the auth class's delete method
        provider_factory, _ = _ensure_providers_loaded()
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        if auth_instance.delete_credential(cred_path):
            console.print(
                Panel(
                    f"Successfully deleted credential for [cyan]{email}[/cyan]",
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    f"Failed to delete credential for [cyan]{email}[/cyan]",
                    style="bold red",
                    title="Error",
                    expand=False,
                )
            )

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


async def _edit_oauth_credential_menu():
    """Menu for editing an OAuth credential's email field."""
    clear_screen("Edit OAuth Credential")
    oauth_summary = _get_oauth_credentials_summary()

    # Check if there are any credentials
    total = sum(len(creds) for creds in oauth_summary.values())
    if total == 0:
        console.print("[bold yellow]No OAuth credentials configured.[/bold yellow]")
        return

    # Show warning about editing
    console.print(
        Panel(
            Text.from_markup(
                "[bold yellow]Warning:[/bold yellow] Editing OAuth credentials is generally not recommended.\n"
                "This is mainly useful for [bold]Qwen Code[/bold] where you manually enter an email identifier.\n\n"
                "For Google OAuth providers (Gemini CLI, Antigravity), the email is automatically\n"
                "retrieved during authentication and changing it may cause confusion."
            ),
            style="yellow",
            title="Edit OAuth Credential",
            expand=False,
        )
    )

    # First, select provider
    console.print("\n[bold cyan]Select OAuth Provider:[/bold cyan]")

    providers_with_creds = [(p, c) for p, c in oauth_summary.items() if c]
    for i, (provider, creds) in enumerate(providers_with_creds, 1):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        recommended = " [green](recommended)[/green]" if provider == "qwen_code" else ""
        console.print(
            f"  {i}. {display_name} ({len(creds)} credential(s)){recommended}"
        )

    provider_choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(providers_with_creds) + 1)] + ["b"],
        show_choices=False,
    )

    if provider_choice.lower() == "b":
        return

    try:
        provider_idx = int(provider_choice) - 1
        provider_name, _ = providers_with_creds[provider_idx]
        await _edit_oauth_credential_email(provider_name)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


def clear_screen(subtitle: str = "Interactive Credential Setup"):
    """
    Cross-platform terminal clear with header display.

    Clears the terminal and displays the application header with an optional subtitle.

    Args:
        subtitle: The subtitle text to display in the header panel.
                  Defaults to "Interactive Credential Setup".

    Uses native OS commands instead of ANSI escape sequences:
    - Windows (conhost & Windows Terminal): cls
    - Unix-like systems (Linux, Mac): clear
    """
    os.system("cls" if os.name == "nt" else "clear")
    console.print(
        Panel(
            f"[bold cyan]{subtitle}[/bold cyan]",
            title="--- API Key Proxy ---",
        )
    )


def ensure_env_defaults():
    """
    Ensures the .env file exists and contains essential default values like PROXY_API_KEY.
    """
    if not _get_env_file().is_file():
        _get_env_file().touch()
        console.print(
            f"Creating a new [bold yellow]{_get_env_file().name}[/bold yellow] file..."
        )

    # Check for PROXY_API_KEY, similar to setup_env.bat
    if get_key(str(_get_env_file()), "PROXY_API_KEY") is None:
        default_key = "VerysecretKey"
        console.print(
            f"Adding default [bold cyan]PROXY_API_KEY[/bold cyan] to [bold yellow]{_get_env_file().name}[/bold yellow]..."
        )
        set_key(str(_get_env_file()), "PROXY_API_KEY", default_key)


# =============================================================================
# LiteLLM Provider Configuration
# Auto-generated from LiteLLM documentation. For full provider docs, visit:
# https://docs.litellm.ai/docs/providers
#
# Structure: Each provider has:
#   - api_key: Environment variable for API key (None if not needed)
#   - category: Provider category for display grouping
#   - note: (optional) Configuration notes shown to user
#   - extra_vars: (optional) Additional env vars needed [(name, label, default), ...]
#
# Note: Adding multiple API base URLs per provider is not yet supported.
# =============================================================================


def _search_providers(query: str, providers: dict) -> list:
    """Search providers by substring match (case-insensitive).

    Searches both the provider key and display name.
    """
    query_lower = query.lower()
    matches = []
    for provider_key, config in providers.items():
        display_name = config.get("display_name", provider_key)
        if query_lower in provider_key.lower() or query_lower in display_name.lower():
            matches.append((provider_key, config))
    return matches


def _get_providers_by_category(providers: dict) -> dict:
    """Group providers by category."""
    by_category = {}
    for name, config in providers.items():
        category = config.get("category", "other")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((name, config))
    return by_category


async def setup_api_key():
    """
    Interactively sets up a new API key for a provider.
    Supports search, categorized display, and additional configuration variables.
    """
    clear_screen("Add API Key")

    # Show info panel
    console.print(
        Panel(
            Text.from_markup(
                "[bold]This list is powered by the LiteLLM library.[/bold]\n"
                "Some providers require additional configuration (API base URL, etc.)\n\n"
                "[dim]Full documentation: https://docs.litellm.ai/docs/providers[/dim]\n"
                "[dim]Note: Adding multiple API base URLs per provider is not yet supported.[/dim]"
            ),
            style="blue",
            title="Provider Information",
            expand=False,
        )
    )
    console.print()

    # -------------------------------------------------------------------------
    # Discover custom providers from project's provider registry
    # -------------------------------------------------------------------------
    _, PROVIDER_PLUGINS = _ensure_providers_loaded()
    from .providers import DynamicOpenAICompatibleProvider
    from .providers.provider_interface import ProviderInterface

    # Build a set of API key env vars already in SCRAPED_PROVIDERS
    litellm_api_keys = set()
    for info in SCRAPED_PROVIDERS.values():
        for api_key_var in info.get("api_key_env_vars", []):
            litellm_api_keys.add(api_key_var)

    # OAuth-only providers to exclude entirely from API key setup
    oauth_only_providers = {
        "gemini_cli",  # OAuth-only
        "antigravity",  # OAuth-only
        "qwen_code",  # OAuth is primary, don't advertise API key
        "iflow",  # OAuth is primary
    }

    # Base classes to exclude
    base_classes = {
        "openai_compatible",
    }

    # Create combined providers dict with scraped data + UI config
    # Key is the provider route key, value includes display_name, api_key, category, etc.
    all_providers = {}

    # Add all scraped providers with their UI config
    for provider_key in SCRAPED_PROVIDERS:
        # Skip blacklisted providers
        if provider_key in PROVIDER_BLACKLIST:
            continue

        scraped_info = SCRAPED_PROVIDERS[provider_key]
        ui_config = LITELLM_PROVIDERS.get(provider_key, {"category": "other"})

        # Skip providers without API keys (OAuth-only or no auth)
        api_key_vars = scraped_info.get("api_key_env_vars", [])
        if not api_key_vars:
            continue

        all_providers[provider_key] = {
            "display_name": scraped_info.get("display_name", provider_key),
            "api_key": api_key_vars[0] if api_key_vars else None,
            "category": ui_config.get("category", "other"),
            "note": ui_config.get("note"),
            "extra_vars": ui_config.get("extra_vars", []),
        }

    # Add custom providers from PROVIDER_PLUGINS
    for provider_key, provider_class in PROVIDER_PLUGINS.items():
        # Skip OAuth-only providers
        if provider_key in oauth_only_providers:
            continue

        # Skip base classes
        if provider_key in base_classes:
            continue

        # Skip if already in scraped providers
        if provider_key in all_providers:
            continue

        # Check if this is a dynamic OpenAI-compatible provider
        try:
            is_dynamic = isinstance(provider_class, type) and issubclass(
                provider_class, DynamicOpenAICompatibleProvider
            )
        except TypeError:
            is_dynamic = False

        env_var = f"{provider_key.upper()}_API_KEY"

        # Skip if API key already covered
        if env_var in litellm_api_keys:
            continue

        display_name = provider_key.replace("_", " ").title()

        if is_dynamic:
            # Dynamic OpenAI-compatible provider uses _API_BASE pattern
            all_providers[provider_key] = {
                "display_name": display_name,
                "api_key": env_var,
                "category": "custom_openai",
                "note": "Custom OpenAI-compatible provider.",
                "extra_vars": [
                    (f"{provider_key.upper()}_API_BASE", "API Base URL", None),
                ],
            }
        else:
            # First-party file-based provider
            all_providers[provider_key] = {
                "display_name": display_name,
                "api_key": env_var,
                "category": "custom",
                "note": "First-party provider from the library.",
            }

    # Search prompt
    search_query = Prompt.ask(
        "[bold]Search providers[/bold] [dim](or press Enter to see all)[/dim]",
        default="",
    )

    # Build provider list based on search
    if search_query.strip():
        # Search mode
        matches = _search_providers(search_query, all_providers)
        if not matches:
            console.print(
                f"[bold yellow]No providers found matching '{search_query}'[/bold yellow]"
            )
            console.print("[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Build numbered list from search results
        provider_list = []
        provider_text = Text()
        provider_text.append(
            f"\nMatching providers for '{search_query}':\n\n", style="bold cyan"
        )

        for i, (provider_key, config) in enumerate(matches, 1):
            provider_list.append((provider_key, config))
            display_name = config.get("display_name", provider_key)
            category = config.get("category", "other")
            category_label = next(
                (label for cat, label in PROVIDER_CATEGORIES if cat == category),
                "Other",
            )
            api_key_var = config.get("api_key")
            if api_key_var:
                key_prefix = (
                    api_key_var.replace("_API_KEY", "")
                    .replace("_TOKEN", "")
                    .replace("_", " ")
                )
                provider_text.append(
                    f"  {i}. {display_name} ({key_prefix}) ", style="white"
                )
            else:
                provider_text.append(f"  {i}. {display_name} ", style="white")
            provider_text.append(f"[{category_label}]\n", style="dim")

        console.print(provider_text)

    else:
        # Full categorized list mode
        by_category = _get_providers_by_category(all_providers)
        provider_list = []
        provider_text = Text()

        for category_key, category_label in PROVIDER_CATEGORIES:
            if category_key not in by_category:
                continue

            providers_in_cat = by_category[category_key]
            provider_text.append(f"\n--- {category_label} ---\n", style="bold cyan")

            for provider_key, config in providers_in_cat:
                idx = len(provider_list) + 1
                provider_list.append((provider_key, config))
                display_name = config.get("display_name", provider_key)
                api_key_var = config.get("api_key")
                if api_key_var:
                    key_prefix = (
                        api_key_var.replace("_API_KEY", "")
                        .replace("_TOKEN", "")
                        .replace("_", " ")
                    )
                    provider_text.append(f"  {idx}. {display_name} ({key_prefix})\n")
                else:
                    provider_text.append(
                        f"  {idx}. {display_name} [dim](no API key)[/dim]\n"
                    )

        console.print(provider_text)

    # Provider selection
    console.print()
    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Select a provider number or type [red]'b'[/red] to go back[/bold]"
        ),
        default="b",
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(provider_list):
            console.print("[bold red]Invalid choice.[/bold red]")
            return

        provider_key, provider_config = provider_list[choice_index]
        display_name = provider_config.get("display_name", provider_key)
        api_key_var = provider_config.get("api_key")
        note = provider_config.get("note")
        extra_vars = provider_config.get("extra_vars", [])

        # Get additional info from scraped data
        scraped_info = SCRAPED_PROVIDERS.get(provider_key, {})
        route = scraped_info.get("route", "").rstrip("/")
        api_base_url = scraped_info.get("api_base_url")

        console.print()

        # Build and show provider info panel
        info_lines = []
        if route:
            info_lines.append(f"Route: [cyan]{route}/[/cyan]")
            info_lines.append(f"Example: [dim]{route}/model-name[/dim]")
        if api_base_url:
            info_lines.append(f"API Base: [dim]{api_base_url}[/dim]")
        if api_key_var:
            info_lines.append(f"Env Variable: [green]{api_key_var}[/green]")

        if info_lines:
            console.print(
                Panel(
                    "\n".join(info_lines),
                    title=f"[bold]{display_name}[/bold]",
                    expand=False,
                    border_style="blue",
                )
            )
            console.print()

        # Show provider note if exists
        if note:
            console.print(
                Panel(
                    note,
                    style="yellow",
                    title="Configuration Note",
                    expand=False,
                )
            )
            console.print()

        saved_vars = []

        # Prompt for API key (if provider has one)
        if api_key_var:
            api_key = Prompt.ask(
                f"[bold]Enter {api_key_var}[/bold] [dim](or press Enter to skip)[/dim]",
                default="",
            )

            if api_key.strip():
                # Find next available key index
                key_index = 1
                while True:
                    key_name = f"{api_key_var}_{key_index}"
                    if _get_env_file().is_file():
                        with open(_get_env_file(), "r") as f:
                            if not any(line.startswith(f"{key_name}=") for line in f):
                                break
                    else:
                        break
                    key_index += 1

                key_name = f"{api_key_var}_{key_index}"
                set_key(str(_get_env_file()), key_name, api_key.strip())
                saved_vars.append((key_name, api_key.strip()))

        # Prompt for extra variables
        if extra_vars:
            console.print("\n[bold]Additional configuration:[/bold]")
            for env_var_name, label, default_value in extra_vars:
                if default_value:
                    # Pre-fill with default
                    value = Prompt.ask(
                        f"  {label}",
                        default=default_value,
                    )
                else:
                    value = Prompt.ask(
                        f"  {label} [dim](or press Enter to skip)[/dim]",
                        default="",
                    )

                if value.strip():
                    set_key(str(_get_env_file()), env_var_name, value.strip())
                    saved_vars.append((env_var_name, value.strip()))

        # Show success message
        if saved_vars:
            success_lines = [f"Successfully configured [bold]{display_name}[/bold]:\n"]
            for var_name, var_value in saved_vars:
                if len(var_value) > 8:
                    masked = f"{var_value[:4]}...{var_value[-4:]}"
                elif len(var_value) > 4:
                    masked = f"****{var_value[-4:]}"
                else:
                    masked = "****"
                success_lines.append(f"  [yellow]{var_name}[/yellow] = {masked}")

            console.print(
                Panel(
                    Text.from_markup("\n".join(success_lines)),
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print("[dim]No values configured (all skipped).[/dim]")

        # Wait for user to read the result
        console.print("\n[dim]Press Enter to continue...[/dim]")
        input()

    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
        console.print("\n[dim]Press Enter to continue...[/dim]")
        input()


async def setup_custom_openai_provider():
    """
    Interactively sets up a custom OpenAI-compatible provider.

    This adds a new provider that uses the standard OpenAI API format but points
    to a custom endpoint (LM Studio, Ollama, vLLM, custom server, etc.).
    """
    clear_screen("Add Custom OpenAI-Compatible Provider")

    # Show info panel
    console.print(
        Panel(
            Text.from_markup(
                "[bold]Custom OpenAI-Compatible Providers[/bold]\n\n"
                "Add a custom endpoint that uses the OpenAI API format.\n"
                "This works with: LM Studio, Ollama, vLLM, text-generation-webui, "
                "and other OpenAI-compatible servers.\n\n"
                "[dim]The library will automatically discover available models from your endpoint.[/dim]\n"
                "[dim]You can also override built-in providers (e.g., OPENAI) to route traffic elsewhere.[/dim]\n\n"
                "[yellow]Please consult the provider's documentation for the correct API base URL.[/yellow]"
            ),
            style="blue",
            title="Custom Provider Setup",
            expand=False,
        )
    )
    console.print()

    # Show existing custom providers
    _display_custom_providers_summary()

    # Prompt for provider name
    console.print("[dim]Provider name will be used for environment variables.[/dim]")
    console.print(
        "[dim]Use alphanumeric characters and underscores only (e.g., MY_LOCAL_LLM).[/dim]\n"
    )

    while True:
        provider_name = Prompt.ask(
            "[bold]Enter provider name[/bold] [dim](or 'b' to go back)[/dim]",
            default="",
        )

        if provider_name.lower() == "b" or not provider_name.strip():
            return

        provider_name = provider_name.strip().upper()

        # Validate name (alphanumeric + underscores only)
        import re

        if not re.match(r"^[A-Z][A-Z0-9_]*$", provider_name):
            console.print(
                "[bold red]Invalid name. Use letters, numbers, and underscores only. "
                "Must start with a letter.[/bold red]"
            )
            continue

        # Check for conflict with built-in LiteLLM providers
        conflict_provider = None
        for litellm_name, config in LITELLM_PROVIDERS.items():
            api_key_var = config.get("api_key", "")
            if api_key_var:
                # Extract prefix from API key var (e.g., OPENAI_API_KEY -> OPENAI)
                prefix = api_key_var.replace("_API_KEY", "").replace("_TOKEN", "")
                if prefix == provider_name:
                    conflict_provider = litellm_name
                    break

        if conflict_provider:
            console.print(
                f"\n[bold yellow]Warning:[/bold yellow] '{provider_name}' matches the built-in "
                f"'{conflict_provider}' provider."
            )
            console.print(
                "If you continue, requests to this provider will be routed to your custom endpoint "
                "instead of the official API.\n"
            )
            override_confirm = Prompt.ask(
                "[bold]Do you want to override the built-in provider?[/bold]",
                choices=["y", "n"],
                default="n",
            )
            if override_confirm.lower() != "y":
                continue

        break

    # Prompt for API Base URL (required)
    console.print()
    console.print("[dim]The API base URL is where requests will be sent.[/dim]")
    console.print(
        "[dim]Common examples: http://localhost:1234/v1, http://localhost:11434/v1[/dim]\n"
    )

    while True:
        api_base = Prompt.ask(
            "[bold]Enter API Base URL[/bold] [dim](required)[/dim]",
            default="",
        )

        if not api_base.strip():
            console.print("[bold red]API Base URL is required.[/bold red]")
            continue

        api_base = api_base.strip()

        # Validate URL format
        if not api_base.startswith(("http://", "https://")):
            console.print(
                "[bold red]Invalid URL. Must start with http:// or https://[/bold red]"
            )
            continue

        break

    # Prompt for API Key (required)
    console.print()
    console.print("[dim]Enter the API key for authentication.[/dim]")
    console.print(
        "[dim]If your server doesn't require authentication, enter any placeholder value.[/dim]\n"
    )

    while True:
        api_key = Prompt.ask(
            "[bold]Enter API Key[/bold] [dim](required)[/dim]",
            default="",
        )

        if not api_key.strip():
            console.print("[bold red]API Key is required.[/bold red]")
            continue

        api_key = api_key.strip()
        break

    # Save to .env file
    env_file = _get_env_file()

    # Save API Base URL
    api_base_var = f"{provider_name}_API_BASE"
    set_key(str(env_file), api_base_var, api_base)

    # Save API Key (find next available index)
    api_key_var_base = f"{provider_name}_API_KEY"
    key_index = 1
    if env_file.is_file():
        with open(env_file, "r") as f:
            content = f.read()
            while f"{api_key_var_base}_{key_index}=" in content:
                key_index += 1

    api_key_var = f"{api_key_var_base}_{key_index}"
    set_key(str(env_file), api_key_var, api_key)

    # Mask the API key for display
    if len(api_key) > 8:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    elif len(api_key) > 4:
        masked_key = f"****{api_key[-4:]}"
    else:
        masked_key = "****"

    # Show success message
    console.print(
        Panel(
            Text.from_markup(
                f"Successfully configured custom provider [bold]{provider_name}[/bold]:\n\n"
                f"  [yellow]{api_base_var}[/yellow] = {api_base}\n"
                f"  [yellow]{api_key_var}[/yellow] = {masked_key}\n\n"
                "[dim]The library will automatically fetch available models from your endpoint.[/dim]\n"
                "[dim]Use launcher menu option 4 'List Available Models' to verify the setup.[/dim]"
            ),
            style="bold green",
            title="Success",
            expand=False,
        )
    )

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()


async def setup_new_credential(provider_name: str):
    """
    Interactively sets up a new OAuth credential for a given provider.

    Delegates all credential management logic to the auth class's setup_credential() method.
    """
    try:
        provider_factory, _ = _ensure_providers_loaded()
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        # Build display name for better user experience
        oauth_friendly_names = {
            "gemini_cli": "Gemini CLI (OAuth)",
            "qwen_code": "Qwen Code (OAuth - also supports API keys)",
            "iflow": "iFlow (OAuth - also supports API keys)",
            "antigravity": "Antigravity (OAuth)",
        }
        display_name = oauth_friendly_names.get(
            provider_name, provider_name.replace("_", " ").title()
        )

        # Call the auth class's setup_credential() method which handles the entire flow:
        # - OAuth authentication
        # - Email extraction for deduplication
        # - File path determination (new or existing)
        # - Credential file saving
        # - Post-auth discovery (tier/project for Google OAuth providers)
        result = await auth_instance.setup_credential(_get_oauth_base_dir())

        if not result.success:
            console.print(
                Panel(
                    f"Credential setup failed: {result.error}",
                    style="bold red",
                    title="Error",
                )
            )
            return

        # Display success message with details
        if result.is_update:
            success_text = Text.from_markup(
                f"Successfully updated credential at [bold yellow]'{Path(result.file_path).name}'[/bold yellow] "
                f"for user [bold cyan]'{result.email}'[/bold cyan]."
            )
        else:
            success_text = Text.from_markup(
                f"Successfully created new credential at [bold yellow]'{Path(result.file_path).name}'[/bold yellow] "
                f"for user [bold cyan]'{result.email}'[/bold cyan]."
            )

        # Add tier/project info if available (Google OAuth providers)
        if hasattr(result, "tier") and result.tier:
            success_text.append(f"\nTier: {result.tier}")
        if hasattr(result, "project_id") and result.project_id:
            success_text.append(f"\nProject: {result.project_id}")

        console.print(Panel(success_text, style="bold green", title="Success"))

    except Exception as e:
        console.print(
            Panel(
                f"An error occurred during setup for {provider_name}: {e}",
                style="bold red",
                title="Error",
            )
        )


async def export_gemini_cli_to_env():
    """
    Export a Gemini CLI credential JSON file to .env format.
    Uses the auth class's build_env_lines() and list_credentials() methods.
    """
    clear_screen("Export Gemini CLI Credential")

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("gemini_cli")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                "No Gemini CLI credentials found. Please add one first using 'Add OAuth Credential'.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_info in enumerate(credentials):
        cred_text.append(
            f"  {i + 1}. {Path(cred_info['file_path']).name} ({cred_info['email']})\n"
        )

    console.print(
        Panel(
            cred_text,
            title="Available Gemini CLI Credentials",
            style="bold blue",
        )
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(credentials))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(credentials):
            cred_info = credentials[choice_index]

            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                numbered_prefix = f"GEMINI_CLI_{cred_info['number']}"
                success_text = Text.from_markup(
                    f"Successfully exported credential to [bold yellow]'{Path(env_path).name}'[/bold yellow]\n\n"
                    f"[bold]Environment variable prefix:[/bold] [cyan]{numbered_prefix}_*[/cyan]\n\n"
                    f"[bold]To use this credential:[/bold]\n"
                    f"1. Copy the contents to your main .env file, OR\n"
                    f"2. Source it: [bold cyan]source {Path(env_path).name}[/bold cyan] (Linux/Mac)\n"
                    f"3. Or on Windows: [bold cyan]Get-Content {Path(env_path).name} | ForEach-Object {{ $_ -replace '^([^#].*)$', 'set $1' }} | cmd[/bold cyan]\n\n"
                    f"[bold]To combine multiple credentials:[/bold]\n"
                    f"Copy lines from multiple .env files into one file.\n"
                    f"Each credential uses a unique number ({numbered_prefix}_*)."
                )
                console.print(Panel(success_text, style="bold green", title="Success"))
            else:
                console.print(
                    Panel(
                        "Failed to export credential", style="bold red", title="Error"
                    )
                )
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
    except Exception as e:
        console.print(
            Panel(
                f"An error occurred during export: {e}", style="bold red", title="Error"
            )
        )


async def export_qwen_code_to_env():
    """
    Export a Qwen Code credential JSON file to .env format.
    Uses the auth class's build_env_lines() and list_credentials() methods.
    """
    clear_screen("Export Qwen Code Credential")

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("qwen_code")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                "No Qwen Code credentials found. Please add one first using 'Add OAuth Credential'.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_info in enumerate(credentials):
        cred_text.append(
            f"  {i + 1}. {Path(cred_info['file_path']).name} ({cred_info['email']})\n"
        )

    console.print(
        Panel(
            cred_text,
            title="Available Qwen Code Credentials",
            style="bold blue",
        )
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(credentials))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(credentials):
            cred_info = credentials[choice_index]

            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                numbered_prefix = f"QWEN_CODE_{cred_info['number']}"
                success_text = Text.from_markup(
                    f"Successfully exported credential to [bold yellow]'{Path(env_path).name}'[/bold yellow]\n\n"
                    f"[bold]Environment variable prefix:[/bold] [cyan]{numbered_prefix}_*[/cyan]\n\n"
                    f"[bold]To use this credential:[/bold]\n"
                    f"1. Copy the contents to your main .env file, OR\n"
                    f"2. Source it: [bold cyan]source {Path(env_path).name}[/bold cyan] (Linux/Mac)\n\n"
                    f"[bold]To combine multiple credentials:[/bold]\n"
                    f"Copy lines from multiple .env files into one file.\n"
                    f"Each credential uses a unique number ({numbered_prefix}_*)."
                )
                console.print(Panel(success_text, style="bold green", title="Success"))
            else:
                console.print(
                    Panel(
                        "Failed to export credential", style="bold red", title="Error"
                    )
                )
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
    except Exception as e:
        console.print(
            Panel(
                f"An error occurred during export: {e}", style="bold red", title="Error"
            )
        )


async def export_iflow_to_env():
    """
    Export an iFlow credential JSON file to .env format.
    Uses the auth class's build_env_lines() and list_credentials() methods.
    """
    clear_screen("Export iFlow Credential")

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("iflow")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                "No iFlow credentials found. Please add one first using 'Add OAuth Credential'.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_info in enumerate(credentials):
        cred_text.append(
            f"  {i + 1}. {Path(cred_info['file_path']).name} ({cred_info['email']})\n"
        )

    console.print(
        Panel(
            cred_text,
            title="Available iFlow Credentials",
            style="bold blue",
        )
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(credentials))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(credentials):
            cred_info = credentials[choice_index]

            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                numbered_prefix = f"IFLOW_{cred_info['number']}"
                success_text = Text.from_markup(
                    f"Successfully exported credential to [bold yellow]'{Path(env_path).name}'[/bold yellow]\n\n"
                    f"[bold]Environment variable prefix:[/bold] [cyan]{numbered_prefix}_*[/cyan]\n\n"
                    f"[bold]To use this credential:[/bold]\n"
                    f"1. Copy the contents to your main .env file, OR\n"
                    f"2. Source it: [bold cyan]source {Path(env_path).name}[/bold cyan] (Linux/Mac)\n\n"
                    f"[bold]To combine multiple credentials:[/bold]\n"
                    f"Copy lines from multiple .env files into one file.\n"
                    f"Each credential uses a unique number ({numbered_prefix}_*)."
                )
                console.print(Panel(success_text, style="bold green", title="Success"))
            else:
                console.print(
                    Panel(
                        "Failed to export credential", style="bold red", title="Error"
                    )
                )
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
    except Exception as e:
        console.print(
            Panel(
                f"An error occurred during export: {e}", style="bold red", title="Error"
            )
        )


async def export_antigravity_to_env():
    """
    Export an Antigravity credential JSON file to .env format.
    Uses the auth class's build_env_lines() and list_credentials() methods.
    """
    clear_screen("Export Antigravity Credential")

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("antigravity")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                "No Antigravity credentials found. Please add one first using 'Add OAuth Credential'.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_info in enumerate(credentials):
        cred_text.append(
            f"  {i + 1}. {Path(cred_info['file_path']).name} ({cred_info['email']})\n"
        )

    console.print(
        Panel(
            cred_text,
            title="Available Antigravity Credentials",
            style="bold blue",
        )
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(credentials))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(credentials):
            cred_info = credentials[choice_index]

            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                numbered_prefix = f"ANTIGRAVITY_{cred_info['number']}"
                success_text = Text.from_markup(
                    f"Successfully exported credential to [bold yellow]'{Path(env_path).name}'[/bold yellow]\n\n"
                    f"[bold]Environment variable prefix:[/bold] [cyan]{numbered_prefix}_*[/cyan]\n\n"
                    f"[bold]To use this credential:[/bold]\n"
                    f"1. Copy the contents to your main .env file, OR\n"
                    f"2. Source it: [bold cyan]source {Path(env_path).name}[/bold cyan] (Linux/Mac)\n"
                    f"3. Or on Windows: [bold cyan]Get-Content {Path(env_path).name} | ForEach-Object {{ $_ -replace '^([^#].*)$', 'set $1' }} | cmd[/bold cyan]\n\n"
                    f"[bold]To combine multiple credentials:[/bold]\n"
                    f"Copy lines from multiple .env files into one file.\n"
                    f"Each credential uses a unique number ({numbered_prefix}_*)."
                )
                console.print(Panel(success_text, style="bold green", title="Success"))
            else:
                console.print(
                    Panel(
                        "Failed to export credential", style="bold red", title="Error"
                    )
                )
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
    except Exception as e:
        console.print(
            Panel(
                f"An error occurred during export: {e}", style="bold red", title="Error"
            )
        )


async def export_all_provider_credentials(provider_name: str):
    """
    Export all credentials for a specific provider to individual .env files.
    Uses the auth class's list_credentials() and export_credential_to_env() methods.
    """
    display_name = provider_name.replace("_", " ").title()
    clear_screen(f"Export All {display_name} Credentials")
    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    try:
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()
    except Exception:
        console.print(f"[bold red]Unknown provider: {provider_name}[/bold red]")
        return

    display_name = provider_name.replace("_", " ").title()

    console.print(
        Panel(
            f"[bold cyan]Export All {display_name} Credentials[/bold cyan]",
            expand=False,
        )
    )

    # List all credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                f"No {display_name} credentials found.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    exported_count = 0
    for cred_info in credentials:
        try:
            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                console.print(
                    f"  ✓ Exported [cyan]{Path(cred_info['file_path']).name}[/cyan] → [yellow]{Path(env_path).name}[/yellow]"
                )
                exported_count += 1
            else:
                console.print(
                    f"  ✗ Failed to export {Path(cred_info['file_path']).name}"
                )

        except Exception as e:
            console.print(
                f"  ✗ Failed to export {Path(cred_info['file_path']).name}: {e}"
            )

    console.print(
        Panel(
            f"Successfully exported {exported_count}/{len(credentials)} {display_name} credentials to individual .env files.",
            style="bold green",
            title="Export Complete",
        )
    )


async def combine_provider_credentials(provider_name: str):
    """
    Combine all credentials for a specific provider into a single .env file.
    Uses the auth class's list_credentials() and build_env_lines() methods.
    """
    display_name = provider_name.replace("_", " ").title()
    clear_screen(f"Combine {display_name} Credentials")
    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    try:
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()
    except Exception:
        console.print(f"[bold red]Unknown provider: {provider_name}[/bold red]")
        return

    display_name = provider_name.replace("_", " ").title()

    console.print(
        Panel(
            f"[bold cyan]Combine All {display_name} Credentials[/bold cyan]",
            expand=False,
        )
    )

    # List all credentials using auth class
    credentials = auth_instance.list_credentials(_get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                f"No {display_name} credentials found.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    combined_lines = [
        f"# Combined {display_name} Credentials",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total credentials: {len(credentials)}",
        "#",
        "# Copy all lines below into your main .env file",
        "",
    ]

    combined_count = 0
    for cred_info in credentials:
        try:
            # Load credential file
            with open(cred_info["file_path"], "r") as f:
                creds = json.load(f)

            # Use auth class to build env lines
            env_lines = auth_instance.build_env_lines(creds, cred_info["number"])

            combined_lines.extend(env_lines)
            combined_lines.append("")  # Blank line between credentials
            combined_count += 1

        except Exception as e:
            console.print(
                f"  ✗ Failed to process {Path(cred_info['file_path']).name}: {e}"
            )

    # Write combined file
    combined_filename = f"{provider_name}_all_combined.env"
    combined_filepath = _get_oauth_base_dir() / combined_filename

    with open(combined_filepath, "w") as f:
        f.write("\n".join(combined_lines))

    console.print(
        Panel(
            Text.from_markup(
                f"Successfully combined {combined_count} {display_name} credentials into:\n"
                f"[bold yellow]{combined_filepath}[/bold yellow]\n\n"
                f"[bold]To use:[/bold] Copy the contents into your main .env file."
            ),
            style="bold green",
            title="Combine Complete",
        )
    )


async def combine_all_credentials():
    """
    Combine ALL credentials from ALL providers into a single .env file.
    Uses auth class list_credentials() and build_env_lines() methods.
    """
    clear_screen("Combine All Credentials")

    # List of providers that support OAuth credentials
    oauth_providers = ["gemini_cli", "qwen_code", "iflow", "antigravity"]

    provider_factory, _ = _ensure_providers_loaded()

    combined_lines = [
        "# Combined All Provider Credentials",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# Copy all lines below into your main .env file",
        "",
    ]

    total_count = 0
    provider_counts = {}

    for provider_name in oauth_providers:
        try:
            auth_class = provider_factory.get_provider_auth_class(provider_name)
            auth_instance = auth_class()
        except Exception:
            continue  # Skip providers that don't have auth classes

        credentials = auth_instance.list_credentials(_get_oauth_base_dir())

        if not credentials:
            continue

        display_name = provider_name.replace("_", " ").title()
        combined_lines.append(f"# ===== {display_name} Credentials =====")
        combined_lines.append("")

        provider_count = 0
        for cred_info in credentials:
            try:
                # Load credential file
                with open(cred_info["file_path"], "r") as f:
                    creds = json.load(f)

                # Use auth class to build env lines
                env_lines = auth_instance.build_env_lines(creds, cred_info["number"])

                combined_lines.extend(env_lines)
                combined_lines.append("")
                provider_count += 1
                total_count += 1

            except Exception as e:
                console.print(
                    f"  ✗ Failed to process {Path(cred_info['file_path']).name}: {e}"
                )

        provider_counts[display_name] = provider_count

    if total_count == 0:
        console.print(
            Panel(
                "No credentials found to combine.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Write combined file
    combined_filename = "all_providers_combined.env"
    combined_filepath = _get_oauth_base_dir() / combined_filename

    with open(combined_filepath, "w") as f:
        f.write("\n".join(combined_lines))

    # Build summary
    summary_lines = [
        f"  • {name}: {count} credential(s)" for name, count in provider_counts.items()
    ]
    summary = "\n".join(summary_lines)

    console.print(
        Panel(
            Text.from_markup(
                f"Successfully combined {total_count} credentials from {len(provider_counts)} providers:\n"
                f"{summary}\n\n"
                f"[bold]Output file:[/bold] [yellow]{combined_filepath}[/yellow]\n\n"
                f"[bold]To use:[/bold] Copy the contents into your main .env file."
            ),
            style="bold green",
            title="Combine Complete",
        )
    )


async def export_credentials_submenu():
    """
    Submenu for credential export options.
    """
    while True:
        clear_screen("Export Credentials")

        console.print(
            Panel(
                Text.from_markup(
                    "[bold]Individual Exports:[/bold]\n"
                    "1. Export Gemini CLI credential\n"
                    "2. Export Qwen Code credential\n"
                    "3. Export iFlow credential\n"
                    "4. Export Antigravity credential\n"
                    "\n"
                    "[bold]Bulk Exports (per provider):[/bold]\n"
                    "5. Export ALL Gemini CLI credentials\n"
                    "6. Export ALL Qwen Code credentials\n"
                    "7. Export ALL iFlow credentials\n"
                    "8. Export ALL Antigravity credentials\n"
                    "\n"
                    "[bold]Combine Credentials:[/bold]\n"
                    "9. Combine all Gemini CLI into one file\n"
                    "10. Combine all Qwen Code into one file\n"
                    "11. Combine all iFlow into one file\n"
                    "12. Combine all Antigravity into one file\n"
                    "13. Combine ALL providers into one file"
                ),
                title="Choose export option",
                style="bold blue",
            )
        )

        export_choice = Prompt.ask(
            Text.from_markup(
                "[bold]Please select an option or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "b",
            ],
            show_choices=False,
        )

        if export_choice.lower() == "b":
            break

        # Individual exports
        if export_choice == "1":
            await export_gemini_cli_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "2":
            await export_qwen_code_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "3":
            await export_iflow_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "4":
            await export_antigravity_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Bulk exports (all credentials for a provider)
        elif export_choice == "5":
            await export_all_provider_credentials("gemini_cli")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "6":
            await export_all_provider_credentials("qwen_code")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "7":
            await export_all_provider_credentials("iflow")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "8":
            await export_all_provider_credentials("antigravity")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Combine per provider
        elif export_choice == "9":
            await combine_provider_credentials("gemini_cli")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "10":
            await combine_provider_credentials("qwen_code")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "11":
            await combine_provider_credentials("iflow")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "12":
            await combine_provider_credentials("antigravity")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Combine all providers
        elif export_choice == "13":
            await combine_all_credentials()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()


async def main(clear_on_start=True):
    """
    An interactive CLI tool to add new credentials.

    Args:
        clear_on_start: If False, skip initial screen clear (used when called from launcher
                       to preserve the loading screen)
    """
    ensure_env_defaults()

    # Only show header if we're clearing (standalone mode)
    if clear_on_start:
        clear_screen()

    while True:
        # Clear screen between menu selections for cleaner UX
        clear_screen()

        # Display credentials summary at the top
        _display_credentials_summary()

        console.print(
            Panel(
                Text.from_markup(
                    "1. Add OAuth Credential\n"
                    "2. Add API Key\n"
                    "3. Add Custom OpenAI-Compatible Provider\n"
                    "4. Export Credentials\n"
                    "5. View Credentials\n"
                    "6. Manage Credentials"
                ),
                title="Choose action",
                style="bold blue",
            )
        )

        setup_type = Prompt.ask(
            Text.from_markup(
                "[bold]Please select an option or type [red]'q'[/red] to quit[/bold]"
            ),
            choices=["1", "2", "3", "4", "5", "6", "q"],
            show_choices=False,
        )

        if setup_type.lower() == "q":
            break

        if setup_type == "1":
            # Clear and show OAuth providers summary before listing providers
            clear_screen("Add OAuth Credential")
            _display_oauth_providers_summary()

            provider_factory, _ = _ensure_providers_loaded()
            available_providers = provider_factory.get_available_providers()

            provider_text = Text()
            for i, provider in enumerate(available_providers):
                display_name = OAUTH_FRIENDLY_NAMES.get(
                    provider, provider.replace("_", " ").title()
                )
                provider_text.append(f"  {i + 1}. {display_name}\n")

            console.print(
                Panel(
                    provider_text,
                    title="Available Providers for OAuth",
                    style="bold blue",
                )
            )

            choice = Prompt.ask(
                Text.from_markup(
                    "[bold]Please select a provider or type [red]'b'[/red] to go back[/bold]"
                ),
                choices=[str(i + 1) for i in range(len(available_providers))] + ["b"],
                show_choices=False,
            )

            if choice.lower() == "b":
                continue

            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_providers):
                    provider_name = available_providers[choice_index]
                    display_name = OAUTH_FRIENDLY_NAMES.get(
                        provider_name, provider_name.replace("_", " ").title()
                    )

                    # Show existing credentials for this provider before proceeding
                    _display_provider_credentials(provider_name)

                    console.print(
                        f"Starting OAuth setup for [bold cyan]{display_name}[/bold cyan]..."
                    )
                    await setup_new_credential(provider_name)
                    # Don't clear after OAuth - user needs to see full flow
                    console.print("\n[dim]Press Enter to return to main menu...[/dim]")
                    input()
                else:
                    console.print(
                        "[bold red]Invalid choice. Please try again.[/bold red]"
                    )
                    await asyncio.sleep(1.5)
            except ValueError:
                console.print(
                    "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
                )
                await asyncio.sleep(1.5)

        elif setup_type == "2":
            await setup_api_key()
            # console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            # input()

        elif setup_type == "3":
            await setup_custom_openai_provider()

        elif setup_type == "4":
            await export_credentials_submenu()

        elif setup_type == "5":
            await view_credentials_menu()

        elif setup_type == "6":
            await manage_credentials_submenu()


def run_credential_tool(from_launcher=False):
    """
    Entry point for credential tool.

    Args:
        from_launcher: If True, skip loading screen (launcher already showed it)
    """
    # Check if we need to show loading screen
    if not from_launcher:
        # Standalone mode - show full loading UI
        os.system("cls" if os.name == "nt" else "clear")

        _start_time = time.time()

        # Phase 1: Show initial message
        print("━" * 70)
        print("Interactive Credential Setup Tool")
        print("GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
        print("━" * 70)
        print("Loading credential management components...")

        # Phase 2: Load dependencies with spinner
        with console.status("Loading authentication providers...", spinner="dots"):
            _ensure_providers_loaded()
        console.print("✓ Authentication providers loaded")

        with console.status("Initializing credential tool...", spinner="dots"):
            time.sleep(0.2)  # Brief pause for UI consistency
        console.print("✓ Credential tool initialized")

        _elapsed = time.time() - _start_time
        _, PROVIDER_PLUGINS = _ensure_providers_loaded()
        print(
            f"✓ Tool ready in {_elapsed:.2f}s ({len(PROVIDER_PLUGINS)} providers available)"
        )

        # Small delay to let user see the ready message
        time.sleep(0.5)

    # Run the main async event loop
    # If from launcher, don't clear screen at start to preserve loading messages
    try:
        asyncio.run(main(clear_on_start=not from_launcher))
        clear_screen()  # Clear terminal when credential tool exits
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting setup.[/bold yellow]")
        clear_screen()  # Clear terminal on keyboard interrupt too
