# src/rotator_library/credential_tool.py

import asyncio
import json
import os
import time
from pathlib import Path
from dotenv import set_key, get_key

# NOTE: Heavy imports (provider_factory, PROVIDER_PLUGINS) are deferred
# to avoid 6-7 second delay before showing loading screen
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)
# Use a direct path to the .env file in the project root
ENV_FILE = Path.cwd() / ".env"

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


def clear_screen():
    """
    Cross-platform terminal clear that works robustly on both
    classic Windows conhost and modern terminals (Windows Terminal, Linux, Mac).

    Uses native OS commands instead of ANSI escape sequences:
    - Windows (conhost & Windows Terminal): cls
    - Unix-like systems (Linux, Mac): clear
    """
    os.system("cls" if os.name == "nt" else "clear")


def ensure_env_defaults():
    """
    Ensures the .env file exists and contains essential default values like PROXY_API_KEY.
    """
    if not ENV_FILE.is_file():
        ENV_FILE.touch()
        console.print(
            f"Creating a new [bold yellow]{ENV_FILE.name}[/bold yellow] file..."
        )

    # Check for PROXY_API_KEY, similar to setup_env.bat
    if get_key(str(ENV_FILE), "PROXY_API_KEY") is None:
        default_key = "VerysecretKey"
        console.print(
            f"Adding default [bold cyan]PROXY_API_KEY[/bold cyan] to [bold yellow]{ENV_FILE.name}[/bold yellow]..."
        )
        set_key(str(ENV_FILE), "PROXY_API_KEY", default_key)


async def setup_api_key():
    """
    Interactively sets up a new API key for a provider.
    """
    console.print(Panel("[bold cyan]API Key Setup[/bold cyan]", expand=False))

    # Debug toggle: Set to True to see env var names next to each provider
    SHOW_ENV_VAR_NAMES = True

    # Verified list of LiteLLM providers with their friendly names and API key variables
    LITELLM_PROVIDERS = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google AI Studio (Gemini)": "GEMINI_API_KEY",
        "Azure OpenAI": "AZURE_API_KEY",
        "Vertex AI": "GOOGLE_API_KEY",
        "AWS Bedrock": "AWS_ACCESS_KEY_ID",
        "Cohere": "COHERE_API_KEY",
        "Chutes": "CHUTES_API_KEY",
        "Mistral AI": "MISTRAL_API_KEY",
        "Codestral (Mistral)": "CODESTRAL_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Perplexity": "PERPLEXITYAI_API_KEY",
        "xAI": "XAI_API_KEY",
        "Together AI": "TOGETHERAI_API_KEY",
        "Fireworks AI": "FIREWORKS_AI_API_KEY",
        "Replicate": "REPLICATE_API_KEY",
        "Hugging Face": "HUGGINGFACE_API_KEY",
        "Anyscale": "ANYSCALE_API_KEY",
        "NVIDIA NIM": "NVIDIA_NIM_API_KEY",
        "Deepseek": "DEEPSEEK_API_KEY",
        "AI21": "AI21_API_KEY",
        "Cerebras": "CEREBRAS_API_KEY",
        "Moonshot": "MOONSHOT_API_KEY",
        "Ollama": "OLLAMA_API_KEY",
        "Xinference": "XINFERENCE_API_KEY",
        "Infinity": "INFINITY_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Deepinfra": "DEEPINFRA_API_KEY",
        "Cloudflare": "CLOUDFLARE_API_KEY",
        "Baseten": "BASETEN_API_KEY",
        "Modal": "MODAL_API_KEY",
        "Databricks": "DATABRICKS_API_KEY",
        "AWS SageMaker": "AWS_ACCESS_KEY_ID",
        "IBM watsonx.ai": "WATSONX_APIKEY",
        "Predibase": "PREDIBASE_API_KEY",
        "Clarifai": "CLARIFAI_API_KEY",
        "NLP Cloud": "NLP_CLOUD_API_KEY",
        "Voyage AI": "VOYAGE_API_KEY",
        "Jina AI": "JINA_API_KEY",
        "Hyperbolic": "HYPERBOLIC_API_KEY",
        "Morph": "MORPH_API_KEY",
        "Lambda AI": "LAMBDA_API_KEY",
        "Novita AI": "NOVITA_API_KEY",
        "Aleph Alpha": "ALEPH_ALPHA_API_KEY",
        "SambaNova": "SAMBANOVA_API_KEY",
        "FriendliAI": "FRIENDLI_TOKEN",
        "Galadriel": "GALADRIEL_API_KEY",
        "CompactifAI": "COMPACTIFAI_API_KEY",
        "Lemonade": "LEMONADE_API_KEY",
        "GradientAI": "GRADIENTAI_API_KEY",
        "Featherless AI": "FEATHERLESS_AI_API_KEY",
        "Nebius AI Studio": "NEBIUS_API_KEY",
        "Dashscope (Qwen)": "DASHSCOPE_API_KEY",
        "Bytez": "BYTEZ_API_KEY",
        "Oracle OCI": "OCI_API_KEY",
        "DataRobot": "DATAROBOT_API_KEY",
        "OVHCloud": "OVHCLOUD_API_KEY",
        "Volcengine": "VOLCENGINE_API_KEY",
        "Snowflake": "SNOWFLAKE_API_KEY",
        "Nscale": "NSCALE_API_KEY",
        "Recraft": "RECRAFT_API_KEY",
        "v0": "V0_API_KEY",
        "Vercel": "VERCEL_AI_GATEWAY_API_KEY",
        "Topaz": "TOPAZ_API_KEY",
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "Deepgram": "DEEPGRAM_API_KEY",
        "GitHub Models": "GITHUB_TOKEN",
        "GitHub Copilot": "GITHUB_COPILOT_API_KEY",
    }

    # Discover custom providers and add them to the list
    # Note: gemini_cli and antigravity are OAuth-only
    # qwen_code API key support is a fallback
    # iflow API key support is a feature
    _, PROVIDER_PLUGINS = _ensure_providers_loaded()

    # Build a set of environment variables already in LITELLM_PROVIDERS
    # to avoid duplicates based on the actual API key names
    litellm_env_vars = set(LITELLM_PROVIDERS.values())

    # Providers to exclude from API key list
    exclude_providers = {
        "gemini_cli",  # OAuth-only
        "antigravity",  # OAuth-only
        "qwen_code",  # API key is fallback, OAuth is primary - don't advertise
        "openai_compatible",  # Base class, not a real provider
    }

    discovered_providers = {}
    for provider_key in PROVIDER_PLUGINS.keys():
        if provider_key in exclude_providers:
            continue

        # Create environment variable name
        env_var = provider_key.upper() + "_API_KEY"

        # Check if this env var already exists in LITELLM_PROVIDERS
        # This catches duplicates like GEMINI_API_KEY, MISTRAL_API_KEY, etc.
        if env_var in litellm_env_vars:
            # Already in LITELLM_PROVIDERS with better name, skip this one
            continue

        # Create display name for this custom provider
        display_name = provider_key.replace("_", " ").title()
        discovered_providers[display_name] = env_var

    # LITELLM_PROVIDERS takes precedence (comes first in merge)
    combined_providers = {**LITELLM_PROVIDERS, **discovered_providers}
    provider_display_list = sorted(combined_providers.keys())

    provider_text = Text()
    for i, provider_name in enumerate(provider_display_list):
        if SHOW_ENV_VAR_NAMES:
            # Extract env var prefix (before _API_KEY)
            env_var = combined_providers[provider_name]
            prefix = env_var.replace("_API_KEY", "").replace("_", " ")
            provider_text.append(f"  {i + 1}. {provider_name} ({prefix})\n")
        else:
            provider_text.append(f"  {i + 1}. {provider_name}\n")

    console.print(
        Panel(provider_text, title="Available Providers for API Key", style="bold blue")
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a provider or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(provider_display_list))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(provider_display_list):
            display_name = provider_display_list[choice_index]
            api_var_base = combined_providers[display_name]

            api_key = Prompt.ask(f"Enter the API key for {display_name}")

            # Check for duplicate API key value
            if ENV_FILE.is_file():
                with open(ENV_FILE, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(api_var_base) and "=" in line:
                            existing_key_name, _, existing_key_value = line.partition(
                                "="
                            )
                            if existing_key_value == api_key:
                                warning_text = Text.from_markup(
                                    f"This API key already exists as [bold yellow]'{existing_key_name}'[/bold yellow]. Overwriting..."
                                )
                                console.print(
                                    Panel(
                                        warning_text,
                                        style="bold yellow",
                                        title="Updating API Key",
                                    )
                                )

                                set_key(str(ENV_FILE), existing_key_name, api_key)

                                success_text = Text.from_markup(
                                    f"Successfully updated existing key [bold yellow]'{existing_key_name}'[/bold yellow]."
                                )
                                console.print(
                                    Panel(
                                        success_text,
                                        style="bold green",
                                        title="Success",
                                    )
                                )
                                return

            # Special handling for AWS
            if display_name in ["AWS Bedrock", "AWS SageMaker"]:
                console.print(
                    Panel(
                        Text.from_markup(
                            "This provider requires both an Access Key ID and a Secret Access Key.\n"
                            f"The key you entered will be saved as [bold yellow]{api_var_base}_1[/bold yellow].\n"
                            "Please manually add the [bold cyan]AWS_SECRET_ACCESS_KEY_1[/bold cyan] to your .env file."
                        ),
                        title="[bold yellow]Additional Step Required[/bold yellow]",
                        border_style="yellow",
                    )
                )

            key_index = 1
            while True:
                key_name = f"{api_var_base}_{key_index}"
                if ENV_FILE.is_file():
                    with open(ENV_FILE, "r") as f:
                        if not any(line.startswith(f"{key_name}=") for line in f):
                            break
                else:
                    break
                key_index += 1

            key_name = f"{api_var_base}_{key_index}"
            set_key(str(ENV_FILE), key_name, api_key)

            success_text = Text.from_markup(
                f"Successfully added {display_name} API key as [bold yellow]'{key_name}'[/bold yellow]."
            )
            console.print(Panel(success_text, style="bold green", title="Success"))

        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )


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
        result = await auth_instance.setup_credential(OAUTH_BASE_DIR)

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
    console.print(
        Panel(
            "[bold cyan]Export Gemini CLI Credential to .env[/bold cyan]", expand=False
        )
    )

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("gemini_cli")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
        Panel(cred_text, title="Available Gemini CLI Credentials", style="bold blue")
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
                cred_info["file_path"], OAUTH_BASE_DIR
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
    console.print(
        Panel(
            "[bold cyan]Export Qwen Code Credential to .env[/bold cyan]", expand=False
        )
    )

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("qwen_code")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
        Panel(cred_text, title="Available Qwen Code Credentials", style="bold blue")
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
                cred_info["file_path"], OAUTH_BASE_DIR
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
    console.print(
        Panel("[bold cyan]Export iFlow Credential to .env[/bold cyan]", expand=False)
    )

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("iflow")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
        Panel(cred_text, title="Available iFlow Credentials", style="bold blue")
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
                cred_info["file_path"], OAUTH_BASE_DIR
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
    console.print(
        Panel(
            "[bold cyan]Export Antigravity Credential to .env[/bold cyan]", expand=False
        )
    )

    # Get auth instance for this provider
    provider_factory, _ = _ensure_providers_loaded()
    auth_class = provider_factory.get_provider_auth_class("antigravity")
    auth_instance = auth_class()

    # List available credentials using auth class
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
        Panel(cred_text, title="Available Antigravity Credentials", style="bold blue")
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
                cred_info["file_path"], OAUTH_BASE_DIR
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
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
                cred_info["file_path"], OAUTH_BASE_DIR
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
    credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
    combined_filepath = OAUTH_BASE_DIR / combined_filename

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
    console.print(
        Panel("[bold cyan]Combine All Provider Credentials[/bold cyan]", expand=False)
    )

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

        credentials = auth_instance.list_credentials(OAUTH_BASE_DIR)

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
    combined_filepath = OAUTH_BASE_DIR / combined_filename

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
        clear_screen()
        console.print(
            Panel(
                "[bold cyan]Export Credentials to .env[/bold cyan]",
                title="--- API Key Proxy ---",
                expand=False,
            )
        )

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
        console.print(
            Panel(
                "[bold cyan]Interactive Credential Setup[/bold cyan]",
                title="--- API Key Proxy ---",
                expand=False,
            )
        )

    while True:
        # Clear screen between menu selections for cleaner UX
        clear_screen()
        console.print(
            Panel(
                "[bold cyan]Interactive Credential Setup[/bold cyan]",
                title="--- API Key Proxy ---",
                expand=False,
            )
        )

        console.print(
            Panel(
                Text.from_markup(
                    "1. Add OAuth Credential\n2. Add API Key\n3. Export Credentials"
                ),
                title="Choose credential type",
                style="bold blue",
            )
        )

        setup_type = Prompt.ask(
            Text.from_markup(
                "[bold]Please select an option or type [red]'q'[/red] to quit[/bold]"
            ),
            choices=["1", "2", "3", "q"],
            show_choices=False,
        )

        if setup_type.lower() == "q":
            break

        if setup_type == "1":
            provider_factory, _ = _ensure_providers_loaded()
            available_providers = provider_factory.get_available_providers()
            oauth_friendly_names = {
                "gemini_cli": "Gemini CLI (OAuth)",
                "qwen_code": "Qwen Code (OAuth - also supports API keys)",
                "iflow": "iFlow (OAuth - also supports API keys)",
                "antigravity": "Antigravity (OAuth)",
            }

            provider_text = Text()
            for i, provider in enumerate(available_providers):
                display_name = oauth_friendly_names.get(
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
                    display_name = oauth_friendly_names.get(
                        provider_name, provider_name.replace("_", " ").title()
                    )
                    console.print(
                        f"\nStarting OAuth setup for [bold cyan]{display_name}[/bold cyan]..."
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
            await export_credentials_submenu()


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
