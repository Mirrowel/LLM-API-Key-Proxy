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

def ensure_env_defaults():
    """
    Ensures the .env file exists and contains essential default values like PROXY_API_KEY.
    """
    if not ENV_FILE.is_file():
        ENV_FILE.touch()
        console.print(f"Creating a new [bold yellow]{ENV_FILE.name}[/bold yellow] file...")

    # Check for PROXY_API_KEY, similar to setup_env.bat
    if get_key(str(ENV_FILE), "PROXY_API_KEY") is None:
        default_key = "VerysecretKey"
        console.print(f"Adding default [bold cyan]PROXY_API_KEY[/bold cyan] to [bold yellow]{ENV_FILE.name}[/bold yellow]...")
        set_key(str(ENV_FILE), "PROXY_API_KEY", default_key)

async def setup_api_key():
    """
    Interactively sets up a new API key for a provider.
    """
    console.print(Panel("[bold cyan]API Key Setup[/bold cyan]", expand=False))

    # Verified list of LiteLLM providers with their friendly names and API key variables
    LITELLM_PROVIDERS = {
        "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY",
        "Google AI Studio (Gemini)": "GEMINI_API_KEY", "Azure OpenAI": "AZURE_API_KEY",
        "Vertex AI": "GOOGLE_API_KEY", "AWS Bedrock": "AWS_ACCESS_KEY_ID",
        "Cohere": "COHERE_API_KEY", "Chutes": "CHUTES_API_KEY",
        "Mistral AI": "MISTRAL_API_KEY",
        "Codestral (Mistral)": "CODESTRAL_API_KEY", "Groq": "GROQ_API_KEY",
        "Perplexity": "PERPLEXITYAI_API_KEY", "xAI": "XAI_API_KEY",
        "Together AI": "TOGETHERAI_API_KEY", "Fireworks AI": "FIREWORKS_AI_API_KEY",
        "Replicate": "REPLICATE_API_KEY", "Hugging Face": "HUGGINGFACE_API_KEY",
        "Anyscale": "ANYSCALE_API_KEY", "NVIDIA NIM": "NVIDIA_NIM_API_KEY",
        "Deepseek": "DEEPSEEK_API_KEY", "AI21": "AI21_API_KEY",
        "Cerebras": "CEREBRAS_API_KEY", "Moonshot": "MOONSHOT_API_KEY",
        "Ollama": "OLLAMA_API_KEY", "Xinference": "XINFERENCE_API_KEY",
        "Infinity": "INFINITY_API_KEY", "OpenRouter": "OPENROUTER_API_KEY",
        "Deepinfra": "DEEPINFRA_API_KEY", "Cloudflare": "CLOUDFLARE_API_KEY",
        "Baseten": "BASETEN_API_KEY", "Modal": "MODAL_API_KEY",
        "Databricks": "DATABRICKS_API_KEY", "AWS SageMaker": "AWS_ACCESS_KEY_ID",
        "IBM watsonx.ai": "WATSONX_APIKEY", "Predibase": "PREDIBASE_API_KEY",
        "Clarifai": "CLARIFAI_API_KEY", "NLP Cloud": "NLP_CLOUD_API_KEY",
        "Voyage AI": "VOYAGE_API_KEY", "Jina AI": "JINA_API_KEY",
        "Hyperbolic": "HYPERBOLIC_API_KEY", "Morph": "MORPH_API_KEY",
        "Lambda AI": "LAMBDA_API_KEY", "Novita AI": "NOVITA_API_KEY",
        "Aleph Alpha": "ALEPH_ALPHA_API_KEY", "SambaNova": "SAMBANOVA_API_KEY",
        "FriendliAI": "FRIENDLI_TOKEN", "Galadriel": "GALADRIEL_API_KEY",
        "CompactifAI": "COMPACTIFAI_API_KEY", "Lemonade": "LEMONADE_API_KEY",
        "GradientAI": "GRADIENTAI_API_KEY", "Featherless AI": "FEATHERLESS_AI_API_KEY",
        "Nebius AI Studio": "NEBIUS_API_KEY", "Dashscope (Qwen)": "DASHSCOPE_API_KEY",
        "Bytez": "BYTEZ_API_KEY", "Oracle OCI": "OCI_API_KEY",
        "DataRobot": "DATAROBOT_API_KEY", "OVHCloud": "OVHCLOUD_API_KEY",
        "Volcengine": "VOLCENGINE_API_KEY", "Snowflake": "SNOWFLAKE_API_KEY",
        "Nscale": "NSCALE_API_KEY", "Recraft": "RECRAFT_API_KEY",
        "v0": "V0_API_KEY", "Vercel": "VERCEL_AI_GATEWAY_API_KEY",
        "Topaz": "TOPAZ_API_KEY", "ElevenLabs": "ELEVENLABS_API_KEY",
        "Deepgram": "DEEPGRAM_API_KEY", "Custom API": "CUSTOM_API_KEY",
        "GitHub Models": "GITHUB_TOKEN", "GitHub Copilot": "GITHUB_COPILOT_API_KEY",
    }

    # Discover custom providers and add them to the list
    # Note: gemini_cli is OAuth-only, but qwen_code and iflow support both OAuth and API keys
    _, PROVIDER_PLUGINS = _ensure_providers_loaded()
    oauth_only_providers = {'gemini_cli'}
    discovered_providers = {
        p.replace('_', ' ').title(): p.upper() + "_API_KEY"
        for p in PROVIDER_PLUGINS.keys()
        if p not in oauth_only_providers and p.replace('_', ' ').title() not in LITELLM_PROVIDERS
    }
    
    combined_providers = {**LITELLM_PROVIDERS, **discovered_providers}
    provider_display_list = sorted(combined_providers.keys())

    provider_text = Text()
    for i, provider_name in enumerate(provider_display_list):
        provider_text.append(f"  {i + 1}. {provider_name}\n")

    console.print(Panel(provider_text, title="Available Providers for API Key", style="bold blue"))

    choice = Prompt.ask(
        Text.from_markup("[bold]Please select a provider or type [red]'b'[/red] to go back[/bold]"),
        choices=[str(i + 1) for i in range(len(provider_display_list))] + ["b"],
        show_choices=False
    )

    if choice.lower() == 'b':
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
                            existing_key_name, _, existing_key_value = line.partition("=")
                            if existing_key_value == api_key:
                                warning_text = Text.from_markup(f"This API key already exists as [bold yellow]'{existing_key_name}'[/bold yellow]. Overwriting...")
                                console.print(Panel(warning_text, style="bold yellow", title="Updating API Key"))

                                set_key(str(ENV_FILE), existing_key_name, api_key)

                                success_text = Text.from_markup(f"Successfully updated existing key [bold yellow]'{existing_key_name}'[/bold yellow].")
                                console.print(Panel(success_text, style="bold green", title="Success"))
                                return

            # Special handling for AWS
            if display_name in ["AWS Bedrock", "AWS SageMaker"]:
                console.print(Panel(
                    Text.from_markup(
                        "This provider requires both an Access Key ID and a Secret Access Key.\n"
                        f"The key you entered will be saved as [bold yellow]{api_var_base}_1[/bold yellow].\n"
                        "Please manually add the [bold cyan]AWS_SECRET_ACCESS_KEY_1[/bold cyan] to your .env file."
                    ),
                    title="[bold yellow]Additional Step Required[/bold yellow]",
                    border_style="yellow"
                ))

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
            
            success_text = Text.from_markup(f"Successfully added {display_name} API key as [bold yellow]'{key_name}'[/bold yellow].")
            console.print(Panel(success_text, style="bold green", title="Success"))

        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")

async def setup_new_credential(provider_name: str):
    """
    Interactively sets up a new OAuth credential for a given provider.
    """
    try:
        provider_factory, _ = _ensure_providers_loaded()
        auth_class = provider_factory.get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        # Build display name for better user experience
        oauth_friendly_names = {
            "gemini_cli": "Gemini CLI (OAuth)",
            "qwen_code": "Qwen Code (OAuth - also supports API keys)",
            "iflow": "iFlow (OAuth - also supports API keys)"
        }
        display_name = oauth_friendly_names.get(provider_name, provider_name.replace('_', ' ').title())

        # Pass provider metadata to auth classes for better display
        temp_creds = {
            "_proxy_metadata": {
                "provider_name": provider_name,
                "display_name": display_name
            }
        }
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
                warning_text = Text.from_markup(f"Found existing credential for [bold cyan]'{email}'[/bold cyan] at [bold yellow]'{cred_file.name}'[/bold yellow]. Overwriting...")
                console.print(Panel(warning_text, style="bold yellow", title="Updating Credential"))

                # Overwrite the existing file in-place
                with open(cred_file, 'w') as f:
                    json.dump(initialized_creds, f, indent=2)

                success_text = Text.from_markup(f"Successfully updated credential at [bold yellow]'{cred_file.name}'[/bold yellow] for user [bold cyan]'{email}'[/bold cyan].")
                console.print(Panel(success_text, style="bold green", title="Success"))
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


async def export_gemini_cli_to_env():
    """
    Export a Gemini CLI credential JSON file to .env format.
    Generates one .env file per credential.
    """
    console.print(Panel("[bold cyan]Export Gemini CLI Credential to .env[/bold cyan]", expand=False))

    # Find all gemini_cli credentials
    gemini_cli_files = list(OAUTH_BASE_DIR.glob("gemini_cli_oauth_*.json"))

    if not gemini_cli_files:
        console.print(Panel("No Gemini CLI credentials found. Please add one first using 'Add OAuth Credential'.",
                          style="bold red", title="No Credentials"))
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_file in enumerate(gemini_cli_files):
        try:
            with open(cred_file, 'r') as f:
                creds = json.load(f)
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")
            cred_text.append(f"  {i + 1}. {cred_file.name} ({email})\n")
        except Exception as e:
            cred_text.append(f"  {i + 1}. {cred_file.name} (error reading: {e})\n")

    console.print(Panel(cred_text, title="Available Gemini CLI Credentials", style="bold blue"))

    choice = Prompt.ask(
        Text.from_markup("[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"),
        choices=[str(i + 1) for i in range(len(gemini_cli_files))] + ["b"],
        show_choices=False
    )

    if choice.lower() == 'b':
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(gemini_cli_files):
            cred_file = gemini_cli_files[choice_index]

            # Load the credential
            with open(cred_file, 'r') as f:
                creds = json.load(f)

            # Extract metadata
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")
            project_id = creds.get("_proxy_metadata", {}).get("project_id", "")
            tier = creds.get("_proxy_metadata", {}).get("tier", "")

            # Generate .env file name
            safe_email = email.replace("@", "_at_").replace(".", "_")
            env_filename = f"gemini_cli_{safe_email}.env"
            env_filepath = OAUTH_BASE_DIR / env_filename

            # Build .env content
            env_lines = [
                f"# Gemini CLI Credential for: {email}",
                f"# Generated from: {cred_file.name}",
                f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"GEMINI_CLI_ACCESS_TOKEN={creds.get('access_token', '')}",
                f"GEMINI_CLI_REFRESH_TOKEN={creds.get('refresh_token', '')}",
                f"GEMINI_CLI_EXPIRY_DATE={creds.get('expiry_date', 0)}",
                f"GEMINI_CLI_CLIENT_ID={creds.get('client_id', '')}",
                f"GEMINI_CLI_CLIENT_SECRET={creds.get('client_secret', '')}",
                f"GEMINI_CLI_TOKEN_URI={creds.get('token_uri', 'https://oauth2.googleapis.com/token')}",
                f"GEMINI_CLI_UNIVERSE_DOMAIN={creds.get('universe_domain', 'googleapis.com')}",
                f"GEMINI_CLI_EMAIL={email}",
            ]

            # Add project_id if present
            if project_id:
                env_lines.append(f"GEMINI_CLI_PROJECT_ID={project_id}")
            
            # Add tier if present
            if tier:
                env_lines.append(f"GEMINI_CLI_TIER={tier}")

            # Write to .env file
            with open(env_filepath, 'w') as f:
                f.write('\n'.join(env_lines))

            success_text = Text.from_markup(
                f"Successfully exported credential to [bold yellow]'{env_filepath}'[/bold yellow]\n\n"
                f"To use this credential:\n"
                f"1. Copy [bold yellow]{env_filepath.name}[/bold yellow] to your deployment environment\n"
                f"2. Load the variables: [bold cyan]export $(cat {env_filepath.name} | grep -v '^#' | xargs)[/bold cyan]\n"
                f"3. Or source it: [bold cyan]source {env_filepath.name}[/bold cyan]\n"
                f"4. The Gemini CLI provider will automatically use these environment variables"
            )
            console.print(Panel(success_text, style="bold green", title="Success"))
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")
    except Exception as e:
        console.print(Panel(f"An error occurred during export: {e}", style="bold red", title="Error"))


async def export_qwen_code_to_env():
    """
    Export a Qwen Code credential JSON file to .env format.
    Generates one .env file per credential.
    """
    console.print(Panel("[bold cyan]Export Qwen Code Credential to .env[/bold cyan]", expand=False))

    # Find all qwen_code credentials
    qwen_code_files = list(OAUTH_BASE_DIR.glob("qwen_code_oauth_*.json"))

    if not qwen_code_files:
        console.print(Panel("No Qwen Code credentials found. Please add one first using 'Add OAuth Credential'.",
                          style="bold red", title="No Credentials"))
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_file in enumerate(qwen_code_files):
        try:
            with open(cred_file, 'r') as f:
                creds = json.load(f)
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")
            cred_text.append(f"  {i + 1}. {cred_file.name} ({email})\n")
        except Exception as e:
            cred_text.append(f"  {i + 1}. {cred_file.name} (error reading: {e})\n")

    console.print(Panel(cred_text, title="Available Qwen Code Credentials", style="bold blue"))

    choice = Prompt.ask(
        Text.from_markup("[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"),
        choices=[str(i + 1) for i in range(len(qwen_code_files))] + ["b"],
        show_choices=False
    )

    if choice.lower() == 'b':
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(qwen_code_files):
            cred_file = qwen_code_files[choice_index]

            # Load the credential
            with open(cred_file, 'r') as f:
                creds = json.load(f)

            # Extract metadata
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")

            # Generate .env file name
            safe_email = email.replace("@", "_at_").replace(".", "_")
            env_filename = f"qwen_code_{safe_email}.env"
            env_filepath = OAUTH_BASE_DIR / env_filename

            # Build .env content
            env_lines = [
                f"# Qwen Code Credential for: {email}",
                f"# Generated from: {cred_file.name}",
                f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"QWEN_CODE_ACCESS_TOKEN={creds.get('access_token', '')}",
                f"QWEN_CODE_REFRESH_TOKEN={creds.get('refresh_token', '')}",
                f"QWEN_CODE_EXPIRY_DATE={creds.get('expiry_date', 0)}",
                f"QWEN_CODE_RESOURCE_URL={creds.get('resource_url', 'https://portal.qwen.ai/v1')}",
                f"QWEN_CODE_EMAIL={email}",
            ]

            # Write to .env file
            with open(env_filepath, 'w') as f:
                f.write('\n'.join(env_lines))

            success_text = Text.from_markup(
                f"Successfully exported credential to [bold yellow]'{env_filepath}'[/bold yellow]\n\n"
                f"To use this credential:\n"
                f"1. Copy [bold yellow]{env_filepath.name}[/bold yellow] to your deployment environment\n"
                f"2. Load the variables: [bold cyan]export $(cat {env_filepath.name} | grep -v '^#' | xargs)[/bold cyan]\n"
                f"3. Or source it: [bold cyan]source {env_filepath.name}[/bold cyan]\n"
                f"4. The Qwen Code provider will automatically use these environment variables"
            )
            console.print(Panel(success_text, style="bold green", title="Success"))
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")
    except Exception as e:
        console.print(Panel(f"An error occurred during export: {e}", style="bold red", title="Error"))


async def export_iflow_to_env():
    """
    Export an iFlow credential JSON file to .env format.
    Generates one .env file per credential.
    """
    console.print(Panel("[bold cyan]Export iFlow Credential to .env[/bold cyan]", expand=False))

    # Find all iflow credentials
    iflow_files = list(OAUTH_BASE_DIR.glob("iflow_oauth_*.json"))

    if not iflow_files:
        console.print(Panel("No iFlow credentials found. Please add one first using 'Add OAuth Credential'.",
                          style="bold red", title="No Credentials"))
        return

    # Display available credentials
    cred_text = Text()
    for i, cred_file in enumerate(iflow_files):
        try:
            with open(cred_file, 'r') as f:
                creds = json.load(f)
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")
            cred_text.append(f"  {i + 1}. {cred_file.name} ({email})\n")
        except Exception as e:
            cred_text.append(f"  {i + 1}. {cred_file.name} (error reading: {e})\n")

    console.print(Panel(cred_text, title="Available iFlow Credentials", style="bold blue"))

    choice = Prompt.ask(
        Text.from_markup("[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"),
        choices=[str(i + 1) for i in range(len(iflow_files))] + ["b"],
        show_choices=False
    )

    if choice.lower() == 'b':
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(iflow_files):
            cred_file = iflow_files[choice_index]

            # Load the credential
            with open(cred_file, 'r') as f:
                creds = json.load(f)

            # Extract metadata
            email = creds.get("_proxy_metadata", {}).get("email", "unknown")

            # Generate .env file name
            safe_email = email.replace("@", "_at_").replace(".", "_")
            env_filename = f"iflow_{safe_email}.env"
            env_filepath = OAUTH_BASE_DIR / env_filename

            # Build .env content
            # IMPORTANT: iFlow requires BOTH OAuth tokens AND the API key for API requests
            env_lines = [
                f"# iFlow Credential for: {email}",
                f"# Generated from: {cred_file.name}",
                f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"IFLOW_ACCESS_TOKEN={creds.get('access_token', '')}",
                f"IFLOW_REFRESH_TOKEN={creds.get('refresh_token', '')}",
                f"IFLOW_API_KEY={creds.get('api_key', '')}",
                f"IFLOW_EXPIRY_DATE={creds.get('expiry_date', '')}",
                f"IFLOW_EMAIL={email}",
                f"IFLOW_TOKEN_TYPE={creds.get('token_type', 'Bearer')}",
                f"IFLOW_SCOPE={creds.get('scope', 'read write')}",
            ]

            # Write to .env file
            with open(env_filepath, 'w') as f:
                f.write('\n'.join(env_lines))

            success_text = Text.from_markup(
                f"Successfully exported credential to [bold yellow]'{env_filepath}'[/bold yellow]\n\n"
                f"To use this credential:\n"
                f"1. Copy [bold yellow]{env_filepath.name}[/bold yellow] to your deployment environment\n"
                f"2. Load the variables: [bold cyan]export $(cat {env_filepath.name} | grep -v '^#' | xargs)[/bold cyan]\n"
                f"3. Or source it: [bold cyan]source {env_filepath.name}[/bold cyan]\n"
                f"4. The iFlow provider will automatically use these environment variables"
            )
            console.print(Panel(success_text, style="bold green", title="Success"))
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")
    except Exception as e:
        console.print(Panel(f"An error occurred during export: {e}", style="bold red", title="Error"))


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
        console.print(Panel("[bold cyan]Interactive Credential Setup[/bold cyan]", title="--- API Key Proxy ---", expand=False))
    
    while True:
        # Clear screen between menu selections for cleaner UX
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Panel("[bold cyan]Interactive Credential Setup[/bold cyan]", title="--- API Key Proxy ---", expand=False))
        
        console.print(Panel(
            Text.from_markup(
                "1. Add OAuth Credential\n"
                "2. Add API Key\n"
                "3. Export Gemini CLI credential to .env\n"
                "4. Export Qwen Code credential to .env\n"
                "5. Export iFlow credential to .env"
            ),
            title="Choose credential type",
            style="bold blue"
        ))

        setup_type = Prompt.ask(
            Text.from_markup("[bold]Please select an option or type [red]'q'[/red] to quit[/bold]"),
            choices=["1", "2", "3", "4", "5", "q"],
            show_choices=False
        )

        if setup_type.lower() == 'q':
            break

        if setup_type == "1":
            provider_factory, _ = _ensure_providers_loaded()
            available_providers = provider_factory.get_available_providers()
            oauth_friendly_names = {
                "gemini_cli": "Gemini CLI (OAuth)",
                "qwen_code": "Qwen Code (OAuth - also supports API keys)",
                "iflow": "iFlow (OAuth - also supports API keys)"
            }
            
            provider_text = Text()
            for i, provider in enumerate(available_providers):
                display_name = oauth_friendly_names.get(provider, provider.replace('_', ' ').title())
                provider_text.append(f"  {i + 1}. {display_name}\n")
            
            console.print(Panel(provider_text, title="Available Providers for OAuth", style="bold blue"))

            choice = Prompt.ask(
                Text.from_markup("[bold]Please select a provider or type [red]'b'[/red] to go back[/bold]"),
                choices=[str(i + 1) for i in range(len(available_providers))] + ["b"],
                show_choices=False
            )

            if choice.lower() == 'b':
                continue
            
            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_providers):
                    provider_name = available_providers[choice_index]
                    display_name = oauth_friendly_names.get(provider_name, provider_name.replace('_', ' ').title())
                    console.print(f"\nStarting OAuth setup for [bold cyan]{display_name}[/bold cyan]...")
                    await setup_new_credential(provider_name)
                    # Don't clear after OAuth - user needs to see full flow
                    console.print("\n[dim]Press Enter to return to main menu...[/dim]")
                    input()
                else:
                    console.print("[bold red]Invalid choice. Please try again.[/bold red]")
                    await asyncio.sleep(1.5)
            except ValueError:
                console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")
                await asyncio.sleep(1.5)

        elif setup_type == "2":
            await setup_api_key()
            #console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()

        elif setup_type == "3":
            await export_gemini_cli_to_env()
            console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()

        elif setup_type == "4":
            await export_qwen_code_to_env()
            console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()

        elif setup_type == "5":
            await export_iflow_to_env()
            console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()

def run_credential_tool(from_launcher=False):
    """
    Entry point for credential tool.
    
    Args:
        from_launcher: If True, skip loading screen (launcher already showed it)
    """
    # Check if we need to show loading screen
    if not from_launcher:
        # Standalone mode - show full loading UI
        os.system('cls' if os.name == 'nt' else 'clear')
        
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
        print(f"✓ Tool ready in {_elapsed:.2f}s ({len(PROVIDER_PLUGINS)} providers available)")
        
        # Small delay to let user see the ready message
        time.sleep(0.5)
    
    # Run the main async event loop
    # If from launcher, don't clear screen at start to preserve loading messages
    try:
        asyncio.run(main(clear_on_start=not from_launcher))
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal when credential tool exits
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting setup.[/bold yellow]")
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal on keyboard interrupt too