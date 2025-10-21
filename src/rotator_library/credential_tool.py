# src/rotator_library/credential_tool.py

import asyncio
import json
import re
from pathlib import Path
from dotenv import set_key, get_key

from .provider_factory import get_provider_auth_class, get_available_providers
from .providers import PROVIDER_PLUGINS
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)
# Use a direct path to the .env file in the project root
ENV_FILE = Path.cwd() / ".env"


console = Console()

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
        "Cohere": "COHERE_API_KEY", "Mistral AI": "MISTRAL_API_KEY",
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
    oauth_providers = {'gemini_cli', 'qwen_code'}
    discovered_providers = {
        p.replace('_', ' ').title(): p.upper() + "_API_KEY"
        for p in PROVIDER_PLUGINS.keys()
        if p not in oauth_providers and p.replace('_', ' ').title() not in LITELLM_PROVIDERS
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
    An interactive CLI tool to add new credentials.
    """
    ensure_env_defaults()
    console.print(Panel("[bold cyan]Interactive Credential Setup[/bold cyan]", title="--- API Key Proxy ---", expand=False))
    
    while True:
        console.print(Panel(
            Text.from_markup("1. Add OAuth Credential\n2. Add API Key"),
            title="Choose credential type",
            style="bold blue"
        ))
        
        setup_type = Prompt.ask(
            Text.from_markup("[bold]Please select an option or type [red]'q'[/red] to quit[/bold]"),
            choices=["1", "2", "q"],
            show_choices=False
        )

        if setup_type.lower() == 'q':
            break
        
        if setup_type == "1":
            available_providers = get_available_providers()
            oauth_friendly_names = {
                "gemini_cli": "Gemini CLI (OAuth)",
                "qwen_code": "Qwen Code (OAuth)"
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
                else:
                    console.print("[bold red]Invalid choice. Please try again.[/bold red]")
            except ValueError:
                console.print("[bold red]Invalid input. Please enter a number or 'b'.[/bold red]")

        elif setup_type == "2":
            await setup_api_key()

        console.print("\n" + "="*50 + "\n")

def run_credential_tool():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting setup.[/bold yellow]")