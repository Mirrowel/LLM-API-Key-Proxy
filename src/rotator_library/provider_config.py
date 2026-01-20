# src/rotator_library/provider_config.py
"""
Centralized provider configuration for the rotator library.

This module handles:
- Known LiteLLM provider definitions
- API base overrides for known providers
- Custom OpenAI-compatible provider detection and routing
"""

import os
import re
import logging
from typing import Dict, Any, Set, Optional

lib_logger = logging.getLogger("rotator_library")

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

LITELLM_PROVIDERS = {
    # =========================================================================
    # POPULAR - Most commonly used providers
    # =========================================================================
    "OpenAI": {
        "api_key": "OPENAI_API_KEY",
        "category": "popular",
    },
    "Anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "category": "popular",
    },
    "Google AI Studio (Gemini)": {
        "api_key": "GEMINI_API_KEY",
        "category": "popular",
    },
    "xAI": {
        "api_key": "XAI_API_KEY",
        "category": "popular",
    },
    "Deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "category": "popular",
    },
    "Mistral AI": {
        "api_key": "MISTRAL_API_KEY",
        "category": "popular",
    },
    "Codestral (Mistral)": {
        "api_key": "CODESTRAL_API_KEY",
        "category": "popular",
    },
    "OpenRouter": {
        "api_key": "OPENROUTER_API_KEY",
        "category": "popular",
        "extra_vars": [
            ("OPENROUTER_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Groq": {
        "api_key": "GROQ_API_KEY",
        "category": "popular",
    },
    "Chutes": {
        "api_key": "CHUTES_API_KEY",
        "category": "popular",
    },
    "NVIDIA NIM": {
        "api_key": "NVIDIA_NIM_API_KEY",
        "category": "popular",
        "extra_vars": [
            ("NVIDIA_NIM_API_BASE", "NIM API Base (optional)", None),
        ],
    },
    "Perplexity AI": {
        "api_key": "PERPLEXITYAI_API_KEY",
        "category": "popular",
    },
    "Moonshot AI": {
        "api_key": "MOONSHOT_API_KEY",
        "category": "popular",
        "extra_vars": [
            ("MOONSHOT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Z.AI (Zhipu AI)": {
        "api_key": "ZAI_API_KEY",
        "category": "popular",
    },
    "MiniMax": {
        "api_key": "MINIMAX_API_KEY",
        "category": "popular",
        "extra_vars": [
            ("MINIMAX_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Xiaomi MiMo": {
        "api_key": "XIAOMI_MIMO_API_KEY",
        "category": "popular",
    },
    "NanoGPT": {
        "api_key": "NANOGPT_API_KEY",
        "category": "popular",
    },
    "Synthetic": {
        "api_key": "SYNTHETIC_API_KEY",
        "category": "popular",
    },
    # =========================================================================
    # CLOUD PLATFORMS - Aggregators & cloud inference platforms
    # =========================================================================
    "Together AI": {
        "api_key": "TOGETHERAI_API_KEY",
        "category": "cloud",
    },
    "Fireworks AI": {
        "api_key": "FIREWORKS_AI_API_KEY",
        "category": "cloud",
        "extra_vars": [
            ("FIREWORKS_AI_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Replicate": {
        "api_key": "REPLICATE_API_KEY",
        "category": "cloud",
    },
    "DeepInfra": {
        "api_key": "DEEPINFRA_API_KEY",
        "category": "cloud",
    },
    "Anyscale": {
        "api_key": "ANYSCALE_API_KEY",
        "category": "cloud",
    },
    "Baseten": {
        "api_key": "BASETEN_API_KEY",
        "category": "cloud",
    },
    "Predibase": {
        "api_key": "PREDIBASE_API_KEY",
        "category": "cloud",
    },
    "Novita AI": {
        "api_key": "NOVITA_API_KEY",
        "category": "cloud",
    },
    "Featherless AI": {
        "api_key": "FEATHERLESS_AI_API_KEY",
        "category": "cloud",
    },
    "Hyperbolic": {
        "api_key": "HYPERBOLIC_API_KEY",
        "category": "cloud",
    },
    "Lambda AI": {
        "api_key": "LAMBDA_API_KEY",
        "category": "cloud",
        "extra_vars": [
            ("LAMBDA_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Nebius AI Studio": {
        "api_key": "NEBIUS_API_KEY",
        "category": "cloud",
    },
    "Galadriel": {
        "api_key": "GALADRIEL_API_KEY",
        "category": "cloud",
    },
    "FriendliAI": {
        "api_key": "FRIENDLI_TOKEN",
        "category": "cloud",
    },
    "SambaNova": {
        "api_key": "SAMBANOVA_API_KEY",
        "category": "cloud",
    },
    "Cerebras": {
        "api_key": "CEREBRAS_API_KEY",
        "category": "cloud",
    },
    "Meta Llama": {
        "api_key": "LLAMA_API_KEY",
        "category": "cloud",
    },
    "AI21": {
        "api_key": "AI21_API_KEY",
        "category": "cloud",
    },
    "Cohere": {
        "api_key": "COHERE_API_KEY",
        "category": "cloud",
    },
    "Aleph Alpha": {
        "api_key": "ALEPHALPHA_API_KEY",
        "category": "cloud",
    },
    "Hugging Face": {
        "api_key": "HF_TOKEN",
        "category": "cloud",
    },
    "GitHub Models": {
        "api_key": "GITHUB_API_KEY",
        "category": "cloud",
    },
    "Helicone": {
        "api_key": "HELICONE_API_KEY",
        "category": "cloud",
        "note": "LLM gateway/proxy with analytics.",
    },
    "Heroku": {
        "api_key": "HEROKU_API_KEY",
        "category": "cloud",
        "extra_vars": [
            (
                "HEROKU_API_BASE",
                "Heroku Inference URL",
                "https://us.inference.heroku.com",
            ),
        ],
    },
    "Morph": {
        "api_key": "MORPH_API_KEY",
        "category": "cloud",
    },
    "Poe": {
        "api_key": "POE_API_KEY",
        "category": "cloud",
    },
    "LlamaGate": {
        "api_key": "LLAMAGATE_API_KEY",
        "category": "cloud",
    },
    "Manus": {
        "api_key": "MANUS_API_KEY",
        "category": "cloud",
    },
    # =========================================================================
    # ENTERPRISE / COMPLEX AUTH - Major cloud providers (may need extra config)
    # =========================================================================
    "Azure OpenAI": {
        "api_key": "AZURE_API_KEY",
        "category": "enterprise",
        "note": "Requires Azure endpoint and API version.",
        "extra_vars": [
            ("AZURE_API_BASE", "Azure endpoint URL", None),
            ("AZURE_API_VERSION", "API version", "2024-02-15-preview"),
        ],
    },
    "Azure AI Studio": {
        "api_key": "AZURE_AI_API_KEY",
        "category": "enterprise",
        "extra_vars": [
            ("AZURE_AI_API_BASE", "Azure AI endpoint URL", None),
        ],
    },
    "Vertex AI": {
        "api_key": "GOOGLE_APPLICATION_CREDENTIALS",
        "category": "enterprise",
        "note": "Uses Google Cloud service account. Enter path to credentials JSON file.",
        "extra_vars": [
            ("VERTEXAI_PROJECT", "GCP Project ID", None),
            ("VERTEXAI_LOCATION", "GCP Location", "us-central1"),
        ],
    },
    "AWS Bedrock": {
        "api_key": "AWS_ACCESS_KEY_ID",
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "AWS Sagemaker": {
        "api_key": "AWS_ACCESS_KEY_ID",
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "Databricks": {
        "api_key": "DATABRICKS_API_KEY",
        "category": "enterprise",
        "extra_vars": [
            ("DATABRICKS_API_BASE", "Databricks workspace URL", None),
        ],
    },
    "Snowflake": {
        "api_key": "SNOWFLAKE_JWT",
        "category": "enterprise",
        "note": "Uses JWT authentication.",
        "extra_vars": [
            ("SNOWFLAKE_ACCOUNT_ID", "Snowflake Account ID", None),
        ],
    },
    "IBM watsonx.ai": {
        "api_key": "WATSONX_APIKEY",
        "category": "enterprise",
        "extra_vars": [
            ("WATSONX_URL", "watsonx.ai URL (optional)", None),
        ],
    },
    "Cloudflare Workers AI": {
        "api_key": "CLOUDFLARE_API_KEY",
        "category": "enterprise",
        "extra_vars": [
            ("CLOUDFLARE_ACCOUNT_ID", "Cloudflare Account ID", None),
        ],
    },
    # =========================================================================
    # SPECIALIZED - Image, audio, embeddings, rerank providers
    # =========================================================================
    "Stability AI": {
        "api_key": "STABILITY_API_KEY",
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "Fal AI": {
        "api_key": "FAL_AI_API_KEY",
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "RunwayML": {
        "api_key": "RUNWAYML_API_KEY",
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "Recraft": {
        "api_key": "RECRAFT_API_KEY",
        "category": "specialized",
        "note": "Image generation and editing.",
        "extra_vars": [
            ("RECRAFT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Topaz": {
        "api_key": "TOPAZ_API_KEY",
        "category": "specialized",
        "note": "Image enhancement provider.",
    },
    "ElevenLabs": {
        "api_key": "ELEVENLABS_API_KEY",
        "category": "specialized",
        "note": "Text-to-speech and audio transcription.",
    },
    "Deepgram": {
        "api_key": "DEEPGRAM_API_KEY",
        "category": "specialized",
        "note": "Audio transcription provider.",
    },
    "Voyage AI": {
        "api_key": "VOYAGE_API_KEY",
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "Jina AI": {
        "api_key": "JINA_AI_API_KEY",
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "Clarifai": {
        "api_key": "CLARIFAI_API_KEY",
        "category": "specialized",
    },
    "NLP Cloud": {
        "api_key": "NLP_CLOUD_API_KEY",
        "category": "specialized",
    },
    "Milvus": {
        "api_key": "MILVUS_API_KEY",
        "category": "specialized",
        "note": "Vector database provider.",
        "extra_vars": [
            ("MILVUS_API_BASE", "Milvus Server URL", None),
        ],
    },
    # =========================================================================
    # REGIONAL - Region-specific or specialized regional providers
    # =========================================================================
    "Dashscope (Qwen)": {
        "api_key": "DASHSCOPE_API_KEY",
        "category": "regional",
        "note": "Alibaba Cloud Qwen models.",
    },
    "Volcano Engine": {
        "api_key": "VOLCENGINE_API_KEY",
        "category": "regional",
        "note": "ByteDance cloud platform.",
    },
    "OVHCloud AI Endpoints": {
        "api_key": "OVHCLOUD_API_KEY",
        "category": "regional",
        "note": "European cloud provider.",
    },
    "Nscale (EU Sovereign)": {
        "api_key": "NSCALE_API_KEY",
        "category": "regional",
        "note": "EU sovereign cloud.",
    },
    # =========================================================================
    # LOCAL / SELF-HOSTED - Run locally or on your own infrastructure
    # =========================================================================
    # NOTE: Providers with no API key are commented out because the library
    # requires credentials (API keys or OAuth files) to function.
    # Use "Add Custom OpenAI-Compatible Provider" for local providers.
    #
    # "Ollama": {
    #     "api_key": None,  # No API key - use custom provider option instead
    #     "category": "local",
    #     "note": "Local provider. No API key required. Make sure Ollama is running.",
    #     "extra_vars": [
    #         ("OLLAMA_API_BASE", "Ollama URL", "http://localhost:11434"),
    #     ],
    # },
    "LM Studio": {
        "api_key": "LM_STUDIO_API_KEY",
        "category": "local",
        "note": "Local provider. API key is optional. Start LM Studio server first.",
        "extra_vars": [
            ("LM_STUDIO_API_BASE", "API Base URL", "http://localhost:1234/v1"),
        ],
    },
    # "Llamafile": {
    #     "api_key": None,  # No API key - use custom provider option instead
    #     "category": "local",
    #     "note": "Local provider. No API key required.",
    #     "extra_vars": [
    #         ("LLAMAFILE_API_BASE", "Llamafile URL", "http://localhost:8080/v1"),
    #     ],
    # },
    "vLLM (Hosted)": {
        "api_key": "HOSTED_VLLM_API_KEY",
        "category": "local",
        "note": "Self-hosted vLLM server. API key is optional.",
        "extra_vars": [
            ("HOSTED_VLLM_API_BASE", "vLLM Server URL", None),
        ],
    },
    "Xinference": {
        "api_key": "XINFERENCE_API_KEY",
        "category": "local",
        "note": "Local Xinference server. API key is optional.",
        "extra_vars": [
            ("XINFERENCE_API_BASE", "Xinference URL", "http://127.0.0.1:9997/v1"),
        ],
    },
    "Infinity": {
        "api_key": "INFINITY_API_KEY",
        "category": "local",
        "note": "Self-hosted embeddings/rerank server. API key is optional.",
        "extra_vars": [
            ("INFINITY_API_BASE", "Infinity Server URL", "http://localhost:8080"),
        ],
    },
    "LiteLLM Proxy": {
        "api_key": "LITELLM_PROXY_API_KEY",
        "category": "local",
        "note": "Self-hosted LiteLLM Proxy gateway.",
        "extra_vars": [
            ("LITELLM_PROXY_API_BASE", "LiteLLM Proxy URL", "http://localhost:4000"),
        ],
    },
    "LangGraph": {
        "api_key": "LANGGRAPH_API_KEY",
        "category": "local",
        "note": "Self-hosted LangGraph server.",
        "extra_vars": [
            ("LANGGRAPH_API_BASE", "LangGraph URL", "http://localhost:2024"),
        ],
    },
    "RAGFlow": {
        "api_key": "RAGFLOW_API_KEY",
        "category": "local",
        "note": "Self-hosted RAGFlow server.",
        "extra_vars": [
            ("RAGFLOW_API_BASE", "RAGFlow URL", "http://localhost:9380"),
        ],
    },
    "Docker Model Runner": {
        "api_key": "DOCKER_MODEL_RUNNER_API_KEY",
        "category": "local",
        "note": "Local Docker Model Runner. API key is optional.",
        "extra_vars": [
            (
                "DOCKER_MODEL_RUNNER_API_BASE",
                "Docker Model Runner URL",
                "http://localhost:22088",
            ),
        ],
    },
    "Lemonade": {
        "api_key": "LEMONADE_API_KEY",
        "category": "local",
        "note": "Local proxy. API key is optional.",
        "extra_vars": [
            ("LEMONADE_API_BASE", "Lemonade URL", "http://localhost:8000/api/v1"),
        ],
    },
    # "Petals": {
    #     "api_key": None,  # No API key - use custom provider option instead
    #     "category": "local",
    #     "note": "Distributed inference network. No API key required.",
    # },
    # "Triton Inference Server": {
    #     "api_key": None,  # No API key - use custom provider option instead
    #     "category": "local",
    #     "note": "NVIDIA Triton server. No API key required.",
    # },
    # =========================================================================
    # OTHER - Miscellaneous providers
    # =========================================================================
    "AI/ML API": {
        "api_key": "AIML_API_KEY",
        "category": "other",
        "extra_vars": [
            ("AIML_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "Abliteration": {
        "api_key": "ABLITERATION_API_KEY",
        "category": "other",
    },
    "Amazon Nova": {
        "api_key": "AMAZON_NOVA_API_KEY",
        "category": "other",
    },
    "Apertis AI (Stima)": {
        "api_key": "STIMA_API_KEY",
        "category": "other",
    },
    "Bytez": {
        "api_key": "BYTEZ_API_KEY",
        "category": "other",
    },
    "CometAPI": {
        "api_key": "COMETAPI_KEY",
        "category": "other",
    },
    "CompactifAI": {
        "api_key": "COMPACTIFAI_API_KEY",
        "category": "other",
    },
    "DataRobot": {
        "api_key": "DATAROBOT_API_KEY",
        "category": "other",
        "extra_vars": [
            ("DATAROBOT_API_BASE", "DataRobot URL", "https://app.datarobot.com"),
        ],
    },
    "GradientAI": {
        "api_key": "GRADIENT_AI_API_KEY",
        "category": "other",
        "extra_vars": [
            ("GRADIENT_AI_AGENT_ENDPOINT", "Gradient AI Endpoint (optional)", None),
        ],
    },
    "PublicAI": {
        "api_key": "PUBLICAI_API_KEY",
        "category": "other",
        "extra_vars": [
            ("PUBLICAI_API_BASE", "PublicAI URL", "https://platform.publicai.co/"),
        ],
    },
    "v0": {
        "api_key": "V0_API_KEY",
        "category": "other",
    },
    "Vercel AI Gateway": {
        "api_key": "VERCEL_AI_GATEWAY_API_KEY",
        "category": "other",
    },
    "Weights & Biases": {
        "api_key": "WANDB_API_KEY",
        "category": "other",
    },
}

# Category display order and labels
PROVIDER_CATEGORIES = [
    ("popular", "Popular"),
    ("cloud", "Cloud Platforms"),
    ("enterprise", "Enterprise / Complex Auth"),
    ("specialized", "Specialized (Image/Audio/Embeddings)"),
    ("regional", "Regional"),
    ("local", "Local / Self-Hosted"),
    ("custom", "Custom (First-Party)"),
    ("custom_openai", "Custom OpenAI-Compatible"),
    ("other", "Other"),
]


def _build_known_providers_set() -> Set[str]:
    """
    Extract provider names from LITELLM_PROVIDERS api_key patterns.

    Examples:
        OPENAI_API_KEY → openai
        ANTHROPIC_API_KEY → anthropic
        GEMINI_API_KEY → gemini

    Returns:
        Set of lowercase provider names known to LiteLLM.
    """
    known = set()
    for provider_info in LITELLM_PROVIDERS.values():
        api_key_var = provider_info.get("api_key", "")
        if not api_key_var:
            continue
        # Extract provider name: OPENAI_API_KEY → openai
        # Handle special cases like HF_TOKEN, FRIENDLI_TOKEN
        if api_key_var.endswith("_API_KEY"):
            provider_name = api_key_var[:-8].lower()  # Remove _API_KEY
            known.add(provider_name)
        elif api_key_var.endswith("_TOKEN"):
            provider_name = api_key_var[:-6].lower()  # Remove _TOKEN
            known.add(provider_name)
        elif api_key_var.endswith("_APIKEY"):
            provider_name = api_key_var[:-7].lower()  # Remove _APIKEY
            known.add(provider_name)
        elif api_key_var.endswith("_JWT"):
            provider_name = api_key_var[:-4].lower()  # Remove _JWT
            known.add(provider_name)
        elif api_key_var.endswith("_KEY"):
            provider_name = api_key_var[:-4].lower()  # Remove _KEY (e.g., COMETAPI_KEY)
            known.add(provider_name)
    return known


# Pre-computed set of known provider names
KNOWN_PROVIDERS: Set[str] = _build_known_providers_set()


class ProviderConfig:
    """
    Centralized provider configuration handling.

    Handles:
    - API base overrides for known LiteLLM providers
    - Custom OpenAI-compatible providers (unknown provider names)

    Usage patterns:

    1. Override existing provider's API base:
       Set OPENAI_API_BASE=http://my-local-llm/v1
       Request: openai/gpt-4 → LiteLLM gets model="openai/gpt-4", api_base="http://..."

    2. Custom OpenAI-compatible provider:
       Set MYSERVER_API_BASE=http://myserver:8000/v1
       Request: myserver/llama-3 → LiteLLM gets model="openai/llama-3",
                api_base="http://...", custom_llm_provider="openai"
    """

    def __init__(self):
        self._api_bases: Dict[str, str] = {}
        self._custom_providers: Set[str] = set()
        self._load_api_bases()

    def _load_api_bases(self) -> None:
        """
        Load all <PROVIDER>_API_BASE environment variables.

        Detects whether each is an override for a known provider
        or defines a new custom provider.
        """
        for key, value in os.environ.items():
            if key.endswith("_API_BASE") and value:
                provider = key[:-9].lower()  # Remove _API_BASE
                self._api_bases[provider] = value.rstrip("/")

                # Track if this is a custom provider (not known to LiteLLM)
                if provider not in KNOWN_PROVIDERS:
                    self._custom_providers.add(provider)
                    lib_logger.info(
                        f"Detected custom OpenAI-compatible provider: {provider} "
                        f"(api_base: {value})"
                    )
                else:
                    lib_logger.info(
                        f"Detected API base override for {provider}: {value}"
                    )

    def is_known_provider(self, provider: str) -> bool:
        """Check if provider is known to LiteLLM."""
        return provider.lower() in KNOWN_PROVIDERS

    def is_custom_provider(self, provider: str) -> bool:
        """Check if provider is a custom OpenAI-compatible provider."""
        return provider.lower() in self._custom_providers

    def get_api_base(self, provider: str) -> Optional[str]:
        """Get configured API base for a provider, if any."""
        return self._api_bases.get(provider.lower())

    def get_custom_providers(self) -> Set[str]:
        """Get the set of detected custom provider names."""
        return self._custom_providers.copy()

    def convert_for_litellm(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model params for LiteLLM call.

        Handles:
        - Known provider with _API_BASE: pass api_base as override
        - Unknown provider with _API_BASE: convert to openai/, set custom_llm_provider
        - No _API_BASE configured: pass through unchanged

        Args:
            **kwargs: LiteLLM call kwargs including 'model'

        Returns:
            Modified kwargs dict ready for LiteLLM
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        # Extract provider from model string (e.g., "openai/gpt-4" → "openai")
        provider = model.split("/")[0].lower()
        api_base = self._api_bases.get(provider)

        if not api_base:
            # No override configured for this provider
            return kwargs

        # Create a copy to avoid modifying the original
        kwargs = kwargs.copy()

        if provider in KNOWN_PROVIDERS:
            # Known provider - just add api_base override
            kwargs["api_base"] = api_base
            lib_logger.debug(
                f"Applying api_base override for known provider {provider}: {api_base}"
            )
        else:
            # Custom provider - route through OpenAI-compatible endpoint
            model_name = model.split("/", 1)[1] if "/" in model else model
            kwargs["model"] = f"openai/{model_name}"
            kwargs["api_base"] = api_base
            kwargs["custom_llm_provider"] = "openai"
            lib_logger.debug(
                f"Routing custom provider {provider} through openai: "
                f"model={kwargs['model']}, api_base={api_base}"
            )

        return kwargs
