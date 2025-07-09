from typing import Optional

# A comprehensive map of provider names to their base URLs.
PROVIDER_URL_MAP = {
    "perplexity": "https://api.perplexity.ai",
    "anyscale": "https://api.endpoints.anyscale.com/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "nvidia_nim": "https://integrate.api.nvidia.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "ai21_chat": "https://api.ai21.com/studio/v1",
    "codestral": "https://codestral.mistral.ai/v1",
    "text-completion-codestral": "https://codestral.mistral.ai/v1",
    "empower": "https://app.empower.dev/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "friendliai": "https://api.friendli.ai/serverless/v1",
    "galadriel": "https://api.galadriel.com/v1",
    "meta_llama": "https://api.llama.com/compat/v1",
    "featherless_ai": "https://api.featherless.ai/v1",
    "nscale": "https://api.nscale.com/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "cohere": "https://api.cohere.ai/v1",
    "bedrock": "https://bedrock-runtime.us-east-1.amazonaws.com",
    "openrouter": "https://openrouter.ai/api/v1",
}

def get_provider_endpoint(provider: str, model_name: str, incoming_path: str) -> Optional[str]:
    """
    Constructs the full provider endpoint URL based on the provider and incoming request path.
    """
    base_url = PROVIDER_URL_MAP.get(provider)
    if not base_url:
        return None

    # Determine the specific action from the incoming path (e.g., 'chat/completions')
    action = incoming_path.split('/v1/', 1)[-1] if '/v1/' in incoming_path else incoming_path

    # --- Provider-specific endpoint structures ---
    if provider == "gemini":
        if action == "chat/completions":
            return f"{base_url}/models/{model_name}:generateContent"
        elif action == "embeddings":
            return f"{base_url}/models/{model_name}:embedContent"
    
    elif provider == "anthropic":
        if action == "chat/completions":
            return f"{base_url}/messages"

    elif provider == "cohere":
        if action == "chat/completions":
            return f"{base_url}/chat"
        elif action == "embeddings":
            return f"{base_url}/embed"

    # Default for OpenAI-compatible providers
    # Most of these have /v1 in the base URL already, so we just append the action.
    if base_url.endswith(("/v1", "/v1/openai")):
        return f"{base_url}/{action}"
    
    # Fallback for other cases
    return f"{base_url}/v1/{action}"