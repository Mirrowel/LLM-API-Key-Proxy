# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import time

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
import os
from pathlib import Path
import sys
import argparse
import logging

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument(
    "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
)
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
parser.add_argument(
    "--enable-request-logging",
    action="store_true",
    help="Enable transaction logging in the library (logs request/response with provider correlation).",
)
parser.add_argument(
    "--enable-raw-logging",
    action="store_true",
    help="Enable raw I/O logging at proxy boundary (captures unmodified HTTP data, disabled by default).",
)
parser.add_argument(
    "--add-credential",
    action="store_true",
    help="Launch the interactive tool to add a new OAuth credential.",
)
args, _ = parser.parse_known_args()

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Check if we should launch TUI (no arguments = TUI mode)
if len(sys.argv) == 1:
    # TUI MODE - Load ONLY what's needed for the launcher (fast path!)
    from proxy_app.launcher_tui import run_launcher_tui

    run_launcher_tui()
    # Launcher modifies sys.argv and returns, or exits if user chose Exit
    # If we get here, user chose "Run Proxy" and sys.argv is modified
    # Re-parse arguments with modified sys.argv
    args = parser.parse_args()

# Check if credential tool mode (also doesn't need heavy proxy imports)
if args.add_credential:
    from rotator_library.credential_tool import run_credential_tool

    run_credential_tool()
    sys.exit(0)

# If we get here, we're ACTUALLY running the proxy - NOW show startup messages and start timer
_start_time = time.time()

# Load all .env files from root folder (main .env first, then any additional *.env files)
from dotenv import load_dotenv
from proxy_app.startup_display import mask_secret_for_display as _mask_secret_for_display

# Get the application root directory (EXE dir if frozen, else CWD)
# Inlined here to avoid triggering heavy rotator_library imports before loading screen
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()

# Load main .env first
load_dotenv(_root_dir / ".env")

# Load any additional .env files (e.g., provider_credentials.env)
_env_files_found = list(_root_dir.glob("*.env"))
for _env_file in sorted(_root_dir.glob("*.env")):
    if _env_file.name != ".env":  # Skip main .env (already loaded)
        load_dotenv(_env_file, override=False)  # Don't override existing values

# Log discovered .env files for deployment verification
if _env_files_found:
    _env_names = [_ef.name for _ef in _env_files_found]
    print(f"📁 Loaded {len(_env_files_found)} .env file(s): {', '.join(_env_names)}")


# Get proxy API key for display
proxy_api_key = os.getenv("PROXY_API_KEY")
if proxy_api_key:
    key_display = f"✓ {_mask_secret_for_display(proxy_api_key)}"
else:
    key_display = "✗ Not Set (INSECURE - anyone can access!)"

print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
print("  → Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.security import APIKeyHeader

print("  → Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    import colorlog
    import json
    from typing import AsyncGenerator, Any, List, Optional, Union
    from pydantic import BaseModel, ConfigDict, Field

    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

print("  → Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    import litellm

litellm.suppress_debug_info = True

# Phase 4: Application imports with granular loading messages
print("  → Initializing proxy core...")
with _console.status("[dim]Initializing proxy core...", spinner="dots"):
    from rotator_library import RotatingClient
    from rotator_library.client.protocol_selection import format_client_protocol_error
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.model_info_service import init_model_info_service
    from proxy_app.request_logger import log_request_to_console
    from proxy_app.batch_manager import EmbeddingBatcher
    from proxy_app.detailed_logger import RawIOLogger
    from rotator_library.responses import ResponsesService, ResponsesServiceError
    from rotator_library.core.errors import StructuredAPIResponseError
    from rotator_library.transaction_logger import TransactionLogger

print("  → Discovering provider plugins...")
# Provider lazy loading happens during import, so time it here
_provider_start = time.time()
with _console.status("[dim]Discovering provider plugins...", spinner="dots"):
    from rotator_library import (
        PROVIDER_PLUGINS,
    )  # This triggers lazy load via __getattr__
_provider_time = time.time() - _provider_start

# Get count after import (without timing to avoid double-counting)
_plugin_count = len(PROVIDER_PLUGINS)


# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    """Basic model card for minimal response."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Mirro-Proxy"


class ModelCapabilities(BaseModel):
    """Model capability flags."""

    tool_choice: bool = False
    function_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    system_messages: bool = True
    prompt_caching: bool = False
    assistant_prefill: bool = False


class EnrichedModelCard(BaseModel):
    """Extended model card with pricing and capabilities."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "unknown"
    # Pricing (optional - may not be available for all models)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    cache_read_input_token_cost: Optional[float] = None
    cache_creation_input_token_cost: Optional[float] = None
    # Limits (optional)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    # Capabilities
    mode: str = "chat"
    supported_modalities: List[str] = Field(default_factory=lambda: ["text"])
    supported_output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    capabilities: Optional[ModelCapabilities] = None
    # Debug info (optional)
    _sources: Optional[List[str]] = None
    _match_type: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields from the service


class ModelList(BaseModel):
    """List of models response."""

    object: str = "list"
    data: List[ModelCard]


class EnrichedModelList(BaseModel):
    """List of enriched models with pricing and capabilities."""

    object: str = "list"
    data: List[EnrichedModelCard]


# Calculate total loading time
_elapsed = time.time() - _start_time
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
import os as _os_module

_os_module.system("cls" if _os_module.name == "nt" else "clear")

# Reprint header
print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.utils.paths import get_logs_dir, get_data_file

LOG_DIR = get_logs_dir(_root_dir)

# Configure a console handler with color (INFO and above only, no DEBUG)
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)

# Configure a file handler for INFO-level logs and higher
info_file_handler = logging.FileHandler(LOG_DIR / "proxy.log", encoding="utf-8")
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure a dedicated file handler for all DEBUG-level logs
debug_file_handler = logging.FileHandler(LOG_DIR / "proxy_debug.log", encoding="utf-8")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)


# Create a filter to ensure the debug handler ONLY gets DEBUG messages from the rotator_library
class RotatorDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG and record.name.startswith(
            "rotator_library"
        )


debug_file_handler.addFilter(RotatorDebugFilter())

# Configure a console handler with color
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)


# Add a filter to prevent any LiteLLM logs from cluttering the console
class NoLiteLLMLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("LiteLLM")


console_handler.addFilter(NoLiteLLMLogFilter())

# Get the root logger and set it to DEBUG to capture all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Add all handlers to the root logger
root_logger.addHandler(info_file_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(debug_file_handler)

# Silence other noisy loggers by setting their level higher than root
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Isolate LiteLLM's logger to prevent it from reaching the console.
# We will capture its logs via the logger_fn callback in the client instead.
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.handlers = []
litellm_logger.propagate = False

# Now that logging is configured, log the module load time to debug file only
logging.debug(f"Modules loaded in {_elapsed:.2f}s")

# Load environment variables from .env file
load_dotenv(_root_dir / ".env")

# --- Configuration ---
USE_EMBEDDING_BATCHER = False
ENABLE_REQUEST_LOGGING = args.enable_request_logging
ENABLE_RAW_LOGGING = args.enable_raw_logging
if ENABLE_REQUEST_LOGGING:
    logging.info(
        "Transaction logging is enabled (library-level with provider correlation)."
    )
if ENABLE_RAW_LOGGING:
    logging.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
# Note: PROXY_API_KEY validation moved to server startup to allow credential tool to run first

# Discover API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    if "_API_KEY" in key and key != "PROXY_API_KEY":
        provider = key.split("_API_KEY")[0].lower()
        if provider not in api_keys:
            api_keys[provider] = []
        api_keys[provider].append(value)

# Load model ignore lists from environment variables
ignore_models = {}
for key, value in os.environ.items():
    if key.startswith("IGNORE_MODELS_"):
        provider = key.replace("IGNORE_MODELS_", "").lower()
        models_to_ignore = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        ignore_models[provider] = models_to_ignore
        logging.debug(
            f"Loaded ignore list for provider '{provider}': {models_to_ignore}"
        )

# Load model whitelist from environment variables
whitelist_models = {}
for key, value in os.environ.items():
    if key.startswith("WHITELIST_MODELS_"):
        provider = key.replace("WHITELIST_MODELS_", "").lower()
        models_to_whitelist = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        whitelist_models[provider] = models_to_whitelist
        logging.debug(
            f"Loaded whitelist for provider '{provider}': {models_to_whitelist}"
        )


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    # [MODIFIED] Perform skippable OAuth initialization at startup
    skip_oauth_init = os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"

    # The CredentialManager now handles all discovery, including .env overrides.
    # We pass all environment variables to it for this purpose.
    cred_manager = CredentialManager(os.environ)
    oauth_credentials = cred_manager.discover_and_prepare()

    oauth_credentials = await bootstrap_oauth_credentials(
        oauth_credentials,
        skip=skip_oauth_init,
    )
    # Provider-specific LiteLLM params. API-key Gemini remains configured through
    # normal provider environment keys.
    litellm_provider_params = {}

    # Load global timeout from environment (default 30 seconds)
    global_timeout = int(os.getenv("GLOBAL_TIMEOUT", "30"))

    # The client now uses the root logger configuration
    client = RotatingClient(
        api_keys=api_keys,
        oauth_credentials=oauth_credentials,  # Pass OAuth config
        configure_logging=True,
        global_timeout=global_timeout,
        litellm_provider_params=litellm_provider_params,
        ignore_models=ignore_models,
        whitelist_models=whitelist_models,
        enable_request_logging=ENABLE_REQUEST_LOGGING,
    )

    await client.initialize_usage_managers()

    # Log loaded credentials summary (compact, always visible for deployment verification)
    # _api_summary = ', '.join([f"{p}:{len(c)}" for p, c in api_keys.items()]) if api_keys else "none"
    # _oauth_summary = ', '.join([f"{p}:{len(c)}" for p, c in oauth_credentials.items()]) if oauth_credentials else "none"
    # _total_summary = ', '.join([f"{p}:{len(c)}" for p, c in client.all_credentials.items()])
    # print(f"🔑 Credentials loaded: {_total_summary} (API: {_api_summary} | OAuth: {_oauth_summary})")
    client.background_refresher.start()  # Start the background task
    app.state.rotating_client = client
    # Phase 4 Responses API compatibility service. It currently bridges through
    # the existing chat-completions client path; later native providers can reuse
    # the same route/storage surface without changing clients.
    from rotator_library.config.experimental import get_responses_store_settings
    from rotator_library.responses import create_configured_responses_store

    app.state.responses_service = ResponsesService(store=create_configured_responses_store(), store_settings=get_responses_store_settings())

    # Warn if no provider credentials are configured
    if not client.all_credentials:
        logging.warning("=" * 70)
        logging.warning("⚠️  NO PROVIDER CREDENTIALS CONFIGURED")
        logging.warning("The proxy is running but cannot serve any LLM requests.")
        logging.warning(
            "Launch the credential tool to add API keys or OAuth credentials."
        )
        logging.warning("  • Executable: Run with --add-credential flag")
        logging.warning("  • Source: python src/proxy_app/main.py --add-credential")
        logging.warning("=" * 70)

    os.environ["LITELLM_LOG"] = "ERROR"
    litellm.set_verbose = False
    litellm.drop_params = True
    if USE_EMBEDDING_BATCHER:
        batcher = EmbeddingBatcher(client=client)
        app.state.embedding_batcher = batcher
        logging.info("RotatingClient and EmbeddingBatcher initialized.")
    else:
        app.state.embedding_batcher = None
        logging.info("RotatingClient initialized (EmbeddingBatcher disabled).")

    # Start model info service in background (fetches pricing/capabilities data)
    # This runs asynchronously and doesn't block proxy startup
    model_info_service = await init_model_info_service()
    app.state.model_info_service = model_info_service
    logging.info("Model info service started (fetching pricing data in background).")

    yield

    await client.background_refresher.stop()  # Stop the background task on shutdown
    if app.state.embedding_batcher:
        await app.state.embedding_batcher.stop()
    responses_service = getattr(app.state, "responses_service", None)
    if responses_service:
        await responses_service.close()
    await client.close()

    # Stop model info service
    if hasattr(app.state, "model_info_service") and app.state.model_info_service:
        await app.state.model_info_service.stop()

    if app.state.embedding_batcher:
        logging.info("RotatingClient and EmbeddingBatcher closed.")
    else:
        logging.info("RotatingClient closed.")


# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


def get_responses_service(request: Request) -> ResponsesService:
    """Dependency to get the Responses API service instance from app state."""

    service = getattr(request.app.state, "responses_service", None)
    if service is None:
        from rotator_library.config.experimental import get_responses_store_settings
        from rotator_library.responses import create_configured_responses_store

        service = ResponsesService(store=create_configured_responses_store(), store_settings=get_responses_store_settings())
        request.app.state.responses_service = service
    return service


async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return auth
    if not auth or auth != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth


# --- Anthropic API Key Header ---
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
gemini_api_key_header = APIKeyHeader(name="x-goog-api-key", auto_error=False)


async def verify_anthropic_api_key(
    x_api_key: str = Depends(anthropic_api_key_header),
    auth: str = Depends(api_key_header),
):
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts either x-api-key header (Anthropic style) or Authorization Bearer (OpenAI style).
    """
    if not PROXY_API_KEY:
        return x_api_key or auth
    # Check x-api-key first (Anthropic style)
    if x_api_key and x_api_key == PROXY_API_KEY:
        return x_api_key
    # Fall back to Bearer token (OpenAI style)
    if auth and auth == f"Bearer {PROXY_API_KEY}":
        return auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


async def verify_gemini_api_key(
    request: Request,
    x_api_key: str = Depends(gemini_api_key_header),
    auth: str = Depends(api_key_header),
):
    """Accept Gemini header/query authentication or the shared Bearer form."""

    if not PROXY_API_KEY:
        return x_api_key or auth or request.query_params.get("key")
    query_key = request.query_params.get("key")
    if (
        x_api_key == PROXY_API_KEY
        or query_key == PROXY_API_KEY
        or auth == f"Bearer {PROXY_API_KEY}"
    ):
        return x_api_key or query_key or auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


# Stream wrapping, request overrides, and embedding fan-out live in route_helpers.
from .route_helpers import (  # noqa: E402
    apply_temperature_override,
    execute_embeddings,
    streaming_response_wrapper,
)
# OAuth credential bootstrap lives in startup.
from .startup import bootstrap_oauth_credentials  # noqa: E402


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    # Raw I/O logger captures unmodified HTTP data at proxy boundary (disabled by default)
    raw_logger = RawIOLogger() if ENABLE_RAW_LOGGING else None
    request_data: dict[str, Any] = {}
    try:
        # Read and parse the request body only once at the beginning.
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            status, content = format_client_protocol_error(
            input_protocol="openai_chat",
                error="Invalid JSON in request body.",
                error_type="invalid_request",
                status_code=400,
            )
            return JSONResponse(status_code=status, content=content)

        # Global temperature=0 override (controlled by .env variable, default: OFF)
        apply_temperature_override(request_data)

        # If raw logging is enabled, capture the unmodified request data.
        if raw_logger:
            raw_logger.log_request(headers=request.headers, body=request_data)

        # Extract and log specific reasoning parameters for monitoring.
        model = request_data.get("model")
        generation_cfg = (
            request_data.get("generationConfig", {})
            or request_data.get("generation_config", {})
            or {}
        )
        reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
            "reasoning_effort"
        )

        logging.getLogger("rotator_library").debug(
            f"Handling reasoning parameters: model={model}, reasoning_effort={reasoning_effort}"
        )

        # Log basic request info to console (this is a separate, simpler logger).
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            response_generator = await client.agenerate(
                request_data,
                input_protocol="openai_chat",
                request=request,
            )
            return StreamingResponse(
                streaming_response_wrapper(
                    request, request_data, response_generator, raw_logger
                ),
                media_type="text/event-stream",
            )
        else:
            response = await client.agenerate(
                request_data,
                input_protocol="openai_chat",
                request=request,
            )

            if raw_logger:
                response_headers = (
                    response.headers if hasattr(response, "headers") else None
                )
                status_code = (
                    response.status_code if hasattr(response, "status_code") else 200
                )
                raw_logger.log_final_response(
                    status_code=status_code,
                    headers=response_headers,
                    body=response if isinstance(response, dict) else response.model_dump(),
                )
            return response

    except StructuredAPIResponseError as e:
        return JSONResponse(status_code=e.http_status, content=e.to_protocol_payload("openai_chat"))
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e,
            error_type=(
                "context_window_exceeded"
                if isinstance(e, litellm.ContextWindowExceededError)
                else "invalid_request"
            ),
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.AuthenticationError as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e, error_type="authentication", status_code=401,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.RateLimitError as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e, error_type="rate_limit", status_code=429,
        )
        return JSONResponse(status_code=status, content=content)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e, error_type="server_error", status_code=503,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.Timeout as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e, error_type="proxy_timeout", status_code=504,
        )
        return JSONResponse(status_code=status, content=content)
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e, error_type="server_error", status_code=502,
        )
        return JSONResponse(status_code=status, content=content)
    except Exception as e:
        logging.error(f"Request failed after all retries: {e}")
        # Optionally log the failed request
        if ENABLE_REQUEST_LOGGING:
            try:
                request_data = await request.json()
            except json.JSONDecodeError:
                request_data = {"error": "Could not parse request body"}
            if raw_logger:
                raw_logger.log_final_response(
                    status_code=500, headers=None, body={"error": str(e)}
                )
        status, content = format_client_protocol_error(
            input_protocol="openai_chat",
            error=e,
            error_type="internal_error",
            status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


def _responses_error_response(
    error: ResponsesServiceError,
    protocol: str = "responses",
) -> dict[str, Any]:
    """Return a Responses service failure in the route's own protocol."""

    return error.to_protocol_payload(protocol)


@app.post("/v1/responses")
async def responses_create(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    service: ResponsesService = Depends(get_responses_service),
    _=Depends(verify_api_key),
):
    """Create, store, and optionally reformat an OpenAI Responses object."""

    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        status, content = format_client_protocol_error(
            input_protocol="responses",
            error="Invalid JSON in request body.",
            error_type="invalid_request",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=service.redact_request_for_logging(request_data),
        )
    transaction_logger = TransactionLogger("responses", request_data.get("model", "unknown")) if ENABLE_REQUEST_LOGGING else None
    try:
        request_scope = service.prepare_request_scope(request_data)
        previous_response_access_token = request.headers.get(
            "X-Proxy-Session-Domain"
        )
        if request_data.get("stream"):
            await service.validate_stream_request(
                request_data,
                request_scope=request_scope,
                previous_response_access_token=previous_response_access_token,
            )
            return StreamingResponse(
                service.stream_response(
                    request_data,
                    client,
                    request=request,
                    transaction_logger=transaction_logger,
                    request_scope=request_scope,
                    previous_response_access_token=previous_response_access_token,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Proxy-Session-Domain": request_scope.access_token,
                },
            )
        result = await service.create_response(
            request_data,
            client,
            request=request,
            transaction_logger=transaction_logger,
            request_scope=request_scope,
            previous_response_access_token=previous_response_access_token,
        )
        if logger:
            logger.log_final_response(status_code=200, headers=None, body=result)
        return JSONResponse(
            content=result,
            headers={"X-Proxy-Session-Domain": request_scope.access_token},
        )
    except ResponsesServiceError as e:
        payload = _responses_error_response(e, "responses")
        if logger:
            logger.log_final_response(status_code=e.status_code, headers=None, body=payload)
        return JSONResponse(status_code=e.status_code, content=payload)
    except ValueError as e:
        payload = _responses_error_response(
            ResponsesServiceError(str(e), status_code=400),
            "responses",
        )
        return JSONResponse(status_code=400, content=payload)
    except Exception as e:
        logging.error(f"Responses endpoint error: {e}")
        payload = _responses_error_response(
            ResponsesServiceError(str(e), status_code=500, error_type="internal_error"),
            "responses",
        )
        if logger:
            logger.log_final_response(status_code=500, headers=None, body=payload)
        return JSONResponse(status_code=500, content=payload)


@app.get("/v1/responses/{response_id}")
async def responses_get(
    response_id: str,
    request: Request,
    service: ResponsesService = Depends(get_responses_service),
    _=Depends(verify_api_key),
):
    """Retrieve a stored Responses object by ID."""

    try:
        return JSONResponse(
            content=await service.get_response_with_access_token(
                response_id,
                request.headers.get("X-Proxy-Session-Domain", "public"),
            )
        )
    except ResponsesServiceError as e:
        return JSONResponse(status_code=e.status_code, content=_responses_error_response(e))


@app.delete("/v1/responses/{response_id}")
async def responses_delete(
    response_id: str,
    request: Request,
    service: ResponsesService = Depends(get_responses_service),
    _=Depends(verify_api_key),
):
    """Delete a stored Responses object by ID."""

    try:
        return JSONResponse(
            content=await service.delete_response_with_access_token(
                response_id,
                request.headers.get("X-Proxy-Session-Domain", "public"),
            )
        )
    except ResponsesServiceError as e:
        return JSONResponse(status_code=e.status_code, content=_responses_error_response(e))


@app.get("/v1/responses/{response_id}/input_items")
async def responses_input_items(
    response_id: str,
    request: Request,
    service: ResponsesService = Depends(get_responses_service),
    _=Depends(verify_api_key),
):
    """Return stored input items for a Responses object."""

    try:
        return JSONResponse(
            content=await service.list_input_items_with_access_token(
                response_id,
                request.headers.get("X-Proxy-Session-Domain", "public"),
            )
        )
    except ResponsesServiceError as e:
        return JSONResponse(status_code=e.status_code, content=_responses_error_response(e))


# --- Anthropic Messages API Endpoint ---
@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible Messages API endpoint.

    Accepts the raw /v1/messages payload; the anthropic_messages protocol
    adapter owns validation and conversion, and the response returns in the
    request's protocol (unknown fields transport verbatim on the raw path).

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    try:
        body = await request.json()
    except Exception as exc:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages",
            error=exc,
            error_type="invalid_request_error",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    if not isinstance(body, dict):
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages",
            error=ValueError("request body must be a JSON object"),
            error_type="invalid_request_error",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)

    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=body,
        )

    try:
        # Log the request to console
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(
                request.client.host if request.client else "unknown",
                request.client.port if request.client else 0,
            ),
            request_data=body,
        )

        # Use the library method to handle the request
        result = await client.anthropic_messages(body, raw_request=request)

        if body.get("stream"):
            # Streaming response
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            if logger:
                logger.log_final_response(
                    status_code=200,
                    headers=None,
                    body=result,
                )
            return JSONResponse(content=result)

    except StructuredAPIResponseError as e:
        return JSONResponse(status_code=e.http_status, content=e.to_protocol_payload("anthropic_messages"))
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type=("context_window_exceeded" if isinstance(e, litellm.ContextWindowExceededError) else "invalid_request"),
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.AuthenticationError as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="authentication", status_code=401,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.RateLimitError as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="rate_limit", status_code=429,
        )
        return JSONResponse(status_code=status, content=content)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="server_error", status_code=503,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.Timeout as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="proxy_timeout", status_code=504,
        )
        return JSONResponse(status_code=status, content=content)
    except Exception as e:
        logging.error(f"Anthropic messages endpoint error: {e}")
        if logger:
            logger.log_final_response(
                status_code=500,
                headers=None,
                body={"error": str(e)},
            )
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="internal_error", status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


# --- Anthropic Count Tokens Endpoint ---
@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible count_tokens endpoint.

    Counts the number of tokens that would be used by a Messages API request.
    This is useful for estimating costs and managing context windows.

    Accepts requests in Anthropic's format and returns token count in Anthropic's format.
    """
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
    except Exception as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="invalid_request_error", status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    try:
        # Use the library method to handle the request
        result = await client.anthropic_count_tokens(body)
        return JSONResponse(content=result)

    except (ValueError, litellm.InvalidRequestError) as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="invalid_request_error", status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.AuthenticationError as e:
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="authentication_error", status_code=401,
        )
        return JSONResponse(status_code=status, content=content)
    except Exception as e:
        logging.error(f"Anthropic count_tokens endpoint error: {e}")
        status, content = format_client_protocol_error(
            input_protocol="anthropic_messages", error=e,
            error_type="api_error", status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


@app.post("/v1beta/models/{model:path}:generateContent")
@app.post("/v1/models/{model:path}:generateContent")
async def gemini_generate_content(
    model: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_gemini_api_key),
):
    """Gemini-compatible generation endpoint using protocol-native routing."""

    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        status, content = format_client_protocol_error(
            input_protocol="gemini",
            error="Invalid JSON in request body.", error_type="invalid_request",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    try:
        if payload.get("stream"):
            raise ValueError(
                "Gemini generateContent does not accept stream=true; use streamGenerateContent"
            )
        result = await client.gemini_generate(payload, model=model, raw_request=request)
        return JSONResponse(content=result)
    except StructuredAPIResponseError as error:
        return JSONResponse(status_code=error.http_status, content=error.to_protocol_payload("gemini"))
    except (ValueError, litellm.InvalidRequestError) as error:
        status, content = format_client_protocol_error(
            input_protocol="gemini", error=error, error_type="invalid_request",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except Exception as error:
        status, content = format_client_protocol_error(
            input_protocol="gemini", error=error, error_type="internal_error",
            status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


@app.post("/v1beta/models/{model:path}:streamGenerateContent")
@app.post("/v1/models/{model:path}:streamGenerateContent")
async def gemini_stream_generate_content(
    model: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_gemini_api_key),
):
    """Gemini-compatible streaming generation through canonical routing."""

    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
        response_stream = await client.gemini_stream_generate(
            payload,
            model=model,
            raw_request=request,
        )
        return StreamingResponse(
            streaming_response_wrapper(request, payload, response_stream, input_protocol="gemini"),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except (json.JSONDecodeError, ValueError, litellm.InvalidRequestError) as error:
        status, content = format_client_protocol_error(
            input_protocol="gemini", error=error, error_type="invalid_request",
            status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except StructuredAPIResponseError as error:
        return JSONResponse(
            status_code=error.http_status,
            content=error.to_protocol_payload("gemini"),
        )
    except Exception as error:
        status, content = format_client_protocol_error(
            input_protocol="gemini", error=error, error_type="internal_error",
            status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


@app.post("/v1beta/models/{model:path}:countTokens")
@app.post("/v1/models/{model:path}:countTokens")
async def gemini_count_tokens(
    model: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_gemini_api_key),
):
    """Gemini-compatible token counting endpoint."""

    try:
        payload = await request.json()
        return JSONResponse(content=client.gemini_count_tokens(payload, model=model))
    except (json.JSONDecodeError, ValueError) as error:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": 400, "message": str(error), "status": "INVALID_ARGUMENT"}},
        )


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    try:
        request_data = body.model_dump(exclude_none=True)
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        response = await execute_embeddings(
            batcher if USE_EMBEDDING_BATCHER else None,
            client,
            request_data,
            raw_request=request,
        )

        return response

    except HTTPException as e:
        # Re-raise HTTPException to ensure it's not caught by the generic Exception handler
        raise e
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="invalid_request", status_code=400,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.AuthenticationError as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="authentication_error", status_code=401,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.RateLimitError as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="rate_limit", status_code=429,
        )
        return JSONResponse(status_code=status, content=content)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="service_unavailable", status_code=503,
        )
        return JSONResponse(status_code=status, content=content)
    except litellm.Timeout as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="timeout", status_code=504,
        )
        return JSONResponse(status_code=status, content=content)
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="server_error", status_code=502,
        )
        return JSONResponse(status_code=status, content=content)
    except Exception as e:
        logging.error(f"Embedding request failed: {e}")
        status, content = format_client_protocol_error(
            input_protocol="openai_chat", error=e,
            error_type="internal_error", status_code=500,
        )
        return JSONResponse(status_code=status, content=content)


@app.get("/")
def read_root():
    return {"Status": "API Key Proxy is running"}


@app.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    enriched: bool = True,
):
    """
    Returns a list of available models in the OpenAI-compatible format.

    Query Parameters:
        enriched: If True (default), returns detailed model info with pricing and capabilities.
                  If False, returns minimal OpenAI-compatible response.
    """
    model_ids = await client.get_all_available_models(grouped=False)

    if enriched and hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            # Return enriched model data
            enriched_data = model_info_service.enrich_model_list(model_ids)
            return {"object": "list", "data": enriched_data}

    # Fallback to basic model cards
    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "Mirro-Proxy",
        }
        for model_id in model_ids
    ]
    return {"object": "list", "data": model_cards}


@app.get("/v1/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns detailed information about a specific model.

    Path Parameters:
        model_id: The model ID (e.g., "anthropic/claude-3-opus", "openrouter/openai/gpt-4")
    """
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            info = model_info_service.get_model_info(model_id)
            if info:
                return info.to_dict()

    # Return basic info if service not ready or model not found
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
    }


@app.get("/v1/model-info/stats")
async def model_info_stats(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns statistics about the model info service (for monitoring/debugging).
    """
    if hasattr(request.app.state, "model_info_service"):
        return request.app.state.model_info_service.get_stats()
    return {"error": "Model info service not initialized"}


@app.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())


@app.get("/v1/quota-stats")
async def get_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    provider: str = None,
):
    """
    Returns quota and usage statistics for all credentials.

    This returns cached data from the proxy without making external API calls.
    Use POST to reload from disk or force refresh from external APIs.

    Query Parameters:
        provider: Optional filter to return stats for a specific provider only

    Returns:
        {
            "providers": {
                "provider_name": {
                    "credential_count": int,
                    "active_count": int,
                    "on_cooldown_count": int,
                    "exhausted_count": int,
                    "total_requests": int,
                    "tokens": {...},
                    "approx_cost": float | null,
                    "quota_groups": {...},
                    "credentials": [...]
                }
            },
            "summary": {...},
            "data_source": "cache",
            "timestamp": float
        }
    """
    try:
        stats = await client.get_quota_stats(provider_filter=provider)
        return stats
    except Exception as e:
        logging.error(f"Failed to get quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/quota-stats")
async def refresh_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Refresh quota and usage statistics.

    Request body:
        {
            "action": "reload" | "force_refresh",
            "scope": "all" | "provider" | "credential",
            "provider": "openai",  // required if scope != "all"
            "credential": "openai_key_1"  // required if scope == "credential"
        }

    Actions:
        - reload: Re-read data from disk (no external API calls)
        - force_refresh: Fetch live quota for providers that support it.
                         For other providers, same as reload.

    Returns:
        Same as GET, plus a "refresh_result" field with operation details.
    """
    try:
        data = await request.json()
        action = data.get("action", "reload")
        scope = data.get("scope", "all")
        provider = data.get("provider")
        credential = data.get("credential")

        # Validate parameters
        if action not in ("reload", "force_refresh"):
            raise HTTPException(
                status_code=400,
                detail="action must be 'reload' or 'force_refresh'",
            )

        if scope not in ("all", "provider", "credential"):
            raise HTTPException(
                status_code=400,
                detail="scope must be 'all', 'provider', or 'credential'",
            )

        if scope in ("provider", "credential") and not provider:
            raise HTTPException(
                status_code=400,
                detail="'provider' is required when scope is 'provider' or 'credential'",
            )

        if scope == "credential" and not credential:
            raise HTTPException(
                status_code=400,
                detail="'credential' is required when scope is 'credential'",
            )

        refresh_result = {
            "action": action,
            "scope": scope,
            "provider": provider,
            "credential": credential,
        }

        if action == "reload":
            # Just reload from disk
            start_time = time.time()
            await client.reload_usage_from_disk()
            refresh_result["duration_ms"] = int((time.time() - start_time) * 1000)
            refresh_result["success"] = True
            refresh_result["message"] = "Reloaded usage data from disk"

        elif action == "force_refresh":
            # Force refresh from external API for supported providers.
            result = await client.force_refresh_quota(
                provider=provider if scope in ("provider", "credential") else None,
                credential=credential if scope == "credential" else None,
            )
            refresh_result.update(result)
            refresh_result["success"] = result["failed_count"] == 0

        # Get updated stats
        stats = await client.get_quota_stats(provider_filter=provider)
        stats["refresh_result"] = refresh_result
        stats["data_source"] = "refreshed"

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to refresh quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/token-count")
async def token_count(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Calculates the token count for a given list of messages and a model.
    """
    try:
        data = await request.json()
        model = data.get("model")
        messages = data.get("messages")

        if not model or not messages:
            raise HTTPException(
                status_code=400, detail="'model' and 'messages' are required."
            )

        count = client.token_count(**data)
        return {"token_count": count}

    except Exception as e:
        logging.error(f"Token count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cost-estimate")
async def cost_estimate(request: Request, _=Depends(verify_api_key)):
    """
    Estimates the cost for a request based on token counts and model pricing.

    Request body:
        {
            "model": "anthropic/claude-3-opus",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_tokens": 0,       # optional
            "cache_creation_tokens": 0    # optional
        }

    Returns:
        {
            "model": "anthropic/claude-3-opus",
            "cost": 0.0375,
            "currency": "USD",
            "pricing": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075
            },
            "source": "model_info_service"  # or "litellm_fallback"
        }
    """
    try:
        data = await request.json()
        model = data.get("model")
        prompt_tokens = data.get("prompt_tokens", 0)
        completion_tokens = data.get("completion_tokens", 0)
        cache_read_tokens = data.get("cache_read_tokens", 0)
        cache_creation_tokens = data.get("cache_creation_tokens", 0)

        if not model:
            raise HTTPException(status_code=400, detail="'model' is required.")

        result = {
            "model": model,
            "cost": None,
            "currency": "USD",
            "pricing": {},
            "source": None,
        }

        # Try model info service first
        if hasattr(request.app.state, "model_info_service"):
            model_info_service = request.app.state.model_info_service
            if model_info_service.is_ready:
                cost = model_info_service.calculate_cost(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if cost is not None:
                    cost_info = model_info_service.get_cost_info(model)
                    result["cost"] = cost
                    result["pricing"] = cost_info or {}
                    result["source"] = "model_info_service"
                    return result

        # Fallback to litellm
        try:
            import litellm

            # Create a mock response for cost calculation
            model_info = litellm.get_model_info(model)
            input_cost = model_info.get("input_cost_per_token", 0)
            output_cost = model_info.get("output_cost_per_token", 0)

            if input_cost or output_cost:
                cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
                result["cost"] = cost
                result["pricing"] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                }
                result["source"] = "litellm_fallback"
                return result
        except Exception:
            pass

        result["source"] = "unknown"
        result["error"] = "Pricing data not available for this model"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Cost estimate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Define ENV_FILE for onboarding checks using centralized path
    ENV_FILE = get_data_file(".env")

    # Check if launcher TUI should be shown (no arguments provided)
    if len(sys.argv) == 1:
        # No arguments - show launcher TUI (lazy import)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        # Launcher modifies sys.argv and returns, or exits if user chose Exit
        # If we get here, user chose "Run Proxy" and sys.argv is modified
        # Re-parse arguments with modified sys.argv
        args = parser.parse_args()

    def needs_onboarding() -> bool:
        """
        Check if the proxy needs onboarding (first-time setup).
        Returns True if onboarding is needed, False otherwise.
        """
        # Only check if .env file exists
        # PROXY_API_KEY is optional (will show warning if not set)
        if not ENV_FILE.is_file():
            return True

        return False

    def show_onboarding_message():
        """Display clear explanatory message for why onboarding is needed."""
        os.system(
            "cls" if os.name == "nt" else "clear"
        )  # Clear terminal for clean presentation
        console.print(
            Panel.fit(
                "[bold cyan]🚀 LLM API Key Proxy - First Time Setup[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print("[bold yellow]:warning:  Configuration Required[/bold yellow]\n")

        console.print("The proxy needs initial configuration:")
        console.print("  [red]:x: No .env file found[/red]")

        console.print("\n[bold]Why this matters:[/bold]")
        console.print("  • The .env file stores your credentials and settings")
        console.print("  • PROXY_API_KEY protects your proxy from unauthorized access")
        console.print("  • Provider API keys enable LLM access")

        console.print("\n[bold]What happens next:[/bold]")
        console.print("  1. We'll create a .env file with PROXY_API_KEY")
        console.print("  2. You can add LLM provider credentials (API keys or OAuth)")
        console.print("  3. The proxy will then start normally")

        console.print(
            "\n[bold yellow]:warning:  Note:[/bold yellow] The credential tool adds PROXY_API_KEY by default."
        )
        console.print("   You can remove it later if you want an unsecured proxy.\n")

        console.input(
            "[bold green]Press Enter to launch the credential setup tool...[/bold green]"
        )

    # Check if user explicitly wants to add credentials
    if args.add_credential:
        # Import and call ensure_env_defaults to create .env and PROXY_API_KEY if needed
        from rotator_library.credential_tool import ensure_env_defaults

        ensure_env_defaults()
        # Reload environment variables after ensure_env_defaults creates/updates .env
        load_dotenv(ENV_FILE, override=True)
        run_credential_tool()
    else:
        # Check if onboarding is needed
        if needs_onboarding():
            # Import console from rich for better messaging
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            # Show clear explanatory message
            show_onboarding_message()

            # Launch credential tool automatically
            from rotator_library.credential_tool import ensure_env_defaults

            ensure_env_defaults()
            load_dotenv(ENV_FILE, override=True)
            run_credential_tool()

            # After credential tool exits, reload and re-check
            load_dotenv(ENV_FILE, override=True)
            # Re-read PROXY_API_KEY from environment
            PROXY_API_KEY = os.getenv("PROXY_API_KEY")

            # Verify onboarding is complete
            if needs_onboarding():
                console.print("\n[bold red]:x: Configuration incomplete.[/bold red]")
                console.print(
                    "The proxy still cannot start. Please ensure PROXY_API_KEY is set in .env\n"
                )
                sys.exit(1)
            else:
                console.print(
                    "\n[bold green]:white_check_mark: Configuration complete![/bold green]"
                )
                console.print("\nStarting proxy server...\n")

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
