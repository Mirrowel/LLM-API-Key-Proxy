import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys

# Add the 'src' directory to the Python path to allow importing 'rotating_api_key_client'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rotator_library import RotatingClient, PROVIDER_PLUGINS

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
if not PROXY_API_KEY:
    raise ValueError("PROXY_API_KEY environment variable not set.")

# Load all provider API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    # Exclude PROXY_API_KEY from being treated as a provider API key
    if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
        parts = key.split("_API_KEY")
        provider = parts[0].lower()
        if provider not in api_keys:
            api_keys[provider] = []
        api_keys[provider].append(value)

if not api_keys:
    raise ValueError("No provider API keys found in environment variables.")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    app.state.rotating_client = RotatingClient(api_keys=api_keys)
    print("RotatingClient initialized.")
    yield
    await app.state.rotating_client.close()
    print("RotatingClient closed.")

# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client

async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    if not auth or auth != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, 
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key)
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses.
    """
    try:
        data = await request.json()
        is_streaming = data.get("stream", False)
        
        response = await client.acompletion(**data)

        if is_streaming:
            return StreamingResponse(response, media_type="text/event-stream")
        else:
            return response

    except Exception as e:
        logging.error(f"Request failed after all retries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Status": "API Key Proxy is running"}

@app.get("/v1/models")
async def list_models(
    grouped: bool = False, 
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key)
):
    """
    Returns a list of available models from all configured providers.
    Optionally returns them as a flat list if grouped=False.
    """
    models = await client.get_all_available_models(grouped=grouped)
    return models

@app.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())

@app.post("/v1/token-count")
async def token_count(
    request: Request, 
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key)
):
    """
    Calculates the token count for a given list of messages and a model.
    """
    try:
        data = await request.json()
        model = data.get("model")
        messages = data.get("messages")

        if not model or not messages:
            raise HTTPException(status_code=400, detail="'model' and 'messages' are required.")

        count = client.token_count(model=model, messages=messages)
        return {"token_count": count}

    except Exception as e:
        logging.error(f"Token count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
