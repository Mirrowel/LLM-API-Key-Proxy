import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys

# This is necessary for the app to find the rotator_library module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.rotator_library.client import RotatingClient

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
if not PROXY_API_KEY:
    raise ValueError("PROXY_API_KEY environment variable not set.")

# Load all Gemini keys from environment variables
gemini_keys = []
i = 1
while True:
    # Start with GEMINI_API_KEY_1, then GEMINI_API_KEY_2, etc.
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if not key and i == 1:
        # Fallback for a single key named just GEMINI_API_KEY
        key = os.getenv("GEMINI_API_KEY")
    
    if key:
        gemini_keys.append(key)
        i += 1
    else:
        break

if not gemini_keys:
    raise ValueError("No GEMINI_API_KEY or GEMINI_API_KEY_n environment variables found.")

# Initialize the rotating client
rotating_client = RotatingClient(api_keys=gemini_keys)

# --- FastAPI App Setup ---
app = FastAPI()
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    if not auth or auth != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _=Depends(verify_api_key)):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses.
    """
    try:
        data = await request.json()
        is_streaming = data.get("stream", False)
        
        response = await rotating_client.acompletion(**data)

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
