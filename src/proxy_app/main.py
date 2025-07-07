print("Proxy starting...")
print("GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys
import json
from typing import AsyncGenerator, Any, List, Optional, Union
from pydantic import BaseModel
import argparse
import litellm

# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to.")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
parser.add_argument("--enable-request-logging", action="store_true", help="Enable request logging.")
args, _ = parser.parse_known_args()


# Add the 'src' directory to the Python path to allow importing 'rotating_api_key_client'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rotator_library import RotatingClient, PROVIDER_PLUGINS
from proxy_app.request_logger import log_request_response
from proxy_app.batch_manager import EmbeddingBatcher

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
ENABLE_REQUEST_LOGGING = args.enable_request_logging
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
    client = RotatingClient(api_keys=api_keys)
    batcher = EmbeddingBatcher(client=client)
    app.state.rotating_client = client
    app.state.embedding_batcher = batcher
    print("RotatingClient and EmbeddingBatcher initialized.")
    yield
    await batcher.stop()
    await client.close()
    print("RotatingClient and EmbeddingBatcher closed.")

# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client

def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher

async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    if not auth or auth != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth

async def streaming_response_wrapper(
    request_data: dict,
    response_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion.
    This function aggregates all data from the stream, including content,
    tool calls, function calls, and any other provider-specific fields.
    """
    response_chunks = []
    full_response = {}
    
    try:
        async for chunk_str in response_stream:
            yield chunk_str
            if chunk_str.strip() and chunk_str.startswith("data:"):
                content = chunk_str[len("data:"):].strip()
                if content != "[DONE]":
                    try:
                        chunk_data = json.loads(content)
                        response_chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        pass  # Ignore non-JSON chunks
    finally:
        if response_chunks:
            # --- Aggregation Logic ---
            final_message = {"role": "assistant"}
            aggregated_tool_calls = {}
            usage_data = None
            finish_reason = None

            for chunk in response_chunks:
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})

                    # Dynamically aggregate all fields from the delta
                    for key, value in delta.items():
                        if value is None:
                            continue

                        if key == "content":
                            if "content" not in final_message:
                                final_message["content"] = ""
                            if value:
                                final_message["content"] += value
                        
                        elif key == "tool_calls":
                            for tc_chunk in value:
                                index = tc_chunk["index"]
                                if index not in aggregated_tool_calls:
                                    aggregated_tool_calls[index] = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                                if tc_chunk.get("id"):
                                    aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                                if "function" in tc_chunk:
                                    if "name" in tc_chunk["function"]:
                                        aggregated_tool_calls[index]["function"]["name"] += tc_chunk["function"]["name"]
                                    if "arguments" in tc_chunk["function"]:
                                        aggregated_tool_calls[index]["function"]["arguments"] += tc_chunk["function"]["arguments"]
                        
                        elif key == "function_call":
                            if "function_call" not in final_message:
                                final_message["function_call"] = {"name": "", "arguments": ""}
                            if "name" in value:
                                final_message["function_call"]["name"] += value["name"]
                            if "arguments" in value:
                                final_message["function_call"]["arguments"] += value["arguments"]
                        
                        else: # Generic key handling for other data like 'reasoning'
                            if key not in final_message:
                                final_message[key] = value
                            elif isinstance(final_message.get(key), str):
                                final_message[key] += value
                            else:
                                final_message[key] = value

                    if "finish_reason" in choice and choice["finish_reason"]:
                        finish_reason = choice["finish_reason"]

                if "usage" in chunk and chunk["usage"]:
                    usage_data = chunk["usage"]

            # --- Final Response Construction ---
            if aggregated_tool_calls:
                final_message["tool_calls"] = list(aggregated_tool_calls.values())

            # Ensure standard fields are present for consistent logging
            for field in ["content", "tool_calls", "function_call"]:
                if field not in final_message:
                    final_message[field] = None

            first_chunk = response_chunks[0]
            final_choice = {
                "index": 0,
                "message": final_message,
                "finish_reason": finish_reason
            }

            full_response = {
                "id": first_chunk.get("id"),
                "object": "chat.completion",
                "created": first_chunk.get("created"),
                "model": first_chunk.get("model"),
                "choices": [final_choice],
                "usage": usage_data
            }

        if ENABLE_REQUEST_LOGGING:
            log_request_response(
                request_data=request_data,
                response_data=full_response,
                is_streaming=True,
                log_type="completion"
            )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _ = Depends(verify_api_key)
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    try:
        request_data = await request.json()
        is_streaming = request_data.get("stream", False)

        response = await client.acompletion(**request_data)

        if is_streaming:
            # Wrap the streaming response to enable logging after it's complete
            return StreamingResponse(
                streaming_response_wrapper(request_data, response),
                media_type="text/event-stream"
            )
        else:
            # For non-streaming, log immediately
            if ENABLE_REQUEST_LOGGING:
                log_request_response(
                    request_data=request_data,
                    response_data=response.model_dump(),
                    is_streaming=False,
                    log_type="completion"
                )
            return response

    except (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request: {str(e)}")
    except litellm.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication Error: {str(e)}")
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {str(e)}")
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except litellm.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {str(e)}")
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
    except Exception as e:
        logging.error(f"Request failed after all retries: {e}")
        # Optionally log the failed request
        if ENABLE_REQUEST_LOGGING:
            try:
                request_data = await request.json()
            except json.JSONDecodeError:
                request_data = {"error": "Could not parse request body"}
            log_request_response(
                request_data=request_data,
                response_data={"error": str(e)},
                is_streaming=request_data.get("stream", False),
                log_type="completion"
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    batcher: EmbeddingBatcher = Depends(get_embedding_batcher),
    _ = Depends(verify_api_key)
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    This endpoint uses a batching manager to group requests for efficiency.
    """
    try:
        request_data = body.model_dump(exclude_none=True)
        
        # The batcher expects a single string input per request
        if isinstance(request_data.get("input"), list):
            if len(request_data["input"]) > 1:
                raise HTTPException(status_code=400, detail="Batching multiple inputs in a single request is not supported when using the server-side batcher. Please send one input string per request.")
            request_data["input"] = request_data["input"][0]

        response_data = await batcher.add_request(request_data)
        
        # The batcher returns a dict, not a Pydantic model, so we construct it
        response = litellm.EmbeddingResponse(**response_data)

        if ENABLE_REQUEST_LOGGING:
            response_summary = {
                "model": response.model,
                "object": response.object,
                "usage": response.usage.model_dump(),
                "data_count": len(response.data),
                "embedding_dimensions": len(response.data[0].embedding) if response.data else 0
            }
            log_request_response(
                request_data=request_data,
                response_data=response_summary,
                is_streaming=False,
                log_type="embedding"
            )
        return response

    except (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request: {str(e)}")
    except litellm.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication Error: {str(e)}")
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {str(e)}")
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except litellm.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {str(e)}")
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
    except Exception as e:
        logging.error(f"Embedding request failed: {e}")
        if ENABLE_REQUEST_LOGGING:
            try:
                request_data = await request.json()
            except json.JSONDecodeError:
                request_data = {"error": "Could not parse request body"}
            log_request_response(
                request_data=request_data,
                response_data={"error": str(e)},
                is_streaming=False,
                log_type="embedding"
            )
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
