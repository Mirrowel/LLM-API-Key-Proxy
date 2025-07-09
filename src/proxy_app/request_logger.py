import json
import os
from datetime import datetime
from pathlib import Path
import uuid
from typing import Literal, Dict
import logging

from .provider_urls import get_provider_endpoint

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
COMPLETIONS_LOGS_DIR = LOGS_DIR / "completions"
EMBEDDINGS_LOGS_DIR = LOGS_DIR / "embeddings"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
COMPLETIONS_LOGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_LOGS_DIR.mkdir(exist_ok=True)

def log_request_to_console(url: str, headers: dict, client_info: tuple, request_data: dict):
    """
    Logs a concise, single-line summary of an incoming request to the console.
    """
    time_str = datetime.now().strftime("%H:%M")
    model_full = request_data.get("model", "N/A")
    
    provider = "N/A"
    model_name = model_full
    endpoint_url = "N/A"

    if '/' in model_full:
        parts = model_full.split('/', 1)
        provider = parts[0]
        model_name = parts[1]
        # Use the helper function to get the full endpoint URL
        endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"

    log_message = f"{time_str} - {client_info[0]}:{client_info[1]} - provider: {provider}, model: {model_name} - {endpoint_url}"
    logging.info(log_message)

def log_request_response(
    request_data: dict,
    response_data: dict,
    is_streaming: bool,
    log_type: Literal["completion", "embedding"]
):
    """
    Logs the request and response data to a file in the appropriate log directory.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4()
        
        if log_type == "completion":
            target_dir = COMPLETIONS_LOGS_DIR
        elif log_type == "embedding":
            target_dir = EMBEDDINGS_LOGS_DIR
        else:
            # Fallback to the main logs directory if log_type is invalid
            target_dir = LOGS_DIR

        filename = target_dir / f"{timestamp}_{unique_id}.json"

        log_content = {
            "request": request_data,
            "response": response_data,
            "is_streaming": is_streaming
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_content, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        # In case of logging failure, we don't want to crash the main application
        # Use the root logger to log the error to the file.
        logging.error(f"Error logging request/response to file: {e}")
