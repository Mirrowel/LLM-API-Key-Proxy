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

