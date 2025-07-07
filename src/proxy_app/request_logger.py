import json
import os
from datetime import datetime
from pathlib import Path
import uuid
from typing import Literal

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
COMPLETIONS_LOGS_DIR = LOGS_DIR / "completions"
EMBEDDINGS_LOGS_DIR = LOGS_DIR / "embeddings"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
COMPLETIONS_LOGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_LOGS_DIR.mkdir(exist_ok=True)

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
        print(f"Error logging request/response: {e}")
