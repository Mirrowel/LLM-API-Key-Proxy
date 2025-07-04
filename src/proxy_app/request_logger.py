import json
import os
from datetime import datetime
from pathlib import Path
import uuid

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def log_request_response(request_data: dict, response_data: dict, is_streaming: bool):
    """
    Logs the request and response data to a single file in the logs directory.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4()
        filename = LOGS_DIR / f"{timestamp}_{unique_id}.json"

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
