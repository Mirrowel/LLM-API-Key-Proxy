import logging
import json
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_failure_logger():
    """Sets up a dedicated JSON logger for writing detailed failure logs to a file."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger specifically for failures.
    # This logger will NOT propagate to the root logger.
    logger = logging.getLogger('failure_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Use a rotating file handler
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'failures.log'),
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=2
    )

    # Custom JSON formatter for structured logs
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # The message is already a dict, so we just format it as a JSON string
            return json.dumps(record.msg)

    handler.setFormatter(JsonFormatter())
    
    # Add handler only if it hasn't been added before
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

# Initialize the dedicated logger for detailed failure logs
failure_logger = setup_failure_logger()

# Get the main library logger for concise, propagated messages
main_lib_logger = logging.getLogger('rotator_library')

def log_failure(api_key: str, model: str, attempt: int, error: Exception, request_headers: dict):
    """
    Logs a detailed failure message to a file and a concise summary to the main logger.
    """
    # 1. Log the full, detailed error to the dedicated failures.log file
    raw_response = None
    if hasattr(error, 'response') and hasattr(error.response, 'text'):
        raw_response = error.response.text

    detailed_log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_ending": api_key[-4:],
        "model": model,
        "attempt_number": attempt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_response": raw_response,
        "request_headers": request_headers,
    }
    failure_logger.error(detailed_log_data)

    # 2. Log a concise summary to the main library logger, which will propagate
    summary_message = (
        f"API call failed for model {model} with key ...{api_key[-4:]}. "
        f"Error: {type(error).__name__}. See failures.log for details."
    )
    main_lib_logger.error(summary_message)
