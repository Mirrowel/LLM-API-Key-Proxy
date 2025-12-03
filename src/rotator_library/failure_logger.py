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

def _extract_response_body(error: Exception) -> str:
    """
    Extract the full response body from various error types.
    
    Handles:
    - httpx.HTTPStatusError: response.text or response.content
    - litellm exceptions: various response attributes
    - Other exceptions: str(error)
    """
    # Try to get response body from httpx errors
    if hasattr(error, 'response') and error.response is not None:
        response = error.response
        # Try .text first (decoded)
        if hasattr(response, 'text') and response.text:
            return response.text
        # Try .content (bytes)
        if hasattr(response, 'content') and response.content:
            try:
                return response.content.decode('utf-8', errors='replace')
            except Exception:
                return str(response.content)
        # Try reading response if it's a streaming response that was read
        if hasattr(response, '_content') and response._content:
            try:
                return response._content.decode('utf-8', errors='replace')
            except Exception:
                return str(response._content)
    
    # Check for litellm's body attribute
    if hasattr(error, 'body') and error.body:
        return str(error.body)
    
    # Check for message attribute that might contain response
    if hasattr(error, 'message') and error.message:
        return str(error.message)
    
    return None


def log_failure(api_key: str, model: str, attempt: int, error: Exception, request_headers: dict, raw_response_text: str = None):
    """
    Logs a detailed failure message to a file and a concise summary to the main logger.
    
    Args:
        api_key: The API key or credential path that was used
        model: The model that was requested
        attempt: The attempt number (1-based)
        error: The exception that occurred
        request_headers: Headers from the original request
        raw_response_text: Optional pre-extracted response body (e.g., from streaming)
    """
    # 1. Log the full, detailed error to the dedicated failures.log file
    # Prioritize the explicitly passed raw response text, as it may contain
    # reassembled data from a stream that is not available on the exception object.
    raw_response = raw_response_text
    if not raw_response:
        raw_response = _extract_response_body(error)

    # Get full error message (not truncated)
    full_error_message = str(error)
    
    # Also capture any nested/wrapped exception info
    error_chain = []
    current_error = error
    while current_error:
        error_chain.append({
            "type": type(current_error).__name__,
            "message": str(current_error)[:2000]  # Limit per-error message size
        })
        current_error = getattr(current_error, '__cause__', None) or getattr(current_error, '__context__', None)
        if len(error_chain) > 5:  # Prevent infinite loops
            break
    
    detailed_log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_ending": api_key[-4:] if len(api_key) >= 4 else "****",
        "model": model,
        "attempt_number": attempt,
        "error_type": type(error).__name__,
        "error_message": full_error_message[:5000],  # Limit total size
        "raw_response": raw_response[:10000] if raw_response else None,  # Limit response size
        "request_headers": request_headers,
        "error_chain": error_chain if len(error_chain) > 1 else None,
    }
    failure_logger.error(detailed_log_data)

    # 2. Log a concise summary to the main library logger, which will propagate
    summary_message = (
        f"API call failed for model {model} with key ...{api_key[-4:] if len(api_key) >= 4 else '****'}. "
        f"Error: {type(error).__name__}. See failures.log for details."
    )
    main_lib_logger.error(summary_message)
