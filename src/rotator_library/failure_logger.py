import logging
import json
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from .error_handler import mask_credential

# Module-level state for resilience
_file_handler = None
_fallback_mode = False


# Custom JSON formatter for structured logs (defined at module level for reuse)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        # The message is already a dict, so we just format it as a JSON string
        return json.dumps(record.msg)


def _create_file_handler():
    """Create file handler with directory auto-recreation."""
    global _file_handler, _fallback_mode
    log_dir = "logs"
    
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        handler = RotatingFileHandler(
            os.path.join(log_dir, "failures.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=2,
        )
        
        handler.setFormatter(JsonFormatter())
        _file_handler = handler
        _fallback_mode = False
        return handler
    except (OSError, PermissionError, IOError) as e:
        logging.warning(f"Cannot create failure log file handler: {e}")
        _fallback_mode = True
        return None


def setup_failure_logger():
    """Sets up a dedicated JSON logger for writing detailed failure logs."""
    logger = logging.getLogger("failure_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Remove existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Try to add file handler
    handler = _create_file_handler()
    if handler:
        logger.addHandler(handler)
    
    # Always add a NullHandler as fallback to prevent "no handlers" warning
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger


def _ensure_handler_valid():
    """Check if file handler is still valid, recreate if needed."""
    global _file_handler, _fallback_mode
    
    if _file_handler is None or _fallback_mode:
        handler = _create_file_handler()
        if handler:
            failure_logger = logging.getLogger("failure_logger")
            failure_logger.handlers.clear()
            failure_logger.addHandler(handler)


# Initialize the dedicated logger for detailed failure logs
failure_logger = setup_failure_logger()

# Get the main library logger for concise, propagated messages
main_lib_logger = logging.getLogger("rotator_library")


def _extract_response_body(error: Exception) -> str:
    """
    Extract the full response body from various error types.

    Handles:
    - httpx.HTTPStatusError: response.text or response.content
    - litellm exceptions: various response attributes
    - Other exceptions: str(error)
    """
    # Try to get response body from httpx errors
    if hasattr(error, "response") and error.response is not None:
        response = error.response
        # Try .text first (decoded)
        if hasattr(response, "text") and response.text:
            return response.text
        # Try .content (bytes)
        if hasattr(response, "content") and response.content:
            try:
                return response.content.decode("utf-8", errors="replace")
            except Exception:
                return str(response.content)

    # Check for litellm's body attribute
    if hasattr(error, "body") and error.body:
        return str(error.body)

    # Check for message attribute that might contain response
    if hasattr(error, "message") and error.message:
        return str(error.message)

    return None


def log_failure(
    api_key: str,
    model: str,
    attempt: int,
    error: Exception,
    request_headers: dict,
    raw_response_text: str = None,
):
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
    visited = set()  # Track visited exceptions to detect circular references
    current_error = error
    while current_error:
        # Check for circular references
        error_id = id(current_error)
        if error_id in visited:
            break
        visited.add(error_id)

        error_chain.append(
            {
                "type": type(current_error).__name__,
                "message": str(current_error)[:2000],  # Limit per-error message size
            }
        )
        current_error = getattr(current_error, "__cause__", None) or getattr(
            current_error, "__context__", None
        )
        if len(error_chain) > 5:  # Prevent excessive chain length
            break

    detailed_log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_ending": mask_credential(api_key),
        "model": model,
        "attempt_number": attempt,
        "error_type": type(error).__name__,
        "error_message": full_error_message[:5000],  # Limit total size
        "raw_response": raw_response[:10000]
        if raw_response
        else None,  # Limit response size
        "request_headers": request_headers,
        "error_chain": error_chain if len(error_chain) > 1 else None,
    }
    
    # 2. Log a concise summary to the main library logger, which will propagate
    summary_message = (
        f"API call failed for model {model} with key {mask_credential(api_key)}. "
        f"Error: {type(error).__name__}. See failures.log for details."
    )
    
    # Attempt to ensure handler is valid before logging
    _ensure_handler_valid()
    
    # Wrap the actual log call with resilience
    try:
        failure_logger.error(detailed_log_data)
    except (OSError, IOError) as e:
        global _fallback_mode
        _fallback_mode = True
        # File logging failed - log to console instead
        logging.error(f"Failed to write to failures.log: {e}")
        logging.error(f"Failure summary: {summary_message}")
    
    # Console log always succeeds
    main_lib_logger.error(summary_message)
