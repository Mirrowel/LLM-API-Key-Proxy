import logging
import json
from logging.handlers import RotatingFileHandler
import os

def setup_failure_logger():
    """Sets up a dedicated JSON logger for failed API calls."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Use the same named logger as the rest of the library
    logger = logging.getLogger('rotator_library')
    logger.setLevel(logging.INFO) # Set to INFO to capture all levels
    
    # Prevent logs from propagating to the root logger
    logger.propagate = False

    # Use a rotating file handler to keep log files from growing too large
    handler = RotatingFileHandler(
        os.path.join(log_dir, 'failures.log'),
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=2
    )

    # Custom JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage()
            }
            return json.dumps(log_record)

    handler.setFormatter(JsonFormatter())
    
    # Add handler only if it hasn't been added before, and is not a NullHandler
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)

    return logger

# Initialize the logger for failures
failure_logger = setup_failure_logger()

def log_failure(api_key: str, model: str, attempt: int, error: Exception, request_data: dict):
    """Logs a structured message for a failed API call."""
    
    # Try to get the raw response from the exception if it exists
    raw_response = None
    if hasattr(error, 'response') and hasattr(error.response, 'text'):
        raw_response = error.response.text

    log_data = {
        "api_key_ending": api_key[-4:],
        "model": model,
        "attempt_number": attempt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_response": raw_response,
        "request_data": request_data,
    }
    failure_logger.error(log_data)
