"""
Rotating API Key Client
"""
from .client import RotatingClient
from .usage_manager import UsageManager
from .error_handler import is_authentication_error, is_rate_limit_error, is_server_error, is_unrecoverable_error
from .failure_logger import log_failure

__all__ = [
    "RotatingClient",
    "UsageManager",
    "is_authentication_error",
    "is_rate_limit_error",
    "is_server_error",
    "is_unrecoverable_error",
    "log_failure",
]
