import re
from typing import Optional, Dict, Any

from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError, AuthenticationError, InvalidRequestError, BadRequestError, OpenAIError, InternalServerError

class ClassifiedError:
    """A structured representation of a classified error."""
    def __init__(self, error_type: str, original_exception: Exception, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        self.error_type = error_type
        self.original_exception = original_exception
        self.status_code = status_code
        self.retry_after = retry_after

    def __str__(self):
        return f"ClassifiedError(type={self.error_type}, status={self.status_code}, retry_after={self.retry_after}, original_exc={self.original_exception})"

def get_retry_after(error: Exception) -> Optional[int]:
    """
    Extracts the 'retry-after' duration in seconds from an exception message.
    Handles both integer and string representations of the duration.
    """
    error_str = str(error).lower()
    
    # Common patterns for 'retry-after'
    patterns = [
        r'retry after:?\s*(\d+)',
        r'retry_after:?\s*(\d+)',
        r'retry in\s*(\d+)\s*seconds',
        r'wait for\s*(\d+)\s*seconds',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_str)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Handle cases where the error object itself has the attribute
    if hasattr(error, 'retry_after'):
        value = getattr(error, 'retry_after')
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
            
    return None

def classify_error(e: Exception) -> ClassifiedError:
    """
    Classifies an exception into a structured ClassifiedError object.
    """
    status_code = getattr(e, 'status_code', None)
    
    if isinstance(e, RateLimitError):
        retry_after = get_retry_after(e)
        return ClassifiedError(
            error_type='rate_limit',
            original_exception=e,
            status_code=status_code or 429,
            retry_after=retry_after
        )
    
    if isinstance(e, (AuthenticationError,)):
        return ClassifiedError(
            error_type='authentication',
            original_exception=e,
            status_code=status_code or 401
        )
        
    if isinstance(e, (InvalidRequestError, BadRequestError)):
        return ClassifiedError(
            error_type='invalid_request',
            original_exception=e,
            status_code=status_code or 400
        )

    if isinstance(e, (ServiceUnavailableError, APIConnectionError, OpenAIError, InternalServerError)):
        # These are often temporary server-side issues
        return ClassifiedError(
            error_type='server_error',
            original_exception=e,
            status_code=status_code or 503
        )

    # Fallback for any other unclassified errors
    return ClassifiedError(
        error_type='unknown',
        original_exception=e,
        status_code=status_code
    )

def is_rate_limit_error(e: Exception) -> bool:
    """Checks if the exception is a rate limit error."""
    return isinstance(e, RateLimitError)

def is_server_error(e: Exception) -> bool:
    """Checks if the exception is a temporary server-side error."""
    return isinstance(e, (ServiceUnavailableError, APIConnectionError, InternalServerError, OpenAIError))

def is_unrecoverable_error(e: Exception) -> bool:
    """
    Checks if the exception is a non-retriable client-side error.
    These are errors that will not resolve on their own.
    """
    return isinstance(e, (InvalidRequestError, AuthenticationError, BadRequestError))

class AllProviders:
    """
    A class to handle provider-specific settings, such as custom API bases.
    """
    def __init__(self):
        self.providers = {
            "chutes": {
                "api_base": "https://llm.chutes.ai/v1",
                "model_prefix": "openai/"
            }
        }

    def get_provider_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Returns provider-specific kwargs for a given model.
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        provider = self._get_provider_from_model(model)
        provider_settings = self.providers.get(provider, {})
        
        if "api_base" in provider_settings:
            kwargs["api_base"] = provider_settings["api_base"]
        
        if "model_prefix" in provider_settings:
            kwargs["model"] = f"{provider_settings['model_prefix']}{model.split('/', 1)[1]}"
            
        return kwargs

    def _get_provider_from_model(self, model: str) -> str:
        """
        Determines the provider from the model name.
        """
        return model.split('/')[0]
