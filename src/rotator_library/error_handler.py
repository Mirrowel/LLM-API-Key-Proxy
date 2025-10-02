import re
import json
from typing import Optional, Dict, Any
import httpx

from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError, AuthenticationError, InvalidRequestError, BadRequestError, OpenAIError, InternalServerError, Timeout, ContextWindowExceededError

class NoAvailableKeysError(Exception):
    """Raised when no API keys are available for a request after waiting."""
    pass

class PreRequestCallbackError(Exception):
    """Raised when a pre-request callback fails."""
    pass

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
    Handles both integer and string representations of the duration, as well as JSON bodies.
    """
    error_str = str(error).lower()

    # 1. Try to parse JSON from the error string to find 'retryDelay'
    try:
        # It's common for the actual JSON to be embedded in the string representation
        json_match = re.search(r'(\{.*\})', error_str)
        if json_match:
            error_json = json.loads(json_match.group(1))
            retry_info = error_json.get('error', {}).get('details', [{}])[0]
            if retry_info.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                delay_str = retry_info.get('retryDelay', {}).get('seconds')
                if delay_str:
                    return int(delay_str)
                # Fallback for the other format
                delay_str = retry_info.get('retryDelay')
                if isinstance(delay_str, str) and delay_str.endswith('s'):
                    return int(delay_str[:-1])

    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        pass # If JSON parsing fails, proceed to regex and attribute checks

    # 2. Common regex patterns for 'retry-after'
    patterns = [
        r'retry after:?\s*(\d+)',
        r'retry_after:?\s*(\d+)',
        r'retry in\s*(\d+)\s*seconds',
        r'wait for\s*(\d+)\s*seconds',
        r'"retryDelay":\s*"(\d+)s"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_str)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # 3. Handle cases where the error object itself has the attribute
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
    Now handles both litellm and httpx exceptions.
    """
    status_code = getattr(e, 'status_code', None)
    if isinstance(e, httpx.HTTPStatusError): # [NEW] Handle httpx errors first
        status_code = e.response.status_code
        if status_code == 401:
            return ClassifiedError(error_type='authentication', original_exception=e, status_code=status_code)
        if status_code == 429:
            retry_after = get_retry_after(e)
            return ClassifiedError(error_type='rate_limit', original_exception=e, status_code=status_code, retry_after=retry_after)
        if 400 <= status_code < 500:
            return ClassifiedError(error_type='invalid_request', original_exception=e, status_code=status_code)
        if 500 <= status_code:
            return ClassifiedError(error_type='server_error', original_exception=e, status_code=status_code)
    
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)): # [NEW]
        return ClassifiedError(error_type='api_connection', original_exception=e, status_code=status_code)

    if isinstance(e, PreRequestCallbackError):
        return ClassifiedError(
            error_type='pre_request_callback_error',
            original_exception=e,
            status_code=400  # Treat as a bad request
        )

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
    
    if isinstance(e, ContextWindowExceededError):
        return ClassifiedError(
            error_type='context_window_exceeded',
            original_exception=e,
            status_code=status_code or 400
        )

    if isinstance(e, (APIConnectionError, Timeout)):
        return ClassifiedError(
            error_type='api_connection',
            original_exception=e,
            status_code=status_code or 503 # Treat like a server error
        )

    if isinstance(e, (ServiceUnavailableError, InternalServerError, OpenAIError)):
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
