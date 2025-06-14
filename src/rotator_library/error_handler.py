from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError, AuthenticationError, InvalidRequestError

def is_authentication_error(e: Exception) -> bool:
    """Checks if the exception is related to authentication."""
    return isinstance(e, AuthenticationError)

def is_rate_limit_error(e: Exception) -> bool:
    """Checks if the exception is a rate limit error."""
    return isinstance(e, RateLimitError)

def is_server_error(e: Exception) -> bool:
    """Checks if the exception is a temporary server-side error."""
    return isinstance(e, (ServiceUnavailableError, APIConnectionError))

def is_unrecoverable_error(e: Exception) -> bool:
    """
    Checks if the exception is a non-retriable client-side error
    (that is not an authentication error).
    """
    return isinstance(e, InvalidRequestError)
