"""
Utility for formatting credential identifiers for display in logs.

This module provides a centralized way to format credentials for logging,
ensuring that file-based credentials (OAuth JSON files) show their full
filename while API keys show only the last 6 characters for security.
"""

import os


def format_credential_for_display(credential: str) -> str:
    """
    Format a credential for display in logs.
    
    For file-based credentials (OAuth JSON files), returns the full basename.
    For API key strings, returns the last 6 characters.
    
    Args:
        credential: The credential string (either a file path or API key)
    
    Returns:
        A display-safe string representation of the credential
    
    Examples:
        >>> format_credential_for_display("/path/to/oauth_cred.json")
        "oauth_cred.json"
        >>> format_credential_for_display("sk-1234567890abcdef")
        "...abcdef"
    """
    if os.path.isfile(credential):
        return os.path.basename(credential)
    else:
        return f"...{credential[-6:]}"