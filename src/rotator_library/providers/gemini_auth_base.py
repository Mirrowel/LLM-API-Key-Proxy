# src/rotator_library/providers/gemini_auth_base.py

from .google_oauth_base import GoogleOAuthBase

class GeminiAuthBase(GoogleOAuthBase):
    """
    Gemini CLI OAuth2 authentication implementation.
    
    Inherits all OAuth functionality from GoogleOAuthBase with Gemini-specific configuration.
    """
    
    CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
    CLIENT_SECRET = "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"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    ENV_PREFIX = "GEMINI_CLI"
    CALLBACK_PORT = 8085
    CALLBACK_PATH = "/oauth2callback"