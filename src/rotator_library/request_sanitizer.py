from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if model not in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]:
            del payload["thinking"]
            
    return payload
