# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import json
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from ..error_handler import extract_retry_after_from_body

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class KilocodeProvider(ProviderInterface):
    """
    Provider implementation for the Kilocode API.
    
    Kilocode routes requests to various providers through model prefixes:
    - minimax/minimax-m2.1:free
    - moonshotai/kimi-k2.5:free
    - z-ai/glm-4.7:free
    - And other provider/model combinations
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Kilocode API.
        """
        try:
            response = await client.get(
                "https://kilocode.ai/api/openrouter/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return [f"kilocode/{model['id']}" for model in response.json().get("data", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Kilocode models: {e}")
            return []

    @staticmethod
    def parse_quota_error(error: Exception, error_body: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parse Kilocode/OpenRouter rate limit errors.

        OpenRouter error format:
        {
          "error": {
            "code": 429,
            "message": "Rate limit exceeded...",
            "metadata": {"retry_after": 60}
          }
        }
        """
        body = error_body
        if not body:
            if hasattr(error, 'response') and hasattr(error.response, 'text'):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, 'body'):
                body = str(error.body) if error.body else None

        if not body:
            return None

        # Try extract_retry_after_from_body first
        retry_after = extract_retry_after_from_body(body)
        if retry_after:
            return {
                "retry_after": retry_after,
                "reason": "RATE_LIMIT_EXCEEDED",
            }

        # Try to parse JSON for OpenRouter/Kilocode format
        try:
            data = json.loads(body)
            error_obj = data.get("error", data)

            # Check for metadata.retry_after
            metadata = error_obj.get("metadata", {})
            if "retry_after" in metadata:
                return {
                    "retry_after": int(metadata["retry_after"]),
                    "reason": "RATE_LIMIT_EXCEEDED",
                }

            # Check for code in error
            if error_obj.get("code") == 429:
                return {
                    "retry_after": 30,  # Default 30s for rate limit
                    "reason": "RATE_LIMIT_EXCEEDED",
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Check for upstream provider errors
        body_lower = body.lower()
        if "upstream error" in body_lower or "provider error" in body_lower:
            return {
                "retry_after": 5,  # Short retry for upstream issues
                "reason": "UPSTREAM_ERROR",
            }

        return None
