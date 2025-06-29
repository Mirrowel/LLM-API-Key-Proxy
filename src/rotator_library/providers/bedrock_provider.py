import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

class BedrockProvider(ProviderInterface):
    """
    Provider implementation for AWS Bedrock.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a hardcoded list of common Bedrock models, as there is no
        simple, unauthenticated API endpoint to list them.
        """
        # Note: Listing Bedrock models typically requires AWS credentials and boto3.
        # For a simple, key-based proxy, we'll list common models.
        # This can be expanded with full AWS authentication if needed.
        lib_logger.info("Returning hardcoded list for Bedrock. Full discovery requires AWS auth.")
        return [
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/cohere.command-r-plus-v1:0",
            "bedrock/mistral.mistral-large-2402-v1:0",
        ]
