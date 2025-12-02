import asyncio
import httpx
from unittest.mock import MagicMock, AsyncMock
from src.rotator_library.client import RotatingClient
from src.rotator_library.providers.provider_interface import ProviderInterface

# Mock Provider that raises HTTP 500
class MockProvider(ProviderInterface):
    def has_custom_logic(self):
        return True
    
    async def get_models(self, api_key, client):
        return ["mock_provider/test-model"]

    async def acompletion(self, client, **kwargs):
        # Simulate an HTTP 500 error from httpx
        request = httpx.Request("POST", "https://example.com/api")
        
        # Simulate streaming iterator that raises error MID-STREAM
        async def error_stream():
             yield litellm.ModelResponse(choices=[{"delta": {"content": "First chunk"}, "index": 0}])
             # Then raise error
             response = httpx.Response(500, request=request)
             raise httpx.HTTPStatusError("Internal Server Error", request=request, response=response)
        
        return error_stream()

import litellm

async def main():
    print("Starting reproduction test...")
    
    # Setup RotatingClient with our mock provider
    client = RotatingClient(
        api_keys={"mock_provider": ["key1", "key2"]},
        max_retries=2,
        configure_logging=False
    )
    
    # Inject our mock provider
    client._provider_plugins["mock_provider"] = MockProvider
    
    # We need to manually inject it into instances because _get_provider_instance uses the class from _provider_plugins
    # but we want to ensure our instance logic works or the class instantiation works.
    # The client instantiates the provider plugin class.
    
    print("Attempting streaming request...")
    try:
        # We use a custom provider 'mock_provider'
        # The client will try to load it. 
        # Since we patched _provider_plugins, it should use our MockProvider class.
        
        async for chunk in client.acompletion(
            model="mock_provider/test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True
        ):
            print(chunk)
            
    except httpx.HTTPStatusError as e:
        print(f"\n[CRASH DETECTED] Caught expected HTTPStatusError: {e}")
        print("Test PASSED: Crash reproduced.")
        return
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {type(e).__name__}: {e}")
        return

    print("\n[FAILURE] No exception raised. The code did not crash as expected.")

if __name__ == "__main__":
    asyncio.run(main())