#!/usr/bin/env python3
"""
Test script for Gitlawb Opengateway (Xiaomi MiMo) provider.

Tests:
1. Provider auto-detection from GITLAWB_API_BASE
2. Model discovery via /models endpoint
3. Chat completion (non-streaming)
4. Streaming chat completion
5. Reasoning content handling
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Set env vars for this test
os.environ.setdefault("GITLAWB_API_BASE", "https://opengateway.gitlawb.com/v1/xiaomi-mimo")
os.environ.setdefault("GITLAWB_API_KEY", "not-needed")
os.environ.setdefault("GITLAWB_MODELS", '["mimo-v2.5-pro"]')
os.environ.setdefault("IGNORE_MODELS_GITLAWB", "*")
os.environ.setdefault("WHITELIST_MODELS_GITLAWB", "mimo-v2.5-pro")
os.environ.setdefault("SKIP_OAUTH_INIT_CHECK", "true")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
# Show rotator_library debug for the transforms
logging.getLogger("rotator_library").setLevel(logging.DEBUG)


async def main():
    from rotator_library import RotatingClient

    print("=" * 70)
    print("Gitlawb Opengateway Provider Test")
    print("=" * 70)

    # Init client with just gitlawb keys
    api_keys = {"gitlawb": ["not-needed"]}

    async with RotatingClient(
        api_keys=api_keys,
        configure_logging=False,
        global_timeout=60,
        ignore_models={"gitlawb": ["*"]},
        whitelist_models={"gitlawb": ["mimo-v2.5-pro"]},
    ) as client:
        # Test 1: Check provider was detected
        print("\n--- Test 1: Provider Detection ---")
        is_custom = client.provider_config.is_custom_provider("gitlawb")
        api_base = client.provider_config.get_api_base("gitlawb")
        print(f"  Custom provider detected: {is_custom}")
        print(f"  API base: {api_base}")
        assert is_custom, "gitlawb should be detected as custom provider"
        assert api_base, "gitlawb API base should be set"
        print("  ✓ PASS")

        # Test 2: Model Discovery
        print("\n--- Test 2: Model Discovery ---")
        models = await client.get_available_models("gitlawb")
        print(f"  Discovered models: {models}")
        has_pro = any("mimo-v2.5-pro" in m for m in models)
        print(f"  Has mimo-v2.5-pro: {has_pro}")
        assert has_pro, "mimo-v2.5-pro should be discovered"
        print("  ✓ PASS")

        # Test 3: Non-streaming completion
        print("\n--- Test 3: Non-Streaming Completion ---")
        try:
            response = await client.acompletion(
                model="gitlawb/mimo-v2.5-pro",
                messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                max_tokens=256,
                stream=False,
            )
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            print(f"  Content: {content}")
            print(f"  Reasoning: {reasoning[:100] if reasoning else 'None'}...")
            print(f"  Usage: {response.usage}")
            print("  ✓ PASS")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()

        # Test 4: Streaming completion
        print("\n--- Test 4: Streaming Completion ---")
        try:
            stream = await client.acompletion(
                model="gitlawb/mimo-v2.5-pro",
                messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
                max_tokens=256,
                stream=True,
            )
            chunks = []
            reasoning_chunks = []
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                # The streaming handler may yield ModelResponse objects or strings
                if isinstance(chunk, str):
                    # SSE format - just count
                    continue
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta:
                    if hasattr(delta, "content") and delta.content:
                        chunks.append(delta.content)
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_chunks.append(delta.reasoning_content)

            full_content = "".join(chunks)
            full_reasoning = "".join(reasoning_chunks)
            print(f"  Streamed content: {full_content}")
            print(f"  Streamed reasoning length: {len(full_reasoning)} chars")
            print(f"  Total chunks: {chunk_count}, content: {len(chunks)}, reasoning: {len(reasoning_chunks)}")
            print("  ✓ PASS")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
