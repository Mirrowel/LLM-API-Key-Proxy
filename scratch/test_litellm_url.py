import litellm
import asyncio
import logging

async def test():
    litellm.set_debug = True
    try:
        await litellm.acompletion(
            model="openai/mimo-v2.5-pro",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://opengateway.gitlawb.com/v1/xiaomi-mimo",
            api_key="sk-test"
        )
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
