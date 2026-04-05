#!/usr/bin/env python3
"""Quick debug test for gitlawb model discovery."""

import asyncio
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["GITLAWB_API_BASE"] = "https://opengateway.gitlawb.com/v1/xiaomi-mimo"
os.environ["GITLAWB_API_KEY"] = "not-needed"
os.environ["IGNORE_MODELS_GITLAWB"] = "*"
os.environ["WHITELIST_MODELS_GITLAWB"] = "mimo-v2.5-pro"
os.environ["SKIP_OAUTH_INIT_CHECK"] = "true"

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")


async def main():
    from rotator_library import RotatingClient
    from rotator_library.client.models import ModelResolver

    api_keys = {"gitlawb": ["not-needed"]}

    async with RotatingClient(
        api_keys=api_keys,
        configure_logging=False,
        global_timeout=60,
        ignore_models={"gitlawb": ["*"]},
        whitelist_models={"gitlawb": ["mimo-v2.5-pro"]},
    ) as client:
        # Test model resolver directly
        resolver = client._model_resolver
        test_models = [
            "gitlawb/mimo-v2.5-pro",
            "gitlawb/mimo-v2-flash",
            "gitlawb/mimo-v2.5",
        ]
        for m in test_models:
            allowed = resolver.is_model_allowed(m, "gitlawb")
            wl = resolver._is_whitelisted(m, "gitlawb")
            bl = resolver._is_blacklisted(m, "gitlawb")
            print(f"  {m}: allowed={allowed}, wl={wl}, bl={bl}")

        # Now test full model discovery
        print("\nFull model discovery:")
        models = await client.get_available_models("gitlawb")
        print(f"  Result: {models}")


if __name__ == "__main__":
    asyncio.run(main())
