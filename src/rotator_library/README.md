# Rotating API Key Client

A simple, thread-safe client that intelligently rotates and retries API keys for use with `litellm`.

## Features

-   **Smart Key Rotation**: Automatically uses the least-used key to distribute load.
-   **Automatic Retries**: Retries requests on transient server errors.
-   **Cooldowns**: Puts keys on a temporary cooldown after rate limit or authentication errors.
-   **Usage Tracking**: Tracks daily and global usage for each key.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.

## Installation

To install the library, you can install it directly from a Git repository or a local path.

### From a local path:

```bash
pip install -e .
```

## Usage

Here's a simple example of how to use the `RotatingClient`:

```python
import asyncio
from rotating_api_key_client import RotatingClient

async def main():
    # List of your API keys
    api_keys = ["key1", "key2", "key3"]

    # Initialize the client
    client = RotatingClient(api_keys=api_keys)

    # Make a request
    response = await client.acompletion(
        model="gemini/gemini-pro",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )

    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

By default, the client will store usage data in a `key_usage.json` file in the current working directory. You can customize this by passing the `usage_file_path` parameter:

```python
client = RotatingClient(api_keys=api_keys, usage_file_path="/path/to/your/usage.json")
