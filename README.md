# API Key Proxy and Rotator [![License](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/) [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)

This project provides a simple and effective solution for managing and rotating API keys for services like Google Gemini, while exposing an OpenAI-compatible endpoint. It's designed to be a lightweight, self-hosted proxy that can help you:

- **Secure your API keys**: Instead of embedding your keys directly in client-side applications, you can keep them on the server where they are more secure.
- **Rotate keys to avoid rate limits**: The proxy can automatically rotate through a pool of API keys, reducing the chance of hitting rate limits on any single key.
- **Monitor key usage**: The system tracks the usage of each key, providing insights into your API consumption.
- **Provide a unified endpoint**: Exposes an OpenAI-compatible `/v1/chat/completions` endpoint, allowing you to use it with a wide range of existing tools and libraries.

## Features

- **OpenAI-Compatible Endpoint**: Drop-in replacement for OpenAI's API.
- **Smart Key Rotation**: Rotates keys based on usage to minimize errors.
- **Usage Tracking**: Logs successes and failures for each key.
- **Streaming and Non-Streaming Support**: Handles both response types seamlessly.
- **Easy to Deploy**: Can be run with a simple `uvicorn` command.

## Getting Started

### Prerequisites

- Python 3.8+
- An `.env` file with your API keys (see `.env.example`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your API keys:**

    Create a `.env` file in the project root and add your keys. You'll need a `PROXY_API_KEY` to secure your proxy and at least one `GEMINI_API_KEY`.

    ```
    # .env
    PROXY_API_KEY="your-secret-proxy-key"
    GEMINI_API_KEY_1="your-gemini-key-1"
    GEMINI_API_KEY_2="your-gemini-key-2"
    # Add more Gemini keys as needed
    ```

### Running the Proxy

You can run the proxy using `uvicorn`:

```bash
uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

The proxy will now be running and accessible at `http://localhost:8000`.

## How to Use

To use the proxy, make a POST request to the `/v1/chat/completions` endpoint, making sure to include your `PROXY_API_KEY` in the `Authorization` header.

Here's an example using `curl`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-proxy-key" \
-d '{
  "model": "gemini-pro",
  "messages": [{"role": "user", "content": "Hello, how are you?"}]
}'
```