# Easy Guide to Deploying LLM-API-Key-Proxy on Render

This guide walks you through deploying the [LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy) as a hosted service on Render.com. The project provides a universal, OpenAI-compatible API endpoint for all your LLM providers (like Gemini or OpenAI), powered by an intelligent key management library. It's perfect for integrating with platforms like JanitorAI, where you can use it as a custom proxy for highly available and resilient chats.

The process is beginner-friendly and takes about 15-30 minutes. We'll use Render's free tier (with limitations like sleep after 15 minutes of inactivity) and upload your `.env` file as a secret for easy key management—no manual entry of variables required.

## Prerequisites

- A free Render.com account (sign up at render.com).
- A GitHub account (for forking the repo).
- Basic terminal access (e.g., Command Prompt, Terminal, or Git Bash).
- API keys from LLM providers (e.g., Gemini, OpenAI—get them from their dashboards). For details on supported providers and how to format their keys (e.g., API key naming conventions), refer to the [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers).

**Note**: You don't need Python installed for initial testing—use the pre-compiled Windows EXE from the repo's releases for a quick local trial.

## Step 1: Test Locally with the Compiled EXE (No Python Required)

Before deploying, try the proxy locally to ensure your keys work. This uses a pre-built executable that's easy to set up.

1. Go to the repo's [GitHub Releases page](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases).
2. Download the latest release ZIP file (e.g., for Windows).
3. Unzip the file.
4. Double-click `setup_env.bat`. A window will open—follow the prompts to add your PROXY_API_KEY (a strong secret you create) and provider keys. Use the [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers) for guidance on key formats (e.g., `GEMINI_API_KEY_1="your-key"`).
5. Double-click `proxy_app.exe` to start the proxy. It runs at `http://127.0.0.1:8000`—visit in a browser to confirm "API Key Proxy is running".
6. Test with curl (replace with your PROXY_API_KEY):

```
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer your-proxy-key" -d '{"model": "gemini/gemini-2.5-flash", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

    - Expected: A JSON response with the answer (e.g., "Paris").

If it works, you're ready to deploy. If not, double-check your keys against LiteLLM docs.

## Step 2: Fork and Prepare the Repository

1. Go to the original repo: [https://github.com/Mirrowel/LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy).
2. Click **Fork** in the top-right to create your own copy (this lets you make changes if needed).
3. Clone your forked repo locally:

```
git clone https://github.com/YOUR-USERNAME/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
```


## Step 3: Assemble Your .env File

The proxy uses a `.env` file to store your API keys securely. We'll create this based on the repo's documentation.

1. In your cloned repo, copy the example: `copy .env.example .env` (Windows) or `cp .env.example .env` (macOS/Linux).
2. Open `.env` in a text editor (e.g., Notepad or VS Code).
3. Add your keys following the format from the repo's README and [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers):
    - **PROXY_API_KEY**: Create a strong, unique secret (e.g., "my-super-secret-proxy-key"). This authenticates requests to your proxy.
    - **Provider Keys**: Add keys for your chosen providers. You can add multiple per provider (e.g., _1, _2) for rotation.

Example `.env` (customize with your real keys):

```
# Your proxy's authentication key (invent a strong one)
PROXY_API_KEY="my-super-secret-proxy-key"

# Provider API keys (get from provider dashboards; see LiteLLM docs for formats)
GEMINI_API_KEY_1="your-gemini-key-here"
GEMINI_API_KEY_2="another-gemini-key"

OPENROUTER_API_KEY_1="your-openrouter-key"
```

    - Supported providers: Check LiteLLM docs for a full list and specifics (e.g., GEMINI, OPENROUTER, NVIDIA_NIM).
    - Tip: Start with 1-2 providers to test. Don't share this file publicly!

### Advanced: Stateless Deployment for OAuth Providers (Gemini CLI, Qwen, iFlow)
If you are using providers that require complex OAuth files (like **Gemini CLI**, **Qwen Code**, or **iFlow**), you don't need to upload the JSON files manually. The proxy includes a tool to "export" these credentials into environment variables.

1.  Run the credential tool locally: `python -m rotator_library.credential_tool`
2.  Select the "Export ... to .env" option for your provider.
3.  The tool will generate a file (e.g., `gemini_cli_user_at_gmail.env`) containing variables like `GEMINI_CLI_ACCESS_TOKEN`, `GEMINI_CLI_REFRESH_TOKEN`, etc.
4.  Copy the contents of this file and paste them directly into your `.env` file or Render's "Environment Variables" section.
5.  The proxy will automatically detect and use these variables—no file upload required!

4. Save the file. (We'll upload it to Render in Step 5.)


## Step 4: Create a New Web Service on Render

1. Log in to render.com and go to your Dashboard.
2. Click **New > Web Service**.
3. Choose **Build and deploy from a Git repository** > **Next**.
4. Connect your GitHub account and select your forked repo.
5. In the setup form:
    - **Name**: Something like "llm-api-key-proxy".
    - **Region**: Choose one close to you (e.g., Oregon for US West).
    - **Branch**: "main" (or your default).
    - **Runtime**: Python 3.
    - **Build Command**: `pip install -r requirements.txt`.
    - **Start Command**: `uvicorn src.proxy_app.main:app --host 0.0.0.0 --port $PORT`.
    - **Instance Type**: Free (for testing; upgrade later for always-on service).
6. Click **Create Web Service**. Render will build and deploy—watch the progress in the Events tab.

## Step 5: Upload .env as a Secret File

Render mounts secret files securely at runtime, making your `.env` available to the app without exposing it.

1. In your new service's Dashboard, go to **Environment > Secret Files**.
2. Click **Add Secret File**.
3. **File Path**: Don't change. Keep it as root directory of the repo.
4. **Contents**: Upload the `.env` file you created previously.
5. Save. This injects the file for the app to load via `dotenv` (already in the code).
6. Trigger a redeploy: Go to **Deploy > Manual Deploy** > **Deploy HEAD** (or push a small change to your repo).

Your keys are now loaded automatically!

## Step 6: Test Your Deployed Proxy

1. Note your service URL: It's in the Dashboard (e.g., https://llm-api-key-proxy.onrender.com).
2. Test with curl (replace with your PROXY_API_KEY):

```
curl -X POST https://your-service.onrender.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer your-proxy-key" -d '{"model": "gemini/gemini-2.5-flash", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

    - Expected: A JSON response with the answer (e.g., "Paris").
3. Check logs in Render's Dashboard for startup messages (e.g., "RotatingClient initialized").

## Step 7: Integrate with JanitorAI

1. Log in to janitorai.com and go to API settings (usually in a chat or account menu).
2. Select "Proxy" mode.
3. **API URL**: `https://your-service.onrender.com/v1`.
4. **API Key**: Your PROXY_API_KEY (from .env).
5. **Model**: Format as "provider/model" (e.g., "gemini/gemini-2.5-flash"; check LiteLLM docs for options).
6. Save and test a chat—messages should route through your proxy.

## Troubleshooting

- **Build Fails**: Check Render logs for missing dependencies—ensure `requirements.txt` is up to date.
- **401 Unauthorized**: Verify your PROXY_API_KEY matches exactly (case-sensitive) and includes "Bearer " in requests. Or you have no keys for the provider/model added that you are trying to use.
- **405 on OPTIONS**: If CORS issues arise, add the middleware from Step 3 and redeploy.
- **Service Sleeps**: Free tier sleeps after inactivity—first requests may delay.
- **Provider Key Issues**: Double-check formats in [LiteLLM Providers Documentation](https://docs.litellm.ai/docs/providers).
- **More Help**: Check Render docs or the repo's README. If stuck, share error logs.

That is it.

