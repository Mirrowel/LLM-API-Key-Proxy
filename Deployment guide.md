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


### Advanced: Antigravity OAuth Provider

The Antigravity provider requires OAuth2 authentication similar to Gemini CLI. It provides access to:
- Gemini 2.5 models (Pro/Flash)
- Gemini 3 models (Pro/Image-preview) - **requires paid-tier Google Cloud project**
- Claude Sonnet 4.5 via Google's Antigravity proxy

**Setting up Antigravity locally:**
1. Run the credential tool: `python -m rotator_library.credential_tool`
2. Select "Add OAuth Credential" and choose "Antigravity"
3. Complete the OAuth flow in your browser
4. The credential is saved to `oauth_creds/antigravity_oauth_1.json`

**Exporting for stateless deployment:**
1. Run: `python -m rotator_library.credential_tool`
2. Select "Export Antigravity to .env"
3. Copy the generated environment variables to your deployment platform:
   ```env
   ANTIGRAVITY_ACCESS_TOKEN="..."
   ANTIGRAVITY_REFRESH_TOKEN="..."
   ANTIGRAVITY_EXPIRY_DATE="..."
   ANTIGRAVITY_EMAIL="your-email@gmail.com"
   ```

**Important Notes:**
- Antigravity uses Google OAuth with additional scopes for cloud platform access
- Gemini 3 models require a paid-tier Google Cloud project (free tier will fail)
- The provider automatically handles thought signature caching for multi-turn conversations
- Tool hallucination prevention is enabled by default for Gemini 3 models

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

---

## Appendix: Deploying to a Custom VPS

If you're deploying the proxy to a **custom VPS** (DigitalOcean, AWS EC2, Linode, etc.) instead of Render.com, you'll encounter special considerations when setting up OAuth providers (Antigravity, Gemini CLI, iFlow). This section covers the professional deployment workflow.

### Understanding the OAuth Callback Problem

OAuth providers like Antigravity, Gemini CLI, and iFlow require an interactive authentication flow that:

1. Opens a browser for you to log in
2. Redirects back to a **local callback server** running on specific ports
3. Receives an authorization code to exchange for tokens

The callback servers bind to `localhost` on these ports:

| Provider      | Port  | Notes                                      |
|---------------|-------|--------------------------------------------|
| **Antigravity**  | 51121 | Google OAuth with extended scopes          |
| **Gemini CLI**   | 8085  | Google OAuth for Gemini API               |
| **iFlow**        | 11451 | Authorization Code flow with API key fetch |
| **Qwen Code**    | N/A   | Uses Device Code flow - works on remote VPS ✅ |

**The Issue**: When running on a remote VPS, your local browser cannot reach `http://localhost:51121` (or other callback ports) on the remote server, causing authentication to fail with a "connection refused" error.

### Recommended Deployment Workflow

There are **three professional approaches** to handle OAuth authentication for VPS deployment, listed from most recommended to least:

---

### **Option 1: Authenticate Locally, Deploy Credentials (RECOMMENDED)**

This is the **cleanest and most secure** approach. Complete OAuth flows on your local machine, export to environment variables, then deploy.

#### Step 1: Clone and Set Up Locally

```bash
# On your local development machine
git clone https://github.com/YOUR-USERNAME/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Run OAuth Authentication Locally

```bash
# Start the credential tool
python -m rotator_library.credential_tool
```

Select **"Add OAuth Credential"** and choose your provider:
- Antigravity
- Gemini CLI  
- iFlow
- Qwen Code (works directly on VPS, but can authenticate locally too)

The tool will:
1. Open your browser automatically
2. Start a local callback server
3. Complete the OAuth flow
4. Save credentials to `oauth_creds/<provider>_oauth_N.json`

#### Step 3: Export Credentials to Environment Variables

Still in the credential tool, select the export option for each provider:
- **"Export Antigravity to .env"**
- **"Export Gemini CLI to .env"**
- **"Export iFlow to .env"**
- **"Export Qwen Code to .env"**

The tool generates a `.env` file snippet like:

```env
# Antigravity OAuth Credentials
ANTIGRAVITY_ACCESS_TOKEN="ya29.a0AfB_byD..."
ANTIGRAVITY_REFRESH_TOKEN="1//0gL6dK9..."
ANTIGRAVITY_EXPIRY_DATE="1735901234567"
ANTIGRAVITY_EMAIL="user@gmail.com"
ANTIGRAVITY_CLIENT_ID="1071006060591-..."
ANTIGRAVITY_CLIENT_SECRET="GOCSPX-..."
ANTIGRAVITY_TOKEN_URI="https://oauth2.googleapis.com/token"
ANTIGRAVITY_UNIVERSE_DOMAIN="googleapis.com"
```

Copy these variables to a file (e.g., `oauth_credentials.env`).

#### Step 4: Deploy to VPS

**Method A: Using Environment Variables (Recommended)**

```bash
# On your VPS
cd /path/to/LLM-API-Key-Proxy

# Create or edit .env file
nano .env

# Paste the exported environment variables
# Also add your PROXY_API_KEY and other provider keys

# Start the proxy
uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

**Method B: Upload Credential Files**

```bash
# On your local machine - copy credential files to VPS
scp -r oauth_creds/ user@your-vps-ip:/path/to/LLM-API-Key-Proxy/

# On VPS - verify files exist
ls -la oauth_creds/

# Start the proxy
uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

> **Note**: Environment variables are preferred for production deployments (more secure, easier to manage, works with container orchestration).

---

### **Option 2: SSH Port Forwarding (For Direct VPS Authentication)**

If you need to authenticate directly on the VPS (e.g., you don't have a local development environment), use SSH port forwarding to create secure tunnels.

#### How It Works

SSH tunnels forward ports from your local machine to the remote VPS, allowing your local browser to reach the callback servers.

#### Step-by-Step Process

**Step 1: Create SSH Tunnels**

From your **local machine**, open a terminal and run:

```bash
# Forward all OAuth callback ports at once
ssh -L 51121:localhost:51121 -L 8085:localhost:8085 -L 11451:localhost:11451 user@your-vps-ip

# Alternative: Forward ports individually as needed
ssh -L 51121:localhost:51121 user@your-vps-ip  # For Antigravity
ssh -L 8085:localhost:8085 user@your-vps-ip    # For Gemini CLI
ssh -L 11451:localhost:11451 user@your-vps-ip  # For iFlow
```

**Keep this SSH session open** during the entire authentication process.

**Step 2: Run Credential Tool on VPS**

In the same SSH terminal (or open a new SSH connection):

```bash
cd /path/to/LLM-API-Key-Proxy

# Ensure Python dependencies are installed
pip install -r requirements.txt

# Run the credential tool
python -m rotator_library.credential_tool
```

**Step 3: Complete OAuth Flow**

1. Select **"Add OAuth Credential"** → Choose your provider
2. The tool displays an authorization URL
3. **Click the URL in your local browser** (works because of the SSH tunnel!)
4. Complete the authentication flow
5. The browser redirects to `localhost:<port>` - **this now routes through the tunnel to your VPS**
6. Credentials are saved to `oauth_creds/` on the VPS

**Step 4: Export to Environment Variables**

Still in the credential tool:
1. Select the export option for each provider
2. Copy the generated environment variables
3. Add them to `/path/to/LLM-API-Key-Proxy/.env` on your VPS

**Step 5: Close Tunnels and Deploy**

```bash
# Exit the SSH session with tunnels (Ctrl+D or type 'exit')
# Tunnels are no longer needed

# Start the proxy on VPS (in a screen/tmux session or as a service)
uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

---

### **Option 3: Copy Credential Files to VPS**

If you've already authenticated locally and have credential files, you can copy them directly.

#### Copy OAuth Credential Files

```bash
# From your local machine
scp -r oauth_creds/ user@your-vps-ip:/path/to/LLM-API-Key-Proxy/

# Verify on VPS
ssh user@your-vps-ip
ls -la /path/to/LLM-API-Key-Proxy/oauth_creds/
```

Expected files:
- `antigravity_oauth_1.json`
- `gemini_cli_oauth_1.json`
- `iflow_oauth_1.json`
- `qwen_code_oauth_1.json`

#### Configure .env to Use Credential Files

On your VPS, edit `.env`:

```env
# Option A: Use credential files directly (not recommended for production)
# No special configuration needed - the proxy auto-detects oauth_creds/ folder

# Option B: Export to environment variables (recommended)
# Run credential tool and export each provider to .env
```

---

### Environment Variables vs. Credential Files

| Aspect                    | Environment Variables                    | Credential Files                           |
|---------------------------|------------------------------------------|--------------------------------------------|
| **Security**              | ✅ More secure (no files on disk)        | ⚠️ Files readable if server compromised   |
| **Container-Friendly**    | ✅ Perfect for Docker/K8s                | ❌ Requires volume mounts                 |
| **Ease of Rotation**      | ✅ Update .env and restart               | ⚠️ Need to regenerate JSON files          |
| **Backup/Version Control**| ✅ Easy to manage with secrets managers  | ❌ Binary files, harder to manage         |
| **Auto-Refresh**          | ✅ Uses refresh tokens                   | ✅ Uses refresh tokens                    |
| **Recommended For**       | Production deployments                   | Local development / testing                |

**Best Practice**: Always export to environment variables for VPS/cloud deployments.

---

### Production Deployment Checklist

#### Security Best Practices

- [ ] Never commit `.env` or `oauth_creds/` to version control
- [ ] Use environment variables instead of credential files in production
- [ ] Secure your VPS firewall - **do not** open OAuth callback ports (51121, 8085, 11451) to public internet
- [ ] Use SSH port forwarding only during initial authentication
- [ ] Rotate credentials regularly using the credential tool's export feature
- [ ] Set file permissions on `.env`: `chmod 600 .env`

#### Firewall Configuration

OAuth callback ports should **never** be publicly exposed:

```bash
# ❌ DO NOT DO THIS - keeps ports closed
# sudo ufw allow 51121/tcp
# sudo ufw allow 8085/tcp
# sudo ufw allow 11451/tcp

# ✅ Only open your proxy API port
sudo ufw allow 8000/tcp

# Check firewall status
sudo ufw status
```

The SSH tunnel method works **without** opening these ports because traffic routes through the SSH connection (port 22).

#### Running as a Service

Create a systemd service file on your VPS:

```bash
# Create service file
sudo nano /etc/systemd/system/llm-proxy.service
```

```ini
[Unit]
Description=LLM API Key Proxy
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/LLM-API-Key-Proxy
Environment="PATH=/path/to/python/bin"
ExecStart=/path/to/python/bin/uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable llm-proxy
sudo systemctl start llm-proxy

# Check status
sudo systemctl status llm-proxy

# View logs
sudo journalctl -u llm-proxy -f
```

---

### Troubleshooting VPS Deployment

#### "localhost:51121 connection refused" Error

**Cause**: Trying to authenticate directly on VPS without SSH tunnel.

**Solution**: Use Option 1 (authenticate locally) or Option 2 (SSH port forwarding).

#### OAuth Credentials Not Loading

```bash
# Check if environment variables are set
printenv | grep -E '(ANTIGRAVITY|GEMINI_CLI|IFLOW|QWEN_CODE)'

# Verify .env file exists and is readable
ls -la .env
cat .env | grep -E '(ANTIGRAVITY|GEMINI_CLI|IFLOW|QWEN_CODE)'

# Check credential files if using file-based approach
ls -la oauth_creds/
```

#### Token Refresh Failing

The proxy automatically refreshes tokens using refresh tokens. If refresh fails:

1. **Re-authenticate**: Run credential tool again and export new credentials
2. **Check token expiry**: Some providers require periodic re-authentication
3. **Verify credentials**: Ensure `REFRESH_TOKEN` is present in environment variables

#### Permission Denied on .env

```bash
# Set correct permissions
chmod 600 .env
chown your-username:your-username .env
```

---

### Summary: VPS Deployment Best Practices

1. **Authenticate locally** on your development machine (easiest, most secure)
2. **Export to environment variables** using the credential tool's built-in export feature
3. **Deploy to VPS** by adding environment variables to `.env`
4. **Never open OAuth callback ports** to the public internet
5. **Use SSH port forwarding** only if you must authenticate directly on VPS
6. **Run as a systemd service** for production reliability
7. **Monitor logs** for authentication errors and token refresh issues

This approach ensures secure, production-ready deployment while maintaining the convenience of OAuth authentication.

