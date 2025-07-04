@echo off
echo Starting API Key Proxy...
uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
pause