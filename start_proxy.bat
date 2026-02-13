@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo LLM API Key Proxy - Запуск
echo ========================================
echo.

cd /d "%~dp0"

echo Активация виртуального окружения...
call venv\Scripts\activate.bat

echo.
echo Запуск прокси-сервера на http://127.0.0.1:8000
echo.
echo.
echo Для остановки нажмите Ctrl+C
echo ========================================
echo.

python src/proxy_app/main.py --host 127.0.0.1 --port 8000

pause
