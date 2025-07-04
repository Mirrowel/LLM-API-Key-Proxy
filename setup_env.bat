@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set "ENV_FILE=.env"
set "DEFAULT_PROXY_KEY=VerysecretKey"

REM --- Provider Name to Variable Name Mapping ---
set "provider_count=0"
set "provider_list[1]=Gemini" & set "provider_vars[1]=GEMINI" & set /a provider_count+=1
set "provider_list[2]=OpenRouter" & set "provider_vars[2]=OPENROUTER" & set /a provider_count+=1
set "provider_list[3]=Chutes" & set "provider_vars[3]=CHUTES" & set /a provider_count+=1
set "provider_list[4]=Nvidia" & set "provider_vars[4]=NVIDIA_NIM" & set /a provider_count+=1
set "provider_list[5]=OpenAI" & set "provider_vars[5]=OPENAI" & set /a provider_count+=1
set "provider_list[6]=Anthropic" & set "provider_vars[6]=ANTHROPIC" & set /a provider_count+=1
set "provider_list[7]=Mistral" & set "provider_vars[7]=MISTRAL" & set /a provider_count+=1
set "provider_list[8]=Groq" & set "provider_vars[8]=GROQ" & set /a provider_count+=1
set "provider_list[9]=Cohere" & set "provider_vars[9]=COHERE" & set /a provider_count+=1
set "provider_list[10]=Bedrock" & set "provider_vars[10]=BEDROCK" & set /a provider_count+=1


:main
cls
echo =================================================================
echo      Welcome to the API Key Setup for Your Proxy Server
echo =================================================================
echo.
echo This script will help you set up your .env file.
echo.

REM --- Ensure .env file exists and has PROXY_API_KEY ---
if not exist "%ENV_FILE%" (
    echo Creating a new %ENV_FILE% file for you...
    echo PROXY_API_KEY="%DEFAULT_PROXY_KEY%" > "%ENV_FILE%"
    echo.
) else (
    findstr /C:"PROXY_API_KEY=" "%ENV_FILE%" >nul
    if errorlevel 1 (
        echo Adding the default proxy key to your .env file...
        echo.>> "%ENV_FILE%"
        echo PROXY_API_KEY="%DEFAULT_PROXY_KEY%" >> "%ENV_FILE%"
        echo.
    )
)

:get_provider
echo -----------------------------------------------------------------
echo Please choose a provider to add an API key for:
echo -----------------------------------------------------------------
echo.
for /L %%i in (1,1,%provider_count%) do (
    echo   %%i. !provider_list[%%i]!
)
echo.
set /p "choice=Type the number of the provider and press Enter: "

REM --- Validate Provider Choice ---
set "VAR_NAME="
set "provider_choice="
if %choice% GTR 0 if %choice% LEQ %provider_count% (
    set "VAR_NAME=!provider_vars[%choice%]!"
    set "provider_choice=!provider_list[%choice%]!"
)

if not defined VAR_NAME (
    cls
    echo =================================================================
    echo      INVALID SELECTION! Please try again.
    echo =================================================================
    echo.
    pause
    goto :get_provider
)

set "API_VAR_BASE=%VAR_NAME%_API_KEY"

:get_key
echo.
echo -----------------------------------------------------------------
set /p "api_key=Enter the API key for %provider_choice%: "
if not defined api_key (
    echo You must enter an API key.
    goto :get_key
)
echo -----------------------------------------------------------------
echo.

REM --- Find the next available key number ---
set /a key_index=1
:find_next_key
findstr /R /C:"^%API_VAR_BASE%_%key_index% *=" "%ENV_FILE%" >nul
if %errorlevel% equ 0 (
    set /a key_index+=1
    goto :find_next_key
)

REM --- Append the new key to the .env file ---
echo Adding your key to %ENV_FILE%...
echo %API_VAR_BASE%_%key_index%="%api_key%" >> "%ENV_FILE%"
echo.
echo Successfully added %provider_choice% API key as %API_VAR_BASE%_%key_index%!
echo.

:ask_another
set /p "another=Do you want to add another key? (yes/no): "
if /i "%another%"=="yes" (
    goto :main
)
if /i "%another%"=="y" (
    goto :main
)

cls
echo =================================================================
echo      Setup Complete! Your .env file is ready.
echo =================================================================
echo.
echo You can now run the proxy server.
echo.
pause
exit /b
