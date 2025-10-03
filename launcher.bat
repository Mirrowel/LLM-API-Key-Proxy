@echo off
setlocal enabledelayedexpansion

:: Default Settings
set "HOST=0.0.0.0"
set "PORT=8000"
set "LOGGING=false"
set "EXECUTION_MODE="
set "EXE_NAME=proxy_app.exe"
set "SOURCE_PATH=src\proxy_app\main.py"

:: --- Phase 1: Detection and Mode Selection ---
set "EXE_EXISTS=false"
set "SOURCE_EXISTS=false"

if exist "%EXE_NAME%" (
    set "EXE_EXISTS=true"
)

if exist "%SOURCE_PATH%" (
    set "SOURCE_EXISTS=true"
)

if "%EXE_EXISTS%"=="true" (
    if "%SOURCE_EXISTS%"=="true" (
        call :SelectModeMenu
    ) else (
        set "EXECUTION_MODE=exe"
    )
) else (
    if "%SOURCE_EXISTS%"=="true" (
        set "EXECUTION_MODE=source"
    ) else (
        call :NoTargetsFound
    )
)

if "%EXECUTION_MODE%"=="" (
    goto :eof
)

:: --- Phase 2: Main Menu ---
:MainMenu
cls
echo ==================================================
echo      LLM API Key Proxy Launcher
echo ==================================================
echo.
echo   Current Configuration:
echo   ----------------------
echo   - Host IP: %HOST%
echo   - Port: %PORT%
echo   - Request Logging: %LOGGING%
echo   - Execution Mode: %EXECUTION_MODE%
echo.
echo   Main Menu:
echo   ----------
echo   1. Run Proxy
echo   2. Configure Proxy
echo   3. Add Credentials
echo   4. Exit
echo.
set /p "CHOICE=Enter your choice: "

if "%CHOICE%"=="1" goto :RunProxy
if "%CHOICE%"=="2" goto :ConfigMenu
if "%CHOICE%"=="3" goto :AddCredentials
if "%CHOICE%"=="4" goto :eof

echo Invalid choice.
pause
goto :MainMenu

:: --- Phase 3: Configuration Sub-Menu ---
:ConfigMenu
cls
echo ==================================================
echo      Configuration Menu
echo ==================================================
echo.
echo   1. Set Host IP (Current: %HOST%)
echo   2. Set Port (Current: %PORT%)
echo   3. Toggle Request Logging (Current: %LOGGING%)
echo   4. Back to Main Menu
echo.
set /p "CHOICE=Enter your choice: "

if "%CHOICE%"=="1" (
    set /p "HOST=Enter new Host IP: "
    goto :ConfigMenu
)
if "%CHOICE%"=="2" (
    set /p "PORT=Enter new Port: "
    goto :ConfigMenu
)
if "%CHOICE%"=="3" (
    if "%LOGGING%"=="true" (
        set "LOGGING=false"
    ) else (
        set "LOGGING=true"
    )
    goto :ConfigMenu
)
if "%CHOICE%"=="4" goto :MainMenu

echo Invalid choice.
pause
goto :ConfigMenu

:: --- Phase 4: Execution ---
:RunProxy
cls
set "ARGS=--host %HOST% --port %PORT%"
if "%LOGGING%"=="true" (
    set "ARGS=%ARGS% --enable-request-logging"
)

echo Starting Proxy...
echo Arguments: %ARGS%
echo.

if "%EXECUTION_MODE%"=="exe" (
    start "%EXE_NAME%" %ARGS%
) else (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
    start python %SOURCE_PATH% %ARGS%
)
goto :eof

:AddCredentials
cls
echo Launching Credential Tool...
echo.

if "%EXECUTION_MODE%"=="exe" (
    start "%EXE_NAME%" --add-credential
) else (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
    start python %SOURCE_PATH% --add-credential
)
goto :eof

:: --- Helper Functions ---
:SelectModeMenu
cls
echo ==================================================
echo      Execution Mode Selection
echo ==================================================
echo.
echo   Both executable and source code found.
echo   Please choose which to use:
echo.
echo   1. Executable (%EXE_NAME%)
echo   2. Source Code (%SOURCE_PATH%)
echo.
set /p "CHOICE=Enter your choice: "

if "%CHOICE%"=="1" (
    set "EXECUTION_MODE=exe"
) else if "%CHOICE%"=="2" (
    set "EXECUTION_MODE=source"
) else (
    echo Invalid choice.
    pause
    goto :SelectModeMenu
)
goto :end_of_function

:NoTargetsFound
cls
echo ==================================================
echo      Error
echo ==================================================
echo.
echo   Could not find the executable (%EXE_NAME%)
echo   or the source code (%SOURCE_PATH%).
echo.
echo   Please ensure the launcher is in the correct
echo   directory or that the project has been built.
echo.
pause
goto :eof

:end_of_function
endlocal