@echo off
:: ================================================================================
:: Universal Instructions for macOS / Linux Users
:: ================================================================================
:: This launcher.bat file is for Windows only.
:: If you are on macOS or Linux, please use the following Python commands directly
:: in your terminal.
::
:: First, ensure you have Python 3.10 or higher installed.
::
:: To run the proxy server (basic command):
:: export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
:: python src/proxy_app/main.py --host 0.0.0.0 --port 8000
::
:: Note: To enable request logging, add the --enable-request-logging flag to the command.
::
:: To add new credentials:
:: export PYTHONPATH=${PYTHONPATH}:$(pwd)/src
:: python src/proxy_app/main.py --add-credential
::
:: To build the executable (requires PyInstaller):
:: pip install -r requirements.txt
:: pip install pyinstaller
:: python src/proxy_app/build.py
:: ================================================================================

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
        call :CheckPython
        if errorlevel 1 goto :eof
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
if "%EXECUTION_MODE%"=="source" (
    echo   4. Build Executable
    echo   5. Exit
) else (
    echo   4. Exit
)
echo.
set /p "CHOICE=Enter your choice: "

if "%CHOICE%"=="1" goto :RunProxy
if "%CHOICE%"=="2" goto :ConfigMenu
if "%CHOICE%"=="3" goto :AddCredentials

if "%EXECUTION_MODE%"=="source" (
    if "%CHOICE%"=="4" goto :BuildExecutable
    if "%CHOICE%"=="5" goto :eof
) else (
    if "%CHOICE%"=="4" goto :eof
)

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
echo   Current Configuration:
echo   ----------------------
echo   - Host IP: %HOST%
echo   - Port: %PORT%
echo   - Request Logging: %LOGGING%
echo   - Execution Mode: %EXECUTION_MODE%
echo.
echo   Configuration Options:
echo   ----------------------
echo   1. Set Host IP
echo   2. Set Port
echo   3. Toggle Request Logging
echo   4. Back to Main Menu
echo.
set /p "CHOICE=Enter your choice: "

if "%CHOICE%"=="1" (
    set /p "NEW_HOST=Enter new Host IP: "
    if defined NEW_HOST (
        set "HOST=!NEW_HOST!"
    )
    goto :ConfigMenu
)
if "%CHOICE%"=="2" (
    set "NEW_PORT="
    set /p "NEW_PORT=Enter new Port: "
    if not defined NEW_PORT goto :ConfigMenu
    set "IS_NUM=true"
    for /f "delims=0123456789" %%i in ("!NEW_PORT!") do set "IS_NUM=false"
    if "!IS_NUM!"=="false" (
        echo Invalid Port. Please enter numbers only.
        pause
    ) else (
        if !NEW_PORT! GTR 65535 (
            echo Invalid Port. Port cannot be greater than 65535.
            pause
        ) else (
            set "PORT=!NEW_PORT!"
        )
    )
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
call :Execute "" "%ARGS%"
goto :eof

:AddCredentials
cls
echo Launching Credential Tool...
echo.
call :Execute "--add-credential" ""
goto :MainMenu

:BuildExecutable
cls
echo ==================================================
echo      Building Executable
echo ==================================================
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.
echo Installing PyInstaller...
pip install pyinstaller
echo.
echo Running build script...
python src/proxy_app/build.py
echo.
echo Build process finished.
pause
goto :MainMenu

:: --- Helper Functions ---
:Execute
set "COMMAND=%~1"
set "ARGS=%~2"
if "%EXECUTION_MODE%"=="exe" (
    start "LLM PROXY" %EXE_NAME% %COMMAND% %ARGS%
) else (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
    start "LLM PROXY" python %SOURCE_PATH% %COMMAND% %ARGS%
)

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
    call :CheckPython
    if errorlevel 1 goto :eof
    set "EXECUTION_MODE=source"
) else (
    echo Invalid choice.
    pause
    goto :SelectModeMenu
)
goto :end_of_function

:CheckPython
where python >nul 2>nul
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

for /f "tokens=1,2" %%a in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if not "%PY_MAJOR%"=="3" (
    call :PythonVersionError
    exit /b 1
)
if %PY_MINOR% lss 10 (
    call :PythonVersionError
    exit /b 1
)

exit /b 0

:PythonVersionError
echo Error: Python 3.10 or higher is required.
echo Found version: %PY_MAJOR%.%PY_MINOR%
echo Please upgrade your Python installation.
pause
goto :eof

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
