@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title CodeAlpha Object Detection & Tracking - START

echo ============================================================
echo   CodeAlpha Object Detection ^& Tracking
 echo   One-click Windows launcher
 echo ============================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found.
        echo Install Python 3.10-3.13 and try again.
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating Python virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo [2/4] Checking backend packages...
"%VENV_PY%" -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Python dependency installation failed.
    pause
    exit /b 1
)

echo [3/4] Detecting laptop Wi-Fi IPv4 address...
set "LAN_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address" /C:"IPv4 Address[ ]*\."') do (
    set "CANDIDATE=%%A"
    set "CANDIDATE=!CANDIDATE: =!"
    echo !CANDIDATE! | findstr /R /B /C:"192\.168\." /C:"10\." /C:"172\.1[6-9]\." /C:"172\.2[0-9]\." /C:"172\.3[0-1]\." >nul
    if not errorlevel 1 if not defined LAN_IP set "LAN_IP=!CANDIDATE!"
)

if not defined LAN_IP (
    for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address" /C:"IPv4 Address[ ]*\."') do (
        set "CANDIDATE=%%A"
        set "CANDIDATE=!CANDIDATE: =!"
        if not defined LAN_IP set "LAN_IP=!CANDIDATE!"
    )
)

if not defined LAN_IP set "LAN_IP=127.0.0.1"
set "DASH_URL=http://localhost:5000/"
set "PHONE_URL=http://%LAN_IP%:5000"

echo.
echo ============================================================
echo   Dashboard : %DASH_URL%
echo   Phone URL : %PHONE_URL%
echo ============================================================
echo.
echo [4/4] Starting the backend...
echo Keep this window open while using the project.
echo.

start "CodeAlpha Dashboard" "%DASH_URL%"

echo Waiting for the server to start...
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo   ANDROID CONNECTION URL
 echo   %PHONE_URL%
echo.
echo   Enter/copy this URL into the Android APK.
echo   Then use Test Ping and Start Camera.
echo ============================================================
echo.

"%VENV_PY%" backend\server.py

set "EXITCODE=%errorlevel%"
echo.
echo Backend stopped with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
