@echo off
setlocal

title CodeAlpha Object Detection & Tracking - STOP

echo ============================================================
echo   Stopping CodeAlpha Object Detection ^& Tracking
echo ============================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match 'backend[\\/]server\.py' }; $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

echo Backend stop request sent.
echo If the browser remains open, you can close its tab normally.
echo.
pause
