@echo off
setlocal

title CodeAlpha Object Detection & Tracking - STOP

echo ============================================================
echo   Stopping CodeAlpha Object Detection & Tracking
 echo ============================================================
echo.

taskkill /FI "WINDOWTITLE eq CodeAlpha Object Detection ^& Tracking - START" /T /F >nul 2>&1

echo Backend stop request sent.
echo.
echo If the browser remains open, you can close its tab normally.
echo.
pause
