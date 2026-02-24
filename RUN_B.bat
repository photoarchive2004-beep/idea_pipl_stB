@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ===================================
echo   STAGE B - Literature Scout (OpenAlex)
echo ===================================
echo.
echo Choose mode:
echo   1) BALANCED (default)
echo   2) WIDE
echo   3) FOCUSED
echo.
set "CH=1"
set /p CH=Enter 1/2/3 (empty = 1): 
if "!CH!"=="" set "CH=1"

set "SCOPE=balanced"
if "!CH!"=="2" set "SCOPE=wide"
if "!CH!"=="3" set "SCOPE=focused"

echo.
echo [INFO] SCOPE=!SCOPE!
echo.

if not exist "%ROOT%tools\run_b_launcher.ps1" (
  echo [ERR] Missing tools\run_b_launcher.ps1
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_b_launcher.ps1" -Scope "!SCOPE!" -N 300
set "RC=%ERRORLEVEL%"

echo.
if "!RC!"=="0" (
  echo [OK] Stage B finished successfully.
) else (
  echo [ERR] Stage B failed. ExitCode=!RC!
  echo [HINT] See launcher_logs\LAST_LOG.txt and idea out\module_B.log / search_log.json
)
echo.
pause
exit /b !RC!
