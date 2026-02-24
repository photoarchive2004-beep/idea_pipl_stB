@echo off
setlocal EnableExtensions

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo [INFO] Running setup (see setup_log.txt)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\setup.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [ERROR] Setup failed with code %RC%. Open: %ROOT%setup_log.txt
  pause
  exit /b %RC%
)
echo [OK] Setup complete.
pause
