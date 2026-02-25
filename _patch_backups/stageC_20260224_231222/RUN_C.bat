@echo off

setlocal EnableExtensions EnableDelayedExpansion

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_c.ps1" -IdeaDir "%~1"
set "RC=%ERRORLEVEL%"
echo.
echo [INFO] ExitCode=%RC%
if exist "launcher_logs\LAST_LOG.txt" (
  for /f "usebackq delims=" %%L in ("launcher_logs\LAST_LOG.txt") do set "LAST=%%L"
  echo [INFO] Last launcher log: %LAST%
)
echo.
pause
exit /b %RC%
