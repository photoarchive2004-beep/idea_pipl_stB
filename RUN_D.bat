@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM If first arg starts with "-" then treat as PowerShell named args (e.g., -All, -IdeaDir ...)
set "FIRST=%~1"
set "FC="
if not "%FIRST%"=="" set "FC=%FIRST:~0,1%"

if "%FC%"=="-" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_d.ps1" %*
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_d.ps1" -IdeaDir "%~1"
)

set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [OK] Module D finished.
) else (
  echo [WARN] Module D finished with ExitCode=%RC% (see launcher_logs\LAST_LOG.txt)
)
echo.
pause
exit /b %RC%
