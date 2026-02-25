@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ================================
echo STAGE C1 - Отбор и сбор материалов
echo ================================

if not exist "%ROOT%tools\run_c1.py" (
  echo [ERR] Не найден tools\run_c1.py
  pause
  exit /b 1
)

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv_xr\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "IDEA_DIR="
if not "%~1"=="" set "IDEA_DIR=%~1"

if "%IDEA_DIR%"=="" (
  "%PY%" "%ROOT%tools\run_c1.py" --screening chatgpt
) else (
  "%PY%" "%ROOT%tools\run_c1.py" --screening chatgpt --idea-dir "%IDEA_DIR%"
)
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Stage C1 завершен успешно.
) else if "%RC%"=="2" (
  echo [WAIT] Жду ответ в RESPONSE.json (первый запуск завершен корректно).
) else (
  echo [ERR] Stage C1 завершился с ошибкой. ExitCode=%RC%
  echo [HINT] Проверьте launcher_logs\LAST_LOG.txt и logs\moduleC1_LAST.log
)

echo.
pause
exit /b %RC%
