@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "IDEA_DIR="
if not "%~1"=="" set "IDEA_DIR=%~1"
if "%IDEA_DIR%"=="" if exist "%ROOT%ideas\_ACTIVE_PATH.txt" (
  set /p IDEA_DIR=<"%ROOT%ideas\_ACTIVE_PATH.txt"
)
if not "%IDEA_DIR%"=="" if not exist "%IDEA_DIR%\logs" set "IDEA_DIR="

if not "%IDEA_DIR%"=="" (
  set "LAUNCHER_LOG=%IDEA_DIR%\logs\launcher_C1_LAST.log"
) else (
  set "LAUNCHER_LOG=%ROOT%launcher_logs\launcher_C1_LAST.log"
)
for %%I in ("%LAUNCHER_LOG%") do if not exist "%%~dpI" mkdir "%%~dpI" >nul 2>&1

call :log "================================"
call :log "STAGE C1 - Отбор и сбор материалов"
call :log "================================"
call :log "Корень проекта: %ROOT%"
call :log "Лог лаунчера: %LAUNCHER_LOG%"

if not exist "%ROOT%tools\run_c1.py" (
  call :log "[ERR] Не найден tools\run_c1.py"
  echo [ERR] Не найден tools\run_c1.py
  echo Лог: %LAUNCHER_LOG%
  echo ExitCode=1
  pause
  exit /b 1
)

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv_xr\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
call :log "Python: %PY%"

if "%IDEA_DIR%"=="" (
  call :log "Команда: %PY% %ROOT%tools\run_c1.py --screening chatgpt"
  "%PY%" "%ROOT%tools\run_c1.py" --screening chatgpt
) else (
  call :log "Команда: %PY% %ROOT%tools\run_c1.py --screening chatgpt --idea-dir %IDEA_DIR%"
  "%PY%" "%ROOT%tools\run_c1.py" --screening chatgpt --idea-dir "%IDEA_DIR%"
)
set "RC=%ERRORLEVEL%"

call :log "ExitCode=%RC%"
echo.
echo ExitCode=%RC%

if "%RC%"=="0" (
  echo [OK] Stage C1 завершен успешно.
  echo [OK] Подробный лог модуля: ^<IDEA^>\logs\moduleC1_LAST.log
  echo [OK] Лог лаунчера: %LAUNCHER_LOG%
  exit /b 0
)

if "%RC%"=="2" (
  echo [WAIT] Ожидается ответ ChatGPT в RESPONSE.json.
  echo [WAIT] После заполнения RESPONSE.json снова запустите RUN_C1.bat.
  echo [WAIT] Лог модуля: ^<IDEA^>\logs\moduleC1_LAST.log
  echo [WAIT] Лог лаунчера: %LAUNCHER_LOG%
  pause
  exit /b 2
)

echo [ERR] Stage C1 завершился с ошибкой.
echo [ERR] Смотрите логи:
echo      1^) %LAUNCHER_LOG%
echo      2^) ^<IDEA^>\logs\moduleC1_LAST.log
echo [HINT] Исправьте причину и запустите RUN_C1.bat повторно.
pause
exit /b %RC%

:log
>>"%LAUNCHER_LOG%" echo [%date% %time%] %~1
exit /b 0
