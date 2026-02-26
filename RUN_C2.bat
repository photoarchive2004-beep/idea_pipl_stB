@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "IDEA_DIR="
if not "%~1"=="" set "IDEA_DIR=%~1"
if "%IDEA_DIR%"=="" if exist "%ROOT%ideas\_ACTIVE_PATH.txt" set /p IDEA_DIR=<"%ROOT%ideas\_ACTIVE_PATH.txt"

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv_xr\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo =================================
echo STAGE C2 - Таблица доказательств
echo =================================
echo IDEA_DIR: %IDEA_DIR%
echo Python: %PY%
echo.

"%PY%" "%ROOT%tools\run_c2.py" --idea-dir "%IDEA_DIR%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo Кратко: C2 завершен успешно, таблицы обновлены.
if "%RC%"=="2" echo Кратко: C2 ожидает RESPONSE.json от ChatGPT.
if not "%RC%"=="0" if not "%RC%"=="2" echo Кратко: C2 завершился с ошибкой, см. out\module_C2.log

echo ExitCode=%RC%
pause
exit /b %RC%
