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
echo STAGE C3 - Evidence Table Engine
echo =================================
echo IDEA_DIR: %IDEA_DIR%
echo Python: %PY%
echo.

"%PY%" "%ROOT%tools\run_c3.py" --idea-dir "%IDEA_DIR%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo Кратко: C3 завершен успешно, выходные файлы обновлены.
if "%RC%"=="2" echo Кратко: C3 ожидает RESPONSE.json от ChatGPT.
if not "%RC%"=="0" if not "%RC%"=="2" echo Кратко: C3 завершился с ошибкой, см. out\module_C3.log

echo ExitCode=%RC%
pause
exit /b %RC%
