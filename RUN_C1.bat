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
if "%IDEA_DIR%"=="" (
  echo [ERR] Не удалось определить папку идеи.
  echo [HINT] Укажите путь первым аргументом или обновите ideas\_ACTIVE_PATH.txt
  set "RC=1"
  goto :finish
)

set "MODULE_LOG=%IDEA_DIR%\logs\moduleC1_LAST.log"
set "LAUNCHER_LOG=%IDEA_DIR%\logs\launcher_C1_LAST.log"
for %%I in ("%LAUNCHER_LOG%") do if not exist "%%~dpI" mkdir "%%~dpI" >nul 2>&1

call :log "================================"
call :log "STAGE C1 - Отбор и сбор материалов"
call :log "================================"
call :log "IDEA_DIR=%IDEA_DIR%"
call :log "MODULE_LOG=%MODULE_LOG%"
call :log "LAUNCHER_LOG=%LAUNCHER_LOG%"

echo =================================
echo STAGE C1 - Отбор и сбор материалов
echo =================================
echo IDEA_DIR: %IDEA_DIR%
echo Лог модуля: %MODULE_LOG%
echo Лог лаунчера: %LAUNCHER_LOG%

if not exist "%ROOT%tools\run_c1.py" (
  echo [ERR] Не найден tools\run_c1.py
  call :log "[ERR] Не найден tools\run_c1.py"
  set "RC=1"
  goto :finish
)

set "TRACE_FOUND=0"
if exist "%IDEA_DIR%\in\papers" set "TRACE_FOUND=1"
if exist "%IDEA_DIR%\in\c1_chatgpt" set "TRACE_FOUND=1"
if exist "%IDEA_DIR%\out\harvest_report.md" set "TRACE_FOUND=1"
if exist "%IDEA_DIR%\out\prisma_c1.md" set "TRACE_FOUND=1"
for %%F in ("%IDEA_DIR%\logs\moduleC1_*.log") do set "TRACE_FOUND=1"

if "%TRACE_FOUND%"=="1" (
  echo.
  echo Обнаружены результаты предыдущего запуска C1 для этой идеи.
  echo Удалить и запустить заново?
  choice /C YNQ /N /M "[Y] Да (очистить) / [N] Нет (продолжить/докачать) / [Q] Отмена: "
  if errorlevel 3 (
    echo [OK] Отмена по запросу пользователя.
    call :log "Пользователь выбрал Q: отмена"
    set "RC=0"
    goto :finish
  )
  if errorlevel 2 (
    echo [OK] Выбран режим resume: без очистки.
    call :log "Пользователь выбрал N: resume"
  ) else (
    echo [OK] Выполняю очистку C1 и архивирование логов...
    call :log "Пользователь выбрал Y: очистка"
    call :cleanup_c1 "%IDEA_DIR%"
  )
)

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv_xr\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
call :log "Python=%PY%"

call :log "Команда: %PY% %ROOT%tools\run_c1.py --screening chatgpt --idea-dir %IDEA_DIR%"
"%PY%" "%ROOT%tools\run_c1.py" --screening chatgpt --idea-dir "%IDEA_DIR%"
set "RC=%ERRORLEVEL%"
call :log "ExitCode=%RC%"

if "%RC%"=="0" (
  echo [OK] Stage C1 завершен успешно.
) else if "%RC%"=="2" (
  echo [WAIT] Жду ответ в RESPONSE.json (первый запуск завершен корректно).
) else (
  echo [ERR] Stage C1 завершился с ошибкой.
)

:finish
echo.
echo ExitCode=%RC%
echo IDEA_DIR: %IDEA_DIR%
echo Лог модуля: %MODULE_LOG%
echo Лог лаунчера: %LAUNCHER_LOG%
pause
exit /b %RC%

:cleanup_c1
setlocal
set "T_IDEA=%~1"
set "ARCHIVE_DIR=%T_IDEA%\logs\archive\C1_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "ARCHIVE_DIR=%ARCHIVE_DIR: =0%"
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%" >nul 2>&1
for %%F in ("%T_IDEA%\logs\moduleC1_*.log") do move /Y "%%~fF" "%ARCHIVE_DIR%\" >nul
if exist "%T_IDEA%\logs\moduleC1_LAST.log" move /Y "%T_IDEA%\logs\moduleC1_LAST.log" "%ARCHIVE_DIR%\" >nul
if exist "%T_IDEA%\in\papers" rmdir /S /Q "%T_IDEA%\in\papers"
if exist "%T_IDEA%\in\c1_chatgpt" rmdir /S /Q "%T_IDEA%\in\c1_chatgpt"
if exist "%T_IDEA%\out\harvest_report.md" del /Q "%T_IDEA%\out\harvest_report.md"
if exist "%T_IDEA%\out\prisma_c1.md" del /Q "%T_IDEA%\out\prisma_c1.md"
endlocal
exit /b 0

:log
>>"%LAUNCHER_LOG%" echo [%date% %time%] %~1
exit /b 0
