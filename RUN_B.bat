@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ===================================
echo   STAGE B - Мульти-источниковый поиск литературы
echo ===================================
echo.
echo Выберите режим:
echo   1) BALANCED (по умолчанию)
echo   2) WIDE
echo   3) FOCUSED
echo.
set "CH=1"
set /p CH=Введите 1/2/3 (пусто = 1): 
if "!CH!"=="" set "CH=1"

set "SCOPE=balanced"
if "!CH!"=="2" set "SCOPE=wide"
if "!CH!"=="3" set "SCOPE=focused"

echo.
echo [INFO] РЕЖИМ=!SCOPE!
echo.

if not exist "%ROOT%tools\run_b_launcher.ps1" (
  echo [ERR] Не найден tools\run_b_launcher.ps1
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_b_launcher.ps1" -Scope "!SCOPE!" -N 300
set "RC=%ERRORLEVEL%"

echo.
if "!RC!"=="0" (
  echo [OK] Stage B завершён.
) else (
  echo [ERR] Stage B завершился с ошибкой. ExitCode=!RC!
  echo [HINT] См. launcher_logs\LAST_LOG.txt и out\stageB_summary.txt
)

echo.
pause
exit /b !RC!
