@echo off

setlocal EnableExtensions EnableDelayedExpansion

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ==========================
echo   STAGE F - RUN SETTINGS
echo ==========================
echo.
echo Choose how Stage F should run:
echo   1) ChatGPT-assisted  (DEFAULT, recommended)
echo   2) Autonomous (no ChatGPT; faster, but less insightful)
echo.
echo If you pick ChatGPT-assisted:
echo   - Stage F may stop and prepare a prompt.
echo   - The prompt will be COPIED to your clipboard.
echo   - The file "in\llm_stageF.json" will open in Notepad.
echo   - Paste the prompt into ChatGPT (or Gemini/Claude), then paste ONLY the JSON reply into that file, save it.
echo   - Run Stage F again to finalize.
echo.
set "F_CHOICE=1"
set /p "F_CHOICE=Enter 1 or 2 and press Enter (default 1): "
for /f "tokens=* delims= " %%A in ("%F_CHOICE%") do set "F_CHOICE=%%A"
if "%F_CHOICE%"=="" set "F_CHOICE=1"

set "MODE=CHATGPT"
if "%F_CHOICE%"=="2" set "MODE=AUTO"

echo.
if "%MODE%"=="CHATGPT" (
  echo [MODE] ChatGPT-assisted (default)
) else (
  echo [MODE] Autonomous (no ChatGPT)
)
echo.

REM Detect target: either -All OR IdeaDir (or empty = newest idea)
set "FIRST=%~1"
set "FC="
if not "%FIRST%"=="" set "FC=%FIRST:~0,1%"

set "HASALL=0"
echo %* | findstr /I "\-All" >nul && set "HASALL=1"

if "%HASALL%"=="1" (
  if "%MODE%"=="CHATGPT" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1" -All -AskLLM
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1" -All
  )
) else (
  if "%FC%"=="-" (
    if "%MODE%"=="CHATGPT" (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1" -AskLLM
    ) else (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1"
    )
  ) else (
    if "%MODE%"=="CHATGPT" (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1" -IdeaDir "%~1" -AskLLM
    ) else (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_f.ps1" -IdeaDir "%~1"
    )
  )
)

set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [OK] Stage F finished.
) else if "%RC%"=="2" (
  echo [NEXT] ChatGPT step prepared. Follow the instructions above, then run Stage F again.
  echo [NEXT] See launcher_logs\LAST_LOG.txt for details.
) else (
  echo [WARN] Stage F finished with ExitCode=%RC% (see launcher_logs\LAST_LOG.txt)
)
echo.
pause
exit /b %RC%
