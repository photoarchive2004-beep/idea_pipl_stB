@echo off

setlocal EnableExtensions EnableDelayedExpansion

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM Optional: override number of ideas via env var G_K (default 8)
set "K=8"
if not "%G_K%"=="" set "K=%G_K%"

echo.
echo ==========================
echo   STAGE G - RUN SETTINGS
echo ==========================
echo.
echo Choose how Stage G should run:
echo   1) ChatGPT-assisted  (DEFAULT, recommended; best quality)
echo   2) Autonomous (no ChatGPT; faster, but more generic)
echo.
echo If you pick ChatGPT-assisted:
echo   - Stage G may stop and prepare a prompt.
echo   - The prompt will be COPIED to your clipboard.
echo   - The file "in\llm_stageG.json" will open in Notepad.
echo   - Paste the prompt into ChatGPT (or Gemini/Claude), then paste ONLY the JSON reply into that file, save it.
echo   - Run Stage G again to finalize.
echo.
echo Tips:
echo   - Set env var G_K to change number of ideas (default 8). Example:
echo       set G_K=10
echo       RUN_G.bat
echo.

set "G_CHOICE=1"
set /p "G_CHOICE=Enter 1 or 2 and press Enter (default 1): "
for /f "tokens=* delims= " %%A in ("%G_CHOICE%") do set "G_CHOICE=%%A"
if "%G_CHOICE%"=="" set "G_CHOICE=1"

set "MODE=CHATGPT"
if "%G_CHOICE%"=="2" set "MODE=AUTO"

echo.
if "%MODE%"=="CHATGPT" (
  echo [MODE] ChatGPT-assisted (default)
) else (
  echo [MODE] Autonomous (no ChatGPT)
)
echo.

REM Detect target: either -All OR IdeaDir (or empty to auto-pick newest)
set "ARG1=%~1"
if /I "%ARG1%"=="-All" (
  if "%MODE%"=="CHATGPT" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -All -AskLLM -K %K%
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -All -K %K%
  )
) else (
  if "%MODE%"=="CHATGPT" (
    if "%ARG1%"=="" (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -AskLLM -K %K%
    ) else (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -IdeaDir "%ARG1%" -AskLLM -K %K%
    )
  ) else (
    if "%ARG1%"=="" (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -K %K%
    ) else (
      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_g.ps1" -IdeaDir "%ARG1%" -K %K%
    )
  )
)

set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Stage G finished.
) else if "%RC%"=="2" (
  echo [NEXT] ChatGPT step prepared. Follow the instructions above, then run Stage G again.
  echo [NEXT] See launcher_logs\LAST_LOG.txt for details.
) else (
  echo [WARN] Stage G finished with ExitCode=%RC% (see launcher_logs\LAST_LOG.txt)
)

echo.
pause
exit /b %RC%
