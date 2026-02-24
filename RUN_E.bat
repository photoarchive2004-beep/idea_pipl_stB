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

echo   STAGE E - RUN SETTINGS

echo ==========================

echo.

echo Choose how Stage E should run:

echo   1) ChatGPT-assisted  (DEFAULT, recommended)

echo   2) Autonomous (no ChatGPT; faster, but less insightful)

echo.

echo If you pick ChatGPT-assisted:

echo   - Stage E may stop and prepare a prompt.

echo   - The prompt will be COPIED to your clipboard.

echo   - The file "in\llm_stageE.json" will open in Notepad.

echo   - Paste the prompt into ChatGPT, then paste ONLY the JSON reply into that file, save it.

echo   - Run Stage E again to finalize.

echo.

set "E_CHOICE=1"

set /p "E_CHOICE=Enter 1 or 2 and press Enter (default 1): "

for /f "tokens=* delims= " %%A in ("%E_CHOICE%") do set "E_CHOICE=%%A"

if "%E_CHOICE%"=="" set "E_CHOICE=1"



set "MODE=CHATGPT"

if "%E_CHOICE%"=="2" set "MODE=AUTO"



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

    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1" -All -AskLLM

  ) else (

    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1" -All

  )

) else (

  if "%FC%"=="-" (

    if "%MODE%"=="CHATGPT" (

      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1" -AskLLM

    ) else (

      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1"

    )

  ) else (

    if "%MODE%"=="CHATGPT" (

      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1" -IdeaDir "%~1" -AskLLM

    ) else (

      powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%tools\run_e.ps1" -IdeaDir "%~1"

    )

  )

)



set "RC=%ERRORLEVEL%"

echo.

if "%RC%"=="0" (

  echo [OK] Stage E finished.

) else if "%RC%"=="2" (

  echo [NEXT] ChatGPT step prepared. Follow the instructions above, then run Stage E again.

  echo [NEXT] See launcher_logs\LAST_LOG.txt for details.

) else (

  echo [WARN] Stage E finished with ExitCode=%RC% (see launcher_logs\LAST_LOG.txt)

)

echo.

pause

exit /b %RC%

