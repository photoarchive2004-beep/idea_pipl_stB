@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Encoding / Unicode safety (do not remove) ---
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
REM -------------------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "ideas" mkdir "ideas"
if not exist "config" mkdir "config"

REM Date (stable)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "DATE=%%i"

set "N=1"
:loop
set "NN=00%N%"
set "NN=%NN:~-3%"
set "IDEA_ID=IDEA-%DATE%-%NN%"
set "DEST=ideas\%IDEA_ID%"
if exist "%DEST%" (
  set /a N+=1
  goto loop
)

mkdir "%DEST%" 2>nul
mkdir "%DEST%\in" 2>nul
mkdir "%DEST%\out" 2>nul
mkdir "%DEST%\logs" 2>nul

REM Ensure template exists
if not exist "config\idea_template.txt" (
  >"config\idea_template.txt" echo # IDEA (fill this text)
)

REM IMPORTANT: idea.txt MUST be in the idea folder AND in\idea.txt (compatibility)
copy /y "config\idea_template.txt" "%DEST%\idea.txt" >nul
copy /y "config\idea_template.txt" "%DEST%\in\idea.txt" >nul

REM Optional: mark "active" idea (some stages may use it)
>"ideas\_ACTIVE_ID.txt" echo %IDEA_ID%
>"ideas\_ACTIVE_PATH.txt" echo %DEST%

echo [OK] Created: %DEST%
echo [NEXT] 1) Fill: %DEST%\in\idea.txt  (Notepad opens now)
echo [NEXT] 2) Then run: RUN_A.bat

start "" notepad.exe "%DEST%\in\idea.txt" >nul 2>nul
start "" explorer.exe "%DEST%" >nul 2>nul
pause