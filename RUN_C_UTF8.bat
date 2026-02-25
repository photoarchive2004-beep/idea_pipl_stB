@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_c.ps1" -IdeaDir "%~1" -Mode "%~2"
pause