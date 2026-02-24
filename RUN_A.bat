@echo off
setlocal EnableExtensions
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_a.ps1"
pause
