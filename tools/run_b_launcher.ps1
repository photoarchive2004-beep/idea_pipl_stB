#requires -Version 5.1
# UTF-8 for Windows PowerShell 5.1 console I/O
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$ErrorActionPreference = "Stop"

function Info([string]$m){ Write-Host ("[INFO] " + $m) }
function Warn([string]$m){ Write-Host ("[WARN] " + $m) -ForegroundColor Yellow }
function Err ([string]$m){ Write-Host ("[ERR]  " + $m) -ForegroundColor Red }

try {
  $root = Split-Path -Parent $PSScriptRoot
  $ideasDir = Join-Path $root "ideas"
  if (!(Test-Path $ideasDir)) { throw "Не найдена папка ideas." }

  # ---- choose idea dir ----
  $ideaDir = $null
  $activeFile = Join-Path $ideasDir "_ACTIVE_PATH.txt"
  if (Test-Path $activeFile) {
    $p = (Get-Content $activeFile -Raw).Trim()
    if ($p) {
      if (Test-Path $p) { $ideaDir = $p }
      else {
        $p2 = Join-Path $root $p
        if (Test-Path $p2) { $ideaDir = $p2 }
      }
    }
  }
  if (-not $ideaDir) {
    $last = Get-ChildItem $ideasDir -Directory -Filter "IDEA-*" | Sort-Object Name -Descending | Select-Object -First 1
    if ($last) { $ideaDir = $last.FullName }
  }
  if (-not $ideaDir) { throw "Не найдена папка идеи (IDEA-*). Создай идею через 1_NEW_IDEA.bat." }

  # ---- mode selection ----
  Write-Host "==============================="
  Write-Host "STAGE B — Мульти-источниковый поиск литературы"
  Write-Host "==============================="
  Write-Host ""
  Write-Host "Выберите режим:"
  Write-Host "  1) BALANCED (по умолчанию)"
  Write-Host "  2) WIDE"
  Write-Host "  3) FOCUSED"
  $sel = Read-Host "Введите 1/2/3 (пусто = 1)"
  if ([string]::IsNullOrWhiteSpace($sel)) { $sel = "1" }

  $scope = "balanced"
  switch ($sel.Trim()) {
    "1" { $scope = "balanced" }
    "2" { $scope = "wide" }
    "3" { $scope = "focused" }
    default { $scope = "balanced" }
  }
  Info ("РЕЖИМ=" + $scope)
  Info ("IDEA=" + $ideaDir)

  # ---- clean out (requested) ----
  $outDir = Join-Path $ideaDir "out"
  if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
  Get-ChildItem -Path $outDir -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Info ("Очищен out: " + $outDir)

  # ---- resolve python ----
  $py = Join-Path $root ".venv\Scripts\python.exe"
  if (!(Test-Path $py)) { $py = Join-Path $root ".venv_xr\Scripts\python.exe" }
  if (!(Test-Path $py)) { $py = "python" }

  # ---- stage B script ----
  $script = Join-Path $root "tools\b_lit_scout.py"
  if (!(Test-Path $script)) {
    $cand = Get-ChildItem (Join-Path $root "tools") -File -Filter "*lit*scout*.py" | Select-Object -First 1
    if ($cand) { $script = $cand.FullName }
  }
  if (!(Test-Path $script)) { throw "Не найден скрипт Stage B (ожидаю tools\b_lit_scout.py)." }

  # ---- logs ----
  $launcherLogs = Join-Path $root "launcher_logs"
  New-Item -ItemType Directory -Force -Path $launcherLogs | Out-Null
  $logFile = Join-Path $launcherLogs ("runB_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

  Info ("PY=" + $py)
  Info ("SCRIPT=" + $script)
  Info ("LOG=" + $logFile)

  # Run with UTF-8 env for python
  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"

  & $py $script "--idea-dir" $ideaDir "--scope" $scope 2>&1 | Tee-Object -FilePath $logFile
  $code = $LASTEXITCODE

  # ---- print summary ----
  $sum = Join-Path $outDir "stageB_summary.txt"
  $prisma = Join-Path $outDir "prisma_lite.md"
  if (!(Test-Path $sum) -and (Test-Path $prisma)) {
    # fallback: create summary from prisma if missing
    Get-Content $prisma -Raw | Set-Content -Encoding UTF8 $sum
  }

  Write-Host ""
  Write-Host "----------------------------"
  Write-Host "ИТОГ Stage B:"
  if (Test-Path $sum) {
    Get-Content $sum -Raw | Write-Host
  } else {
    Warn "Сводка stageB_summary.txt не найдена. Смотри лог запуска: $logFile"
  }
  Write-Host "----------------------------"

  if ($code -ne 0) {
    Err ("Stage B завершился с ошибкой. ExitCode=" + $code)
    Warn ("Смотри: " + $logFile + " и " + (Join-Path $outDir "search_log.json"))
  } else {
    Info "Stage B завершился успешно."
  }

  Read-Host "Нажмите Enter для выхода"
  exit $code
}
catch {
  Err $_.Exception.Message
  Read-Host "Нажмите Enter для выхода"
  exit 1
}
