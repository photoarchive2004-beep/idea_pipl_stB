param(
  [string]$IdeaDir = ""
)

try { chcp 65001 | Out-Null } catch {}
[Console]::InputEncoding  = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$ErrorActionPreference = "Stop"

function Say([string]$s){ Write-Host $s }

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$LogDir = Join-Path $Root "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "runC_last.log"
"" | Out-File -FilePath $Log -Encoding UTF8

function Resolve-IdeaDir([string]$arg) {
  $ideas = Join-Path $Root "ideas"
  if ($arg -and $arg.Trim()) {
    $p = $arg
    if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $Root $p }
    return [IO.Path]::GetFullPath($p)
  }
  $active = Join-Path $ideas "_ACTIVE_PATH.txt"
  if (Test-Path $active) {
    $p = (Get-Content -LiteralPath $active -Raw).Trim()
    if ($p) {
      if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $Root $p }
      if (Test-Path $p) { return [IO.Path]::GetFullPath($p) }
    }
  }
  $cands = Get-ChildItem -LiteralPath $ideas -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "IDEA-*" } | Sort-Object Name -Descending
  $first = $cands | Select-Object -First 1
  if ($first) { return $first.FullName }
  throw "В папке ideas нет ни одной IDEA-*"
}

try {
  $IdeaDir = Resolve-IdeaDir $IdeaDir
  $IdeaDir = (Resolve-Path $IdeaDir).Path

  $py = Join-Path $Root ".venv\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = "python" }

  $script = Join-Path $Root "tools\run_c3.py"
  if (-not (Test-Path $script)) { throw "Missing tools\run_c3.py" }

  Say "Stage C (через C3): $(Split-Path $IdeaDir -Leaf)"
  & $py $script --idea-dir $IdeaDir 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
  $rc = $LASTEXITCODE

  if ($rc -eq 0) {
    Say ""
    Say "Готово: Stage C3 завершён."
    Say "Файлы: out\evidence_table.csv ; out\evidence_table.md ; out\evidence_summary.md ; out\evidence_bundle.json"
    exit 0
  }
  if ($rc -eq 2) {
    Say ""
    Say "Ожидание ChatGPT: откройте in\c3_chatgpt\PROMPT.txt, вставьте ответ в RESPONSE.json и запустите RUN_C.bat снова."
    exit 2
  }

  Say "ОШИБКА: Stage C3 завершился с кодом=$rc. Смотрите launcher_logs\runC_last.log"
  exit 1
}
catch {
  $_ | Out-String | Out-File -FilePath $Log -Append -Encoding UTF8
  Say "ОШИБКА: launcher аварийно завершился. Смотрите launcher_logs\runC_last.log"
  exit 1
}
