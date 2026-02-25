param(
  [string]$IdeaDir = "",
  [string]$Mode = ""
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

function LogLine([string]$s){ $s | Out-File -FilePath $Log -Append -Encoding UTF8 }

function Resolve-IdeaDir([string]$arg) {
  $ideas = Join-Path $Root "ideas"
  if (-not (Test-Path $ideas)) { throw "Нет папки ideas." }

  if ([string]::IsNullOrWhiteSpace($arg)) {
    $active = Join-Path $ideas "_ACTIVE_PATH.txt"
    if (Test-Path $active) {
      $p = (Get-Content -LiteralPath $active -Raw).Trim()
      if ($p) {
        if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $Root $p }
        $p = [IO.Path]::GetFullPath($p)
        if (Test-Path $p) { return $p }
      }
    }
  }

  if ($arg -and $arg.Trim()) {
    $p = $arg
    if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $Root $p }
    $p = [IO.Path]::GetFullPath($p)
    if ((Split-Path -Leaf $p) -ieq "out") { $p = Split-Path -Parent $p }
    if (-not (Test-Path $p)) { throw "Папка идеи не найдена: $p" }
    return $p
  }

  $cands = Get-ChildItem -LiteralPath $ideas -Directory -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -like "IDEA-*" } |
           Sort-Object Name -Descending
  $first = $cands | Select-Object -First 1
  if ($first) { return $first.FullName }
  throw "В папке ideas нет ни одной IDEA-*"
}

function Ensure-IdeaLayout([string]$ideaDir){
  New-Item -ItemType Directory -Force -Path (Join-Path $ideaDir "in"),(Join-Path $ideaDir "out"),(Join-Path $ideaDir "logs") | Out-Null
}

function Restore-StructuredIdeaIfMissing([string]$ideaDir){
  $dst = Join-Path $ideaDir "out\structured_idea.json"
  if (Test-Path $dst) { return $true }
  $src = Join-Path $ideaDir "in\llm_response.json"
  if (-not (Test-Path $src)) { return $false }
  try {
    $raw = [System.IO.File]::ReadAllText($src, [System.Text.Encoding]::UTF8)
    $i0 = $raw.IndexOf("{"); $i1 = $raw.LastIndexOf("}")
    if ($i0 -lt 0 -or $i1 -le $i0) { return $false }
    $json = $raw.Substring($i0, ($i1-$i0+1))
    [System.IO.File]::WriteAllText($dst, $json, (New-Object System.Text.UTF8Encoding($false)))
    LogLine "[INFO] Восстановлен out\structured_idea.json из in\llm_response.json"
    return $true
  } catch { return $false }
}

try {
  $IdeaDir = Resolve-IdeaDir $IdeaDir
  $IdeaDir = (Resolve-Path $IdeaDir).Path
  Ensure-IdeaLayout $IdeaDir

  $ideaName = Split-Path $IdeaDir -Leaf
  Say "Stage C (Evidence): $ideaName"
  Say ""

  $noLLM = $false
  if ([string]::IsNullOrWhiteSpace($Mode)) {
    Say "Выберите режим:"
    Say "  1) ChatGPT Deep (по умолчанию, лучшее качество)"
    Say "  2) Autonomous (без ChatGPT)"
    $sel = Read-Host "Введите 1 или 2 (по умолчанию 1)"
    if ([string]::IsNullOrWhiteSpace($sel)) { $sel = "1" }
    if ($sel.Trim() -eq "2") { $Mode="fast"; $noLLM=$true } else { $Mode="deep"; $noLLM=$false }
  } else {
    $Mode = ($Mode + "").Trim().ToLower()
    if ($Mode -ne "fast" -and $Mode -ne "deep") { $Mode="deep" }
  }

  Say "Режим: $Mode" + $(if($noLLM){" + no-LLM"}else{""})
  LogLine "[INFO] IDEA=$IdeaDir"
  LogLine "[INFO] MODE=$Mode"
  LogLine "[INFO] noLLM=$noLLM"

  $corpus = Join-Path $IdeaDir "out\corpus.csv"
  if (-not (Test-Path $corpus)) {
    Say "ОШИБКА: out\corpus.csv не найден. Сначала запустите RUN_B.bat."
    exit 1
  }

  if (-not (Restore-StructuredIdeaIfMissing $IdeaDir)) {
    Say "ОШИБКА: out\structured_idea.json не найден и восстановление не удалось. Запустите RUN_A, затем RUN_B и снова RUN_C."
    exit 1
  }

  $py = Join-Path $Root ".venv\Scripts\python.exe"
  if (-not (Test-Path $py)) { throw "Python venv not found (.venv). Run 0_SETUP.bat" }

  $script = Join-Path $Root "tools\c_evidence_engine.py"
  if (-not (Test-Path $script)) { throw "Missing tools\c_evidence_engine.py" }

  $args = @($script, "--idea", $IdeaDir, "--mode", $Mode)
  if ($noLLM) { $args += "--no-llm" }

  LogLine ("[CMD] " + $py + " " + ($args -join " "))
  & $py @args 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
  $rc = $LASTEXITCODE

  if ($rc -eq 0) {
    Say ""
    Say "Готово: Stage C завершён."
    Say "Файлы: out\evidence_table.csv ; out\evidence_summary.md"
    exit 0
  }

  if ($rc -eq 2) {
    $prompt = Join-Path $IdeaDir "out\llm_prompt_C.txt"
    $resp   = Join-Path $IdeaDir "in\llm_evidence.json"

    Say ""
    Say "Нужен один ответ ChatGPT (режим Deep):"
    Say "1) Промпт скопирован в буфер (вставьте Ctrl+V в ChatGPT)."
    Say "2) Скопируйте ИСКЛЮЧИТЕЛЬНО JSON из ответа ChatGPT."
    Say "3) Вставьте JSON в in\llm_evidence.json и сохраните файл."
    Say "4) Запустите RUN_C.bat повторно."

    if (Test-Path $prompt) {
      Set-Clipboard -Value ([System.IO.File]::ReadAllText($prompt,[System.Text.Encoding]::UTF8))
      Start-Process notepad.exe -ArgumentList $prompt | Out-Null
    }
    if (-not (Test-Path $resp)) { New-Item -ItemType File -Force -Path $resp | Out-Null }
    Start-Process notepad.exe -ArgumentList $resp | Out-Null
    Start-Process explorer.exe -ArgumentList $IdeaDir | Out-Null
    exit 2
  }

  Say ("ОШИБКА: Stage C завершился с кодом=" + $rc + ". Смотрите launcher_logs\runC_last.log")
  Start-Process notepad.exe -ArgumentList $Log | Out-Null
  exit 1
}
catch {
  Say "ОШИБКА: launcher аварийно завершился. Открываю лог."
  $_ | Out-String | Out-File -FilePath $Log -Append -Encoding UTF8
  Start-Process notepad.exe -ArgumentList $Log | Out-Null
  exit 1
}
