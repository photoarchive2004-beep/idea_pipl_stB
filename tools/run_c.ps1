param([string]$IdeaDir = "")

# --- Encoding / Unicode safety ---
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { chcp 65001 | Out-Null } catch {}
# ---------------------------------

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir ("runC_" + $stamp + ".log")
$Log | Out-File -FilePath (Join-Path $LogDir "LAST_LOG.txt") -Encoding UTF8

function LogLine([string]$s) {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  "$ts $s" | Out-File -FilePath $Log -Append -Encoding UTF8
  Write-Host $s
}

try {
  LogLine "[DIAG] Root=$Root"
  LogLine "[DIAG] IdeaDir(arg)='$IdeaDir'"

  if ([string]::IsNullOrWhiteSpace($IdeaDir)) {
    $ideas = Join-Path $Root "ideas"
    if (-not (Test-Path $ideas)) { throw "Folder 'ideas' not found. Run 1_NEW_IDEA.bat first." }
    $latest = Get-ChildItem -Directory $ideas | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { throw "No idea folders inside 'ideas'. Run 1_NEW_IDEA.bat first." }
    $IdeaDir = $latest.FullName
    LogLine "[INFO] Using newest idea folder: $IdeaDir"
  } else {
    if (-not (Test-Path $IdeaDir)) {
      $cand = Join-Path $Root $IdeaDir
      if (Test-Path $cand) { $IdeaDir = $cand }
    }
    $IdeaDir = (Resolve-Path $IdeaDir).Path
    LogLine "[INFO] Using idea folder: $IdeaDir"
  }

  $py = Join-Path $Root ".venv\Scripts\python.exe"
  $script = Join-Path $Root "tools\c_evidence_engine.py"
  if (-not (Test-Path $py)) { throw "Python venv not found: $py (run 0_SETUP.bat)" }
  if (-not (Test-Path $script)) { throw "Not found: $script" }

  LogLine "[CMD] $py $script --idea `"$IdeaDir`""

  $oldEAP = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $py $script --idea $IdeaDir 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
  $ErrorActionPreference = $oldEAP

  $rc = $LASTEXITCODE
  LogLine "[DIAG] ExitCode=$rc"

  if ($rc -eq 2) {
    $prompt = Join-Path $IdeaDir "out\llm_prompt_C.txt"
    $target = Join-Path $IdeaDir "in\llm_evidence.json"
    if (Test-Path $prompt) {
      Set-Clipboard -Value (Get-Content -Raw $prompt)
      LogLine "[OK] Prompt copied to clipboard."
      LogLine "[NEXT] 1) Paste prompt into ChatGPT."
      LogLine "[NEXT] 2) Copy ONLY JSON from ChatGPT and paste into: $target"
      LogLine "[NEXT] 3) Save file, close Notepad, rerun RUN_C.bat"

      if (-not (Test-Path $target)) {
        New-Item -ItemType File -Force -Path $target | Out-Null
      }
      Start-Process notepad.exe -ArgumentList $target | Out-Null
      Start-Process explorer.exe -ArgumentList $IdeaDir | Out-Null
    } else {
      LogLine "[WARN] Prompt not found: $prompt"
    }
  }

  exit $rc
}
catch {
  LogLine ("[FATAL] " + $_.Exception.Message)
  LogLine ($_.ScriptStackTrace | Out-String)
  exit 1
}
