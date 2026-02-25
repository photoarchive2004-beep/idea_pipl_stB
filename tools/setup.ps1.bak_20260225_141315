param(
  [string]$Root = "",
  [string]$Log = ""
)

# ASCII-only setup script (PowerShell 5.1 safe in any codepage).
# Bootstraps Python venv and installs requirements for ALL stages.

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
  $Root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
} else {
  $Root = (Resolve-Path $Root).Path
}

if ([string]::IsNullOrWhiteSpace($Log)) {
  $Log = Join-Path $Root "setup_log.txt"
}

function LogLine([string]$Msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "$ts $Msg"
  try { $line | Out-File -FilePath $Log -Append -Encoding utf8 } catch {}
  Write-Host $Msg
}

function DownloadFile([string]$Url, [string]$OutFile, [int]$Retries = 3) {
  for ($i=1; $i -le $Retries; $i++) {
    try {
      Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
      return
    } catch {
      if ($i -eq $Retries) { throw }
      Start-Sleep -Seconds (2 * $i)
    }
  }
}

function EnsureSecretsEnv([string]$CfgDir) {
  $secrets = Join-Path $CfgDir "secrets.env"
  $example = Join-Path $CfgDir "secrets.env.example"
  if (Test-Path $secrets) { return }
  if (Test-Path $example) { Copy-Item -Force -LiteralPath $example -Destination $secrets; return }
  @"
# API keys for Idea-pipeline
# OPENALEX_API_KEY=your_key_here
OPENALEX_API_KEY=
"@ | Out-File -FilePath $secrets -Encoding utf8
}

function GetBasePython() {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($null -ne $cmd) { return $cmd.Source }

  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($null -ne $py) {
    try {
      $exe = & $py.Source -3 -c "import sys; print(sys.executable)" 2>$null
      if ($exe -and (Test-Path $exe)) { return $exe }
    } catch {}
  }

  $ver = "3.11.9"
  $pkgUrl = "https://www.nuget.org/api/v2/package/python/$ver"
  $pkgDir = Join-Path $Root ".cache"
  New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null

  $pkgFile = Join-Path $pkgDir ("python." + $ver + ".nupkg")
  if (-not (Test-Path $pkgFile)) {
    LogLine ("[INFO] Downloading portable Python from NuGet: " + $ver)
    DownloadFile -Url $pkgUrl -OutFile $pkgFile -Retries 3
  } else {
    LogLine ("[INFO] Using cached NuGet package: " + $pkgFile)
  }

  $extractDir = Join-Path $pkgDir ("python." + $ver)
  if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
  New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

  $zipFile = Join-Path $pkgDir ("python." + $ver + ".zip")
  Copy-Item -Force -LiteralPath $pkgFile -Destination $zipFile
  try {
    Expand-Archive -LiteralPath $zipFile -DestinationPath $extractDir -Force
  } finally {
    Remove-Item -Force -LiteralPath $zipFile -ErrorAction SilentlyContinue
  }

  $pyExe = Join-Path $extractDir "tools\python.exe"
  if (-not (Test-Path $pyExe)) {
    $found = Get-ChildItem -Path $extractDir -Filter python.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $found) { throw "python.exe not found in NuGet package." }
    $pyExe = $found.FullName
  }

  return $pyExe
}

function EnsureVenv([string]$BasePy) {
  $venvDir = Join-Path $Root ".venv"
  $venvPy  = Join-Path $venvDir "Scripts\python.exe"
  if (-not (Test-Path $venvPy)) {
    LogLine "[INFO] Creating .venv ..."
    & $BasePy -m venv $venvDir
  }
  if (-not (Test-Path $venvPy)) {
    throw "Failed to create venv: .venv\\Scripts\\python.exe not found."
  }
  return $venvPy
}

try {
  Remove-Item -Force $Log -ErrorAction SilentlyContinue
  LogLine ("[DIAG] Root=" + $Root)

  $cfgDir = Join-Path $Root "config"
  $ideasDir = Join-Path $Root "ideas"
  New-Item -ItemType Directory -Force -Path $cfgDir,$ideasDir | Out-Null
  EnsureSecretsEnv -CfgDir $cfgDir

  $need = @(
    "requirements.txt",
    "tools\module_a.py",
    "tools\b_lit_scout.py",
    "tools\c_evidence_engine.py",
    "tools\d_gap_map.py",
    "tools\e_novelty_breakthrough.py",
    "tools\f_dual_scoring.py",
    "tools\g_better_ideas.py"
  )
  foreach ($rel in $need) {
    $p = Join-Path $Root $rel
    if (-not (Test-Path $p)) { throw ("Missing file: " + $rel) }
  }

  $basePy = GetBasePython
  LogLine ("[DIAG] basePy=" + $basePy)

  $venvPy = EnsureVenv -BasePy $basePy
  LogLine ("[DIAG] venvPy=" + $venvPy)

  LogLine "[INFO] Upgrading pip ..."
  & $venvPy -m pip install --upgrade pip

  $req = Join-Path $Root "requirements.txt"
  LogLine ("[INFO] Installing requirements from: " + $req)
  & $venvPy -m pip install -r $req

  LogLine "[OK] Setup finished successfully."
  LogLine "[NEXT] If you plan to run Stage B, put your OPENALEX_API_KEY into config\\secrets.env"
  exit 0

} catch {
  LogLine "[ERROR] Setup failed."
  $err = ($_ | Out-String)
  if ($null -ne $err) { $err = $err.Trim() }
  LogLine ("[ERROR] " + $err)
  exit 1
}
