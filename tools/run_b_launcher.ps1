param(
  [ValidateSet("balanced","wide","focused")]
  [string]$Scope = "balanced",
  [string]$IdeaArg = "",
  [int]$N = 300
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir ("runB_" + $stamp + ".log")
$Last = Join-Path $LogDir "LAST_LOG.txt"
"" | Out-File -FilePath $Log -Encoding UTF8
$Log | Out-File -FilePath $Last -Encoding UTF8

function Log([string]$s){
  $s | Out-File -FilePath $Log -Append -Encoding UTF8
  Write-Host $s
}

function Resolve-IdeaDir([string]$arg) {
  $ideas = Join-Path $Root "ideas"
  if (-not (Test-Path $ideas)) { throw "Папка ideas не найдена. Сначала запустите 1_NEW_IDEA.bat." }

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

  throw "В ideas не найдены папки IDEA-*"
}

try {
  $IdeaDir = Resolve-IdeaDir $IdeaArg
  $IdeaDir = (Resolve-Path $IdeaDir).Path
  $outDir = Join-Path $IdeaDir "out"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  Log ("[INFO] ROOT=" + $Root)
  Log ("[INFO] IDEA_DIR=" + $IdeaDir)
  Log ("[INFO] SCOPE=" + $Scope)
  Log ("[INFO] N=" + $N)
  Log ("[INFO] LOG=" + $Log)

  $script = Join-Path $Root "tools\b_lit_scout.py"
  if (-not (Test-Path $script)) { throw "Missing: $script" }

  $py = Join-Path $Root ".venv\Scripts\python.exe"
  if (-not (Test-Path $py)) { $py = "python" }

  Log ("[INFO] PY=" + $py)
  Log ("[INFO] CMD: " + $py + " " + $script + " --idea-dir " + $IdeaDir + " --n " + $N + " --scope " + $Scope)

  & $py $script --idea-dir $IdeaDir --n $N --scope $Scope
  $code = $LASTEXITCODE

  $searchLog = Join-Path $outDir "search_log.json"
  $summaryPath = Join-Path $outDir "stageB_summary.txt"
  if (Test-Path $searchLog) {
    $j = Get-Content -LiteralPath $searchLog -Raw | ConvertFrom-Json
    $st = if ($j.status) { $j.status } else { if ($code -eq 0) { "OK" } else { "FAILED" } }
    $oa = 0; $nc = 0
    if ($j.source_stats) {
      if ($j.source_stats.openalex) { $oa = [int]$j.source_stats.openalex }
      if ($j.source_stats.ncbi) { $nc = [int]$j.source_stats.ncbi }
    }
    $dedup = 0
    if ($j.dedup_keys) { $dedup = [int]$j.dedup_keys }
    $corpus = Join-Path $outDir "corpus.csv"
    $lines = @(
      "=== Краткая сводка Stage B ===",
      "Статус: $st",
      "Источник OpenAlex добавил: $oa",
      "Источник NCBI добавил: $nc",
      "Дедуп-ключей: $dedup",
      "Корпус: $corpus"
    )
    $lines | Out-File -FilePath $summaryPath -Encoding UTF8
    foreach($ln in $lines){ Log $ln }
  } else {
    "=== Краткая сводка Stage B ===`nСтатус: FAILED`nsearch_log.json не найден" | Out-File -FilePath $summaryPath -Encoding UTF8
    Log "[ERR] search_log.json не найден, сводка неполная"
  }
  exit $code
}
catch {
  Log ("[ERR] " + $_.Exception.Message)
  exit 1
}
