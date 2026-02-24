param([string]$IdeaDir = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "runA_last.log"
"" | Out-File -FilePath $Log -Encoding UTF8

function Say([string]$s){ Write-Host $s }
function Log([string]$s){ $s | Out-File -FilePath $Log -Append -Encoding UTF8 }

function Resolve-IdeaDir([string]$arg) {
  $ideas = Join-Path $Root "ideas"
  if (-not (Test-Path $ideas)) { throw "Нет папки ideas. Сначала запусти 1_NEW_IDEA.bat" }

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
  $inDir = Join-Path $ideaDir "in"
  $outDir = Join-Path $ideaDir "out"
  $logsDir = Join-Path $ideaDir "logs"
  New-Item -ItemType Directory -Force -Path $inDir,$outDir,$logsDir | Out-Null

  $ideaTop = Join-Path $ideaDir "idea.txt"
  $ideaIn  = Join-Path $inDir "idea.txt"
  if (Test-Path $ideaIn) {
    if ((-not (Test-Path $ideaTop)) -or ((Get-Item $ideaIn).LastWriteTime -gt (Get-Item $ideaTop).LastWriteTime)) {
      Copy-Item -Force -LiteralPath $ideaIn -Destination $ideaTop
    }
  }
  if (-not (Test-Path $ideaTop)) { throw "Не найден текст идеи. Заполни in\idea.txt" }
}

function Remove-Utf8BomIfAny([string]$path){
  if (-not (Test-Path $path)) { return }
  $b = [IO.File]::ReadAllBytes($path)
  if ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF) {
    $b2 = $b[3..($b.Length-1)]
    $txt = [Text.Encoding]::UTF8.GetString($b2)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($path, $txt, $utf8NoBom)
  }
}

function Get-ExpectedResponseNames([string]$modulePath){
  # пытаемся вытащить имена .json прямо из module_a.py (без догадок)
  $set = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
  try {
    $t = Get-Content -LiteralPath $modulePath -Raw
    $ms = [regex]::Matches($t, "(?i)(['""])([^'""]+?\.json)\1")
    foreach($m in $ms){
      $p = $m.Groups[2].Value
      $bn = Split-Path $p -Leaf
      if ($bn -match '(?i)response|reply|llm|stage') { [void]$set.Add($bn) }
    }
  } catch {}

  if ($set.Count -eq 0) {
    # fallback (минимум лишнего)
    [void]$set.Add("llm_response.json")
    [void]$set.Add("response.json")
    [void]$set.Add("llm_response_A.json")
  }
  return $set
}

function Find-Prompt([string]$outDir){
  if (-not (Test-Path $outDir)) { return $null }
  $common = @("llm_prompt_A.txt","llm_prompt.txt","prompt.txt","chatgpt_prompt.txt")
  foreach($n in $common){
    $p = Join-Path $outDir $n
    if (Test-Path $p) { return $p }
  }
  $cand = Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -match '(?i)prompt' -and $_.Extension -in '.txt','.md' } |
          Sort-Object LastWriteTime -Descending |
          Select-Object -First 1
  if ($cand) { return $cand.FullName }
  return $null
}

try {
  $IdeaDir = Resolve-IdeaDir $IdeaDir
  $IdeaDir = (Resolve-Path $IdeaDir).Path
  $ideaName = Split-Path $IdeaDir -Leaf

  Ensure-IdeaLayout $IdeaDir

  $py = Join-Path $Root ".venv\Scripts\python.exe"
  $module = Join-Path $Root "tools\module_a.py"
  if (-not (Test-Path $py))     { throw "Не найден .venv. Запусти 0_SETUP.bat" }
  if (-not (Test-Path $module)) { throw "Не найден tools\module_a.py" }

  $respMain = Join-Path $IdeaDir "in\llm_response.json"
  Remove-Utf8BomIfAny $respMain

  # Если ответ уже есть — делаем “умные” временные копии под то имя, которое ждёт module_a.py
  $tempCopies = @()
  if ((Test-Path $respMain) -and (Get-Item $respMain).Length -gt 5) {
    $names = Get-ExpectedResponseNames $module
    foreach($n in $names){
      $dst = Join-Path (Join-Path $IdeaDir "in") $n
      if ($dst -ne $respMain) {
        Copy-Item -Force -LiteralPath $respMain -Destination $dst
        $tempCopies += $dst
      }
    }
  }

  Say "Stage A: выполняю (идея: $ideaName)..."
  Log "[CMD] $py $module --idea `"$IdeaDir`""
  & $py $module --idea $IdeaDir *> $Log
  $rc = $LASTEXITCODE

  if ($rc -eq 0) {
    # чистим временные копии (чтобы не плодить файлы)
    foreach($p in $tempCopies){
      if (Test-Path $p) { Remove-Item -Force $p -ErrorAction SilentlyContinue }
    }
    Say ""
    Say "✅ Stage A готова."
    Say "Дальше: запусти RUN_B.bat"
    exit 0
  }

  if ($rc -eq 2) {
    $prompt = Find-Prompt (Join-Path $IdeaDir "out")
    $resp   = Join-Path $IdeaDir "in\llm_response.json"
    if (-not (Test-Path $resp)) { New-Item -ItemType File -Force -Path $resp | Out-Null }

    Say ""
    Say "Нужен ChatGPT (1 раз). Делай так:"
    Say "1) Откроется PROMPT и файл ответа."
    Say "2) PROMPT уже в буфере (Ctrl+V в ChatGPT)."
    Say "3) Из ответа скопируй ТОЛЬКО JSON."
    Say "4) Вставь JSON в llm_response.json, сохрани."
    Say "5) Запусти RUN_A.bat ещё раз."

    if ($prompt) {
      Set-Clipboard -Value (Get-Content -Raw -LiteralPath $prompt)
      Start-Process notepad.exe -ArgumentList $prompt | Out-Null
    }
    Start-Process notepad.exe -ArgumentList $resp | Out-Null
    exit 2
  }

  Say ""
  Say "❌ Stage A: ошибка. Открою лог."
  Start-Process notepad.exe -ArgumentList $Log | Out-Null
  exit 1
}
catch {
  Say ""
  Say "❌ Stage A: ошибка запуска. Открою лог."
  $_ | Out-String | Out-File -FilePath $Log -Append -Encoding UTF8
  Start-Process notepad.exe -ArgumentList $Log | Out-Null
  exit 1
}
