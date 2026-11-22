# Runs the full test suite (C++ + Python via ctest) under gpu_guard and saves
# the evidence logs.
#
# Usage (after building):   .\scripts\run-tests.ps1 [-Preset release] [-Name m1]
#
# Writes results\raw\<Name>-ctest.log and results\raw\<Name>-ctest-gpu.csv.
# Before saving, the log is redacted: the repo root becomes <repo> and the
# user profile folder becomes <home>, so committed logs carry no user paths.
param([string]$Preset = 'release', [string]$Name = 'latest')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "missing project venv: $python" }
$outDir = Join-Path $root 'results\raw'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$log = Join-Path $outDir "$Name-ctest.log"
$gpuLog = "results\raw\$Name-ctest-gpu.csv"

Push-Location $root
try {
    $raw = Join-Path $env:TEMP "kernellens-$Name-ctest.tmp"
    cmd /c "`"$python`" scripts\gpu_guard.py --log $gpuLog --max-seconds 600 -- ctest --preset $Preset --verbose > `"$raw`" 2>&1"
    $code = $LASTEXITCODE
} finally { Pop-Location }

$text = Get-Content $raw -Raw
Remove-Item $raw
# Repo root first (it lies inside the user profile), both slash styles.
foreach ($pair in @(@($root, '<repo>'), @($env:USERPROFILE, '<home>'))) {
    foreach ($p in @($pair[0], $pair[0].Replace('\', '/'))) {
        $text = $text -ireplace [regex]::Escape($p), $pair[1]
    }
}
Set-Content -Path $log -Value $text -Encoding utf8
$text -split "`r?`n" | Select-String -Pattern 'Test #|tests passed|tests failed|passed in|failed|skipped|SKIP|gpu_guard' | ForEach-Object { $_.Line }
Write-Host "exit=$code  log: $log"
exit $code
