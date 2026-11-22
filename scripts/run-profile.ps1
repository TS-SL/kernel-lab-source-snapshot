# Profiles the kernels launched by build\release\profile_kernels.exe with
# Nsight Compute, under gpu_guard (docs/profiling.md).
#
# Usage (after building, in a shell prepared by scripts\dev-env.ps1):
#   .\scripts\run-profile.ps1 [-Name m5]
#
# Writes results\raw\<Name>-ncu-raw.csv (every collected metric, one row per
# profiled kernel launch; user paths redacted to <repo> / <home>) and the
# guard log. ncu runs with --clock-control none: it does NOT lock the GPU
# clocks (the project never changes clocks). Requires GPU performance-counter
# access for the user (an administrator setting in the NVIDIA Control Panel); without
# it ncu reports ERR_NVGPUCTRPERM and this script fails.
param([string]$Name = 'latest', [int]$MaxSeconds = 600)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$exe = Join-Path $root 'build\release\profile_kernels.exe'
$ncuDir = Get-ChildItem 'C:\Program Files\NVIDIA Corporation' -Filter 'Nsight Compute*' -Directory |
          Sort-Object Name -Descending | Select-Object -First 1
if (-not $ncuDir) { throw 'Nsight Compute not found' }
$ncu = Join-Path $ncuDir.FullName 'ncu.bat'
foreach ($p in @($python, $exe, $ncu)) { if (-not (Test-Path $p)) { throw "missing: $p" } }

$out = Join-Path $root "results\raw\$Name-ncu-raw.csv"
$err = Join-Path $env:TEMP "kernellens-$Name-ncu.stderr"
$gpuLog = "results\raw\$Name-ncu-gpu.csv"
Push-Location $root
try {
    # --set detailed: speed-of-light, memory workload, compute workload,
    # occupancy, launch statistics, warp-state (stall) statistics.
    cmd /c "`"$python`" scripts\gpu_guard.py --log $gpuLog --max-seconds $MaxSeconds -- `"$ncu`" --clock-control none --cache-control all --set detailed --csv --page raw `"$exe`" > `"$out`" 2> `"$err`""
    $code = $LASTEXITCODE
} finally { Pop-Location }

$text = Get-Content $out -Raw
foreach ($pair in @(@($root, '<repo>'), @($env:USERPROFILE, '<home>'))) {
    foreach ($p in @($pair[0], $pair[0].Replace('\', '/'))) { $text = $text -ireplace [regex]::Escape($p), $pair[1] }
}
Set-Content -Path $out -Value $text -Encoding utf8
Get-Content $err | Select-Object -Last 5
Write-Host "ncu exit=$code  csv: $out"
exit $code
