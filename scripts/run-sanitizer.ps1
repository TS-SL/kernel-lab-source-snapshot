# Runs the four Compute Sanitizer tools over the C++ tests, under gpu_guard.
#
# Usage (after building):   .\scripts\run-sanitizer.ps1 [-Build build\release] [-Name m1] [-MaxN 1048576]
#
# Writes one log per tool to results\raw\<Name>-sanitizer-<tool>.log (Name
# defaults to "latest", which is gitignored) and prints a per-tool verdict,
# checked in this order:
#   FAILED   the test program did not complete with "0 of N tests failed"
#            (crash, abnormal exit, test failure), or gpu_guard stopped the
#            run - regardless of anything else in the log
#   ERRORS   the log contains a sanitizer error other than the two known
#            debugger-interface start-up messages - a defect to investigate,
#            even if device-side checking was unavailable
#   BLOCKED  the program completed and the only errors are the start-up
#            messages: the WDDM debugger interface is disabled, so kernels
#            were NOT checked. NVIDIA's fix is to run, as administrator,
#            %CUDA_PATH%\compute-sanitizer\EnableDebuggerInterface.bat
#            (sets HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface=1).
#            This script never changes that setting.
#   CLEAN    tool ran with device-side checking and reported 0 errors
#
# The error_paths test is excluded because it provokes CUDA errors on purpose;
# kernel sizes are capped with --max-n because sanitizers slow kernels 10-100x
# and the Windows watchdog is 2 s.
param([string]$Build = 'build\release', [string]$Name = 'latest', [long]$MaxN = 1048576)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine') }
$sanitizer = Join-Path $cudaPath 'compute-sanitizer\compute-sanitizer.exe'
$python = Join-Path $root '.venv\Scripts\python.exe'
# Test programs and the tests to exclude from sanitizer runs (they provoke
# CUDA errors on purpose).
$programs = [ordered]@{
    smoke   = @{ exe = Join-Path $root "$Build\test_smoke.exe"; exclude = @('--exclude', 'error_paths') }
    softmax = @{ exe = Join-Path $root "$Build\test_softmax.exe"; exclude = @() }
    layernorm = @{ exe = Join-Path $root "$Build\test_layernorm.exe"; exclude = @() }
    gemm    = @{ exe = Join-Path $root "$Build\test_gemm.exe"; exclude = @() }
}
foreach ($p in @($sanitizer, $python) + @($programs.Values | ForEach-Object { $_.exe })) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

$tools = [ordered]@{
    memcheck  = @('--tool', 'memcheck', '--leak-check', 'full')
    racecheck = @('--tool', 'racecheck')
    initcheck = @('--tool', 'initcheck')
    synccheck = @('--tool', 'synccheck')
}
$outDir = Join-Path $root 'results\raw'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Push-Location $root  # gpu_guard --log paths below are repo-relative
$overall = 0
# Note: PowerShell variable names are case-insensitive, so the loop variable
# must not be called $name (it would overwrite the -Name parameter).
foreach ($prog in $programs.Keys) {
foreach ($tool in $tools.Keys) {
    $log = Join-Path $outDir "$Name-sanitizer-$prog-$tool.log"
    $gpuLog = "results\raw\$Name-sanitizer-$prog-$tool-gpu.csv"
    $cmdArgs = @($python, (Join-Path $root 'scripts\gpu_guard.py'), '--log', $gpuLog, '--max-seconds', '300', '--',
                 $sanitizer) + $tools[$tool] + @('--error-exitcode', '99', $programs[$prog].exe) +
                 $programs[$prog].exclude + @('--max-n', "$MaxN")
    $quoted = ($cmdArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    cmd /c "$quoted > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    $text = Get-Content $log -Raw
    # Redact user paths before the log is kept as evidence (repo root first;
    # it lies inside the user profile).
    foreach ($pair in @(@($root, '<repo>'), @($env:USERPROFILE, '<home>'))) {
        foreach ($p in @($pair[0], $pair[0].Replace('\', '/'))) {
            $text = $text -ireplace [regex]::Escape($p), $pair[1]
        }
    }
    Set-Content -Path $log -Value $text -Encoding utf8
    # Verdict rules (FAILED > ERRORS > BLOCKED > CLEAN): scripts/sanitizer_verdict.py,
    # unit-tested in tests/python/test_sanitizer_verdict.py.
    $res = (& $python (Join-Path $root 'scripts\sanitizer_verdict.py') $log $code) -split "`t", 2
    $verdict = $res[0]
    $otherErrors = @(if ($verdict -eq 'ERRORS') { $res[1] })
    if ($verdict -ne 'CLEAN') { $overall = 1 }
    $detail = if ($verdict -eq 'ERRORS') { "  first unexpected: $($otherErrors[0].Trim())" } else { '' }
    Write-Host ("{0,-8} {1,-10} {2,-8} exit={3}  log: {4}{5}" -f $prog, $tool, $verdict, $code, $log, $detail)
}
}
Pop-Location
exit $overall
