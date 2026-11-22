# Runs a benchmark series for one operator: N rounds of the C++ harness
# (bench_<op>.exe: KernelLens implementations vs each other) and the
# in-process comparison harness (bench_<op>_compare.py: KernelLens vs PyTorch
# in one process), each under gpu_guard, in ABBA order across rounds
# (cpp,py / py,cpp / cpp,py ...), with a cool-down before every run.
#
# Usage:   .\scripts\run-bench.ps1 -Op softmax|layernorm [-Name m3] [-Processes 3]
#
# Rebuilds first, so the measured binary matches the work tree. The commit
# recorded in each row is `git rev-parse --short HEAD`, suffixed "-dirty" if
# there are ANY uncommitted or untracked files (the analysis refuses to label
# such runs MEASURED). Writes results\raw\<Name>-bench-<op>-{cpp,py}-p<N>.csv
# (+ .meta.json sidecars and guard logs).
param([Parameter(Mandatory = $true)][ValidateSet('softmax', 'layernorm', 'gemm')][string]$Op,
      [string]$Name = 'latest', [int]$Processes = 3, [int]$CoolTempC = 45, [int]$MinCoolS = 20,
      [int]$MaxWaitS = 180, [int]$MaxRunS = 600)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$bench = Join-Path $root "build\release\bench_$Op.exe"
$compare = "benchmarks\bench_${Op}_compare.py"
# GEMM has no Python harness: cuBLAS, its reference, runs inside the C++ harness.
# @(...) keeps a one-element list an array (PowerShell would unroll it to a
# string, and indexing a string yields characters).
$harnesses = @(if ($Op -eq 'gemm') { 'cpp' } else { 'cpp', 'py' })

Push-Location $root
try {
    cmd /c "cmake --build --preset release 2>&1" | Select-String -Pattern 'error|warning' | ForEach-Object { $_.Line }
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
    $needed = @($python, $bench)
    if ($harnesses -contains 'py') { $needed += Join-Path $root $compare }
    foreach ($p in $needed) { if (-not (Test-Path $p)) { throw "missing: $p" } }

    $commit = (git rev-parse --short HEAD).Trim()
    if (git status --porcelain) { $commit = "$commit-dirty" }

    function Wait-Cool {
        Start-Sleep -Seconds $MinCoolS
        $t0 = Get-Date
        while ($true) {
            $temp = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
            if ($temp -le $CoolTempC -or ((Get-Date) - $t0).TotalSeconds -ge $MaxWaitS) { return $temp }
            Start-Sleep -Seconds 5
        }
    }

    $failed = 0
    for ($p = 1; $p -le $Processes; $p++) {
        $order = @(if ($p % 2 -eq 1) { $harnesses } else { $harnesses[($harnesses.Count - 1)..0] })
        foreach ($kind in $order) {
            $temp = Wait-Cool
            $csv = "results\raw\$Name-bench-$Op-$kind-p$p.csv"
            $gpuLog = "results\raw\$Name-bench-$Op-$kind-p$p-gpu.csv"
            # Remove this run's outputs from any earlier series of the same name,
            # so a process that dies cannot leave a new CSV next to an old sidecar.
            foreach ($old in @($csv, "$csv.meta.json", $gpuLog)) {
                if (Test-Path $old) { Remove-Item -LiteralPath $old -Force }
            }
            $cmd = if ($kind -eq 'cpp') {
                "`"$bench`" --out $csv --process $p --commit $commit"
            } else {
                "`"$python`" $compare --out $csv --process $p --commit $commit"
            }
            Write-Host "[run-bench] $Op round $p $kind (start temp $temp C, commit $commit)"
            # 2>&1 inside cmd: Windows PowerShell 5.1 would otherwise turn the
            # tools' informational stderr lines into terminating errors.
            $env:KERNELLENS_LIB = Join-Path $root 'build\release\kernellens.dll'
            cmd /c "`"$python`" scripts\gpu_guard.py --log $gpuLog --max-seconds $MaxRunS -- $cmd 2>&1"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[run-bench] $Op $kind round $p FAILED (exit $LASTEXITCODE) - stopping the series"
                $failed = 1
                break
            }
        }
        if ($failed) { break }
    }
} finally { Pop-Location }
exit $failed
