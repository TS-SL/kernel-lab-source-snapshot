# Read-only environment report for KernelLens. Changes nothing.
#
# Usage:   . .\scripts\dev-env.ps1 ; .\scripts\check-env.ps1 [-Out results\raw\environment.txt]
#
# Records the exact hardware and toolchain behind every result in this repo,
# and flags the two settings that block tools (debugger interface, profiler
# counters) without touching them.
param([string]$Out = '')

$ErrorActionPreference = 'Continue'
$lines = New-Object System.Collections.Generic.List[string]
function Add([string]$k, $v) { $lines.Add(('{0,-28} {1}' -f $k, $v)) }
function FirstLine($s) { if ($null -eq $s) { '(not found)' } else { (($s | Out-String).Trim() -split "`r?`n")[0] } }

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$release = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').DisplayVersion
Add 'date (UTC)' ((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm'))
Add 'os' "$($os.Caption) $release (build $($os.BuildNumber)), $($os.OSArchitecture)"
Add 'cpu' "$($cpu.Name.Trim()) ($($cpu.NumberOfCores) cores / $($cpu.NumberOfLogicalProcessors) threads)"
Add 'host memory (GiB)' ('{0:N1}' -f ($os.TotalVisibleMemorySize / 1MB))

$gpu = nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,power.default_limit,vbios_version --format=csv,noheader 2>$null
Add 'gpu (nvidia-smi)' (FirstLine $gpu)
Add 'gpu display active' (FirstLine (nvidia-smi --query-gpu=display_active --format=csv,noheader 2>$null))
Add 'gpu state now' (FirstLine (nvidia-smi --query-gpu=temperature.gpu,fan.speed,pstate,power.draw,clocks.sm,clocks.max.sm,utilization.gpu --format=csv,noheader 2>$null))
$q = (nvidia-smi -q -d TEMPERATURE 2>$null | Out-String)
$temps = foreach ($k in 'GPU Target Temperature Specification', 'GPU Max Operating Temp', 'GPU Slowdown Temp', 'GPU Shutdown Temp') {
    if ($q -match "$k\s*:\s*([^\r\n]+)") { "$($k -replace 'GPU ', ''): $($matches[1].Trim())" }
}
Add 'gpu temperature limits' ($temps -join '; ')
$smi = (nvidia-smi 2>$null | Out-String)
Add 'driver CUDA version' $(if ($smi -match 'CUDA (?:UMD )?Version:\s*([\d.]+)') { $matches[1] } else { '(not reported)' })
Add 'gpu driver model' (FirstLine (nvidia-smi --query-gpu=driver_model.current --format=csv,noheader 2>$null))

$cudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
$verJson = Join-Path $cudaRoot 'version.json'
if (Test-Path $verJson) {
    $vj = Get-Content $verJson -Raw | ConvertFrom-Json
    Add 'cuda toolkit (version.json)' $vj.cuda.version
}
$nvcc = if ($env:CUDACXX) { $env:CUDACXX } else { Join-Path $cudaRoot 'bin\nvcc.exe' }
Add 'nvcc' ((& $nvcc --version 2>$null | Select-String 'release') -join '').Trim()
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    $vsv = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion
    $vsn = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName
    Add 'visual studio' "$vsn $vsv"
}
$sdkInc = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\Include'
if (Test-Path $sdkInc) { Add 'windows sdk' ((Get-ChildItem $sdkInc -Directory | Select-Object -ExpandProperty Name) -join ', ') }
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
    $banner = (cmd /c "cl.exe 2>&1" | Select-Object -First 1)
    Add 'msvc' $banner
} else { Add 'msvc' '(cl.exe not on PATH; dot-source scripts\dev-env.ps1 first)' }
Add 'cmake' (FirstLine (cmake --version 2>$null))
Add 'ninja' (FirstLine (ninja --version 2>$null))

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
if (Test-Path $py) {
    $pyInfo = & $py -c "import sys, numpy, pytest; print(sys.version.split()[0], 'numpy', numpy.__version__, 'pytest', pytest.__version__)" 2>$null
    Add 'python (venv)' (FirstLine $pyInfo)
    $torchInfo = & $py -c "import torch; print(torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())" 2>$null
    Add 'torch (venv)' (FirstLine $torchInfo)
    $l2 = & $py -c "import torch; p = torch.cuda.get_device_properties(0); print(p.L2_cache_size // 1024, 'KiB')" 2>$null
    Add 'gpu L2 cache (torch query)' (FirstLine $l2)
} else { Add 'python (venv)' "(missing: $py)" }

$cudaPath = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
$san = Join-Path $cudaPath 'compute-sanitizer\compute-sanitizer.exe'
Add 'compute-sanitizer' ((& $san --version 2>$null | Select-String 'Version') -join '').Trim()
$ncu = Get-ChildItem 'C:\Program Files\NVIDIA Corporation' -Filter 'Nsight Compute*' -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
Add 'nsight compute' $(if ($ncu) { $ncu.Name } else { '(not found)' })

$dbg = (Get-ItemProperty 'HKLM:\SOFTWARE\NVIDIA Corporation\GPUDebugger' -ErrorAction SilentlyContinue).EnableInterface
Add 'GPUDebugger EnableInterface' $(if ($null -eq $dbg) { 'not set -> Compute Sanitizer device checks BLOCKED' } else { $dbg })
$prof = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak' -ErrorAction SilentlyContinue).RmProfilingAdminOnly
Add 'GPU perf counters' $(if ($null -eq $prof) { 'registry value absent -> driver default (admin only); non-elevated ncu verified BLOCKED, see results/raw/m2-ncu-probe.log' } elseif ($prof -eq 0) { 'all users (registry)' } else { 'admin only (registry)' })
$tdr = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -ErrorAction SilentlyContinue).TdrDelay
Add 'Windows TDR delay' $(if ($null -eq $tdr) { 'default (2 s)' } else { "$tdr s" })
$lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction SilentlyContinue).LongPathsEnabled
Add 'LongPathsEnabled' $(if ($lp -eq 1) { 'yes' } else { 'no (keep the repo at a short path)' })

$text = $lines -join "`r`n"
Write-Output $text
if ($Out) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
    Set-Content -Path $Out -Value $text -Encoding utf8
}
