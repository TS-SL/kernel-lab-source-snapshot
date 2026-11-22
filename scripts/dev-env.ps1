# Sets up the KernelLens build environment in the CURRENT PowerShell session
# only (MSVC x64 tools, CUDA, CMake, Ninja). Changes nothing system-wide.
#
# Usage (dot-source it):   . .\scripts\dev-env.ps1
#
# Tool locations, overridable with environment variables:
#   KERNELLENS_TOOLS  folder holding cmake-4.4.5\ and ninja-1.13.2\
#                    (default: %LOCALAPPDATA%\Programs\kernellens-tools)
#   CUDA_PATH        CUDA 13.3 install (set by the CUDA installer)

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found: install Visual Studio 2022 Build Tools (see README)." }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'No Visual Studio installation with the MSVC x64 tools was found.' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'

# vcvars64 calls vswhere by name, so its folder must be on PATH.
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
# Import the environment that vcvars64.bat produces into this session.
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

if (-not $env:CUDA_PATH) {
    # Shells opened before the CUDA install do not have it yet.
    $env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
}
if (-not $env:CUDA_PATH -or -not (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
    throw 'CUDA toolkit not found (CUDA_PATH unset or nvcc.exe missing).'
}

$tools = if ($env:KERNELLENS_TOOLS) { $env:KERNELLENS_TOOLS } else { Join-Path $env:LOCALAPPDATA 'Programs\kernellens-tools' }
$cmakeBin = Join-Path $tools 'cmake-4.4.5\cmake-4.4.5-windows-x86_64\bin'
$ninjaDir = Join-Path $tools 'ninja-1.13.2'
foreach ($p in @($cmakeBin, $ninjaDir)) {
    if (-not (Test-Path $p)) { throw "Tool folder not found: $p (see README 'Prerequisites')." }
}
$env:PATH = "$cmakeBin;$ninjaDir;$(Join-Path $env:CUDA_PATH 'bin');$env:PATH"
# Let CMake's CUDA detection find this exact nvcc.
$env:CUDACXX = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'

Write-Host "KernelLens dev environment ready:"
Write-Host ("  cl     : " + (Get-Command cl.exe).Source)
Write-Host ("  nvcc   : " + $env:CUDACXX)
Write-Host ("  cmake  : " + (Get-Command cmake.exe).Source)
Write-Host ("  ninja  : " + (Get-Command ninja.exe).Source)
