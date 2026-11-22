# Deletes the preset's build folder, configures and builds from scratch, and
# saves a redacted build log as evidence.
#
# Usage:   . .\scripts\dev-env.ps1 ; .\scripts\clean-build.ps1 [-Preset release] [-Name m1] [-PtxasVerbose]
#
# Writes results\raw\<Name>-build.log (repo root -> <repo>, profile -> <home>)
# and prints the number of compiler warnings found in it. -PtxasVerbose adds
# ptxas' per-kernel register / shared-memory / spill report to the log.
param([string]$Preset = 'release', [string]$Name = 'latest', [switch]$PtxasVerbose)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build\$Preset"
if (Test-Path $buildDir) { Remove-Item -LiteralPath $buildDir -Recurse -Force }

Push-Location $root
try {
    $raw = Join-Path $env:TEMP "kernellens-$Name-build.tmp"
    $verbose = if ($PtxasVerbose) { 'ON' } else { 'OFF' }
    cmd /c "cmake --preset $Preset -DKERNELLENS_PTXAS_VERBOSE=$verbose > `"$raw`" 2>&1 && cmake --build --preset $Preset >> `"$raw`" 2>&1"
    $code = $LASTEXITCODE
} finally { Pop-Location }

$text = Get-Content $raw -Raw
Remove-Item $raw
foreach ($pair in @(@($root, '<repo>'), @($env:USERPROFILE, '<home>'))) {
    foreach ($p in @($pair[0], $pair[0].Replace('\', '/'))) {
        $text = $text -ireplace [regex]::Escape($p), $pair[1]
    }
}
$log = Join-Path $root "results\raw\$Name-build.log"
Set-Content -Path $log -Value $text -Encoding utf8
$warnings = ([regex]::Matches($text, '(?im)\bwarning\b(?!s? for)')).Count
Write-Host "clean build exit=$code, lines mentioning 'warning': $warnings, log: $log"
exit $code
