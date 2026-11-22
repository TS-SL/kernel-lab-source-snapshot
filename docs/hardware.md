# Hardware and toolchain

All results in this repository come from the single machine below unless a
result says otherwise. Raw report: `results/raw/environment.txt` (generated
by `scripts/check-env.ps1`); C++ test output (`results/raw/m1-ctest.log`)
records the CUDA device query.

## Machine (VERIFIED by inspection)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB, compute capability 8.6 (sm_86) | environment.txt |
| GPU details (CUDA query) | 28 SMs, 1536 max resident threads per SM, kernel execution timeout enabled | m1-ctest.log (`device_info`) |
| L2 cache (CUDA device property, read through `torch.cuda.get_device_properties(0).L2_cache_size`) | 2304 KiB | environment.txt |
| Driver | 610.60 (R610), CUDA 13.3, WDDM driver model | environment.txt |
| Display / watchdog | This is the machine's only GPU and drives its monitor. nvidia-smi `display_active` changes with the monitor state (Enabled at one inspection, Disabled in the committed report). The Windows watchdog applies either way: the CUDA query reports kernel execution timeout = 1. | environment.txt, m1-ctest.log |
| VBIOS / power | 94.06.2f.40.11, 170 W default power limit (never modified) | environment.txt |
| GPU temperature limits (nvidia-smi) | target 83 C, max operating 93 C, slowdown 95 C, shutdown 98 C | environment.txt |
| GPU state when the report was taken | see the `gpu state now` line (temperature, fan, P-state, power, SM clock, max SM clock, utilization) - a snapshot, not a claim that the GPU was idle | environment.txt |
| CPU | Intel Core i5-12400 (6 cores / 12 threads) | environment.txt |
| Host memory | 15.7 GiB | environment.txt |
| OS | Windows 11 Home 25H2, build 26200, x64 | environment.txt |
| Windows TDR | default (2 s); not changed | environment.txt |
| Long paths | disabled (`LongPathsEnabled = 0`) - the repo must live at a short path | environment.txt |

Project stop threshold: 80 C (below the 83 C target), enforced by
`scripts/gpu_guard.py`.

## Toolchain (VERIFIED: versions printed by the installed tools)

Download sizes, hashes and signature results below are *unarchived
observations*: they were computed when each file was downloaded, and the
installers were not kept, so they cannot be re-checked
from this repository.

| Tool | Version | Source and verification of the download |
|---|---|---|
| CUDA Toolkit | 13.3.1 (`version.json`), nvcc 13.3.73 | `developer.download.nvidia.com/compute/cuda/13.3.1/network_installers/cuda_13.3.1_windows_network.exe`: MD5 `c7ad7bdbb732eb32002061d823bf8624` matched NVIDIA's published `md5sum.txt`; SHA-256 `0a4964db02d01a80f897605b0cb2c18fddaf277c44940a0f3436301a577b3c57`; valid NVIDIA Corporation Authenticode signature. The components it then downloaded (compiler, cudart, cuBLAS, Compute Sanitizer, Nsight Compute, NVTX, NVML headers, CCCL, binary utilities, VS integration) were verified only by the installer itself. No driver component (Windows toolkits do not bundle the driver since 13.1). |
| Host compiler | MSVC 19.44.35230 (Visual Studio Build Tools 2022 17.14.37906.3), Windows SDK 10.0.26100.0 | `aka.ms/vs/17/release/vs_buildtools.exe` (4,474,072 bytes): SHA-256 `8aaf70d164808d313ee6226af5d1e3b2db43c60be551bfeb2b70ce139d792c37` matched the hash in Microsoft's download URL; valid Microsoft Corporation signature. Components: VC tools workload (required parts), MSVC v143 x64/x86, Windows 11 SDK 26100. |
| CMake | 4.4.5 | GitHub release `cmake-4.4.5-windows-x86_64.zip`: SHA-256 `326c826486d46a23cf9d5a1c9ee3a567d427bdd02c4a34071937a2a675cff004` matched GitHub's asset digest and Kitware's `cmake-4.4.5-SHA-256.txt`. Unpacked to `%LOCALAPPDATA%\Programs\kernellens-tools` (no admin, no PATH change). |
| Ninja | 1.13.2 | GitHub release `ninja-win.zip`: SHA-256 `07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65` matched GitHub's asset digest. Same location. |
| Python | 3.12.10 (project venv `.venv`) | Already installed on the machine; its origin was not verified. |
| Python packages | exact set in `requirements-lock.txt` (numpy 2.5.5, pytest 9.2.0, torch 2.15.0+cu132, and their dependencies) | PyPI and `download.pytorch.org/whl/cu132` over HTTPS, versions pinned, hashes not pinned. torch is a reference only; `torch.cuda.is_available()` is True on this GPU. |
| Compute Sanitizer | 2026.2.1 | part of CUDA 13.3.1 |
| Nsight Compute | 2026.2.1 | part of CUDA 13.3.1 |

Why CUDA 13.3 and not the newer 13.4/13.5: 13.3 belongs to the same R610
branch as the installed driver, so no driver update is needed and no feature
depends on minor-version compatibility. PyTorch's CUDA 13.2 build is at or
below the driver's CUDA version.

Compatibility references consulted at setup: NVIDIA CUDA 13.3.1 release
notes and Windows installation guide (supported MSVC: VS 2019/2022/2026),
CUDA toolkit archive, Compute Sanitizer and Nsight Compute documentation,
Microsoft VS 2022 release history, PyTorch wheel index.

## Tool limitations on this machine

| Capability | Status | Reason / how to unblock (user action only) |
|---|---|---|
| Compute Sanitizer device-side checks | needs the WDDM debugger interface: BLOCKED while it is disabled (racecheck also exited abnormally then); CLEAN on all four test programs with it enabled (archived, `results/raw/m5-sanitizer-*.log`; docs/limitations.md) | A machine-wide setting, changed only by an administrator, never by the project. NVIDIA's fix: run `%CUDA_PATH%\compute-sanitizer\EnableDebuggerInterface.bat` as administrator; its single command is `REG ADD "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /f /v EnableInterface /t REG_DWORD /d 1`. |
| Nsight Compute metrics | needs GPU performance-counter access: BLOCKED without it (`ERR_NVGPUCTRPERM`, `results/raw/m2-ncu-probe.log`); available for the archived profile | Enabled for all users in the NVIDIA Control Panel by an administrator, not by the project. Profiles run with `--clock-control none` (clocks are never locked): `results/raw/m5-ncu-raw.csv`, docs/profiling.md. |
| Compute Sanitizer messages | - | While the interface is disabled, each sanitizer log also says "Device not supported" right after the debugger-interface error; it appears to be a consequence of the interface failure (UNVERIFIED), since the RTX 3060 / sm_86 is a supported architecture. |
| PowerShell scripts | blocked by default in a normal shell | No execution policy is set in any scope, so Windows' default (scripts disabled) applies. Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the PowerShell window first; it affects only that window. |
| Long-running kernels | limited by design | The display GPU is under the 2 s TDR watchdog; the project keeps launches short instead of changing TDR. |
