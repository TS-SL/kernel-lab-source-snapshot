"""Tests for scripts/gpu_guard.py (no GPU compute; nvidia-smi must work).

The guard is the project's thermal/time safety net, so its stop paths are
tested directly: a time-limit stop must kill grandchildren too (regression
test for an earlier version whose terminate() left them running).

The tree-kill test needs normal user rights over the processes it starts:
the guard calls `taskkill /T /F`, which a restricted sandbox may forbid.
"""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "scripts" / "gpu_guard.py"
sys.path.insert(0, str(REPO / "scripts"))

import gpu_guard  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="guard process handling is Windows-specific")


def _alive(pid: int) -> bool:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not h:
        return False
    try:
        return k.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT -> still running
    finally:
        k.CloseHandle(h)


def _run_guard(tmp_path, *guard_args, command):
    return subprocess.run(
        [sys.executable, str(GUARD), "--log", str(tmp_path / "gpu.csv"), *guard_args, "--", *command],
        capture_output=True, text=True, timeout=60,
    )


def test_time_limit_kills_whole_process_tree(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    child_code = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(g.pid))\n"
        "time.sleep(60)\n"
    )
    # 5 s leaves the child ample time to start the grandchild and write its
    # PID before the stop (if it did not, reading the file would error, not pass).
    result = _run_guard(tmp_path, "--max-seconds", "5", command=[sys.executable, "-c", child_code])
    grandchild = int(pid_file.read_text())
    try:
        assert result.returncode == gpu_guard.EXIT_SAFETY_STOP, result.stderr
        assert "exceeded --max-seconds" in result.stderr
        deadline = time.monotonic() + 5
        while _alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(grandchild), "grandchild survived the safety stop"
    finally:
        subprocess.run(["taskkill", "/F", "/PID", str(grandchild)], capture_output=True)


def test_exit_code_is_passed_through(tmp_path):
    result = _run_guard(tmp_path, command=[sys.executable, "-c", "import sys; sys.exit(5)"])
    assert result.returncode == 5, result.stderr


def test_refuses_to_start_above_temperature_limit(tmp_path):
    # Any working GPU is warmer than 1 C, so this exercises the temperature trip.
    result = _run_guard(tmp_path, "--max-temp", "1", command=[sys.executable, "-c", "pass"])
    assert result.returncode == gpu_guard.EXIT_SAFETY_STOP
    assert "not starting" in result.stderr


def _sample(**overrides):
    s = {f: "Not Active" for f in gpu_guard.FIELDS}
    s.update({"timestamp": "t", "temperature.gpu": "50", "utilization.gpu": "0", "memory.used": "0",
              "power.draw": "10", "clocks.sm": "210", "clocks_event_reasons.active": "0x0"})
    s.update(overrides)
    return s


def test_violation_rules():
    assert gpu_guard.violation(_sample(), 80) is None
    assert "temperature 80" in gpu_guard.violation(_sample(**{"temperature.gpu": "80"}), 80)
    hot = _sample(**{"clocks_event_reasons.sw_thermal_slowdown": "Active"})
    assert "sw_thermal_slowdown" in gpu_guard.violation(hot, 80)
    # Unreadable values must stop the run, not crash the guard.
    assert "unreadable" in gpu_guard.violation(_sample(**{"temperature.gpu": "[N/A]"}), 80)
    assert "unreadable" in gpu_guard.violation(_sample(**{"clocks_event_reasons.hw_slowdown": "[N/A]"}), 80)
    # Power capping is logged but is not a stop condition.
    assert gpu_guard.violation(_sample(**{"clocks_event_reasons.sw_power_cap": "Active"}), 80) is None
