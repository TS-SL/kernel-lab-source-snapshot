"""Run a command while watching the GPU; stop it if a safety limit is hit.

Every GPU workload in this project (tests, sanitizer runs, benchmarks) is
launched through this guard. Once per second it polls nvidia-smi and appends a
CSV row to --log. The command's WHOLE process tree is killed, and the guard
exits with status 3, if any of these happen:

  * GPU temperature >= --max-temp (default 80 C, the project stop threshold;
    the RTX 3060 reports a 83 C target and 93 C max operating temperature)
  * a thermal or hardware slowdown clock-event reason becomes active
  * the run exceeds --max-seconds (default 600)
  * nvidia-smi stops answering or returns an unreadable value (treated as
    possible driver instability: the guard cannot see the GPU, so it stops)

Process-tree handling (Windows): on a stop, `taskkill /T /F` kills the
command and every descendant (e.g. the test binary under ctest or
compute-sanitizer) by walking parent-process links while the tree is intact.
A nested Job Object was tried first and does NOT work here: the shell this
project runs in is itself inside a job with JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
(observed on the development machine), so grandchildren silently leave any nested job.
Residual gap: a descendant whose parent already exited (an orphan) is not
found by the tree walk; the guard also does not chase processes left behind
after a command exits normally.

The guard never changes any GPU setting; it only reads nvidia-smi.

Usage:
  python scripts/gpu_guard.py --log results/raw/run.csv [--max-temp 80] \
      [--max-seconds 600] -- <command> [args...]
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

FIELDS = [
    "timestamp",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "power.draw",
    "clocks.sm",
    "clocks_event_reasons.active",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    # Logged for benchmark context (power capping is normal under load and is
    # not a stop condition).
    "clocks_event_reasons.sw_power_cap",
]
THERMAL_FLAGS = (
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)
EXIT_SAFETY_STOP = 3


def query_gpu() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    values = [v.strip() for v in out.split(",")]
    if len(values) != len(FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi output: {out!r}")
    return dict(zip(FIELDS, values))


def violation(sample: dict[str, str], max_temp: float) -> str | None:
    """Reason to stop, or None. Unreadable values are a reason to stop."""
    try:
        temp = float(sample["temperature.gpu"])
    except ValueError:
        return f"unreadable GPU temperature {sample['temperature.gpu']!r}"
    if temp >= max_temp:
        return f"GPU temperature {temp:.0f} C >= limit {max_temp:.0f} C"
    for flag in THERMAL_FLAGS:
        value = sample[flag].strip().lower()
        if value == "active":
            return f"{flag} is active"
        if value != "not active":
            return f"unreadable {flag} value {sample[flag]!r}"
    return None


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and all its descendants (see module docstring)."""
    if os.name == "nt":
        # Must run while `proc` is alive: the walk starts from its PID.
        result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                                text=True)
        if result.returncode != 0:
            print(f"[gpu_guard] warning: taskkill /T failed ({result.returncode}): "
                  f"{(result.stderr or result.stdout).strip()}", file=sys.stderr)
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, type=Path, help="CSV file for 1 Hz GPU samples")
    parser.add_argument("--max-temp", type=float, default=80.0)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command [args...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given (use: -- <command> [args...])")

    # Refuse to start on an already hot or throttling GPU.
    first = query_gpu()
    reason = violation(first, args.max_temp)
    if reason:
        print(f"[gpu_guard] not starting: {reason}", file=sys.stderr)
        return EXIT_SAFETY_STOP

    args.log.parent.mkdir(parents=True, exist_ok=True)
    stop_reason = None
    peak_temp = float(first["temperature.gpu"])
    start = time.monotonic()
    with args.log.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(first)
        child = subprocess.Popen(command)
        try:
            while child.poll() is None:
                time.sleep(1.0)
                if time.monotonic() - start > args.max_seconds:
                    stop_reason = f"run exceeded --max-seconds={args.max_seconds:.0f}"
                    break
                try:
                    sample = query_gpu()
                except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
                    stop_reason = f"nvidia-smi query failed ({exc}); possible driver instability"
                    break
                writer.writerow(sample)
                fh.flush()
                stop_reason = violation(sample, args.max_temp)
                if stop_reason:
                    break
                peak_temp = max(peak_temp, float(sample["temperature.gpu"]))
        finally:
            if child.poll() is None:
                kill_tree(child)
        # One sample after the child exits, so short runs still log an end state.
        try:
            last = query_gpu()
            writer.writerow(last)
            peak_temp = max(peak_temp, float(last["temperature.gpu"]))
        except (subprocess.SubprocessError, RuntimeError, OSError, ValueError):
            pass

    elapsed = time.monotonic() - start
    print(f"[gpu_guard] {elapsed:.1f} s, peak {peak_temp:.0f} C, log: {args.log}", file=sys.stderr)
    if stop_reason:
        print(f"[gpu_guard] SAFETY STOP: {stop_reason}", file=sys.stderr)
        return EXIT_SAFETY_STOP
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
