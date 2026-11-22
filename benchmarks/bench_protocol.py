"""Shared Python benchmark protocol (docs/methodology.md section 2).

Used by bench_softmax_compare.py and bench_layernorm_compare.py so that
KernelLens and PyTorch are measured with the same procedure as the C++
harness (benchmarks/bench_harness.hpp):

* one process, one CUDA stream, one set of buffers per shape; the
  implementations (KernelLens via ctypes on torch-owned memory, and torch)
  run interleaved ABBA, sample by sample, with the same batch size K
  (benchmarks/bench_common.hpp formula) and the same CUDA events;
* buffers rotate so one rotation touches >= 4x L2;
* per batch: GPU time per launch (CUDA events), host enqueue time for the
  whole batch (to flag host-bound launches), NVML SM/memory clock, P-state,
  temperature, clock-event reasons (read-only queries);
* correctness: before timing and after the series, outputs are poisoned and
  recomputed by the implementation, then checked; linearity B(2K) vs 2 B(K);
* a ~300 ms re-ramp after the CPU-side checks of each shape.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from typing import Callable

import torch

HEADER = ("harness,process,impl,rows,cols,elements,dtype,layout,sample,batch_k,ms_per_launch,host_enqueue_ms,"
          "cache,buffers,sm_clock_mhz,mem_clock_mhz,pstate,temp_c,clock_event_reasons,gpu,driver,cuda_driver,"
          "cuda_runtime,commit,build,timestamp_utc,timing,includes_transfers")


def batch_k(elements: int, target_ms: float) -> int:
    """Same formula as kl_bench::batch_k in benchmarks/bench_common.hpp."""
    return int(min(2000, max(1, math.ceil(target_ms * 1e-3 * 300e9 / (8 * elements)))))


def rotation_pairs(elements: int, l2_bytes: int) -> int:
    """Same rule as kl_bench::rotation_pairs: one rotation touches >= 4x L2."""
    return max(1, math.ceil(4 * l2_bytes / (2 * elements * 4)))


class Nvml:
    """Minimal read-only NVML access (nvml.dll ships with the driver)."""

    class _Temp(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("sensorType", ctypes.c_int), ("temperature", ctypes.c_int)]

    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.WinDLL("nvml.dll")
            self.dev = ctypes.c_void_p()
            self.ok = (self.lib.nvmlInit_v2() == 0 and
                       self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.dev)) == 0)
        except OSError:
            self.ok = False

    def driver(self) -> tuple[str, int]:
        if not self.ok:
            return "unknown", 0
        buf = ctypes.create_string_buffer(96)
        v = ctypes.c_int()
        self.lib.nvmlSystemGetDriverVersion(buf, 96)
        self.lib.nvmlSystemGetCudaDriverVersion(ctypes.byref(v))
        return buf.value.decode(), v.value

    def sample(self) -> dict:
        if not self.ok:
            return {"sm": 0, "mem": 0, "pstate": -1, "temp": 0, "reasons": 0}
        sm, mem, ps, reasons = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_int(), ctypes.c_ulonglong()
        t = self._Temp()
        t.version = ctypes.sizeof(self._Temp) | (1 << 24)  # NVML_STRUCT_VERSION(Temperature, 1)
        self.lib.nvmlDeviceGetClockInfo(self.dev, 1, ctypes.byref(sm))   # NVML_CLOCK_SM
        self.lib.nvmlDeviceGetClockInfo(self.dev, 2, ctypes.byref(mem))  # NVML_CLOCK_MEM
        if self.lib.nvmlDeviceGetPerformanceState(self.dev, ctypes.byref(ps)) != 0:
            ps.value = -1
        temp = t.temperature if self.lib.nvmlDeviceGetTemperatureV(self.dev, ctypes.byref(t)) == 0 else 0
        self.lib.nvmlDeviceGetCurrentClocksEventReasons(self.dev, ctypes.byref(reasons))
        return {"sm": sm.value, "mem": mem.value, "pstate": ps.value, "temp": temp, "reasons": reasons.value}


class Run:
    """Per-process state: options, output CSV, metadata and the stream."""

    def __init__(self, args, out, nvml: Nvml, meta: dict, stream):
        self.args, self.out, self.nvml, self.meta, self.stream = args, out, nvml, meta, stream
        self.items: list[dict] = []


def run_protocol(run: Run, rows: int, cols: int, pairs: int, impls: dict[str, Callable[[int], None]],
                 check: Callable[[str, int], dict], poison: Callable[[], None]) -> None:
    """impls: name -> launch(p) enqueueing one launch on buffer pair p (on
    run.stream). check(name, p) checks pair p's output (raising on failure)
    and returns fields for the JSON sidecar. poison() fills every output
    with NaN on run.stream."""
    stream, args = run.stream, run.args
    k = batch_k(rows * cols, args.target_ms)
    names = list(impls)
    rot = [0]
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)  # torch creates CUDA events lazily: do it outside timed batches
        stop.record(stream)
    stream.synchronize()

    def launch(name):
        p = rot[0] % pairs
        rot[0] += 1
        impls[name](p)

    def time_batch(name, batch):
        with torch.cuda.stream(stream):
            h0 = time.perf_counter()
            start.record(stream)
            for _ in range(batch):
                launch(name)
            stop.record(stream)
            h1 = time.perf_counter()
        stop.synchronize()
        return start.elapsed_time(stop) / batch, (h1 - h0) * 1e3

    def poison_run_check(name, when):
        # Outputs poisoned (NaN) outside any timed region, so a check can only
        # pass on output this implementation itself just wrote.
        with torch.cuda.stream(stream):
            poison()
            for _ in range(pairs):
                launch(name)
        stream.synchronize()
        item = {"type": "check", "when": when, "impl": name, "rows": rows, "cols": cols}
        item.update(check(name, (rot[0] - 1) % pairs))
        run.items.append(item)

    stream.wait_stream(torch.cuda.current_stream())  # inputs were copied on the default stream
    for name in names:
        poison_run_check(name, "before_timing")
    # The CPU-side checks leave the GPU idle long enough to drop its clocks
    # (observed in the first measured softmax series): re-ramp with ~300 ms
    # of this shape's own launches before warm-up, as in bench_harness.hpp.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        for name in names:
            time_batch(name, k)
    for _ in range(args.warmup):
        for name in names:
            time_batch(name, k)
    for name in names:
        tk, _ = time_batch(name, k)
        t2k, _ = time_batch(name, 2 * k)
        run.items.append({"type": "linearity", "impl": name, "rows": rows, "cols": cols, "k": k,
                          "ms_per_launch_k": tk, "ms_per_launch_2k": t2k, "fixed_batch_cost_ms": 2 * k * (tk - t2k)})
    meta = run.meta
    for smp in range(args.samples):
        order = names if smp % 2 == 0 else names[::-1]  # ABBA
        for name in order:
            ms, host_ms = time_batch(name, k)
            g = run.nvml.sample()
            run.out.write(f"python,{args.process},{name},{rows},{cols},{rows * cols},fp32,row_major,{smp},{k},"
                          f"{ms:.6f},{host_ms:.6f},rotated_cold_between_launches,{pairs},{g['sm']},{g['mem']},"
                          f"{g['pstate']},{g['temp']},0x{g['reasons']:x},\"{meta['gpu']}\",{meta['driver']},"
                          f"{meta['cuda_driver']},{meta['runtime']},{args.commit},\"{meta['build']}\","
                          f"{meta['stamp']},cuda_events_kernel_only,no\n")
    for name in names:
        poison_run_check(name, "after_timing_fresh_pass")
    run.out.flush()


def parse_args(doc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--process", type=int, default=1)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--target-ms", type=float, default=1.0)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not (1 <= args.samples <= 1000 and 0 <= args.warmup <= 1000 and 0 < args.target_ms <= 5):
        ap.error("out-of-range --samples/--warmup/--target-ms")
    return args


def main(doc: str, prog: str, lib, ramp_batch: Callable[[], None],
         body: Callable[[Run, int], None]) -> None:
    """Process skeleton: metadata, a ~300 ms clock ramp on the stream
    (ramp_batch enqueues ~20 short launches), then body(run, l2_bytes), then
    the JSON sidecar."""
    import sys

    from kl import DeviceInfo  # tests/python is on sys.path (set by the caller)

    args = parse_args(doc)
    props = torch.cuda.get_device_properties(0)
    nvml = Nvml()
    driver, cuda_driver = nvml.driver()
    major, minor = (int(v) for v in torch.version.cuda.split(".")[:2])
    info = DeviceInfo()
    lib.kl_get_device_info(info)
    meta = {"gpu": props.name, "driver": driver, "cuda_driver": cuda_driver,
            "runtime": f"torch:{major * 1000 + minor * 10}/kernellens:{info.runtime_version}",
            "build": f"torch {torch.__version__}; kernellens.dll via ctypes",
            "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stream = torch.cuda.Stream()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        while time.perf_counter() - t0 < 0.3:
            ramp_batch()
            stream.synchronize()
    g = nvml.sample()
    with open(args.out, "w", newline="") as out:
        run = Run(args, out, nvml, meta, stream)
        run.items.append({"type": "after_clock_ramp", "ramp_ms": 300, "sm_mhz": g["sm"], "mem_mhz": g["mem"],
                          "pstate": g["pstate"], "temp_c": g["temp"], "reasons": hex(g["reasons"])})
        print(f"{prog}: {props.name}, driver {driver}, L2 {props.L2_cache_size // 1024} KiB, "
              f"torch {torch.__version__}; after ramp SM {g['sm']} MHz, mem {g['mem']} MHz, P{g['pstate']}",
              file=sys.stderr)
        out.write(HEADER + "\n")
        body(run, props.L2_cache_size)
    with open(args.out + ".meta.json", "w") as fh:
        json.dump({"harness": "python", "process": args.process, "commit": args.commit, **meta,
                   "l2_bytes": props.L2_cache_size, "nvml": nvml.ok, "items": run.items}, fh, indent=2)
