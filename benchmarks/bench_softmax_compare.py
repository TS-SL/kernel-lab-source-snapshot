"""In-process comparison of KernelLens softmax with torch.softmax.

One process, one CUDA stream, one set of buffers: torch.softmax(x, -1, out=y)
and the KernelLens kernels (kl_softmax_device on the same torch tensors and
stream, via ctypes) run interleaved ABBA with the same batch size K, the same
CUDA events and therefore the same clocks (protocol: bench_protocol.py).
This removes the cross-process clock confound of comparing separate runs.

Usage: python benchmarks/bench_softmax_compare.py --out FILE.csv [--process N]
       [--samples N] [--warmup N] [--target-ms X] [--commit SHA] [--quick]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_protocol as bp  # noqa: E402
import kl  # noqa: E402

SHAPES = [(4096, 32), (4096, 128), (4096, 512), (4096, 1024), (2048, 2048),
          (1024, 4096), (256, 16384), (64, 65536), (16, 262144), (4, 1048576),
          (4, 65536), (8, 65536), (16, 65536), (28, 65536), (56, 65536), (112, 65536), (224, 65536)]
QUICK_SHAPES = [(4096, 128), (1024, 4096)]


def bench_shape(run: bp.Run, lib, rows: int, cols: int, l2_bytes: int) -> None:
    elems = rows * cols
    pairs = bp.rotation_pairs(elems, l2_bytes)
    host_x, xs, ys = [], [], []
    for p in range(pairs):
        h = kl.fill_uniform(1000 + 31 * p + cols, elems, -10.0, 10.0).reshape(rows, cols)
        host_x.append(h)
        xs.append(torch.from_numpy(h).cuda())
        ys.append(torch.empty_like(xs[-1]))
    ws = torch.empty(max(1, 2 * rows), device="cuda")  # MULTIPASS workspace (2 floats per row)
    handle = ctypes.c_void_p(run.stream.cuda_stream)

    def kl_launch(impl):
        def launch(p):
            st = lib.kl_softmax_device(rows, cols, xs[p].data_ptr(), ys[p].data_ptr(), impl, ws.data_ptr(), handle)
            if st != kl.KL_OK:
                raise RuntimeError(f"kl_softmax_device failed: {lib.kl_last_error_message()!r}")
        return launch

    impls = {name: kl_launch(kl.SOFTMAX_IMPLS[name]) for name in ("multipass", "fused_online")}
    if cols <= kl.SOFTMAX_WARP_MAX_COLS:
        impls["fused_warp"] = kl_launch(kl.SOFTMAX_FUSED_WARP)
    impls["torch_softmax"] = lambda p: torch.softmax(xs[p], -1, out=ys[p])

    def check(name, p):
        y = ys[p].cpu().numpy()
        if name == "torch_softmax":
            ref, _ = kl.softmax_reference(host_x[p])
            return {"max_rel_err_vs_float64": float(np.max(np.abs(y - ref) / np.maximum(ref, kl.FLT_MIN)))}
        return {"worst_ratio_to_bound": kl.softmax_check(host_x[p], y, kl.SOFTMAX_IMPLS[name])}  # asserts

    def poison():
        for y in ys:
            y.fill_(float("nan"))

    bp.run_protocol(run, rows, cols, pairs, impls, check, poison)


def main():
    lib = kl.load_library()
    ramp_bufs = {"x": torch.zeros(4096, 1024, device="cuda")}
    ramp_bufs["y"] = torch.empty_like(ramp_bufs["x"])

    def ramp():
        for _ in range(20):
            torch.softmax(ramp_bufs["x"], -1, out=ramp_bufs["y"])

    def body(run, l2_bytes):
        ramp_bufs.clear()  # release the ramp tensors before measuring
        for rows, cols in (QUICK_SHAPES if run.args.quick else SHAPES):
            bench_shape(run, lib, rows, cols, l2_bytes)

    bp.main(__doc__, "bench_softmax_compare", lib, ramp, body)


if __name__ == "__main__":
    main()
