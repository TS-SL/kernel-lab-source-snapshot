r"""Mutation checks for an operator's kernels (layer norm, GEMM).

Each mutation is a deliberate, plausible bug in the operator's .cu file. The
script applies one at a time, rebuilds, runs the operator's C++ and Python
tests under scripts/gpu_guard.py, and records which tests failed (failure =
the mutation is caught). The original file is always restored byte for byte
and rebuilt, and the suites are re-run on it, also when a step fails.

Usage (in a shell prepared by scripts/dev-env.ps1, from the repo root):
    .venv\Scripts\python.exe scripts\mutation_check.py --op layernorm|gemm [--name latest]

Writes results/raw/<name>-mutation-<op>.txt (latest-* is gitignored).
GPU use: two short test runs per mutation, one at a time.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
GUARD = str(REPO / "scripts" / "gpu_guard.py")
BUILD = REPO / "build" / "release"

# (name, what it models, old text, new text, expected occurrences of old)
LAYERNORM_MUTATIONS = [
    ("rsqrtf", "scale via rsqrtf (2 ulp) instead of 1.0f / sqrtf",
     "return 1.0f / sqrtf(var + eps);", "return rsqrtf(var + eps);", 1),
    ("unbiased_variance", "divide the sum of squares by n - 1 (all kernels)",
     "/ static_cast<float>(cols), eps)", "/ static_cast<float>(cols - 1), eps)", 4),
    ("naive_variance_fused_block", "E[x^2] - E[x]^2 in FUSED_BLOCK",
     "        const float d = row[j] - m1;\n        q += d * d;\n        sd += d;\n    }\n"
     "    const float r = inv_std(block_reduce(q, SumOp{}, 0.0f) / static_cast<float>(cols), eps);",
     "        q += row[j] * row[j];\n        sd += row[j] - m1;\n    }\n"
     "    const float r = inv_std(block_reduce(q, SumOp{}, 0.0f) / static_cast<float>(cols) - m1 * m1, eps);", 1),
    ("dropped_last_element_fused_block_mean", "FUSED_BLOCK mean loop skips the last element",
     "j < cols; j += kReduceBlock) s += row[j] - pivot;\n    const float m1 = shifted_mean(pivot, block_reduce",
     "j + 1 < cols; j += kReduceBlock) s += row[j] - pivot;\n    const float m1 = shifted_mean(pivot, block_reduce", 1),
    ("no_pivot_fused_block", "FUSED_BLOCK forms the mean without the pivot shift (sum of x, pivot 0)",
     "s += row[j] - pivot;\n    const float m1 = shifted_mean(pivot, block_reduce",
     "s += row[j];\n    const float m1 = shifted_mean(0.0f, block_reduce", 1),
    ("no_mean_correction_fused_block", "FUSED_BLOCK outputs with the pivot-shifted mean m1, not the corrected mean",
     "const float m = corrected_mean(m1, block_reduce(sd, SumOp{}, 0.0f), cols);\n"
     "    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {\n        y[offset + j] = affine_out((row[j]",
     "const float m = m1 + 0.0f * block_reduce(sd, SumOp{}, 0.0f);\n"
     "    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {\n        y[offset + j] = affine_out((row[j]", 1),
    ("no_mean_correction_warp", "FUSED_WARP outputs with the pivot-shifted mean m1, not the corrected mean",
     "const float m = corrected_mean(m1, warp_reduce(sd, SumOp{}), cols);",
     "const float m = m1 + 0.0f * warp_reduce(sd, SumOp{});", 1),
    ("warp_pivot_per_lane", "FUSED_WARP lanes use different pivots (row[lane] instead of row[0])",
     "const float pivot = row[0];  // same address for all lanes: one broadcast load",
     "const float pivot = row[lane];", 1),
    ("warp_padding_nonzero_deviation", "FUSED_WARP padding lanes contribute (0 - m1)^2 and -m1",
     "const float d = j < cols ? v[i] - m1 : 0.0f;", "const float d = j < cols ? v[i] - m1 : -m1;", 1),
    ("eps_outside_sqrt", "1 / (sqrt(var) + eps) instead of 1 / sqrt(var + eps)",
     "return 1.0f / sqrtf(var + eps);", "return 1.0f / (sqrtf(var) + eps);", 1),
    ("smem_output_wrong_index", "FUSED_SMEM output pass reads row_s[threadIdx.x] instead of row_s[j]",
     "affine_out((row_s[j] - m) * r, gamma, beta, j);", "affine_out((row_s[threadIdx.x] - m) * r, gamma, beta, j);", 1),
]

GEMM_MUTATIONS = [
    ("tiled_missing_load_barrier", "TILED reads its shared tiles before every thread has stored them",
     "        __syncthreads();  // tiles complete before anyone reads them\n        const int pmax = min(kTile, k - t);",
     "        const int pmax = min(kTile, k - t);", 1),
    ("tiled_missing_reuse_barrier", "TILED overwrites its tiles while other threads may still read them",
     "        for (int p = 0; p < pmax; ++p) acc = fmaf(as[ty][p], bs[p][tx], acc);\n"
     "        __syncthreads();  // everyone is done with the tiles before the next load",
     "        for (int p = 0; p < pmax; ++p) acc = fmaf(as[ty][p], bs[p][tx], acc);", 1),
    ("tiled_padded_tail", "TILED runs its inner loop over the whole zero-padded last tile (extra fmaf(0, 0, acc))",
     "for (int p = 0; p < pmax; ++p) acc = fmaf(as[ty][p], bs[p][tx], acc);",
     "for (int p = 0; p < kTile; ++p) acc = fmaf(as[ty][p], bs[p][tx], acc);", 1),
    ("tiled_drops_last_element", "TILED's inner loop stops one element early in the last tile",
     "const int pmax = min(kTile, k - t);", "const int pmax = min(kTile, k - t - 1);", 1),
    ("naive_edge_guard_off_by_one", "NAIVE (and its L1-bypass control copy) writes one row/column past the matrix",
     "if (row >= m || col >= n) return;", "if (row > m || col > n) return;", 2),
    ("regblock_wrong_row_stride", "REGBLOCK stores outputs with a wrong row step (15 instead of 16)",
     "const int r = row0 + ty + 16 * i;", "const int r = row0 + ty + 15 * i;", 1),
    ("regblock_b_tile_column_offset", "REGBLOCK loads B's tile with a one-column shift",
     "const int gpb = t + pb, gc = col0 + cc;", "const int gpb = t + pb, gc = col0 + cc + 1;", 1),
    ("regblock_wrong_a_index", "REGBLOCK reads A values from the wrong tile row (ty + 15 i)",
     "for (int i = 0; i < 4; ++i) av[i] = as[p][ty + 16 * i];", "for (int i = 0; i < 4; ++i) av[i] = as[p][ty + 15 * i];", 1),
    ("regblock_k_step_skips", "REGBLOCK advances K by 2 tiles' worth after a full tile (skips data)",
     "for (int t = 0; t < k; t += kRbK) {", "for (int t = 0; t < k; t += kRbK + (k > 2 * kRbK ? kRbK : 0)) {", 1),
]

OPS = {
    "layernorm": {"title": "Layer-norm", "src": REPO / "src" / "cuda" / "layernorm.cu", "mutations": LAYERNORM_MUTATIONS,
                  "cpp": "test_layernorm.exe", "py": "tests/python/test_layernorm.py"},
    "gemm": {"title": "GEMM", "src": REPO / "src" / "cuda" / "gemm.cu", "mutations": GEMM_MUTATIONS,
             "cpp": "test_gemm.exe", "py": "tests/python/test_gemm.py"},
}



def build() -> tuple[bool, str]:
    r = subprocess.run(["cmake", "--build", str(BUILD)], capture_output=True, text=True, cwd=REPO)
    return r.returncode == 0, r.stdout + r.stderr


def guarded(args: list[str], gpu_log: Path) -> tuple[int, str]:
    cmd = [PY, GUARD, "--log", str(gpu_log), "--max-seconds", "600", "--"] + args
    env = dict(os.environ, KERNELLENS_LIB=str(BUILD / "kernellens.dll"))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
    return r.returncode, r.stdout + r.stderr


def run_suites(cfg: dict, tmp: Path, tag: str) -> dict:
    code_c, out_c = guarded([str(BUILD / cfg["cpp"])], tmp / f"{tag}-cpp-gpu.csv")
    code_p, out_p = guarded([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                             cfg["py"]], tmp / f"{tag}-py-gpu.csv")
    m = re.search(r"^(\d+) of (\d+) tests failed", out_c, re.M)
    cpp_failed = sorted(set(re.findall(r"^\[ FAIL \] (\S+)", out_c, re.M)))
    py_failed = re.findall(r"^FAILED \S+::(\w+)", out_p, re.M)
    py_summary = re.findall(r"^(?:=+ )?(\d+ (?:passed|failed).*?)(?: =+)?$", out_p, re.M)
    return {
        "code_c": code_c, "code_p": code_p,
        "cpp": f"{m.group(1)} of {m.group(2)} C++ tests failed" if m else "C++: no summary line",
        "cpp_failed": cpp_failed,
        "py": py_summary[-1] if py_summary else "pytest: no summary line",
        "py_failed": sorted(set(py_failed)),
        "stopped": "SAFETY STOP" in out_c or "SAFETY STOP" in out_p,
    }


def main(argv=None, op=None) -> int:
    ap = argparse.ArgumentParser()
    if op is None:
        ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--name", default="latest")
    args = ap.parse_args(argv)
    op = op or args.op
    cfg = OPS[op]
    SRC, MUTATIONS = cfg["src"], cfg["mutations"]
    if shutil.which("cmake") is None:
        sys.exit("cmake not on PATH: run in a shell prepared by scripts/dev-env.ps1")
    original = SRC.read_bytes()
    text = original.decode("utf-8")
    lines = [f"# {cfg['title']} mutation checks ({len(MUTATIONS)} mutations of src/cuda/{SRC.name})",
             "# CAUGHT = at least one C++ or Python test failed with the mutation applied."]
    overall_ok = True
    with tempfile.TemporaryDirectory(prefix="kl-mutation-") as tmp_name:
        tmp = Path(tmp_name)
        backup = tmp / (SRC.name + ".orig")
        backup.write_bytes(original)
        try:
            for name, what, old, new, count in MUTATIONS:
                n = text.count(old)
                if n != count:
                    lines.append(f"{name}: NOT APPLIED (pattern found {n} times, expected {count})")
                    overall_ok = False
                    continue
                SRC.write_bytes(text.replace(old, new).encode("utf-8"))
                ok, _ = build()
                if not ok:
                    lines.append(f"{name}: BUILD FAILED (mutation invalid)")
                    overall_ok = False
                    continue
                r = run_suites(cfg, tmp, name)
                if r["stopped"]:
                    lines.append(f"{name}: GPU GUARD SAFETY STOP - mutation run aborted")
                    overall_ok = False
                    break
                verdict = "CAUGHT" if (r["code_c"] != 0 or r["code_p"] != 0) else "SURVIVED"
                lines.append(f"{name}: {verdict} - {what}")
                lines.append(f"    {r['cpp']}: {', '.join(r['cpp_failed']) or '-'}")
                lines.append(f"    pytest: {r['py']}; failing tests: {', '.join(r['py_failed']) or '-'}")
                print(lines[-3], flush=True)
        finally:
            shutil.copyfile(backup, SRC)
            restored = SRC.read_bytes() == original
            ok, _ = build()
            r = run_suites(cfg, tmp, "restored")
            clean = restored and ok and r["code_c"] == 0 and r["code_p"] == 0
            overall_ok = overall_ok and clean
            lines.append(f"restored original: byte-identical={restored}, rebuild ok={ok}, "
                         f"{r['cpp']}, pytest: {r['py']}")
    out = REPO / "results" / "raw" / f"{args.name}-mutation-{op}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"written: results/raw/{out.name}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
