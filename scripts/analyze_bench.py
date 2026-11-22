"""Summarize benchmark runs of one operator (protocol: docs/methodology.md section 2).

Reads results/raw/<name>-bench-<op>-{cpp,py}-p*.csv (+ .meta.json) and
writes results/processed/<name>-<op>-summary.csv and .md. Every number in
the summary is computed here from raw samples; nothing is typed in by hand.

* Two harnesses: "cpp" (bench_<op>.exe) compares KernelLens
  implementations with each other without Python overhead; "python"
  (bench_<op>_compare.py) runs KernelLens and PyTorch interleaved in ONE
  process, so their ratio is free of the cross-process clock confound.
* Medians with hierarchical-bootstrap 95 % CIs: resample processes, then
  samples within each resampled process (samples are autocorrelated, and
  between-process variation is real). Each CI is computed once and reused.
* Speedups: ratio of medians, hierarchical-bootstrap CI (both sides
  resampled within the same resampled processes) plus the per-process range.
* Effective bandwidth: minimal traffic / median time - 8 bytes per element
  for softmax (one read of x, one write of y), 8 bytes per element + 8 per
  column for layer norm (gamma and beta read once). MEASURED-derived, not
  counters.
* Flags: CV > 5 % ("noisy"), IQR/median, host-bound risk when the host needs
  >= 80 % of the GPU time to enqueue a batch, fixed per-batch cost from the
  linearity check, thermal / power-cap / idle clock-event bits.
* Status: MEASURED only with a clean commit, >= 3 processes per harness and,
  per result group, >= 3 processes, a sidecar from the same run as each CSV
  (same commit and start stamp) and, for kernel timings, both correctness
  checks in every process; failing groups are marked NOT EVIDENCE, and so
  is the whole summary.
* Layer norm only: verdicts for the pre-registered predictions (decision
  rules in docs/layernorm.md), AUTO vs the fastest implementation, control
  kernels and API-reported occupancy.

Usage: python scripts/analyze_bench.py --op softmax|layernorm [--name m3] [--boot 2000]
(scripts/analyze_softmax_bench.py is an equivalent entry point for --op softmax.)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "multipass"
# Per operator: KernelLens implementations (baseline first), internal study
# kernels (C++ harness only, compared with `study_vs`), the PyTorch label,
# the two implementations of the rows-sweep table, and the host entry point.
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "tests" / "python"))
import kl  # noqa: E402  (AUTO dispatch rules, independent copies of the C++ ones)

OPS = {
    "softmax": {"title": "Softmax", "ours": ["multipass", "fused_online", "fused_warp"], "study": [],
                "torch": "torch_softmax", "torch_name": "torch.softmax",
                "sweep": ("multipass", "fused_online"), "host": "kl_softmax_host",
                "bytes": lambda rows, cols: 8 * rows * cols,
                "bytes_label": "8 bytes/element: one read of x + one write of y",
                "auto": lambda rows, cols: {kl.SOFTMAX_MULTIPASS: "multipass", kl.SOFTMAX_FUSED_ONLINE: "fused_online",
                                            kl.SOFTMAX_FUSED_WARP: "fused_warp"}[
                    kl.softmax_resolve(rows, cols, kl.SOFTMAX_AUTO)],
                "starved_rows": None, "predictions": None},
    # Study kernels (C++ harness only): (study kernel, compared with, what the pair isolates).
    "layernorm": {"title": "Layer-norm", "ours": ["multipass", "fused_block", "fused_smem", "fused_warp"],
                  "study": [("naive_study", "fused_block", "what the unstable E[x^2] - E[x]^2 formula saves"),
                            ("fused_block_maxsmem", "fused_block",
                             "the largest shared-memory carveout (small L1), same blocks per SM"),
                            ("fused_block_occ", "fused_block_maxsmem",
                             "fewer blocks per SM (FUSED_SMEM's), same carveout"),
                            ("fused_block_occ", "fused_smem",
                             "one read of x vs three, same carveout and blocks per SM"),
                            ("fused_smem_pad32k", "fused_smem", "3 blocks per SM (as at 8192 cols)")],
                  "torch": "torch_layer_norm", "torch_name": "torch.nn.functional.layer_norm",
                  "sweep": ("multipass", "fused_block"), "host": "kl_layernorm_host",
                  "bytes": lambda rows, cols: 8 * rows * cols + 8 * cols,
                  "bytes_label": "one read of x, gamma and beta + one write of y: 8 bytes/element + 8 bytes/column",
                  "auto": lambda rows, cols: {kl.LAYERNORM_FUSED_WARP: "fused_warp", kl.LAYERNORM_FUSED_SMEM: "fused_smem",
                                              kl.LAYERNORM_FUSED_BLOCK: "fused_block"}[
                      kl.layernorm_resolve(cols, kl.LAYERNORM_AUTO)],
                  # Fewer than 2 rows per SM (28 SMs): one-block-per-row kernels leave SMs idle.
                  "starved_rows": 56, "predictions": "layernorm"},
}
THERMAL_BITS = 0x8 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake
POWER_CAP_BIT, IDLE_BIT, DISPLAY_BIT = 0x4, 0x1, 0x100
NOISY_CV = 0.05
HOST_BOUND = 0.8
KERNEL = "cuda_events_kernel_only"


def load(name, op):
    files = sorted(f for f in glob.glob(str(ROOT / "results" / "raw" / f"{name}-bench-{op}-*-p*.csv"))
                   if not f.endswith("-gpu.csv"))
    if not files:
        raise SystemExit(f"no {op} benchmark CSVs for '{name}'")
    rows, metas, missing_meta = [], [], {}
    for f in files:
        with open(f, newline="") as fh:
            part = list(csv.DictReader(fh))
        rows.extend(part)
        procs = {(r["harness"], r["process"]) for r in part}
        meta = Path(f + ".meta.json")
        why = None
        if not meta.exists():
            why = "no sidecar"  # a process that died writes CSV rows but no sidecar
        else:
            try:
                m = json.loads(meta.read_text())
            except json.JSONDecodeError:
                m, why = None, "unreadable (truncated) sidecar"
            if m is not None:
                # The sidecar must come from the same run as the CSV (not left
                # over from an earlier series of the same name): same commit
                # and the same start stamp. The first softmax series' C++ sidecars have no
                # stamp; for those only the commit is compared.
                commits = {r["commit"] for r in part}
                stamps = {r["timestamp_utc"] for r in part}
                if commits != {m.get("commit")}:
                    why = "sidecar from a different commit"
                elif "stamp" in m and stamps != {m["stamp"]}:
                    why = "sidecar from a different run (start stamp differs)"
                else:
                    metas.append(m)
        if why:
            missing_meta.update({pr: why for pr in procs})
    return files, rows, metas, missing_meta


class Group:
    """Samples of one (harness, timing, impl, rows, cols), kept per process."""

    def __init__(self, rs):
        self.by_proc = defaultdict(list)
        for r in rs:
            self.by_proc[r["process"]].append(float(r["ms_per_launch"]))
        self.procs = sorted(self.by_proc)
        self.all = np.array([v for p in self.procs for v in self.by_proc[p]])
        self.k = sorted({int(r["batch_k"]) for r in rs})
        host = np.array([float(r["host_enqueue_ms"]) for r in rs])
        gpu_batch = np.array([float(r["ms_per_launch"]) * int(r["batch_k"]) for r in rs])
        self.host_ratio = float(np.median(host / gpu_batch))
        reasons = [int(r["clock_event_reasons"], 16) for r in rs]
        self.thermal = sum(1 for b in reasons if b & THERMAL_BITS)
        self.power_cap = sum(1 for b in reasons if b & POWER_CAP_BIT)
        self.idle = sum(1 for b in reasons if b & IDLE_BIT)
        self.display = sum(1 for b in reasons if b & DISPLAY_BIT)
        self.sm = [int(r["sm_clock_mhz"]) for r in rs]
        self.mem = [int(r["mem_clock_mhz"]) for r in rs]
        self.temp_max = max(int(r["temp_c"]) for r in rs)
        self.commits = {r["commit"] for r in rs}
        self.labels = {r["impl"] for r in rs}

    @property
    def median(self):
        return float(np.median(self.all))

    def proc_medians(self):
        return {p: float(np.median(self.by_proc[p])) for p in self.procs}


def hboot_medians(rng, groups, n_boot):
    """Hierarchical bootstrap: same resampled process list for all groups
    (they share processes), then samples within each process. Returns an
    (n_boot, len(groups)) array of medians."""
    procs = groups[0].procs
    out = np.empty((n_boot, len(groups)))
    for b in range(n_boot):
        pick = rng.choice(procs, size=len(procs), replace=True)
        for gi, g in enumerate(groups):
            parts = []
            for p in pick:
                v = np.asarray(g.by_proc.get(p) or g.all)
                parts.append(v[rng.integers(0, v.size, v.size)])
            out[b, gi] = np.median(np.concatenate(parts))
    return out


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def layernorm_predictions(cpp, speedup, stream, main_shapes, auto_rows):
    """PASS / FAIL / INCONCLUSIVE (or TIE) for the layer-norm predictions, with
    the decision rules of docs/layernorm.md ("Decision rules"). Ratios are
    time(other) / time(candidate) from the C++ harness. A prediction holds
    only if it is PASS on every shape where it is judged."""
    rows_out = []
    verdicts = defaultdict(list)  # prediction id -> verdicts on judged shapes

    def ratio(a, b, rows, cols):
        if (a, rows, cols) not in cpp or (b, rows, cols) not in cpp:
            return None
        return speedup(cpp[(a, rows, cols)], cpp[(b, rows, cols)], stream)

    def row(pid, name, rows, cols, r, verdict, judged=True):
        v, (lo, hi), (pmin, pmax) = r
        shown = verdict if judged else f"not judged ({verdict})"
        rows_out.append(f"| {pid} {name} | {rows} x {cols} | {v:.2f} [{lo:.2f}, {hi:.2f}] | {pmin:.2f}-{pmax:.2f} | "
                        f"{shown} |")
        if judged:
            verdicts[pid].append(verdict)

    def faster_verdict(lo, hi):  # ratio > 1 means the candidate is faster
        return "PASS" if lo > 1.0 else ("FAIL" if hi < 1.0 else "TIE")

    public = ["multipass", "fused_block", "fused_smem", "fused_warp"]
    for rows, cols in main_shapes:
        # x <= 2 MiB can stay in L2 between multipass's kernels: the traffic
        # model does not derive P2 or P3a there (docs/layernorm.md).
        in_l2 = rows * cols * 4 <= 2 * 1024 * 1024
        note = " (x fits in L2)" if in_l2 else ""
        # P1: fused_warp is the fastest public implementation where it applies.
        if ("fused_warp", rows, cols) in cpp:
            for other in public[:-1]:
                r = ratio(other, "fused_warp", rows, cols)
                if r:
                    row("P1", f"fused_warp faster than {other}", rows, cols, r, faster_verdict(*r[1]))
        # P2: multipass is the slowest.
        for other in public[1:]:
            r = ratio("multipass", other, rows, cols)
            if r:
                row("P2", f"{other} faster than multipass{note}", rows, cols, r, faster_verdict(*r[1]), not in_l2)
        # P3: fused_block re-reads hit L2 at cols <= 2048 (rows >= 168), miss from 8192 cols on.
        r = ratio("fused_block", "multipass", rows, cols)
        if r and cols <= 2048 and rows >= 168:
            lo, hi = r[1]
            row("P3a", f"fused_block / multipass time <= 0.80{note}", rows, cols, r,
                "PASS" if hi <= 0.80 else ("FAIL" if lo > 0.80 else "INCONCLUSIVE"), not in_l2)
        if r and cols >= 8192:
            lo, hi = r[1]
            row("P3b", "fused_block / multipass time >= 0.85", rows, cols, r,
                "PASS" if lo >= 0.85 else ("FAIL" if hi < 0.85 else "INCONCLUSIVE"))
        # P4: the two-read naive kernel is >= 1.15x faster than fused_block from 8192 cols on.
        r = ratio("fused_block", "naive_study", rows, cols)
        if r and cols >= 8192:
            lo, hi = r[1]
            row("P4", "naive_study >= 1.15x faster than fused_block", rows, cols, r,
                "PASS" if lo >= 1.15 else ("FAIL" if hi < 1.15 else "INCONCLUSIVE"))
        # P5: fused_smem >= 1.3x faster than fused_block at 4096 and 8192 cols.
        r = ratio("fused_block", "fused_smem", rows, cols)
        if r and cols in (4096, 8192):
            (lo, hi), (pmin, _) = r[1], r[2]
            row("P5", "fused_smem >= 1.30x faster than fused_block", rows, cols, r,
                "PASS" if lo >= 1.30 and pmin >= 1.30 else ("FAIL" if hi < 1.30 else "INCONCLUSIVE"))
        # P6: fused_smem about equal to fused_block at cols <= 2048.
        if r and cols <= 2048:
            lo, hi = r[1]
            row("P6", "fused_block / fused_smem within [0.90, 1.10]", rows, cols, r,
                "PASS" if lo >= 0.90 and hi <= 1.10 else ("FAIL" if hi < 0.90 or lo > 1.10 else "INCONCLUSIVE"))
    # P7: AUTO runs the fastest implementation or one tied with it (judged
    # shapes only: the rows sweep and few-row shapes are not judged).
    judged = set(main_shapes)
    for r_, c_, auto, best, _, vd in auto_rows:
        if (r_, c_) in judged:
            verdicts["P7"].append("PASS" if vd in ("fastest", "tie (CI includes 1)") else "FAIL")

    names = {"P1": "fused_warp fastest where it applies", "P2": "multipass slowest",
             "P3a": "fused_block re-reads served from cache at cols <= 2048",
             "P3b": "fused_block no cheaper than multipass per byte from 8192 cols",
             "P4": "naive_study >= 1.15x faster than fused_block from 8192 cols",
             "P5": "fused_smem >= 1.30x faster than fused_block at 4096 and 8192 cols",
             "P6": "fused_smem about equal to fused_block at cols <= 2048",
             "P7": "AUTO runs the fastest implementation (or a tie)"}
    L = ["", "## Predictions (docs/layernorm.md) and their verdicts", "",
         "Decision rules as written in docs/layernorm.md before this series. P1-P4 restate predictions about "
         "kernels the dry run had already measured (a replication, not a new test); P5-P7 concern FUSED_SMEM "
         "and AUTO. A prediction **holds** only if its verdict is PASS on every shape where it is judged; "
         "few-row shapes, the rows sweep (except 64 x 65536) and, for P2/P3a, shapes whose x fits in L2 are "
         "not judged.", "",
         "| prediction | judged shapes | PASS | TIE | INCONCLUSIVE | FAIL | holds |", "|---|---|---|---|---|---|---|"]
    for pid in ("P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7"):
        v = verdicts.get(pid, [])
        holds = "no shapes" if not v else ("yes" if all(x == "PASS" for x in v) else "no")
        L.append(f"| {pid} {names[pid]} | {len(v)} | {v.count('PASS')} | {v.count('TIE')} | "
                 f"{v.count('INCONCLUSIVE')} | {v.count('FAIL')} | {holds} |")
    L += ["", "Per shape (C++ harness):", "",
          "| prediction | shape | ratio [95 % CI] | per-process range | verdict |", "|---|---|---|---|---|"]
    L += rows_out
    return L


def main(argv=None, op=None, generator=None):
    ap = argparse.ArgumentParser()
    if op is None:
        ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--name", default="latest")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--note", default="", help="text printed under the status line, e.g. a superseded marker")
    args = ap.parse_args(argv)
    op = op or args.op
    cfg = OPS[op]
    STUDY, TORCH = cfg["study"], cfg["torch"]
    generator = generator or f"scripts/analyze_bench.py --op {op}"
    rng = np.random.default_rng(12345)
    # Separate stream for statistics added after the softmax series (AUTO
    # columns, predictions), so the softmax numbers are reproduced exactly.
    rng2 = np.random.default_rng(54321)
    files, raw, metas, missing_meta = load(args.name, op)

    by_key = defaultdict(list)
    for r in raw:
        impl = r["impl"].split("=")[0] if r["impl"].startswith("host_api_auto") else r["impl"]
        by_key[(r["harness"], r["timing"], impl, int(r["rows"]), int(r["cols"]))].append(r)
    groups = {k: Group(v) for k, v in by_key.items()}
    # Only implementations that were measured appear in the tables.
    measured = {k[2] for k in groups}
    OURS = [i for i in cfg["ours"] if i in measured]

    # ---- evidence status --------------------------------------------------
    commits = sorted({c for g in groups.values() for c in g.commits})
    nprocs = {h: len({p for (hh, *_), g in groups.items() if hh == h for p in g.procs}) for h in ("cpp", "python")}
    problems = []
    if any(c.endswith("-dirty") or c == "unknown" for c in commits):
        problems.append(f"commit {', '.join(commits)} is not a clean commit")
    if len(commits) > 1:
        problems.append("runs come from different commits")
    for h, n in nprocs.items():
        if n < 3:
            problems.append(f"{h} harness has {n} process(es), fewer than 3")
    # Per-group evidence: every process of a group must have its sidecar and,
    # for kernel timings, both correctness checks of that implementation and
    # shape (before timing and after the series); and >= 3 processes.
    checked = set()
    for m in metas:
        for it in m.get("items", []):
            if it["type"] == "check":
                checked.add((m["harness"], str(m["process"]), it["impl"], it["rows"], it["cols"], it["when"]))
    group_problem = {}
    for key, g in groups.items():
        h, timing, impl, rows, cols = key
        why = []
        if len(g.procs) < 3:
            why.append(f"{len(g.procs)} process(es)")
        lost = [p for p in g.procs if (h, p) in missing_meta]
        if lost:
            why.append("; ".join(f"process {p}: {missing_meta[(h, p)]}" for p in lost))
        if timing == KERNEL:
            unchecked = [p for p in g.procs if (h, p) not in missing_meta and not all(
                (h, p, impl, rows, cols, w) in checked for w in ("before_timing", "after_timing_fresh_pass"))]
            if unchecked:
                why.append(f"missing correctness checks in process(es) {', '.join(unchecked)}")
        if why:
            group_problem[key] = "; ".join(why)
    if group_problem:
        problems.append(f"{len(group_problem)} result group(s) incomplete (listed below)")
    status = "MEASURED" if not problems else "NOT EVIDENCE (" + "; ".join(problems) + ")"

    def row_status(key):
        if key in group_problem:
            return f"NOT EVIDENCE ({group_problem[key]})"
        return status

    # ---- statistics (each CI computed once) --------------------------------
    stats = {}
    for key, g in groups.items():
        med = hboot_medians(rng, [g], args.boot)[:, 0]
        q75, q25 = np.percentile(g.all, [75, 25])
        pm = g.proc_medians()
        stats[key] = {
            "n": g.all.size, "processes": len(g.procs), "median": g.median, "ci": ci(med),
            "p10": float(np.percentile(g.all, 10)), "p90": float(np.percentile(g.all, 90)),
            "min": float(g.all.min()), "max": float(g.all.max()),
            "cv": float(g.all.std(ddof=1) / g.all.mean()) if g.all.size > 1 else 0.0,
            "iqr_rel": float((q75 - q25) / g.median),
            "proc_spread": (max(pm.values()) / min(pm.values()) - 1) if len(pm) > 1 else None,
        }

    def speedup(num_key, den_key, stream=None):
        """median(num)/median(den) with hierarchical CI and per-process range."""
        gn, gd = groups[num_key], groups[den_key]
        meds = hboot_medians(stream or rng, [gn, gd], args.boot)
        r = meds[:, 0] / meds[:, 1]
        pn, pd = gn.proc_medians(), gd.proc_medians()
        per = [pn[p] / pd[p] for p in pn if p in pd]
        return gn.median / gd.median, ci(r), (min(per), max(per))

    cpp = {(k[2], k[3], k[4]): k for k in groups if k[0] == "cpp" and k[1] == KERNEL}
    py = {(k[2], k[3], k[4]): k for k in groups if k[0] == "python" and k[1] == KERNEL}
    shapes = sorted({(r, c) for (_, r, c) in cpp} | {(r, c) for (_, r, c) in py}, key=lambda s: (s[1], s[0]))
    sweep_cols = 65536
    main_shapes = [s for s in shapes if not (s[1] == sweep_cols and s[0] != 64)]
    starved = cfg["starved_rows"]
    starved_shapes = [s for s in main_shapes if starved and s[0] < starved]
    main_shapes = [s for s in main_shapes if s not in starved_shapes]
    sweep_rows = sorted(r for (r, c) in shapes if c == sweep_cols)

    # linearity and checks from the sidecars
    lin = defaultdict(list)
    checks = defaultdict(list)
    ramps = []
    torch_kernels = defaultdict(list)
    occupancy = defaultdict(set)  # (impl, rows, cols) -> blocks per SM values seen
    dispatch = []
    for m in metas:
        for it in m.get("items", []):
            key = (m["harness"], it.get("impl"), it.get("rows"), it.get("cols"))
            if it["type"] == "linearity":
                lin[key].append(it)
            elif it["type"] == "check":
                checks[(m["harness"], it["impl"])].append(it)
            elif it["type"] == "after_clock_ramp":
                ramps.append((m["harness"], m["process"], it))
            elif it["type"] == "torch_kernels":
                torch_kernels[tuple(it["kernels"])].append((it["rows"], it["cols"]))
            elif it["type"] == "occupancy":
                occupancy[(it["impl"], it["rows"], it["cols"])].add(it["blocks_per_sm"])
            elif it["type"] == "host_dispatch_us":
                dispatch.append((m["process"], it))

    def fixed_cost_frac(harness, impl, rows, cols):
        """Largest per-process estimate of fixed per-batch cost / batch time.

        Signed: a fixed cost makes the K batch slower per launch than the 2K
        batch (positive). One K/2K pair per process, so a single slow batch
        can give a large NEGATIVE value (the 2K batch slower), which is noise,
        not a fixed cost; only the positive side is meaningful."""
        items = lin.get((harness, impl, rows, cols), [])
        fr = [i["fixed_batch_cost_ms"] / (i["k"] * i["ms_per_launch_k"]) for i in items if i["ms_per_launch_k"] > 0]
        return max(fr) if fr else None

    def gbs(key):
        _, _, _, rows, cols = key
        return cfg["bytes"](rows, cols) / (stats[key]["median"] * 1e-3) / 1e9

    # ---- CSV --------------------------------------------------------------
    out_dir = ROOT / "results" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}-{op}-summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["status", "harness", "timing", "impl", "rows", "cols", "n", "processes", "median_ms",
                    "ci95_lo_ms", "ci95_hi_ms", "p10_ms", "p90_ms", "min_ms", "max_ms", "cv", "iqr_over_median",
                    "process_median_spread", "eff_bandwidth_gbs", "host_enqueue_over_gpu", "fixed_batch_cost_frac",
                    "sm_clock_median", "mem_clock_median", "temp_max_c", "thermal_samples", "power_cap_samples",
                    "idle_flag_samples"])
        for key in sorted(stats):
            h, timing, impl, rows, cols = key
            s, g = stats[key], groups[key]
            bw = gbs(key) if timing == KERNEL else ""
            w.writerow([row_status(key), h, timing, impl, rows, cols, s["n"], s["processes"], s["median"], *s["ci"],
                        s["p10"],
                        s["p90"], s["min"], s["max"], s["cv"], s["iqr_rel"],
                        "" if s["proc_spread"] is None else s["proc_spread"], bw, g.host_ratio,
                        fixed_cost_frac(h, impl, rows, cols) or "", int(np.median(g.sm)), int(np.median(g.mem)),
                        g.temp_max, g.thermal, g.power_cap, g.idle])

    # ---- Markdown ---------------------------------------------------------
    us = lambda s: f"{s['median'] * 1e3:.1f} [{s['ci'][0] * 1e3:.1f}, {s['ci'][1] * 1e3:.1f}]"  # noqa: E731
    L = [f"# {cfg['title']} benchmark summary ({args.name})", "", f"**Status: {status}.**", ""]
    if args.note:
        L += [f"**{args.note}**", ""]
    L += [
         f"Generated by `{generator}` from:", ""]
    L += [f"- `{Path(f).relative_to(ROOT).as_posix()}`" for f in files]
    builds = sorted({r["build"] for r in raw})
    L += ["", f"Commit: {', '.join(commits)}. Builds: {'; '.join(builds)}.",
          f"Processes: cpp {nprocs['cpp']}, python {nprocs['python']}. Clocks after the 300 ms ramp: " +
          ", ".join(f"{h} p{p}: SM {it['sm_mhz']} / mem {it['mem_mhz']} MHz" for h, p, it in sorted(ramps,
                                                                                               key=lambda t: (t[0], t[1]))) + "."]
    if group_problem:
        L += ["", "Result groups that are NOT EVIDENCE (each needs >= 3 processes, a sidecar for every process and, "
              "for kernel timings, both correctness checks in every process):", ""]
        L += [f"- {h} {impl} {rows}x{cols} ({timing}): {why}"
              for (h, timing, impl, rows, cols), why in sorted(group_problem.items())]
    L += ["", "## KernelLens implementations (C++ harness)", "",
          "Kernel time per launch, microseconds: median [95 % hierarchical-bootstrap CI]. CUDA events around "
          "batches of K launches (K fixed per shape), inputs rotated so each launch is cold in L2 relative to "
          "the previous ones; FP32, row-major.", "",
          "| rows x cols | " + " | ".join(OURS) + " | " + " | ".join(f"speedup {i}" for i in OURS[1:]) + " |",
          "|---|" + "---|" * (2 * len(OURS) - 1)]
    for rows, cols in main_shapes:
        cells = [us(stats[cpp[(i, rows, cols)]]) if (i, rows, cols) in cpp else "n/a" for i in OURS]
        sp = []
        for impl in OURS[1:]:
            if (impl, rows, cols) in cpp and (BASELINE, rows, cols) in cpp:
                v, (lo, hi), (pmin, pmax) = speedup(cpp[(BASELINE, rows, cols)], cpp[(impl, rows, cols)])
                sp.append(f"{v:.2f}x [{lo:.2f}, {hi:.2f}]")
            else:
                sp.append("n/a")
        L.append(f"| {rows} x {cols} | " + " | ".join(cells) + " | " + " | ".join(sp) + " |")

    study_names = list(dict.fromkeys(s for s, _, _ in STUDY))
    bw_impls = OURS + [i for i in study_names if i in measured]
    bw_text = ("8 bytes/element / median" if op == "softmax" else f"minimal traffic ({cfg['bytes_label']}) / median")
    L += ["", f"Effective bandwidth ({bw_text}; MEASURED-derived; the RTX 3060's ~360 GB/s DRAM "
          "peak is an ESTIMATE from its specification):", "",
          "| rows x cols | " + " | ".join(f"{i} GB/s" for i in bw_impls) + " |", "|---|" + "---|" * len(bw_impls)]
    for rows, cols in main_shapes:
        cells = [f"{gbs(cpp[(i, rows, cols)]):.0f}" if (i, rows, cols) in cpp else "n/a" for i in bw_impls]
        L.append(f"| {rows} x {cols} | " + " | ".join(cells) + " |")

    L += ["", "Speedup over multipass across the main shapes (geometric mean over the shapes where the "
          "implementation runs - the shape sets differ), and the shapes where it is not shown faster (95 % CI "
          "lower bound <= 1.00) and where it is slower (CI upper bound < 1.00):", ""]
    for impl in OURS[1:]:
        vals, not_faster, slower = [], [], []
        for rows, cols in main_shapes:
            if (impl, rows, cols) in cpp and (BASELINE, rows, cols) in cpp:
                v, (lo, hi), _ = speedup(cpp[(BASELINE, rows, cols)], cpp[(impl, rows, cols)])
                vals.append(v)
                if lo <= 1.0:
                    not_faster.append(f"{rows}x{cols} ({v:.2f}x)")
                if hi < 1.0:
                    slower.append(f"{rows}x{cols} ({v:.2f}x)")
        gm = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
        L.append(f"- {impl}: geometric mean {gm:.2f}x over {len(vals)} shapes; not shown faster on: "
                 f"{', '.join(not_faster) if not_faster else 'none'}; slower on: "
                 f"{', '.join(slower) if slower else 'none'}")

    if starved_shapes:
        L += ["", f"### Few-row shapes (rows < {starved}: fewer than 2 rows per SM, so one-block-per-row kernels "
              "leave SMs idle; kept out of the geometric means)", "",
              "| rows x cols | " + " | ".join(f"{i} us (GB/s)" for i in bw_impls) + " |",
              "|---|" + "---|" * len(bw_impls)]
        for rows, cols in starved_shapes:
            cells = [f"{stats[cpp[(i, rows, cols)]]['median'] * 1e3:.1f} ({gbs(cpp[(i, rows, cols)]):.0f})"
                     if (i, rows, cols) in cpp else "n/a" for i in bw_impls]
            L.append(f"| {rows} x {cols} | " + " | ".join(cells) + " |")

    for study, vs, what in STUDY:
        if study not in measured or vs not in measured:
            continue
        L += ["", f"## Study kernel `{study}` vs `{vs}` (C++ harness)", "",
              f"Isolates: {what}. Ratio {vs} time / {study} time (> 1: `{study}` is faster): "
              "median [95 % hierarchical CI].",
              "", "| rows x cols | ratio |", "|---|---|"]
        for rows, cols in shapes:
            if (study, rows, cols) in cpp and (vs, rows, cols) in cpp:
                v, (lo, hi), _ = speedup(cpp[(vs, rows, cols)], cpp[(study, rows, cols)])
                L.append(f"| {rows} x {cols} | {v:.2f} [{lo:.2f}, {hi:.2f}] |")

    if occupancy:
        occ_impls = sorted({i for (i, _, _) in occupancy}, key=lambda i: (i not in OURS, i))
        L += ["", "Resident blocks per SM as reported by cudaOccupancyMaxActiveBlocksPerMultiprocessor (recorded by "
              "the C++ harness after the shared-memory carveout preference is set). API-reported, not verified: "
              "the carveout is a hint, and which L1/shared split the driver applied at launch is not observable "
              "here (profiler counters are BLOCKED).", "",
              "| rows x cols | " + " | ".join(occ_impls) + " |", "|---|" + "---|" * len(occ_impls)]
        for rows, cols in shapes:
            if not any((i, rows, cols) in occupancy for i in occ_impls):
                continue
            cells = ["/".join(map(str, sorted(occupancy[(i, rows, cols)]))) if (i, rows, cols) in occupancy else "-"
                     for i in occ_impls]
            L.append(f"| {rows} x {cols} | " + " | ".join(cells) + " |")

    # AUTO vs the fastest public implementation (C++ harness), per shape.
    auto_rows = []
    for rows, cols in shapes:
        ours = [i for i in OURS if (i, rows, cols) in cpp]
        if len(ours) < 2:
            continue
        auto = cfg["auto"](rows, cols)
        best = min(ours, key=lambda i: stats[cpp[(i, rows, cols)]]["median"])
        if auto == best or (auto, rows, cols) not in cpp:
            verdict, ratio = ("fastest" if auto == best else "not measured"), ""
        else:
            v, (lo, hi), _ = speedup(cpp[(auto, rows, cols)], cpp[(best, rows, cols)], rng2)
            verdict = "tie (CI includes 1)" if lo <= 1.0 else f"{best} faster"
            ratio = f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
        auto_rows.append((rows, cols, auto, best, ratio, verdict))
    L += ["", "## AUTO dispatch vs the fastest implementation (C++ harness)", "",
          "AUTO's choice per shape (rule fixed before this series), the fastest measured public implementation, "
          "and the ratio AUTO time / fastest time (> 1: AUTO slower) when they differ. All shapes are listed; "
          "for layer norm, P7 judges only the main shapes (not the few-row shapes or the rows sweep).", "",
          "| rows x cols | AUTO runs | fastest | AUTO / fastest | verdict |", "|---|---|---|---|---|"]
    L += [f"| {r} x {c} | {a} | {b} | {ratio} | {vd} |" for r, c, a, b, ratio, vd in auto_rows]

    if cfg["predictions"] == "layernorm":
        L += layernorm_predictions(cpp, speedup, rng2, main_shapes, auto_rows)

    if len(sweep_rows) > 2:
        a, b = cfg["sweep"]
        L += ["", f"## Rows sweep at cols = {sweep_cols} (C++ harness; one block per row; 28 SMs)", "",
              "Median microseconds, and in parentheses the ratio to the same implementation's time at "
              f"{sweep_rows[0]} rows.", "", f"| rows | {a} | {b} | {b} / {a} |",
              "|---|---|---|---|"]
        for rows in sweep_rows:
            cells = []
            for impl in (a, b):
                k = cpp.get((impl, rows, sweep_cols))
                k0 = cpp.get((impl, sweep_rows[0], sweep_cols))
                cells.append(f"{stats[k]['median'] * 1e3:.1f} ({stats[k]['median'] / stats[k0]['median']:.2f}x)"
                             if k and k0 else "n/a")
            ratio = ""
            if (a, rows, sweep_cols) in cpp and (b, rows, sweep_cols) in cpp:
                v, (lo, hi), _ = speedup(cpp[(b, rows, sweep_cols)], cpp[(a, rows, sweep_cols)])
                ratio = f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
            L.append(f"| {rows} | " + " | ".join(cells) + f" | {ratio} |")

    L += ["", f"## KernelLens vs {cfg['torch_name']} (Python harness, same process, same stream, interleaved ABBA)",
          "",
          "Ratio torch time / KernelLens time (> 1: KernelLens faster), for the fastest KernelLens implementation "
          "of the shape in this harness (the minimum of several noisy medians: slightly optimistic) and for "
          "the implementation AUTO runs (what a caller of the library gets): median [95 % hierarchical CI] and "
          "the range over the per-process ratios. Host-bound risk: the host needed >= 80 % of the GPU time to "
          "enqueue a batch, so launch overhead (Python dispatch / ctypes) may be inflating both sides.", "",
          "| rows x cols | fastest KernelLens | torch us | KernelLens us | torch / KernelLens | per-process range | "
          "AUTO runs | torch / AUTO | SM / mem MHz (median) | host-bound risk |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for rows, cols in shapes:
        if (TORCH, rows, cols) not in py:
            continue
        ours = [i for i in OURS if (i, rows, cols) in py]
        best = min(ours, key=lambda i: stats[py[(i, rows, cols)]]["median"])
        v, (lo, hi), (pmin, pmax) = speedup(py[(TORCH, rows, cols)], py[(best, rows, cols)])
        auto = cfg["auto"](rows, cols)
        if auto == best:
            auto_cell = "same"
        elif (auto, rows, cols) in py:
            va, (la, ha), _ = speedup(py[(TORCH, rows, cols)], py[(auto, rows, cols)], rng2)
            auto_cell = f"{va:.2f} [{la:.2f}, {ha:.2f}]"
        else:
            auto_cell = "not measured"
        gt, gb = groups[py[(TORCH, rows, cols)]], groups[py[(best, rows, cols)]]
        risk = [n for n, g in (("torch", gt), (best, gb)) if g.host_ratio >= HOST_BOUND]
        L.append(f"| {rows} x {cols} | {best} | {stats[py[(TORCH, rows, cols)]]['median'] * 1e3:.1f} | "
                 f"{stats[py[(best, rows, cols)]]['median'] * 1e3:.1f} | {v:.2f} [{lo:.2f}, {hi:.2f}] | "
                 f"{pmin:.2f}-{pmax:.2f} | {auto} | {auto_cell} | {int(np.median(gt.sm))} / {int(np.median(gt.mem))} | "
                 f"{', '.join(risk) if risk else 'no'} |")
    if dispatch:
        L += ["", "Host cost of one rows = 0 call (validation only, no launch), per process: " + "; ".join(
            f"p{p}: KernelLens ctypes {it['kernellens_device_rows0']:.1f} us, torch {it['torch_layer_norm_empty']:.1f} us"
            for p, it in sorted(dispatch, key=lambda t: t[0])) + "."]
    if torch_kernels:
        L += ["", "Kernels PyTorch launched per call (torch profiler, one untimed call per shape and process):", ""]
        for names, where in sorted(torch_kernels.items()):
            shapes_txt = ", ".join(f"{r}x{c}" for r, c in sorted(set(where), key=lambda s: (s[1], s[0])))
            L.append(f"- {' + '.join(f'`{n}`' for n in names)}: {shapes_txt}")
    L += ["", "Agreement of the two harnesses for KernelLens kernels (python median / C++ median; > 1 means the "
          "Python-launched run was slower, e.g. from launch overhead or different clocks):", ""]
    agree = []
    for (impl, rows, cols), k in sorted(py.items(), key=lambda kv: (kv[0][2], kv[0][1])):
        if impl in OURS and (impl, rows, cols) in cpp:
            agree.append(f"{impl} {rows}x{cols}: {stats[k]['median'] / stats[cpp[(impl, rows, cols)]]['median']:.2f}")
    L.append("; ".join(agree) if agree else "n/a")

    e2e = {k: v for k, v in stats.items() if k[1] != KERNEL}
    if e2e:
        L += ["", f"## End-to-end `{cfg['host']}` (C++ harness)", "",
              "Host steady clock; pageable host memory, per-call device allocation, H2D + kernel + D2H. "
              "Compared with the kernel-only time of the implementation AUTO dispatched to.", "",
              "| rows x cols | AUTO ran | end-to-end ms (median [CI]) | CV | kernel-only ms | ratio |",
              "|---|---|---|---|---|---|"]
        for key, s in sorted(e2e.items(), key=lambda kv: (kv[0][4], kv[0][3])):
            h, timing, _, rows, cols = key
            ran = sorted(groups[key].labels)[0].split("=")[-1]
            kern = stats[cpp[(ran, rows, cols)]]["median"] if (ran, rows, cols) in cpp else float("nan")
            L.append(f"| {rows} x {cols} | {ran} | {s['median']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] | "
                     f"{s['cv']:.1%} | {kern:.4f} | {s['median'] / kern:.1f}x |")

    L += ["", "## Variability, conditions and harness checks", "",
          "\"fixed batch cost\": largest of the per-process estimates 2K(t_K - t_2K) / (K t_K) from one batch of "
          "K and one of 2K launches; positive values estimate a fixed per-batch cost included in the timing, "
          "negative values mean the single 2K batch happened to be slower (noise).", "",
          "| harness | impl | rows x cols | n | proc | CV | IQR/median | process spread | K | host/GPU | "
          "fixed batch cost | SM MHz | mem MHz | max C | thermal | power cap | idle flag |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    noisy = []
    for key in sorted(stats, key=lambda k: (k[0], k[4], k[3], k[2])):
        h, timing, impl, rows, cols = key
        if timing != KERNEL:
            continue
        s, g = stats[key], groups[key]
        if s["cv"] > NOISY_CV:
            noisy.append(f"{h}:{impl} {rows}x{cols} (CV {s['cv']:.1%}, IQR/median {s['iqr_rel']:.1%})")
        fc = fixed_cost_frac(h, impl, rows, cols)
        L.append(f"| {h} | {impl} | {rows} x {cols} | {s['n']} | {s['processes']} | {s['cv']:.1%} | "
                 f"{s['iqr_rel']:.1%} | {'n/a' if s['proc_spread'] is None else format(s['proc_spread'], '.1%')} | "
                 f"{'/'.join(map(str, g.k))} | {g.host_ratio:.2f} | {'n/a' if fc is None else format(fc, '.1%')} | "
                 f"{min(g.sm)}-{max(g.sm)} | {min(g.mem)}-{max(g.mem)} | {g.temp_max} | {g.thermal} | "
                 f"{g.power_cap} | {g.idle} |")
    # Do low NVML clock readings coincide with slower samples? (NVML values
    # are driver-sampled and read right after each batch, so a low reading
    # may be a post-batch idle snapshot rather than the clock during it.)
    low_rows = []
    for key in sorted(by_key, key=lambda k: (k[0], k[4], k[3], k[2])):
        h, timing, impl, rows, cols = key
        if timing != KERNEL:
            continue
        lo = [float(r["ms_per_launch"]) for r in by_key[key] if int(r["sm_clock_mhz"]) < 1500]
        hi = [float(r["ms_per_launch"]) for r in by_key[key] if int(r["sm_clock_mhz"]) >= 1500]
        if len(lo) >= 5 and len(hi) >= 5:
            low_rows.append(f"| {h} | {impl} | {rows} x {cols} | {len(lo)} | {float(np.median(lo)) * 1e3:.1f} | "
                            f"{float(np.median(hi)) * 1e3:.1f} | {np.median(lo) / np.median(hi):.3f} |")
    if low_rows:
        L += ["", "Samples whose NVML SM-clock reading was below 1500 MHz, compared with the other samples of "
              "the same group (groups with >= 5 of each):", "",
              "| harness | impl | rows x cols | low-reading samples | their median us | others' median us | ratio |",
              "|---|---|---|---|---|---|---|"] + low_rows

    L += ["", f"Noisy (CV > {NOISY_CV:.0%}): {', '.join(noisy) if noisy else 'none'}.", "",
          "Correctness of benchmarked outputs (before timing and again after the timed series, each time on a "
          "poisoned output buffer recomputed by one untimed pass of the implementation): "]
    for (h, impl), items in sorted(checks.items()):
        if impl == TORCH and "max_rel_err_vs_float64" in items[0]:
            worst = max(i["max_rel_err_vs_float64"] for i in items)
            L.append(f"- {h} {impl}: {len(items)} checks, max relative error vs float64 {worst:.2e} "
                     "(no bound is claimed for PyTorch's algorithm)")
        elif impl == TORCH:
            worst = max(i["max_abs_err_vs_float64"] for i in items)
            ratio = max(i["max_ratio_to_fused_block_bound"] for i in items)
            fresh = sum(1 for i in items if i.get("fresh_write_verified"))
            L.append(f"- {h} {impl}: {len(items)} checks, max absolute error vs float64 {worst:.2e}, "
                     f"{ratio:.3f} of KernelLens's fused_block bound (for information: no bound is claimed for "
                     f"PyTorch's algorithm); output verified as freshly written into a NaN-filled block in "
                     f"{fresh} of {len(items)} checks (informational, not gated: for outputs <= 1 MiB torch's "
                     "small allocator pool can place the output outside the NaN-filled blocks, a false negative)")
        else:
            worst = max(i["worst_ratio_to_bound"] for i in items)
            viol = sum(i.get("violations", 0) for i in items)
            L.append(f"- {h} {impl}: {len(items)} checks, worst error {worst:.3f} of the derived bound, "
                     f"{viol} violations")
    md_path = out_dir / f"{args.name}-{op}-summary.md"
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{status}\nwrote {csv_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)} "
          f"({len(raw)} samples from {len(files)} files)")


if __name__ == "__main__":
    main()
