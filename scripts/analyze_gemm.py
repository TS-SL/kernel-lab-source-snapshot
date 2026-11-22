"""Summarize GEMM benchmark runs (docs/gemm.md, docs/methodology.md section 2).

Reads results/raw/<name>-bench-gemm-cpp-p*.csv (+ .meta.json) and writes
results/processed/<name>-gemm-summary.csv and .md. Every number is computed
here from raw samples, with the statistics of scripts/analyze_bench.py
(hierarchical bootstrap over processes then samples). The shape is
(M, N, K) = (rows, cols, depth) in the CSV.

Gating (per result group = implementation x shape): >= 3 processes, a
sidecar from the same run as each CSV (same commit and start stamp), both
correctness checks in every process, and no thermal / HW-slowdown
clock-event sample; MEASURED overall only with a clean commit and every
group passing. Host-bound risk (host enqueue >= 80 % of GPU time), power-cap
samples, the fixed per-batch cost and the SM clock range are reported per
group.

Usage: python scripts/analyze_gemm.py [--name m4] [--boot 2000]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_bench import HOST_BOUND, KERNEL, NOISY_CV, Group, ci, hboot_medians  # noqa: E402

IMPLS = ["naive", "tiled", "regblock", "cublas"]
PEAK_GFLOPS = 12700.0      # ESTIMATED FP32 peak (docs/gemm.md): 28 SMs x 128 x 2 x 1.777 GHz
PEAK_CLOCK_MHZ = 1777.0
SMALL_WORK = 512 ** 3      # launches of a few to tens of us: per-launch gaps (UNVERIFIED size) inflate times

# Pre-registered predictions (docs/gemm.md): expected shapes per prediction,
# so a missing or invalid shape is INCOMPLETE instead of silently dropped.
CUBES = [(s, s, s) for s in (256, 512, 1024, 2048, 4096)]
SHAPES_ALL = CUBES + [(1000, 1000, 1000), (1023, 1023, 1023), (8192, 768, 768), (8192, 3072, 768),
                      (8192, 768, 3072), (64, 4096, 4096), (4096, 4096, 64)]
EXPECTED = {
    "P1": [(1024, 1024, 1024), (2048, 2048, 2048)],
    # 4096^3 dropped: TILED is capped at 4e10 FLOP (safety change after the dry run, docs/gemm.md).
    "P2": [(1024, 1024, 1024), (2048, 2048, 2048)],
    "P3": [s for s in SHAPES_ALL if s[0] * s[1] * s[2] >= 2 ** 30],
    "P4": [(2048, 2048, 2048), (4096, 4096, 4096)],
}


def load(name):
    files = sorted(f for f in glob.glob(str(ROOT / "results" / "raw" / f"{name}-bench-gemm-cpp-p*.csv"))
                   if not f.endswith("-gpu.csv"))
    if not files:
        raise SystemExit(f"no GEMM benchmark CSVs for '{name}'")
    rows, metas, missing = [], [], {}
    for f in files:
        with open(f, newline="") as fh:
            part = list(csv.DictReader(fh))
        rows.extend(part)
        procs = {r["process"] for r in part}
        meta = Path(f + ".meta.json")
        why = None
        if not meta.exists():
            why = "no sidecar"
        else:
            try:
                m = json.loads(meta.read_text())
            except json.JSONDecodeError:
                m, why = None, "unreadable (truncated) sidecar"
            if m is not None:
                if {r["commit"] for r in part} != {m.get("commit")}:
                    why = "sidecar from a different commit"
                elif {r["timestamp_utc"] for r in part} != {m.get("stamp")}:
                    why = "sidecar from a different run (start stamp differs)"
                else:
                    metas.append(m)
        if why:
            missing.update({p: why for p in procs})
    return files, rows, metas, missing


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="latest")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(12345)
    files, raw, metas, missing = load(args.name)

    by_key = defaultdict(list)
    for r in raw:
        if r["timing"] == KERNEL:
            by_key[(r["impl"], int(r["rows"]), int(r["cols"]), int(r["depth"]))].append(r)
    groups = {k: Group(v) for k, v in by_key.items()}
    shapes = sorted({k[1:] for k in groups}, key=lambda s: (s[0] * s[1] * s[2], s))

    checked, check_items, lin, math_modes = set(), defaultdict(list), defaultdict(list), set()
    for m in metas:
        for it in m.get("items", []):
            key = (it.get("impl"), it.get("rows"), it.get("cols"), it.get("depth", 0))
            if it["type"] == "check":
                checked.add((str(m["process"]), *key, it["when"]))
                check_items[key].append(it)
            elif it["type"] == "linearity":
                lin[key].append(it)
            elif it["type"] == "cublas_config":
                math_modes.add((it["math_mode"], it["tf32_override_env"]))
    commits = sorted({c for g in groups.values() for c in g.commits})
    nprocs = len({p for g in groups.values() for p in g.procs})
    problems = []
    if any(c.endswith("-dirty") or c == "unknown" for c in commits):
        problems.append(f"commit {', '.join(commits)} is not a clean commit")
    if len(commits) > 1:
        problems.append("runs come from different commits")
    if nprocs < 3:
        problems.append(f"{nprocs} process(es), fewer than 3")
    group_problem = {}
    for key, g in groups.items():
        why = []
        if len(g.procs) < 3:
            why.append(f"{len(g.procs)} process(es)")
        lost = [p for p in g.procs if p in missing]
        if lost:
            why.append("; ".join(f"process {p}: {missing[p]}" for p in lost))
        unchecked = [p for p in g.procs if p not in missing and not all(
            (p, *key, w) in checked for w in ("before_timing", "after_timing_fresh_pass"))]
        if unchecked:
            why.append(f"missing correctness checks in process(es) {', '.join(unchecked)}")
        if g.thermal:
            why.append(f"{g.thermal} thermal / HW-slowdown samples")
        if why:
            group_problem[key] = "; ".join(why)
    if group_problem:
        problems.append(f"{len(group_problem)} result group(s) not evidence (listed below)")
    status = "MEASURED" if not problems else "NOT EVIDENCE (" + "; ".join(problems) + ")"

    stats = {}
    for key, g in groups.items():
        med = hboot_medians(rng, [g], args.boot)[:, 0]
        q75, q25 = np.percentile(g.all, [75, 25])
        stats[key] = {"median": g.median, "ci": ci(med), "n": g.all.size, "procs": len(g.procs),
                      "cv": float(g.all.std(ddof=1) / g.all.mean()) if g.all.size > 1 else 0.0,
                      "iqr_rel": float((q75 - q25) / g.median)}

    def gflops(key, t_ms=None):
        _, mm, nn, kk = key
        t = stats[key]["median"] if t_ms is None else t_ms
        return 2.0 * mm * nn * kk / (t * 1e-3) / 1e9

    def ratio(slow, fast, shape):
        """time(slow) / time(fast): (value, CI, per-process range), or None."""
        ks, kf = (slow, *shape), (fast, *shape)
        if ks not in groups or kf not in groups:
            return None
        meds = hboot_medians(rng, [groups[ks], groups[kf]], args.boot)
        r = meds[:, 0] / meds[:, 1]
        ps, pf = groups[ks].proc_medians(), groups[kf].proc_medians()
        per = [ps[p] / pf[p] for p in ps if p in pf]
        if not per:
            return None
        return groups[ks].median / groups[kf].median, ci(r), (min(per), max(per))

    def fixed_cost(key):
        fr = [i["fixed_batch_cost_ms"] / (i["k"] * i["ms_per_launch_k"]) for i in lin.get(key, [])
              if i["ms_per_launch_k"] > 0]
        return max(fr) if fr else None

    out_dir = ROOT / "results" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}-gemm-summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["status", "impl", "m", "n", "k", "n_samples", "processes", "median_ms", "ci95_lo_ms", "ci95_hi_ms",
                    "gflops", "cv", "iqr_over_median", "host_enqueue_over_gpu", "fixed_batch_cost_frac",
                    "sm_clock_median", "power_cap_samples", "thermal_samples"])
        for key in sorted(stats):
            s, g = stats[key], groups[key]
            w.writerow([f"NOT EVIDENCE ({group_problem[key]})" if key in group_problem else status, *key, s["n"],
                        s["procs"], s["median"], *s["ci"], gflops(key), s["cv"], s["iqr_rel"], g.host_ratio,
                        fixed_cost(key) or "", int(np.median(g.sm)), g.power_cap, g.thermal])

    L = [f"# GEMM benchmark summary ({args.name})", "", f"**Status: {status}.**", "",
         "Generated by `scripts/analyze_gemm.py` from:", ""]
    L += [f"- `{Path(f).relative_to(ROOT).as_posix()}`" for f in files]
    L += ["", f"Commit: {', '.join(commits)}. Processes: {nprocs}. cuBLAS configuration recorded per process "
              f"(math mode, NVIDIA_TF32_OVERRIDE): {sorted(math_modes) if math_modes else 'not recorded'}."]
    if group_problem:
        L += ["", "Result groups that are NOT EVIDENCE:", ""]
        L += [f"- {k[0]} {k[1]}x{k[2]}x{k[3]}: {why}" for k, why in sorted(group_problem.items())]
    L += ["", "## Kernel time and throughput", "",
          "Median milliseconds per launch [95 % hierarchical-bootstrap CI] and GFLOP/s = 2 M N K / median time "
          "(MEASURED-derived). The FP32 peak of ~12.7 TFLOP/s is an ESTIMATE (specification, 1.777 GHz boost; "
          "reachable only without INT32 issue); \"at clock\" rescales it to the group's median recorded SM clock "
          "(ESTIMATED). Shapes marked * have launches of a few to tens of microseconds, where gaps between "
          "back-to-back launches (UNVERIFIED size) inflate times.", "",
          "| M x N x K | " + " | ".join(f"{i} ms" for i in IMPLS) + " | " + " | ".join(f"{i} GFLOP/s" for i in IMPLS)
          + " | regblock % of peak (at clock) | cublas % of peak (at clock) |", "|---|" + "---|" * (2 * len(IMPLS) + 2)]
    for shape in shapes:
        cells, rates = [], []
        for i in IMPLS:
            k = (i, *shape)
            if k in stats:
                s = stats[k]
                cells.append(f"{s['median']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
                rates.append(f"{gflops(k):.0f}")
            else:
                cells.append("n/a")
                rates.append("n/a")
        pct = []
        for i in ("regblock", "cublas"):
            k = (i, *shape)
            if k in stats:
                clk = float(np.median(groups[k].sm))
                at_clock = PEAK_GFLOPS * clk / PEAK_CLOCK_MHZ if clk > 0 else float("nan")
                pct.append(f"{100 * gflops(k) / PEAK_GFLOPS:.0f} % ({100 * gflops(k) / at_clock:.0f} % at {clk:.0f} MHz)")
            else:
                pct.append("n/a")
        mark = " *" if shape[0] * shape[1] * shape[2] <= SMALL_WORK else ""
        L.append(f"| {shape[0]} x {shape[1]} x {shape[2]}{mark} | " + " | ".join(cells) + " | " + " | ".join(rates)
                 + " | " + " | ".join(pct) + " |")

    L += ["", "## Control: NAIVE with L1-bypassing loads", "",
          "Ratio naive_l2only time / naive time (> 1: bypassing L1 slows the naive kernel). Confirmatory (its "
          "direction was seen in a quick check before it was registered, docs/gemm.md); rule: supported where the "
          "CI lower bound > 1.10, refuted where the upper bound < 1.10. Same arithmetic (bitwise identical, tested).",
          "", "| M x N x K | naive_l2only / naive | verdict |", "|---|---|---|"]
    for shape in shapes:
        r = ratio("naive_l2only", "naive", shape)
        if r:
            lo, hi = r[1]
            v = "supported" if lo > 1.10 else ("refuted" if hi < 1.10 else "inconclusive")
            L.append(f"| {shape[0]} x {shape[1]} x {shape[2]} | {r[0]:.2f} [{lo:.2f}, {hi:.2f}] | {v} |")

    L += ["", "## Speedups", "", "Ratio time(slower) / time(faster): median [95 % CI] (per-process range).", "",
          "| M x N x K | tiled vs naive | regblock vs tiled | cublas vs regblock |", "|---|---|---|---|"]
    ratios = {}
    for shape in shapes:
        cells = []
        for slow, fast in (("naive", "tiled"), ("tiled", "regblock"), ("regblock", "cublas")):
            r = ratio(slow, fast, shape)
            ratios[(slow, fast, shape)] = r
            cells.append("n/a" if r is None else f"{r[0]:.2f}x [{r[1][0]:.2f}, {r[1][1]:.2f}] ({r[2][0]:.2f}-{r[2][1]:.2f})")
        L.append(f"| {shape[0]} x {shape[1]} x {shape[2]} | " + " | ".join(cells) + " |")

    # Predictions (docs/gemm.md): every expected shape gets a verdict; a missing
    # or not-evidence group is INCOMPLETE and cannot PASS.
    verdicts = defaultdict(list)
    rows_out = []

    def incomplete(pid, name, shape, why):
        verdicts[pid].append("INCOMPLETE")
        rows_out.append(f"| {pid} {name} | {shape[0]} x {shape[1]} x {shape[2]} | - | - | INCOMPLETE ({why}) |")

    def judge_ratio(pid, name, shape, slow, fast, threshold, strict=False):
        r = ratios.get((slow, fast, shape)) or ratio(slow, fast, shape)
        if r is None:
            return incomplete(pid, name, shape, "group missing")
        if any(k in group_problem for k in ((slow, *shape), (fast, *shape))):
            return incomplete(pid, name, shape, "group not evidence")
        (lo, hi), (pmin, pmax) = r[1], r[2]
        if strict:  # "faster": strictly above the threshold
            v = "PASS" if lo > threshold and pmin > threshold else ("FAIL" if hi <= threshold else "INCONCLUSIVE")
        else:
            v = "PASS" if lo >= threshold and pmin >= threshold else ("FAIL" if hi < threshold else "INCONCLUSIVE")
        verdicts[pid].append(v)
        rows_out.append(f"| {pid} {name} | {shape[0]} x {shape[1]} x {shape[2]} | {r[0]:.2f} [{lo:.2f}, {hi:.2f}] | "
                        f"{pmin:.2f}-{pmax:.2f} | {v} |")

    for shape in EXPECTED["P1"]:
        judge_ratio("P1", "tiled >= 3x faster than naive", shape, "naive", "tiled", 3.0)
    for shape in EXPECTED["P2"]:
        judge_ratio("P2", "regblock >= 1.5x faster than tiled", shape, "tiled", "regblock", 1.5)
    for shape in EXPECTED["P3"]:
        judge_ratio("P3", "cublas faster than regblock", shape, "regblock", "cublas", 1.0, strict=True)
    need = 0.25 * PEAK_GFLOPS
    for shape in EXPECTED["P4"]:
        key = ("regblock", *shape)
        if key not in stats:
            incomplete("P4", "regblock >= 25 % of peak", shape, "group missing")
            continue
        if key in group_problem:
            incomplete("P4", "regblock >= 25 % of peak", shape, "group not evidence")
            continue
        lo_rate = gflops(key, stats[key]["ci"][1])  # slowest time in the CI -> lowest rate
        hi_rate = gflops(key, stats[key]["ci"][0])
        v = "PASS" if lo_rate >= need else ("FAIL" if hi_rate < need else "INCONCLUSIVE")
        verdicts["P4"].append(v)
        rows_out.append(f"| P4 regblock >= 25 % of peak ({need:.0f} GFLOP/s) | {shape[0]} x {shape[1]} x {shape[2]} | "
                        f"{gflops(key):.0f} GFLOP/s [{lo_rate:.0f}, {hi_rate:.0f}] | - | {v} |")
    names = {"P1": "tiled >= 3x naive at 1024^3, 2048^3", "P2": "regblock >= 1.5x tiled at 1024^3, 2048^3",
             "P3": "cuBLAS faster than regblock where M N K >= 2^30", "P4": "regblock >= 25 % of peak at 2048^3, 4096^3"}
    L += ["", "## Predictions (docs/gemm.md) and their verdicts", "",
          "Written before any GEMM timing. A prediction holds only if it is PASS on every expected shape; a "
          "missing or not-evidence group is INCOMPLETE.", "",
          "| prediction | shapes | PASS | INCONCLUSIVE | INCOMPLETE | FAIL | holds |", "|---|---|---|---|---|---|---|"]
    for pid in ("P1", "P2", "P3", "P4"):
        v = verdicts.get(pid, [])
        holds = "yes" if v and all(x == "PASS" for x in v) else "no"
        L.append(f"| {pid} {names[pid]} | {len(v)} | {v.count('PASS')} | {v.count('INCONCLUSIVE')} | "
                 f"{v.count('INCOMPLETE')} | {v.count('FAIL')} | {holds} |")
    L += ["", "| prediction | shape | value [95 % CI] | per-process range | verdict |", "|---|---|---|---|---|"] + rows_out

    L += ["", "## Conditions per group", "",
          "Host/GPU: host enqueue time / GPU time per batch (median; >= 0.80 = host-bound risk). Fixed batch cost: "
          "largest per-process 2K(t_K - t_2K)/(K t_K) (positive = fixed per-batch cost; at K = 1 a single noisy "
          "pair). Thermal samples make a group NOT EVIDENCE.", "",
          "| impl | M x N x K | K batch | host/GPU | fixed batch cost | SM MHz p10-p90 | power-cap samples | thermal | "
          "worst check ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(stats, key=lambda k: (k[1] * k[2] * k[3], k)):
        g = groups[key]
        fc = fixed_cost(key)
        worst = max((i["worst_ratio_to_bound"] for i in check_items.get(key, [])), default=float("nan"))
        risk = " (risk)" if g.host_ratio >= HOST_BOUND else ""
        L.append(f"| {key[0]} | {key[1]} x {key[2]} x {key[3]} | {'/'.join(map(str, g.k))} | {g.host_ratio:.2f}{risk} | "
                 f"{'n/a' if fc is None else format(fc, '.1%')} | {np.percentile(g.sm, 10):.0f}-{np.percentile(g.sm, 90):.0f} | "
                 f"{g.power_cap} | {g.thermal} | {worst:.3f} |")

    noisy = [f"{k[0]} {k[1]}x{k[2]}x{k[3]} (CV {s['cv']:.1%}, IQR/median {s['iqr_rel']:.1%})"
             for k, s in sorted(stats.items()) if s["cv"] > NOISY_CV]
    L += ["", f"Noisy (CV > {NOISY_CV:.0%}): {', '.join(noisy) if noisy else 'none'}.", "",
          "Correctness of benchmarked outputs (poisoned output recomputed before timing and after the series; 32 "
          "evenly spaced rows per check, first and last included, against the float64 reference and the derived "
          "bound - a check that the timed build is correct, not a replacement for the full-output tests):"]
    by_impl = defaultdict(list)
    for key, items in check_items.items():
        by_impl[key[0]].extend(items)
    for impl, items in sorted(by_impl.items()):
        L.append(f"- {impl}: {len(items)} checks, worst error {max(i['worst_ratio_to_bound'] for i in items):.3f} of "
                 f"the bound, {sum(i.get('violations', 0) for i in items)} violations")
    md_path = out_dir / f"{args.name}-gemm-summary.md"
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{status}\nwrote {csv_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)} ({len(raw)} samples)")


if __name__ == "__main__":
    main()
