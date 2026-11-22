# Evidence

`raw/` holds the archived logs and raw samples behind every number in the
documentation. `processed/` holds the summaries that the scripts in
`scripts/` generate from them. Nothing in `processed/` is edited by hand.

## Redacted copy

The files here are a **redacted copy** of the original evidence. Two kinds
of text were changed: the project's earlier name in recorded identifiers
(library, header and environment-variable names) was replaced by the
current name, KernelLens, and wall-clock dates and times were removed:

- **GPU guard logs (`*-gpu.csv`).** The nvidia-smi `timestamp` column is
  replaced by `elapsed_s`, the seconds since the log's first sample. Sample
  order and spacing are unchanged.
- **Benchmark CSVs and `.meta.json` sidecars.** Each run's start time
  (`timestamp_utc` / `stamp`) is replaced by a token `run-001` ...
  `run-027`, numbered in time order. A CSV and the sidecar from the same run
  still carry the same token, which is what the analysis checks.
- **Three free-text headers** (`environment.txt`, `m2-ncu-probe.log`,
  `m5-repro.log`) read `[date redacted]`.

No measured value depends on these names. Every other value is unchanged: timings, clocks, temperatures, sample order,
seeds, shapes, tool and driver versions, exit codes, skipped tests and
failures. The summaries in `processed/` regenerate from this copy with
identical numbers: `scripts/analyze_softmax_bench.py --name m2`,
`scripts/analyze_bench.py --op layernorm --name m3`,
`scripts/analyze_gemm.py --name m4`, `scripts/analyze_profile.py --name m5`.
The softmax summary's header predates one explanatory sentence that the
current script adds.

The scripts also replaced local paths when each log was saved: `<repo>` is
the repository folder, `<home>` the user profile, and `<clone>` a temporary
clone.

## Series names

The file prefixes `m1`-`m5` name the evidence series in the order they were
recorded:

| Prefix | Content | Code revision (development history) |
|---|---|---|
| `m1` | toolchain, SAXPY, error paths, first sanitizer attempt (BLOCKED) | - |
| `m2` | softmax tests and benchmark series. `m2-run1` and `m2-run2` are superseded series, kept as evidence of a benchmark clock-drop flaw and its fix | 64bbf58 (final series) |
| `m3` | layer norm tests, benchmark series, mutation checks | b309644 |
| `m4` | GEMM benchmark series, mutation checks | 79a9ab2 |
| `m5` | Nsight Compute profile, fresh-clone reproduction, final tests and sanitizer runs | 79a9ab2 (profile, repro); the final tests and sanitizer runs were on the last development revision |

The revision IDs refer to the private development history. Each benchmark
row and sidecar records the commit it was measured at.
