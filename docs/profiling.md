# Profiling

Written before any profile was taken; results are added only from executed
runs. Nsight Compute needs GPU performance-counter access; without it `ncu`
fails with `ERR_NVGPUCTRPERM` (`results/raw/m2-ncu-probe.log`). The access
was enabled for all users in the NVIDIA Control Panel by the machine's
administrator; the project never changes that setting.

## Method

- `benchmarks/profile_kernels.cpp` launches each kernel of interest a few
  times at fixed shapes and inputs (`kl_fill_uniform` seeds), nothing else.
  `scripts/run-profile.ps1` runs it under `scripts/gpu_guard.py` with
  Nsight Compute (`ncu`), one profile pass per kernel instance, and saves the
  raw metric values as CSV (`results/raw/m5-ncu-*.csv`, user paths redacted).
- **`--clock-control none`**: ncu does not lock the GPU clocks (by default it
  would lock them to base clocks for the duration of a profile; the project
  never changes clocks). Counter values that depend on time (throughputs,
  % of peak) are therefore at whatever clock the GPU runs; counts (bytes,
  instructions, hit rates) do not depend on it.
- `--cache-control all` (ncu's default): caches are flushed before each
  replay pass, so every profiled launch starts cold - as in the benchmarks,
  whose buffers rotate over >= 4x L2.
- Profiler durations are **not** compared with the event timings (different
  conditions; docs/methodology.md); profiles explain, events measure.
- Bounded: kernels take < 50 ms each natively; ncu replays each one a few
  dozen times; the whole profile run has a guard time limit.

## Hypotheses and the counter signatures that would support or refute them

Each is a hypothesis stated earlier without counters (label HYPOTHESIS in
docs/softmax.md, docs/layernorm.md, docs/gemm.md). "Supports" and "refutes"
are fixed here, before profiling.

**H1 (softmax):** with few long rows (4 x 65536) the fused
online kernel is slower than the multipass baseline because its first pass
is latency-bound (each step depends on the previous running max through an
`expf`).
- Supports: for `softmax_online_kernel` at 4 x 65536, the largest stall
  reason is a dependency stall (short/long scoreboard or "wait"), and its
  DRAM throughput (% of peak) is below the multipass kernels'.
- Refutes: the online kernel's dominant stall is memory throttling or its
  DRAM throughput is equal to or higher than multipass's.

**H2 (layer norm):** `fused_block`'s second and third reads of a
row are served from cache at <= 2048 cols (rows in flight fit), partly by L1
(the control `fused_block_maxsmem` with a small L1 was up to 1.33x slower),
and come from DRAM at >= 8192 cols.
- Supports: DRAM bytes read by `ln_fused_block_kernel` are close to one read
  of x (~4 B/element) at 2048 cols and close to three reads (~12 B/element)
  at 8192 cols, with the L1 hit rate clearly above zero at
  2048 / 4096 cols; `ln_fused_smem_kernel` reads ~4 B/element at every
  measured width.
- Refutes: DRAM reads near 12 B/element already at 2048 cols, or an L1 hit
  rate near zero where the control showed an L1 effect.

**H3 (GEMM):** `NAIVE` and `TILED` run at about the same rate
because both are limited by load-instruction issue (about 2 loads per FMA,
from L1 for NAIVE and from shared memory for TILED), not by DRAM; TILED's
1024-thread blocks also limit it to 32 warps per SM.
- Supports: for both kernels at 1024^3, DRAM throughput is far below peak
  (< 30 %), the L1 (NAIVE) or shared-memory (TILED) pipe utilization is high,
  the dominant stalls are load-related (MIO throttle / short scoreboard for
  TILED, long scoreboard / LG throttle for NAIVE), and TILED's achieved
  occupancy is <= 50 %.
- Refutes: DRAM throughput near peak for either, or dominant stalls
  unrelated to loads.

**H4 (GEMM):** `REGBLOCK` is limited by shared-memory loads and
FMA issue, not memory; cuBLAS reaches higher FMA-pipe utilization.
- Supports: REGBLOCK's FMA-pipe utilization is clearly higher than TILED's
  and lower than cuBLAS's kernel's, with low DRAM throughput.
- Refutes: REGBLOCK is DRAM-bound (DRAM throughput near peak).

## Numeric rules

The profile ran first (to confirm that counters are accessible and which
metric names this Nsight Compute version reports); these thresholds were
then written down before any metric value was read (*unarchived observation*:
git cannot show it - this section and `scripts/analyze_profile.py` were
first committed together with the values, in 9d3a11d; what was committed
before the profile, in 79a9ab2, is the qualitative signatures above,
including the "< 30 %" DRAM and "<= 50 %" occupancy thresholds of H3). No
verdict depends on a threshold close to its value except H3's 30 % line
(30.4 %), which was committed beforehand.

- "Close to one read" = DRAM bytes read <= 6 B per element (1.5x of 4);
  "close to three reads" = >= 9 B per element (0.75x of 12).
- "L1 hit rate clearly above zero" = `l1tex__t_sector_hit_rate.pct` >= 10 %.
- "Far below peak" DRAM = `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` < 30 %;
  "near peak" = >= 70 %.
- Dominant stall = the largest of the `smsp__pcsamp_warps_issue_stalled_*`
  sample counts (excluding "selected", the issuing state); dependency
  stalls = short/long scoreboard and wait; load-related = MIO throttle,
  short scoreboard (shared memory), long scoreboard and LG throttle
  (global/L1).
- Achieved occupancy = `sm__warps_active.avg.pct_of_peak_sustained_active`.
- FMA-pipe utilization = `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active`.

Verdicts: H1 supported if the online kernel's dominant stall is a
dependency stall AND its DRAM throughput is below the mean of the three
multipass kernels'; refuted if its dominant stall is memory throttling (LG
or MIO throttle) or its DRAM throughput is >= theirs. H2 supported if
fused_block reads <= 6 B/element at 2048 cols and >= 9 at 8192, its L1 hit
rate is >= 10 % at 2048 and 4096 cols, and fused_smem reads <= 6 B/element
at all three widths; refuted if fused_block reads >= 9 B/element at 2048
cols or its L1 hit rate is < 10 % at both 2048 and 4096. H3 supported if
both NAIVE and TILED have DRAM throughput < 30 %, load-related dominant
stalls, and TILED's achieved occupancy is <= 50 %; refuted if either has
DRAM throughput >= 70 % or a dominant stall that is not load-related. H4
supported if REGBLOCK's FMA-pipe utilization is above TILED's and below
cuBLAS's with DRAM throughput < 70 %; refuted if its DRAM throughput is
>= 70 %. Anything else: INCONCLUSIVE.

## Results (MEASURED by Nsight Compute, commit 79a9ab2)

Source: `results/processed/m5-profile-summary.md`, generated by
`scripts/analyze_profile.py` from `results/raw/m5-ncu-raw.csv` (18 profiled
launches, `--set detailed`, guard peak 63 C). Counter readings are the
profiler's (MEASURED); they come from one profiled launch each (cold caches,
unlocked clocks), not from repeated runs.

| Hypothesis | Key counters | Verdict (rule) |
|---|---|---|
| H1 softmax online kernel latency-bound at 4 x 65536 | dominant stall long scoreboard (76 % of samples); DRAM 3.8 % of peak vs the multipass kernels' mean 15.4 %; achieved occupancy ~17 % | SUPPORTED |
| H2 fused_block re-reads cached at <= 2048 cols, from DRAM at 8192; fused_smem one read | fused_block DRAM reads 4.88 / 9.08 / 11.99 B per element at 2048 / 4096 / 8192 cols; L1 hit 56 / 41 / 26 %; fused_smem 4.01-4.03 B per element | SUPPORTED |
| H3 GEMM naive and tiled load-issue bound, not DRAM; tiled occupancy-limited | dominant stalls LG throttle (naive, 56 %) and MIO throttle (tiled, 52 %); DRAM 30.4 % (naive) and 15.4 % (tiled); tiled achieved occupancy 66.5 % | INCONCLUSIVE |
| H4 regblock between tiled and cuBLAS in FMA-pipe use, not DRAM-bound | FMA pipe 6.8 % (tiled) / 22.4 % (regblock) / 59.5 % (cuBLAS); regblock DRAM 12.8 %, dominant stall MIO throttle (43 %) | SUPPORTED |

**What the counters add:**
- **H1** passes its rule, but only partly in the sense intended: the
  dominant stall is the long scoreboard (waiting for global-memory loads),
  and it dominates every kernel at 4 x 65536, with ~17 % achieved occupancy
  (4 blocks for 28 SMs). The counters therefore show a latency-bound kernel
  starved of parallelism; they do not single out the `expf` dependency
  chain that the softmax timing study proposed (the long scoreboard is a memory wait, not
  an arithmetic one). The specific mechanism stays a HYPOTHESIS.
- **H2** is confirmed in detail, including the L1 share that the layer-norm
  timing controls could only infer from timing: with the largest shared-memory carveout
  (small L1), the same fused_block kernel's L1 hit rate falls from 56 % to
  9 % at 2048 cols (41 % to 4 % at 4096) and its DRAM reads rise from 4.88
  to 6.76 B per element (9.08 to 11.70). fused_smem reads x exactly once
  (4.01 B per element) at every width. All layer-norm kernels run at
  82-95 % of the DRAM peak (memory-bound, as the traffic model assumed).
- **H3** is INCONCLUSIVE by its rule for two reasons, one of them an error
  in the rule: NAIVE's DRAM throughput is 30.4 %, just above the 30 % line,
  and TILED's achieved occupancy is 66.5 % - one 1024-thread block per SM is
  32 of 48 warps, i.e. 66.7 %, so the pre-registered "<= 50 %" could not
  have passed (a mistake in the pre-registration, reported, not corrected).
  The load-issue half is consistent with the counters: both kernels are
  dominated by load-throttle stalls (LG for NAIVE's global/L1 loads, MIO for
  TILED's shared-memory loads), NAIVE's L1 hit rate is 88 % (bypassing L1
  drops it to 0 % and halves its SM throughput), and NAIVE moves ~257 B of
  DRAM reads per output element against TILED's ~136.
- **H4** is confirmed: REGBLOCK uses the FMA pipe 3.3x more than TILED but
  2.7x less than cuBLAS, and is held back mostly by shared-memory (MIO)
  throttling, not DRAM. cuBLAS (`ampere_sgemm_128x64_nn`, launched with a
  K-split grid of 8 x 16 x 5) runs at 31.8 % achieved occupancy with far
  higher FMA-pipe use - larger register tiles, not more warps.

**Caveat on DRAM writes:** the softmax kernels at 4 x 65536 report DRAM
writes of 1-3 B per element although each writes 4: their 1 MiB outputs fit
in L2, and write-backs can complete after the profiled kernel ends. Write
counts for small outputs are therefore not used.
