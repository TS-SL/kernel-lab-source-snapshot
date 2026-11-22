# Softmax

Written before the kernels; results sections are filled only from executed
runs.

## Definition and supported domain

Row-wise softmax over a contiguous, row-major FP32 matrix `x[rows][cols]`:

    m_r    = max_j x[r][j]
    y[r][j] = exp(x[r][j] - m_r) / sum_k exp(x[r][k] - m_r)

Subtracting the row maximum makes every exponent <= 0, so no term overflows
(numerical stability); it does not change the mathematical result.

| Item | Supported | Otherwise |
|---|---|---|
| dtype / layout | FP32, contiguous row-major, softmax over the last dimension | (no other entry points) |
| rows, cols | `rows == 0` or `cols == 0`: no-op, `KL_OK` | negative: `KL_ERR_INVALID_ARGUMENT` |
| cols | 1 ... 2^20 | `KL_ERR_UNSUPPORTED` |
| rows | 1 ... 2^20 | `KL_ERR_UNSUPPORTED` |
| rows x cols | <= 2^26 elements (256 MiB per tensor) | `KL_ERR_UNSUPPORTED` |
| `KL_SOFTMAX_FUSED_WARP` | cols <= 1024 | `KL_ERR_UNSUPPORTED` |
| device memory | within the library's VRAM cap | `KL_ERR_RESOURCE_LIMIT` |

The bounds keep every launch far below the 2 s Windows watchdog (one block
per row; at most 2^20 blocks and 2^20 elements per block) and inside the
memory cap.

**Special values** (same as PyTorch's `torch.softmax`):
- `-inf` entries give 0, as long as the row has at least one finite value;
- a row that is entirely `-inf`, or contains `NaN` or `+inf`, gives `NaN`
  for the whole row.

## Implementations

| Enum | Kernels | Global reads of x | Notes |
|---|---|---|---|
| `KL_SOFTMAX_MULTIPASS` (baseline) | 3: row max, row sum, normalize | 3 | Writes `max` and `sum` (2 floats per row) to global memory between kernels. Block per row, 256 threads. |
| `KL_SOFTMAX_FUSED_ONLINE` | 1 | 2 | Block per row, 256 threads. Pass 1 computes max and sum together in one read with the online-softmax recurrence (Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018); pass 2 writes the output. |
| `KL_SOFTMAX_FUSED_WARP` | 1 | 1 | One warp per row (4 rows per 128-thread block); each lane keeps its share of the row (up to 32 values) in registers, so x is read once. cols <= 1024. |
| `KL_SOFTMAX_AUTO` | - | - | `FUSED_WARP` for cols <= 1024; otherwise `FUSED_ONLINE` if rows >= 64, else `MULTIPASS`. The first version chose `FUSED_ONLINE` for every cols > 1024; a dry-run benchmark showed it 0.67-0.72x the speed of `MULTIPASS` at 4 and 16 rows, so the row threshold was added (dry-run numbers are *unarchived observations*; files not kept). It was chosen on dry-run points at 16 rows (multipass faster) and 64 rows (online faster); the measured rows sweep, not used to choose it, agrees (see Results). |

Minimal global traffic for any softmax is one read and one write of the
matrix (8 bytes per element). The baseline moves about 16 bytes per element
(3 reads + 1 write) plus the per-row intermediates.

## Error bound (derived; used as the test tolerance)

Notation: `u = 2^-24` (FP32 unit roundoff). `+ - * /` round correctly
(`|rel err| <= u`; division is IEEE with the default `-prec-div=true`).
`expf` has at most 2 ulp error (CUDA Programming Guide, mathematical
functions table), i.e. relative error `<= 4u` for normal results. `fmaxf` is
exact. No fast-math flags are used. nvcc's default `--fmad=true` may fuse
`s * r + t` into one FMA; that removes a rounding, so the bound still holds.

For element `i` of a row, let `d_i = m - x_i >= 0` over the finite entries,
and `Dw = sum_j e_j d_j / sum_j e_j` (the exp-weighted mean of `d_j`); `k` =
elements accumulated per thread (`ceil(cols / T)` for `T` threads per row:
256 for the block kernels, 32 for the warp kernel) and `L` = levels of the
cross-thread reduction tree that combine real data. The tree has
`log2(T)` levels (5 warp-shuffle levels, plus 3 across the 8 warp partials
of a 256-thread block), but combining with an identity partial is exact
(`x + 0`; `s * 1 + 0 * 0`), so only `L = ceil(log2(min(cols, T)))` of them
can round.

- Computing `x_i - m` rounds once; through `exp` this becomes a relative
  error of at most `d_i * u`. With `expf`'s 4u, each term
  `e_i = exp(x_i - m)` has relative error `<= (d_i + 4) u`.
- **Two-pass sum** (`MULTIPASS`, `FUSED_WARP`): a positive term passes
  through at most `k - 1 + L` additions, so the sum's relative error is at
  most the exp-weighted mean of the term errors plus `(k - 1 + L) u`:
  `<= (Dw + 4 + k - 1 + L) u`.
- **Online sum** (`FUSED_ONLINE`): a term computed against a running
  maximum `m'` is later rescaled by `exp(m' - m'')` factors whose arguments
  telescope to `m - m'`, so the subtraction errors still total `d_i u`; each
  of at most `k - 1 + L` rescales adds `4u` (expf) `+ u` (multiply), and each
  addition `u`: sum error `<= (Dw + 4 + 6 (k - 1 + L)) u`.
- **Output** `y_i = e_i / s` adds the term error, the sum error and `u` for
  the division:

      two-pass:  |y_i - y_i*| / y_i*  <=  (d_i + Dw + k + L + 8) u
      online:    |y_i - y_i*| / y_i*  <=  (d_i + Dw + 6 (k - 1 + L) + 9) u

**Exact differences.** The `d_i` and `Dw` terms come only from rounding in
float subtractions of inputs (`x - m`, and in the online kernel the rescale
arguments `m' - m''`, which are also differences of inputs). If every input
is a multiple of `2^-10` with `|x| <= 128`, every such difference is exactly
representable, those terms vanish, and a tighter bound applies. Tests with
such inputs check the premise per row rather than assume it.

**Second order and underflow.** These are first-order bounds. An output
that does not underflow has `d_i <= ~104` (exp(-104) is below the smallest
subnormal), and `Dw` is a weighted mean dominated by such terms, so the
second-order terms are below 1 % of the bound over the whole domain; tests
multiply the bound by 1.01. In the subnormal range two error sources stop
being relative: `expf`'s 2 ulp become `2 * 2^-149` absolute, and the
division's rounding becomes half a subnormal ulp (dividing by the sum,
which is `>= 1` in every kernel because the maximum contributes exactly 1,
cannot enlarge either). The subtraction-amplification part stays relative
to the true value, so the relative term is kept in the formula and only
these absolute parts are added, `2.5 * 2^-149 < 2^-147 = 4 * 2^-149`:

    |y - y_ref| <= 1.01 * B_i * y_ref + 2^-147      (y_ref in float64)

**What the tests can and cannot detect** (each claim was checked by
deliberately breaking the code and re-running the tests; the mutation
results are *unarchived observations*: no logs were kept):
- removing the online recurrence's `-inf` guard: caught (NaN rows);
- `__expf` (fast-math exp) in the online kernel's output pass: caught at
  4.05x the bound by the two-element exp-accuracy rows (an earlier, looser
  version of the bound caught it at only 1.02x);
- a flush-to-zero build (`-ftz=true`): caught at up to 1.5e5x the bound;
- a wrong row maximum mostly cancels out of softmax, so it is only caught
  when it causes overflow: the max-spike test puts a +100 spike at warp and
  block boundaries, where a dropped partial gives `exp(~99) = inf`;
- a dropped or duplicated element changes a large row's result by far less
  than the bound (e.g. ~1e-6 vs ~2e-4 at cols = 2^20); only the bit-exact
  constant-row test catches such counting errors;
- out-of-bounds writes are checked with guard bands around the output and
  workspace; out-of-bounds reads are not detectable without Compute
  Sanitizer, which is unavailable on this machine;
- the device entry point refuses overlapping x / y / workspace ranges and
  misaligned pointers (a misaligned access would be a sticky CUDA error that
  also breaks other runtimes sharing the context, e.g. PyTorch).

## Results

All numbers are MEASURED on the RTX 3060 at commit `64bbf58` (3 processes
per harness, 50 samples each, interleaved), from
`results/processed/m2-softmax-summary.md`, which
`scripts/analyze_softmax_bench.py` regenerates from `results/raw/m2-bench-*`.
Kernel times: CUDA events around batches of launches, inputs cold in L2
between launches. Effective bandwidth = 8 bytes per element / time
(MEASURED-derived; the ~360 GB/s peak is an ESTIMATE from the RTX 3060's
specification). Correctness of every benchmarked output was checked against
the derived bound on a poisoned buffer before and after timing: 0
violations, worst error 0.63 of the bound.

### KernelLens implementations compared (C++ harness)

| rows x cols | multipass us | fused_online us | fused_warp us | online speedup | warp speedup | warp GB/s |
|---|---|---|---|---|---|---|
| 4096 x 32 | 52.5 | 54.0 | 6.3 (*) | 0.97x | 8.34x (*) | 167 (*) |
| 4096 x 128 | 59.3 | 60.2 | 15.5 | 0.99x | 3.82x | 270 |
| 4096 x 512 | 121.4 | 91.9 | 53.4 | 1.32x | 2.27x | 314 |
| 4096 x 1024 | 209.7 | 127.1 | 104.3 | 1.65x | 2.01x | 322 |
| 2048 x 2048 | 210.3 | 139.7 | - | 1.50x | - | - |
| 1024 x 4096 | 213.0 | 158.0 | - | 1.35x | - | - |
| 256 x 16384 | 213.4 | 169.3 | - | 1.26x | - | - |
| 64 x 65536 | 233.7 | 223.0 | - | 1.05x | - | - |
| 16 x 262144 | 525.4 | 728.4 | - | **0.72x** | - | - |
| 4 x 1048576 | 1866.8 | 2773.9 | - | **0.67x** | - | - |

95 % hierarchical-bootstrap CIs of the speedups are within +-0.01 of the
value, except fused_warp at 4096 x 32 ([8.27, 8.42]). (*) At 4096 x 32 the
host needs about as long to enqueue a launch of the
warp kernel as the GPU needs to run it (host/GPU ratio 0.99, positive fixed
per-batch cost 28 %), so 6.3 us is an upper bound on its kernel time.

- **The warp kernel** (row in registers, one read) is 2.0-8.3x faster than
  the baseline wherever it applies (cols <= 1024) and reaches 314-322 GB/s
  at 512-1024 cols, i.e. 87-89 % of the ESTIMATED peak.
- **The fused online kernel** is 1.05-1.65x faster than the baseline from
  512 to 65536 cols with 64 or more rows; it is slightly slower at 32 and
  128 cols (0.97x, 0.99x; CI upper bounds <= 1.00) and clearly **slower**
  with few long rows (0.72x at
  16 x 262144, 0.67x at 4 x 1048576). Geometric mean over the 10 shapes:
  1.11x.
- *Traffic model (ESTIMATE):* the baseline reads the matrix three times
  (~16 bytes per element), so its 157-160 GB/s effective at 2048-16384 cols
  corresponds to ~315-320 GB/s of modelled DRAM traffic - close to the
  peak, i.e. memory-bound, and fusion wins by removing traffic. At
  4096 x 1024 the online kernel's 264 GB/s effective would mean 396 GB/s if
  its second read of each row went to DRAM, which exceeds the peak; the
  second read is therefore at least partly served from L2 (a 4 KiB row is
  re-read by the same block right after the first pass). This is an
  inference from timings and the specification, not a counter measurement.

### Rows sweep and the AUTO threshold

At cols = 65536 (one block per row, 28 SMs):

| rows | multipass us (x vs 4 rows) | fused_online us (x vs 4 rows) | online / multipass time |
|---|---|---|---|
| 4 | 71.9 (1.00x) | 138.3 (1.00x) | 1.92 |
| 8 | 116.7 (1.62x) | 181.6 (1.31x) | 1.56 |
| 16 | 140.4 (1.95x) | 188.4 (1.36x) | 1.34 |
| 28 | 157.0 (2.18x) | 194.7 (1.41x) | 1.24 |
| 56 | 213.6 (2.97x) | 215.9 (1.56x) | 1.01 |
| 64 | 233.7 (3.25x) | 223.0 (1.61x) | 0.95 |
| 112 | 368.8 (5.13x) | 294.6 (2.13x) | 0.80 |
| 224 | 735.2 (10.23x) | 611.0 (4.42x) | 0.83 |

- **AUTO threshold, held-out check:** the threshold (online from 64 rows)
  was chosen on a dry run at 16 and 64 rows; this sweep was not used to pick
  it. The crossover lies between 28 rows (online 1.24x slower) and 64 rows
  (online faster), with the two equal at 56 - consistent with the threshold.
- **A refuted hypothesis:** if few-row shapes were slow only because one
  block per row leaves SMs idle, the time should stay flat up to 28 rows
  (one block per SM). Multipass time grows 2.18x from 4 to 28 rows, so that
  explanation does not hold for it. The online kernel's time is closer to
  flat (1.41x) and starts far higher (138 us vs 72 us at 4 rows): consistent
  with the HYPOTHESIS that its first pass is latency-bound - each step of the
  recurrence depends on the previous running max through an `exp` - but
  stall-reason counters (BLOCKED here) would be needed to confirm it.

### KernelLens vs `torch.softmax` (one process, same tensors, stream and clocks)

Ratio = torch time / KernelLens time for the fastest KernelLens implementation
of the shape (> 1: KernelLens faster); per-process ratios in brackets.

| rows x cols | KernelLens | torch us | KernelLens us | ratio |
|---|---|---|---|---|
| 4096 x 32 | fused_warp | 7.1 | 6.2 | 1.14 (1.12-1.15) - both host-bound, no claim |
| 4096 x 128 | fused_warp | 15.7 | 15.5 | 1.01 (1.01-1.01) |
| 4096 x 512 | fused_warp | 53.2 | 53.5 | 0.99 (0.99-1.00) |
| 4096 x 1024 | fused_warp | 104.9 | 104.6 | 1.00 (1.00-1.00) |
| 2048 x 2048 | fused_online | 105.4 | 139.5 | 0.76 (0.75-0.76) |
| 1024 x 4096 | fused_online | 112.4 | 158.4 | 0.71 (0.71-0.71) |
| 256 x 16384 | fused_online | 136.4 | 169.7 | 0.80 (0.80-0.80) |
| 4 x 65536 | multipass | 23.9 | 72.1 | 0.33 |
| 8 x 65536 | multipass | 30.7 | 115.3 | 0.27 |
| 16 x 65536 | multipass | 55.2 | 140.6 | 0.39 |
| 28 x 65536 | multipass | 91.8 | 157.2 | 0.58 |
| 56 x 65536 | multipass | 179.7 | 212.8 | 0.84 |
| 64 x 65536 | fused_online | 208.1 | 222.5 | 0.94 |
| 112 x 65536 | fused_online | 355.5 | 294.6 | 1.21 (1.20-1.21) |
| 224 x 65536 | fused_online | 707.2 | 614.4 | 1.15 (1.14-1.15) |
| 16 x 262144 | multipass | 207.2 | 526.6 | 0.39 |
| 4 x 1048576 | multipass | 357.4 | 1867.2 | 0.19 |

- At 128-1024 cols the warp kernel is **level with torch.softmax** (within
  1.3 %); no winner is claimed.
- From 2048 to 16384 cols (256-2048 rows) torch is 1.24-1.41x faster.
- With few long rows torch is far faster (1.71-5.22x at <= 28 rows). Its
  advantage shrinks as rows grow, and at 112 and 224 rows of 65536 the
  online kernel is 1.15-1.21x faster than torch (two points of one shape
  family). How torch processes these shapes (kernel choice, grid) was not
  inspected: UNVERIFIED.
- NVML reported SM clocks below 1500 MHz for some Python-harness samples of
  the 2048 x 2048, 112 x 65536 and 224 x 65536 shapes. The summary's
  low-reading table shows those samples' median times equal the others'
  (ratio 0.999-1.007), so the readings are post-batch idle snapshots, not
  slow execution; in the superseded first series the same comparison gave
  1.04-1.27 at 112 x 65536 (a real slowdown). All three Python processes
  also read idle clocks (210 MHz) right after their initial 300 ms ramp, yet
  per-sample clocks during timing were ~1920 MHz - consistent with NVML
  sampling lag.
- Noisy groups (CV > 5 %) besides 4096 x 32: Python multipass 2048 x 2048
  (CV 8.0 %, one sample at ~2x the median; IQR/median 0.4 %) and torch
  4096 x 128 (CV 5.5 %, IQR/median 0.9 %). Medians are unaffected.
- The two harnesses agree on KernelLens kernel times within 2 % (0.98-1.01).

### End-to-end `kl_softmax_host`

Pageable host memory, per-call device allocation, H2D + kernel + D2H:
0.71 ms at 4096 x 128 (46x the kernel), 3.90 ms at 1024 x 4096 (25x) and
5.93 ms at 4 x 1048576 (3.2x). The non-kernel time grows with the data
size (0.70 ms at 2 MiB, 3.7-4.1 ms at 16 MiB), consistent with pageable
copies plus allocation; the breakdown was not measured. The host API is
meant for tests, not for performance.

### Register usage

`ptxas -v` for each kernel is in `results/raw/m2-build.log`.
