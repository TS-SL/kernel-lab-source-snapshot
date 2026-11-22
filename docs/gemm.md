# Matrix multiplication

Written before the kernels; results are added only from executed runs.

## Definition and supported domain

`C = A B` for contiguous row-major FP32 matrices `A[M][K]`, `B[K][N]`,
`C[M][N]`:

    C[i][j] = sum_{p=0}^{K-1} A[i][p] * B[p][j]

No scaling factors (`alpha = 1`, `beta = 0`); `C` is overwritten.

| Item | Supported | Otherwise |
|---|---|---|
| dtype / layout | FP32, contiguous row-major, no transposes | - |
| M, N, K | each `>= 0`; `M == 0` or `N == 0`: no-op, `KL_OK`; `K == 0`: `C = 0` | negative: `KL_ERR_INVALID_ARGUMENT` |
| sizes | each dimension `<= 2^14`, each matrix `<= 2^26` elements | `KL_ERR_UNSUPPORTED` |
| pointers (device API) | 4-byte aligned; `C` must not overlap `A` or `B` (`A` and `B` may overlap) | `KL_ERR_INVALID_ARGUMENT` |

**Special values** follow IEEE arithmetic of the sequential sum below
(NaN and inf propagate; `inf * 0` and `inf - inf` give NaN), compared with a
float64 reference that has the same IEEE semantics - as long as the finite
products cannot overflow FP32 partial sums (inside the bound's domain).
Outside it the two can differ: for the row `(-FLT_MAX, -FLT_MAX, +inf)`
times ones, the FP32 sum overflows to -inf and then gives NaN, the float64
sum gives +inf.

## Implementations

All three KernelLens kernels compute every output with the **same
operation sequence**: `acc = 0; for p = 0..K-1: acc = fmaf(A[i][p], B[p][j], acc)`.
They differ only in how A and B reach the arithmetic units, so their results
are **bitwise identical** (tested) and share one error bound. Edge tiles
stop the last tile's inner loop at `K`. Padding the sum with
`fmaf(0, 0, acc)` terms instead would be observable: an `fmaf` whose exact
result underflows below half the smallest subnormal rounds to a signed zero,
so `acc` can be -0 (e.g. `fmaf(-2^-80, 2^-80, +0) = -0`), and `-0 + 0 = +0`.
A test with products near 2^-150 checks that all kernels agree bit for bit,
signed zeros included, and a mutation pads the tail. (An earlier version of
this paragraph claimed `acc` could never be -0; the review found the
counterexample.) For K = 1 the single product rounding reaches the bound
exactly, so ratios close to 1 there are expected.

| Enum | Mapping | Data reuse |
|---|---|---|
| `KL_GEMM_NAIVE` (baseline) | one thread per output, 16 x 16 blocks | none: each thread reads a row of A and a column of B from global memory (2K loads per output) |
| `KL_GEMM_TILED` | 32 x 32 output tile per block (1024 threads), K in steps of 32 | A and B tiles staged in shared memory: each loaded element is used 32 times |
| `KL_GEMM_REGBLOCK` | 64 x 64 output tile per block, 256 threads, each thread 4 x 4 outputs, K in steps of 16 | shared-memory tiles plus register blocking: per inner step a thread loads 4 + 4 values and does 16 FMAs |
| `KL_GEMM_AUTO` | - | `REGBLOCK` (fixed before measuring) |

`REGBLOCK` thread `(ty, tx)` owns rows `ty + 16 a` and columns `tx + 16 b`
(`a, b = 0..3`) of the block tile, so in the inner loop a warp reads two
distinct A words (broadcast) and 16 consecutive B words: no shared-memory
bank conflicts.

**Not detectable here:** an edge-tile load that reads past the end of A or
B (a dropped `row < m` or `p < K` guard) only fills shared-memory slots that
are never used, so results, bitwise checks and guard bands all pass; only
Compute Sanitizer memcheck could see it. memcheck reported no error on the
test shapes (capped at `M N K / 64 <= 1048576`, every tile boundary
included; `results/raw/m5-sanitizer-gemm-memcheck.log`); larger shapes rely
on review.

**cuBLAS** (`cublasSgemm`, `CUBLAS_DEFAULT_MATH`, i.e. no TF32 tensor-op
math; row-major handled as `C^T = B^T A^T`) is the performance reference and
a second, independent numerical reference in the tests. Its summation order
is unknown, so it is checked against the order-independent bound below, not
bitwise.

## Error bound (derived; used as the test tolerance)

With `u = 2^-24` and `gamma_K = K u / (1 - K u)`, the sequential FMA sum
satisfies, for *any* summation order in which each of the `K` products
enters through one rounding and each partial sum is rounded once
(Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1;
with FMA the product and the add share one rounding):

    |C_hat[i][j] - C[i][j]| <= gamma_K * S[i][j] + K * 2^-150,
    S[i][j] = sum_p |A[i][p]| |B[p][j]|

The `K * 2^-150` term covers roundings whose result is subnormal (absolute
error <= 2^-150 instead of relative u). `gamma_K` is exact, not first order,
so no safety factor is needed. The tests add `K * 2^-52 * S` for the
float64 reference's own rounding (a sequential float64 sum errs by up to
about `K * 2^-53 * S`), and multiply the subnormal term by `1 + gamma_K`
(an underflow error can be amplified by later roundings). **Validity domain:** no overflow, `S < 2^126`
(then no partial sum overflows); `K <= 2^14` gives `K u <= 2^-10`. The same
bound applies to cuBLAS (any order, any blocking).

**A tighter, exact check.** With integer inputs in [-8, 8] and
`K <= 2^14`, every product and partial sum is an integer below 2^24 in
magnitude, so every FP32 operation is exact: all implementations, including
cuBLAS, must equal the float64 reference bit for bit. This catches indexing
errors that a relative bound could hide (e.g. a wrong but similar-magnitude
element).

## Tests (plan)

Shapes around tile boundaries (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127,
128, 129) in every dimension independently, K = 0 and K = 1, rectangular and
skinny shapes, a 2^14 dimension; uniform data (bound), integer data (exact),
special values (NaN/inf rows of A and columns of B, `inf * 0`), signed zeros;
bitwise equality of the three kernels (also with products near 2^-150, where
sums can round to -0); cuBLAS within the bound; guard bands around C; overlap/alignment/invalid-argument checks; the host API with
in-place refusal. Mutation checks: transposed B index, missing barrier,
off-by-one edge guard, a wrong tile offset, a dropped last element of a tile.

## Benchmark plan (pre-registered; predictions written before any timing)

Same protocol as the other operators (C++ harness, `bench_harness.hpp`),
with the batch size chosen from the work instead of the bytes: `K_batch` so
that one batch lasts about 1 ms at an assumed 5 TFLOP/s, clamped to 1..2000;
each launch stays < 100 ms (NAIVE is not run above 2048^3 work). Metric:
GFLOP/s = 2 M N K / median time. cuBLAS runs in the same process, stream and
batches (ABBA). 3 processes, clean commit.

**Shapes:** square 256, 512, 1024, 2048, 4096 (no NAIVE at 4096);
non-multiples 1000 and 1023 (edge handling); transformer-like
8192 x 768 x 768, 8192 x 3072 x 768, 8192 x 768 x 3072; skinny 64 x 4096 x
4096 and 4096 x 4096 x 64.

**Predictions (ESTIMATED from the design, before any timing).** The RTX
3060's FP32 peak is about 12.7 TFLOP/s (28 SMs x 128 FP32 lanes x 2 FLOP x
~1.78 GHz boost; ESTIMATED from the specification).
- P1: `TILED` is >= 3x faster than `NAIVE` at 1024^3 and 2048^3 (global
  loads per FMA fall from 2 to 2/32).
- P2: `REGBLOCK` is >= 1.5x faster than `TILED` at 1024^3 and above
  (shared-memory loads per FMA fall from 2 to 0.5).
- P3: cuBLAS is faster than `REGBLOCK` on every shape with `M N K >= 2^30`
  (the reading pinned before the series: all such shapes, cubes or not; it uses
  far larger register tiles, vector loads and double buffering, none of
  which `REGBLOCK` has); no factor predicted.
- P4: `REGBLOCK` reaches >= 25 % of the ESTIMATED FP32 peak at 2048^3 and
  4096^3.

Decision rules (as for layer norm): a ratio prediction PASSes on a shape if
the 95 % CI lower bound clears the threshold and every per-process ratio
does, FAILs if the CI upper bound is below it, else INCONCLUSIVE; P4 PASSes
if the CI lower bound of the GFLOP/s clears 25 % of 12.7 TFLOP/s. A
prediction holds only if it is PASS on every listed shape.

### Change after the dry run (before any measured series)

A dry run (one process, uncommitted code; *unarchived observations*, not
evidence; raw files kept only outside the repository) ran every shape. All
outputs passed the bound (worst 0.07 of it). Two observations:
- **Safety:** `TILED` took about 170 ms per launch at 4096^3, below the 2 s
  watchdog but above the project's 100 ms launch target. `TILED` is
  therefore capped like `NAIVE`: it runs only where `2 M N K <= 4e10`
  (about 50 ms at the dry-run rate). This removes 4096^3 from P2's judged
  shapes; no other prediction or rule changes.
- **Tiling alone did not help:** `TILED` ran at about 800 GFLOP/s, no faster
  than `NAIVE` (about 830). P1 is expected to FAIL; it stays as written. Its
  likely mechanism (a hypothesis, not a result) has two halves: `NAIVE`'s
  reuse is already served by L1 (a 16 x 16 block's warp is 2 rows by 16
  columns, so it reads 2 A words as broadcasts and 16 consecutive B words,
  which neighbouring warps reuse from L1/L2), and `TILED` still issues two
  shared-memory loads per FMA, so both are bound by load issue, not by DRAM
  (ESTIMATED cap for 2 loads per FMA at one load instruction per clock per
  SM: 28 x 32 FLOP x 1.78 GHz ~ 1.6 TFLOP/s). A confound: `TILED`'s
  1024-thread blocks fit once per SM (32 warps) against `NAIVE`'s 48. The control
  added to test it: `naive_l2only`, the naive kernel with L1-bypassing
  `__ldcg` loads (same arithmetic, bitwise identical, tested), at the shapes
  where `NAIVE` runs. It tests only the first half of the hypothesis
  (`NAIVE` relies on L1), and it is **confirmatory, not predictive**: a quick
  harness check (one process, 2 shapes) had already shown it 1.9-2.3x slower
  than `NAIVE` before it was written down here. Rule fixed now, before the
  series: the first half is supported where the ratio naive_l2only / naive
  has a CI lower bound > 1.10, refuted where its upper bound < 1.10. The
  second half (TILED load-issue bound, and the occupancy confound) is left to
  the Nsight Compute counters (docs/profiling.md, H3).

## Results (MEASURED, commit 79a9ab2)

Source: `results/processed/m4-gemm-summary.md` (generated by
`scripts/analyze_gemm.py` from `results/raw/m4-bench-gemm-cpp-p*`: 3
processes, 7,950 samples, status MEASURED; peak GPU temperature 74 C, no
guard stop, no thermal clock-event sample). All 318 benchmark correctness
checks passed (worst 0.07 of the bound). cuBLAS ran with math mode 0
(`CUBLAS_DEFAULT_MATH`, no TF32) and `NVIDIA_TF32_OVERRIDE` unset in every
process. The board's power limit was reached in 2,434 of 7,950 samples (38
of 53 groups had at least one power-cap sample); median SM clocks per group
stayed at 1852-1935 MHz. A first attempt at the series stopped after one
process (a runner bug with a single harness, fixed in 79a9ab2); its files
were deleted unanalyzed (*unarchived observation*).

| Shape (M x N x K) | NAIVE | TILED | REGBLOCK | cuBLAS |
|---|---|---|---|---|
| 1024^3 | 830 GFLOP/s | 790 | 2620 (21 % of peak) | 7133 (56 %) |
| 2048^3 | 827 | 791 | 2918 (23 %) | 7910 (62 %) |
| 4096^3 | - | - | 2759 (22 %) | 8133 (64 %) |
| 8192 x 3072 x 768 | - | 805 | 2967 (23 %) | 9306 (73 %) |

(% of the ESTIMATED 12.7 TFLOP/s FP32 peak; the summary also gives it at
the recorded clock.)

**Predictions:**

| Prediction | Shapes | Verdict |
|---|---|---|
| P1 TILED >= 3x faster than NAIVE | 1024^3, 2048^3 | **FAIL** on both: TILED is 0.95-0.96x NAIVE (slightly slower) |
| P2 REGBLOCK >= 1.5x faster than TILED | 1024^3, 2048^3 | PASS (3.32x, 3.69x) |
| P3 cuBLAS faster than REGBLOCK | 8 shapes with M N K >= 2^30 | PASS on all (2.35-3.93x) |
| P4 REGBLOCK >= 25 % of the ESTIMATED peak | 2048^3, 4096^3 | **FAIL** on both (23 %, 22 %; CI below 3175 GFLOP/s) |

**What the measurements say:**
- **Shared-memory tiling alone bought nothing here.** TILED (32 x 32 tiles)
  is 0.84-0.97x NAIVE on every shape. NAIVE is not a DRAM-bound strawman:
  bypassing L1 slows it 1.62-3.93x (`naive_l2only`, "supported" on all 9
  shapes), so its operand reuse is already served by L1. What limits TILED
  (load issue, occupancy) is tested with counters in docs/profiling.md.
- **Register blocking is what pays:** 4 x 4 outputs per thread make
  REGBLOCK 2.65-3.74x faster than TILED from 1000^3 work up (2.65x on the
  skinny 64 x 4096 x 4096 shape, 3.3-3.7x on the others), at 1.9-3.0 TFLOP/s
  (15-23 % of the ESTIMATED peak; 15 % on 64 x 4096 x 4096).
- **cuBLAS is 2.35-3.93x faster than REGBLOCK** (6.6-9.3 TFLOP/s, 52-73 % of
  the ESTIMATED peak), most on the skinny 64 x 4096 x 4096 shape (64 rows:
  REGBLOCK has only 64 blocks for 28 SMs) and least on 4096 x 4096 x 64.
- **Small shapes** (256^3, 512^3; launches of 13-611 us) are marked in the
  summary: per-launch gaps of unknown size inflate them; no prediction
  uses them.
- The cuBLAS fixed per-batch cost estimate is ~10 % at 1000^3 (batches of 3
  launches); the other implementations' estimates are a few percent.
