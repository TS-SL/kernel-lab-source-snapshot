"""Python reference tests for kernellens.dll (via ctypes).

Inputs come from fixed seeds. The GPU computes fmaf (one rounding of the
exact a*x+y). Checks, strongest first:
  * bit-for-bit equality wherever the float64 reference is provably exact
    (TwoSum error == 0; see kl.saxpy_reference);
  * <= 1 ulp everywhere against the float64 reference (double-rounding bound,
    docs/methodology.md), shown to be tight by the hand-built midpoint case.
"""

import numpy as np
import pytest

import kl

# Straddle warp/block boundaries and the 65535-block grid cap (16,776,960).
SIZES = [1, 2, 31, 32, 33, 255, 256, 257, 1000, 1 << 20, 16_776_961]
ALPHAS = [0.0, 1.0, -2.5, 1.0e-3]
SENTINEL = np.float32(-12345.0)


def run_saxpy(lib, a, x, y):
    """Runs the GPU on copies; returns the result. Inputs are never modified."""
    x = np.array(x, dtype=np.float32, copy=True)
    out = np.array(y, dtype=np.float32, copy=True)
    status = lib.kl_saxpy_host(out.size, a, kl.float_ptr(x), kl.float_ptr(out))
    assert status == kl.KL_OK, lib.kl_last_error_message()
    return out


def check_against_reference(a, x, y, got, min_exact_fraction=0.0):
    ref, exact = kl.saxpy_reference(a, x, y)
    assert exact.mean() >= min_exact_fraction, "test input does not exercise the exact check"
    bad = exact & ~kl.same_float(got, ref)
    assert not bad.any(), f"{bad.sum()} provably-exact elements differ from correctly rounded fmaf"
    assert kl.ulp_distance(got, ref).max() <= 1


def test_device_info(lib):
    info = kl.DeviceInfo()
    assert lib.kl_get_device_info(info) == kl.KL_OK, lib.kl_last_error_message()
    assert info.compute_major * 10 + info.compute_minor >= 86
    assert info.multiprocessor_count > 0
    assert info.total_global_mem > info.free_global_mem > 0
    assert lib.kl_get_device_info(None) == kl.KL_ERR_INVALID_ARGUMENT


@pytest.mark.parametrize("seed", [0, 1, 42, 2**64 - 1])
@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-100.0, 100.0), (-3.0, 5.0), (-1e-3, 1e-3),
                                   (-3.4e38, 3.4e38), (1e-38, 2e-38)])
def test_prng_bit_identical_across_languages(lib, seed, lo, hi):
    n = 10_000
    out = np.empty(n, dtype=np.float32)
    assert lib.kl_fill_uniform(seed, n, lo, hi, kl.float_ptr(out)) == kl.KL_OK
    expected = kl.fill_uniform(seed, n, lo, hi)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
    assert out.min() >= np.float32(lo) and out.max() <= np.float32(hi)


@pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (2.0, 1.0), (-np.inf, 0.0), (0.0, np.inf),
                                   (np.nan, 1.0), (0.0, np.nan)])
def test_prng_rejects_bad_bounds_without_writing(lib, lo, hi):
    out = np.full(4, SENTINEL, dtype=np.float32)
    assert lib.kl_fill_uniform(1, 4, lo, hi, kl.float_ptr(out)) == kl.KL_ERR_INVALID_ARGUMENT
    assert np.all(out == SENTINEL)
    with pytest.raises(ValueError):
        kl.fill_uniform(1, 4, lo, hi)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("a", ALPHAS)
def test_saxpy_matches_reference(lib, n, a):
    seed = 1000 * n + int(abs(a) * 1000)
    # y much smaller than a*x: most float sums are inexact, so a kernel that
    # rounded in the wrong direction would fail the bitwise check.
    x = kl.fill_uniform(seed, n, -100.0, 100.0)
    y = kl.fill_uniform(seed + 1, n, -1e-3, 1e-3)
    got = run_saxpy(lib, a, x, y)
    check_against_reference(a, x, y, got, min_exact_fraction=0.99)


def test_saxpy_double_rounding_midpoint(lib):
    # a*x = 1 + 2^-24 exactly (24929 * 673 = 2^24 + 1): the midpoint between
    # 1 and 1 + 2^-23. Adding +2^-60 must round up; the float64 reference
    # loses 2^-60, lands on the midpoint and rounds to even (1.0).
    a = np.float32(24929 / 2**15)
    x = np.array([673 / 2**9, 673 / 2**9, -673 / 2**9], dtype=np.float32)
    y = np.array([2.0**-60, -(2.0**-60), -(2.0**-60)], dtype=np.float32)
    got = run_saxpy(lib, float(a), x, y)
    expected = np.array([1 + 2**-23, 1.0, -(1 + 2**-23)], dtype=np.float32)
    assert kl.same_float(got, expected).all(), got
    ref, exact = kl.saxpy_reference(float(a), x, y)
    assert not exact[0] and ref[0] == np.float32(1.0)  # reference is the one that is off
    assert kl.ulp_distance(got, ref)[0] == 1           # ... by exactly the bound


def test_saxpy_matches_torch_float64(lib):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    n, a = 1_000_003, -1.75
    x = kl.fill_uniform(7, n, -10.0, 10.0)
    y = kl.fill_uniform(8, n, -10.0, 10.0)
    # Independent implementation: PyTorch's float64 kernels on the GPU.
    xt = torch.from_numpy(x).cuda().double()
    yt = torch.from_numpy(y).cuda().double()
    want = (a * xt + yt).float().cpu().numpy()
    got = run_saxpy(lib, a, x, y)
    assert kl.ulp_distance(got, want).max() <= 1


def test_saxpy_special_values(lib):
    f = np.finfo(np.float32)
    specials = np.array(
        [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, f.max, -f.max, f.tiny,
         f.smallest_subnormal, -f.smallest_subnormal],
        dtype=np.float32,
    )
    xs, ys = np.meshgrid(specials, specials, indexing="ij")
    x = xs.ravel().copy()
    y = ys.ravel().copy()
    for a in [0.0, -0.0, 1.0, -2.5, 3.0e38, 1.0e-30]:
        got = run_saxpy(lib, a, x, y)
        check_against_reference(a, x, y, got)
    # The inputs really are the special grid for every alpha (regression
    # test: an earlier version wrote GPU results back into its input grid).
    assert kl.same_float(y, ys.ravel()).all()


def test_saxpy_zero_length_is_noop(lib):
    assert lib.kl_saxpy_host(0, 1.0, None, None) == kl.KL_OK


def test_invalid_arguments_are_rejected(lib):
    x = np.ones(4, dtype=np.float32)
    y = np.full(4, SENTINEL, dtype=np.float32)
    assert lib.kl_saxpy_host(-1, 1.0, kl.float_ptr(x), kl.float_ptr(y)) == kl.KL_ERR_INVALID_ARGUMENT
    assert lib.kl_last_error_message()  # a reason is recorded
    assert lib.kl_saxpy_host(4, 1.0, None, kl.float_ptr(y)) == kl.KL_ERR_INVALID_ARGUMENT
    assert lib.kl_saxpy_host(4, 1.0, kl.float_ptr(x), None) == kl.KL_ERR_INVALID_ARGUMENT
    assert np.all(y == SENTINEL)


@pytest.mark.parametrize("buffers_over_cap", ["one", "two"])
def test_vram_cap_refuses_before_touching_memory(lib, buffers_over_cap):
    cap = lib.kl_device_alloc_hard_cap()
    assert cap == 2 << 30
    x = np.ones(4, dtype=np.float32)
    # "one": each buffer alone exceeds the cap; "two": each fits, both do not.
    n = cap // 4 + 1 if buffers_over_cap == "one" else cap // 8 + 1
    assert lib.kl_saxpy_host(n, 1.0, kl.float_ptr(x), kl.float_ptr(x)) == kl.KL_ERR_RESOURCE_LIMIT
    assert b"safety cap" in lib.kl_last_error_message()
    assert lib.kl_device_live_bytes() == 0


def test_no_device_memory_leaks_between_calls(lib):
    x = kl.fill_uniform(5, 4096, -1.0, 1.0)
    y = kl.fill_uniform(6, 4096, -1.0, 1.0)
    for _ in range(3):
        run_saxpy(lib, 0.5, x, y)
    assert lib.kl_device_live_bytes() == 0
