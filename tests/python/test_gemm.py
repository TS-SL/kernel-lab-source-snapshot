"""Python tests for kernellens GEMM (via ctypes).

References are independent of the CUDA code: numpy float64 with its own
implementation of the bound from docs/gemm.md, and PyTorch float64 matmul
on the GPU.
"""

import numpy as np
import pytest

import kl

IMPLS = kl.GEMM_IMPLS
SENTINEL = np.float32(-12345.0)
SHAPES = [(1, 1, 1), (15, 17, 33), (64, 64, 64), (65, 63, 129), (100, 1, 300), (1, 100, 300), (129, 130, 16),
          (300, 200, 1000)]


def run(lib, a: np.ndarray, b: np.ndarray, impl: int) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    m, k = a.shape
    n = b.shape[1]
    c = np.full((m, n), SENTINEL, dtype=np.float32)
    status = lib.kl_gemm_host(m, n, k, kl.float_ptr(a), kl.float_ptr(b), kl.float_ptr(c), impl)
    assert status == kl.KL_OK, lib.kl_last_error_message()
    return c


def matrices(m, n, k, seed, lo=-1.0, hi=1.0):
    a = kl.fill_uniform(seed, m * k, lo, hi).reshape(m, k)
    b = kl.fill_uniform(seed + 1, k * n, lo, hi).reshape(k, n)
    return a, b


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_gemm_matches_float64_reference(lib, m, n, k, impl_name):
    a, b = matrices(m, n, k, 10 + m + n + k)
    kl.gemm_check(a, b, run(lib, a, b, IMPLS[impl_name]))


@pytest.mark.parametrize("m,n,k", SHAPES + [(64, 64, 16384)])
@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_gemm_integer_inputs_exact(lib, m, n, k, impl_name):
    a, b = matrices(m, n, k, 20 + m + n + k, -8.0, 8.0)
    a, b = np.round(a), np.round(b)
    kl.gemm_check(a, b, run(lib, a, b, IMPLS[impl_name]), exact=True)


@pytest.mark.parametrize("m,n,k", [(77, 1000, 333), (512, 512, 512), (16, 16, 16384), (16384, 16, 16)])
def test_gemm_implementations_bitwise_identical(lib, m, n, k):
    a, b = matrices(m, n, k, 30 + m + k, -2.0, 2.0)
    want = run(lib, a, b, kl.GEMM_NAIVE)
    for impl in (kl.GEMM_TILED, kl.GEMM_REGBLOCK, kl.GEMM_AUTO):
        assert kl.same_float(run(lib, a, b, impl), want).all()


def test_gemm_matches_torch_float64(lib):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    a, b = matrices(300, 257, 1000, 40)
    want = (torch.from_numpy(a).cuda().double() @ torch.from_numpy(b).cuda().double()).cpu().numpy()
    s = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    k = a.shape[1]
    g = k * kl.U / (1 - k * kl.U)
    allowed = (g + k * 2.0**-52) * s + k * 2.0**-150 * (1 + g)
    for impl in IMPLS.values():
        assert np.all(np.abs(run(lib, a, b, impl) - want) <= allowed)


@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_gemm_special_values(lib, impl_name):
    a, b = matrices(9, 11, 13, 50, -2.0, 2.0)
    a[0, 3] = np.nan
    a[1, 2] = np.inf
    b[2, 5] = 0.0     # C[1][5]: inf * 0 = NaN
    b[4, 6] = -np.inf
    a[3, 4] = 0.0     # C[3][6]: 0 * -inf = NaN
    kl.gemm_check(a, b, run(lib, a, b, IMPLS[impl_name]))


@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_gemm_k_zero_gives_zeros(lib, impl_name):
    c = np.full((5, 7), SENTINEL, dtype=np.float32)
    assert lib.kl_gemm_host(5, 7, 0, None, None, kl.float_ptr(c), IMPLS[impl_name]) == kl.KL_OK
    assert kl.same_float(c, np.zeros_like(c)).all()


def test_gemm_rejects_bad_input_without_writing(lib):
    a = np.ones((4, 4), dtype=np.float32)
    c = np.full((4, 4), SENTINEL, dtype=np.float32)
    ap, cp = kl.float_ptr(a), kl.float_ptr(c)
    host = lib.kl_gemm_host
    assert host(-1, 4, 4, ap, ap, cp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, 4, ap, ap, cp, 9) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, 4, None, ap, cp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(kl.GEMM_MAX_DIM + 1, 1, 1, ap, ap, cp, 0) == kl.KL_ERR_UNSUPPORTED
    assert host(1 << 14, 1 << 13, 1, ap, ap, cp, 0) == kl.KL_ERR_UNSUPPORTED
    assert b"supported domain" in lib.kl_last_error_message()
    assert np.all(c == SENTINEL)
    assert lib.kl_device_live_bytes() == 0
    assert host(0, 4, 4, None, None, None, 0) == kl.KL_OK
    assert lib.kl_gemm_supported(1 << 14, 1 << 12, 1 << 12, 0) == kl.KL_OK


def test_gemm_device_api_on_torch_tensors(lib):
    """kl_gemm_device on PyTorch-owned memory and stream gives bit-identical
    results to the host API; overlap and alignment are refused."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    stream = torch.cuda.Stream()
    a, b = matrices(130, 70, 31, 60)
    at, bt = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    for impl in IMPLS.values():
        ct = torch.full((130, 70), float("nan"), device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        assert lib.kl_gemm_device(130, 70, 31, at.data_ptr(), bt.data_ptr(), ct.data_ptr(), impl,
                                  stream.cuda_stream) == kl.KL_OK
        stream.synchronize()
        assert kl.same_float(ct.cpu().numpy(), run(lib, a, b, impl)).all()
    t = torch.zeros(256, device="cuda")
    p = t.data_ptr()
    dev = lib.kl_gemm_device
    assert dev(4, 4, 4, p, p + 64, p + 40, 0, None) == kl.KL_ERR_INVALID_ARGUMENT   # C overlaps A
    assert dev(4, 4, 4, p, p + 64, p + 100, 0, None) == kl.KL_ERR_INVALID_ARGUMENT  # C overlaps B
    assert dev(4, 4, 4, p, p + 64, p + 514, 0, None) == kl.KL_ERR_INVALID_ARGUMENT  # misaligned C
    torch.cuda.synchronize()
