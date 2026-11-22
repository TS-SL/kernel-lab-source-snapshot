"""Python tests for kernellens softmax (via ctypes).

References are independent of the CUDA code: numpy float64 (with its own
implementation of the bound from docs/softmax.md) and PyTorch (float64 on the
GPU as a second reference; float32 torch.softmax for special-value semantics).
"""

import numpy as np
import pytest

import kl

IMPLS = kl.SOFTMAX_IMPLS
SHAPES = [(7, 1), (7, 32), (7, 33), (7, 257), (7, 1024), (7, 1025), (5, 4097), (2, 65536), (1, 1 << 20)]
SENTINEL = np.float32(-12345.0)


def run(lib, x: np.ndarray, impl: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.full_like(x, SENTINEL)
    rows, cols = x.shape
    status = lib.kl_softmax_host(rows, cols, kl.float_ptr(x), kl.float_ptr(y), impl)
    assert status == kl.KL_OK, lib.kl_last_error_message()
    return y


def supported(impl: int, cols: int) -> bool:
    return impl != kl.SOFTMAX_FUSED_WARP or cols <= kl.SOFTMAX_WARP_MAX_COLS


# Only valid (shape, implementation) pairs are generated; FUSED_WARP's
# rejection of cols > 1024 is tested in test_softmax_rejects_bad_input...
CASES = [(r, c, name) for (r, c) in SHAPES for name in IMPLS if supported(IMPLS[name], c)]
SPECIAL_COLS = [1, 31, 33, 1024, 3000]
SPECIAL_CASES = [(c, name) for c in SPECIAL_COLS for name in IMPLS if supported(IMPLS[name], c)]


@pytest.mark.parametrize("rows,cols,impl_name", CASES)
@pytest.mark.parametrize("lo,hi", [(-10.0, 10.0), (1000.0, 1010.0), (-100.0, 100.0)])
def test_softmax_matches_float64_reference(lib, rows, cols, impl_name, lo, hi):
    impl = IMPLS[impl_name]
    x = kl.fill_uniform(rows * cols + int(lo), rows * cols, lo, hi).reshape(rows, cols)
    kl.softmax_check(x, run(lib, x, impl), impl)


@pytest.mark.parametrize("rows,cols,impl_name", CASES)
def test_softmax_grid_inputs_tight_bound(lib, rows, cols, impl_name):
    impl = IMPLS[impl_name]
    # Multiples of 2^-10 in [-16, 16]: all input differences are exact floats,
    # so the bound without the subtraction terms applies.
    x = np.round(kl.fill_uniform(900 + cols, rows * cols, -16.0, 16.0) * 1024) / 1024
    x = x.astype(np.float32).reshape(rows, cols)
    assert np.all(x * 1024 == np.round(x * 1024)) and np.abs(x).max() <= 128
    kl.softmax_check(x, run(lib, x, impl), impl, exact_differences=True)


@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_softmax_exp_accuracy_two_element_rows(lib, impl_name):
    impl = IMPLS[impl_name]
    t = np.arange(87 * 64, dtype=np.float32) / 64  # exact grid values
    x = np.stack([np.zeros_like(t), -t], axis=1)
    x[1::2] = x[1::2, ::-1]  # alternate where the maximum is
    kl.softmax_check(x, run(lib, x, impl), impl, exact_differences=True)


def test_softmax_matches_torch_float64(lib):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    rows, cols = 33, 3001
    x = kl.fill_uniform(42, rows * cols, -20.0, 20.0).reshape(rows, cols)
    want = torch.softmax(torch.from_numpy(x).cuda().double(), dim=-1).cpu().numpy()
    for impl in IMPLS.values():
        if not supported(impl, cols):
            continue
        y = run(lib, x, impl)
        allowed = 1.01 * kl.softmax_bound(x, impl) * want + kl.ABS_SLACK
        assert np.all(np.abs(y - want) <= allowed)


def special_matrix(cols: int) -> np.ndarray:
    x = kl.fill_uniform(500 + cols, 6 * cols, -5.0, 5.0).reshape(6, cols)
    x[0, ::3] = -np.inf          # masked entries
    x[1, :] = -np.inf            # all masked -> NaN
    x[2, cols // 2] = np.nan     # NaN -> NaN row
    x[3, -1] = np.inf            # +inf -> NaN row
    x[4, :] = -np.inf            # only the last entry is finite
    x[4, -1] = 2.0
    x[5, 0] = -np.inf
    return x


@pytest.mark.parametrize("cols,impl_name", SPECIAL_CASES)
def test_softmax_special_values_match_torch_semantics(lib, cols, impl_name):
    torch = pytest.importorskip("torch")
    impl = IMPLS[impl_name]
    x = special_matrix(cols)
    y = run(lib, x, impl)
    kl.softmax_check(x, y, impl)
    # Same NaN pattern and same zeros for masked entries as PyTorch.
    t = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    np.testing.assert_array_equal(np.isnan(y), np.isnan(t))
    np.testing.assert_array_equal(y[~np.isnan(y)] == 0, t[~np.isnan(t)] == 0)


@pytest.mark.parametrize("rows,cols,expected", [
    (9, 1, kl.SOFTMAX_FUSED_WARP), (9, 1024, kl.SOFTMAX_FUSED_WARP),
    (9, 2048, kl.SOFTMAX_MULTIPASS), (63, 1025, kl.SOFTMAX_MULTIPASS), (64, 1025, kl.SOFTMAX_FUSED_ONLINE)])
def test_softmax_auto_is_bitwise_the_resolved_impl(lib, rows, cols, expected):
    assert kl.softmax_resolve(rows, cols, kl.SOFTMAX_AUTO) == expected
    x = kl.fill_uniform(77 + rows, rows * cols, -8.0, 8.0).reshape(rows, cols)
    assert kl.same_float(run(lib, x, kl.SOFTMAX_AUTO), run(lib, x, expected)).all()


def test_softmax_rejects_bad_input_without_writing(lib):
    x = np.ones((4, 4), dtype=np.float32)
    y = np.full_like(x, SENTINEL)
    xp, yp = kl.float_ptr(x), kl.float_ptr(y)
    assert lib.kl_softmax_host(-1, 4, xp, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert lib.kl_softmax_host(4, 4, xp, yp, 9) == kl.KL_ERR_INVALID_ARGUMENT
    assert lib.kl_softmax_host(4, 4, None, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert lib.kl_softmax_host(1, kl.SOFTMAX_MAX_COLS + 1, xp, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert lib.kl_softmax_host(kl.SOFTMAX_MAX_ROWS + 1, 1, xp, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert lib.kl_softmax_host(65, 1 << 20, xp, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert b"supported domain" in lib.kl_last_error_message()
    assert lib.kl_softmax_host(1, 1025, xp, yp, kl.SOFTMAX_FUSED_WARP) == kl.KL_ERR_UNSUPPORTED
    assert b"cols <= 1024" in lib.kl_last_error_message()
    assert np.all(y == SENTINEL)
    assert lib.kl_device_live_bytes() == 0
    assert lib.kl_softmax_host(0, 4, None, None, 0) == kl.KL_OK
    assert lib.kl_softmax_supported(64, 1 << 20, 0) == kl.KL_OK
    assert lib.kl_softmax_supported(65, 1 << 20, 0) == kl.KL_ERR_UNSUPPORTED


@pytest.mark.parametrize("cols", [33, 257, 1024, 4097, 70000])
def test_softmax_max_spike_positions(lib, cols):
    # A reduction that dropped the spike's partial would compute exp(~99) = inf.
    positions = [j for j in (0, 31, 32, 63, 255, 256, cols - 1) if j < cols]
    x = kl.fill_uniform(1100 + cols, len(positions) * cols, -1.0, 1.0).reshape(len(positions), cols)
    for r, j in enumerate(positions):
        x[r, j] = 100.0
    for name, impl in IMPLS.items():
        if supported(impl, cols):
            kl.softmax_check(x, run(lib, x, impl), impl)


def test_softmax_extreme_finite_values(lib):
    f = np.finfo(np.float32)
    row = np.array([f.max, -f.max, 0.0, -0.0, f.smallest_subnormal, -f.smallest_subnormal, 1.0, -1.0],
                   dtype=np.float32)
    x = np.stack([row, row, row])
    x[1, 0], x[2, 0] = 0.25, -f.max
    for impl in IMPLS.values():
        y = run(lib, x, impl)
        kl.softmax_check(x, y, impl)
        assert y[0, 0] == 1.0 and np.all(y[0, 1:] == 0) and not np.signbit(y[0, 1:]).any()


def test_softmax_workspace_bytes(lib):
    assert kl.softmax_workspace_bytes(lib, 10, 2048, kl.SOFTMAX_MULTIPASS) == 80
    assert kl.softmax_workspace_bytes(lib, 10, 2048, kl.SOFTMAX_AUTO) == 80   # few rows -> multipass
    assert kl.softmax_workspace_bytes(lib, 100, 2048, kl.SOFTMAX_AUTO) == 0   # online
    with pytest.raises(ValueError):
        kl.softmax_workspace_bytes(lib, 1, 1025, kl.SOFTMAX_FUSED_WARP)


def test_softmax_device_api_on_torch_tensors(lib):
    """kl_softmax_device on PyTorch-owned memory and stream (the path the
    in-process benchmark uses) gives bit-identical results to the host API."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    stream = torch.cuda.Stream()
    for rows, cols in [(37, 100), (9, 2048), (100, 3000)]:
        x = kl.fill_uniform(2000 + cols, rows * cols, -8.0, 8.0).reshape(rows, cols)
        for impl in IMPLS.values():
            if not supported(impl, cols):
                continue
            xt = torch.from_numpy(x).cuda()
            yt = torch.full_like(xt, float("nan"))
            ws = torch.empty(max(1, kl.softmax_workspace_bytes(lib, rows, cols, impl) // 4), device="cuda")
            # xt/yt were written on the default stream; order the side stream
            # after them (it is non-blocking with respect to the default stream).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                status = lib.kl_softmax_device(rows, cols, xt.data_ptr(), yt.data_ptr(), impl,
                                               ws.data_ptr(), stream.cuda_stream)
            assert status == kl.KL_OK, lib.kl_last_error_message()
            stream.synchronize()
            assert kl.same_float(yt.cpu().numpy(), run(lib, x, impl)).all()
    t = torch.zeros(256, device="cuda")
    p = t.data_ptr()
    assert lib.kl_softmax_device(4, 16, p, p, kl.SOFTMAX_FUSED_ONLINE, None,
                                 None) == kl.KL_ERR_INVALID_ARGUMENT  # x == y overlap
    assert lib.kl_softmax_device(4, 16, p, p + 256, kl.SOFTMAX_MULTIPASS, p + 280,
                                 None) == kl.KL_ERR_INVALID_ARGUMENT  # workspace overlaps y
    assert lib.kl_softmax_device(4, 16, p, p + 258, kl.SOFTMAX_FUSED_ONLINE, None,
                                 None) == kl.KL_ERR_INVALID_ARGUMENT  # misaligned y
    assert lib.kl_softmax_device(4, 16, p, p + 256, -1, None, None) == kl.KL_ERR_INVALID_ARGUMENT
    torch.cuda.synchronize()


def test_softmax_inplace_host(lib):
    x = kl.fill_uniform(3, 5 * 777, -3.0, 3.0).reshape(5, 777)
    want = run(lib, x, kl.SOFTMAX_FUSED_ONLINE)
    xi = x.copy()
    assert lib.kl_softmax_host(5, 777, kl.float_ptr(xi), kl.float_ptr(xi), kl.SOFTMAX_FUSED_ONLINE) == kl.KL_OK
    assert kl.same_float(xi, want).all()
