"""Python tests for kernellens layer normalization (via ctypes).

References are independent of the CUDA code: numpy float64 with its own
implementation of the bound from docs/layernorm.md, PyTorch float64 on the
GPU as a second reference, and float32 torch layer_norm for special-value
semantics.
"""

import numpy as np
import pytest

import kl

IMPLS = kl.LAYERNORM_IMPLS
SHAPES = [(7, 1), (7, 2), (7, 32), (7, 33), (7, 257), (7, 1000), (7, 1024), (7, 1025), (5, 4097),
          (5, 8192), (5, 8193), (2, 65536), (1, 1 << 20)]
SENTINEL = np.float32(-12345.0)
EPS = 1e-5


def run(lib, x, gamma=None, beta=None, eps=EPS, impl=kl.LAYERNORM_AUTO) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.full_like(x, SENTINEL)
    rows, cols = x.shape
    status = lib.kl_layernorm_host(rows, cols, kl.float_ptr(x), kl.float_ptr(gamma), kl.float_ptr(beta), eps,
                                   kl.float_ptr(y), impl)
    assert status == kl.KL_OK, lib.kl_last_error_message()
    return y


def supported(impl: int, cols: int) -> bool:
    if impl == kl.LAYERNORM_FUSED_WARP:
        return cols <= kl.LAYERNORM_WARP_MAX_COLS
    if impl == kl.LAYERNORM_FUSED_SMEM:
        return cols <= kl.LAYERNORM_SMEM_MAX_COLS
    return True


def affine_params(cols: int, seed: int):
    return kl.fill_uniform(seed, cols, 0.5, 1.5), kl.fill_uniform(seed + 1, cols, -1.0, 1.0)


# Only valid (cols, implementation) pairs are generated; the rejection of
# larger rows by FUSED_WARP / FUSED_SMEM is tested in
# test_layernorm_rejects_bad_input...
CASES = [(r, c, name) for (r, c) in SHAPES for name in IMPLS if supported(IMPLS[name], c)]


def cols_cases(cols_list):
    return [(c, name) for c in cols_list for name in IMPLS if supported(IMPLS[name], c)]


@pytest.mark.parametrize("rows,cols,impl_name", CASES)
@pytest.mark.parametrize("lo,hi,affine", [(-10.0, 10.0, True), (-10.0, 10.0, False), (999.0, 1001.0, True),
                                          (-1e-3, 1e-3, True)])
def test_layernorm_matches_float64_reference(lib, rows, cols, impl_name, lo, hi, affine):
    impl = IMPLS[impl_name]
    x = kl.fill_uniform(rows * cols + int(lo), rows * cols, lo, hi).reshape(rows, cols)
    g, b = affine_params(cols, 70 + cols) if affine else (None, None)
    kl.layernorm_check(x, g, b, EPS, run(lib, x, g, b, EPS, impl), impl)


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-5, 1.0, 1000.0])
@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_layernorm_eps_values(lib, eps, impl_name):
    impl = IMPLS[impl_name]
    x = kl.fill_uniform(5, 9 * 777, -3.0, 3.0).reshape(9, 777)
    kl.layernorm_check(x, None, None, eps, run(lib, x, None, None, eps, impl), impl)


def test_layernorm_matches_torch_float64(lib):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    for rows, cols in [(33, 1000), (33, 3001)]:
        x = kl.fill_uniform(42 + cols, rows * cols, -20.0, 20.0).reshape(rows, cols)
        g, b = affine_params(cols, 44 + cols)
        want = torch.nn.functional.layer_norm(
            torch.from_numpy(x).cuda().double(), (cols,), torch.from_numpy(g).cuda().double(),
            torch.from_numpy(b).cuda().double(), EPS).cpu().numpy()
        for impl in IMPLS.values():
            if not supported(impl, cols):
                continue
            y = run(lib, x, g, b, EPS, impl)
            allowed = kl.layernorm_allowed(x, g, b, EPS, impl)
            assert np.all(np.abs(y - want) <= allowed)


def special_matrix(cols: int):
    x = kl.fill_uniform(500 + cols, 6 * cols, -5.0, 5.0).reshape(6, cols)
    x[0, cols // 2] = np.nan   # NaN row
    x[1, 0] = np.inf           # +inf -> NaN row
    x[2, -1] = -np.inf         # -inf -> NaN row
    g, b = affine_params(cols, 600 + cols)
    g[0] = np.nan              # NaN column
    if cols > 1:
        b[1] = np.inf          # +inf column (finite rows)
    return x, g, b


@pytest.mark.parametrize("cols,impl_name", cols_cases([1, 33, 257, 1024, 3000]))
def test_layernorm_special_values_match_torch_semantics(lib, cols, impl_name):
    torch = pytest.importorskip("torch")
    impl = IMPLS[impl_name]
    x, g, b = special_matrix(cols)
    y = run(lib, x, g, b, EPS, impl)
    kl.layernorm_check(x, g, b, EPS, y, impl)
    t = torch.nn.functional.layer_norm(torch.from_numpy(x), (cols,), torch.from_numpy(g), torch.from_numpy(b),
                                       EPS).numpy()
    np.testing.assert_array_equal(np.isnan(y), np.isnan(t))
    np.testing.assert_array_equal(np.isinf(y), np.isinf(t))


@pytest.mark.parametrize("cols,impl_name", cols_cases([1, 33, 777, 1000, 1024, 1025, 4096, 10007]))
@pytest.mark.parametrize("c", [3.25, 0.1, 1 / 3, 7.77, -1e-3, 1e36, -2.5e30, float(np.finfo(np.float32).max), 1e-40])
def test_layernorm_constant_rows(lib, cols, impl_name, c):
    """Pivot-shifted mean (docs/layernorm.md): every x_j - x_0 is 0, so the mean
    is exactly c for any c (0.1 was a counterexample for the unshifted mean),
    x - mean = +0 and y is beta bit for bit (+0 without affine); eps = 0 gives
    NaN. PyTorch's CPU kernel gives the same; its CUDA kernel was observed to
    differ for some lengths (docs/layernorm.md)."""
    impl = IMPLS[impl_name]
    x = np.full((5, cols), c, dtype=np.float32)
    g, b = affine_params(cols, 800 + cols)
    assert kl.same_float(run(lib, x, g, b, EPS, impl), np.broadcast_to(b, x.shape)).all()
    assert kl.same_float(run(lib, x, None, None, EPS, impl), np.zeros_like(x)).all()
    assert np.isnan(run(lib, x, None, None, 0.0, impl)).all()


EXTREME_CASES = [
    # (label, x generator, gamma kind, eps) - inside the validity domain
    ("1e18*U[-1,1]", lambda n, s: kl.fill_uniform(s, n, -1.0, 1.0) * np.float32(1e18), "affine", 1e-5),
    ("1e20*U[0.999,1.001]", lambda n, s: kl.fill_uniform(s, n, 0.999, 1.001) * np.float32(1e20), "affine", 1e-5),
    ("subnormal x", lambda n, s: np.ldexp(np.floor(kl.fill_uniform(s, n, 0.0, 1000.0)), -149).astype(np.float32),
     None, 1e-5),
    ("subnormal x, eps 1e-12", lambda n, s: np.ldexp(np.floor(kl.fill_uniform(s, n, 0.0, 1000.0)),
                                                     -149).astype(np.float32), None, 1e-12),
    ("subnormal t, gamma 1e6", lambda n, s: kl.fill_uniform(s, n, -1.0, 1.0) * np.float32(1e-37), "big", 1e4),
]


@pytest.mark.parametrize("cols,impl_name", cols_cases([2, 33, 128]))
@pytest.mark.parametrize("case", range(len(EXTREME_CASES)))
def test_layernorm_extreme_in_domain(lib, cols, impl_name, case):
    """Far from U[-10, 10] but inside the bound's validity domain: huge and
    offset magnitudes, subnormal inputs (the mean's 2^-150 term), subnormal
    outputs scaled by a large gamma (the (1 + |gamma|) 2^-147 term)."""
    label, gen, gk, eps = EXTREME_CASES[case]
    rows = 16
    x = gen(rows * cols, 2200 + cols).reshape(rows, cols).astype(np.float32)
    if label.startswith("subnormal x"):
        x[:, 0] = np.float32(2.0**-149)
    g, b = affine_params(cols, 2300 + cols)
    if gk is None:
        g = b = None
    elif gk == "big":
        g = np.full(cols, 1e6, dtype=np.float32)
    kl.layernorm_check(x, g, b, eps, run(lib, x, g, b, eps, IMPLS[impl_name]), IMPLS[impl_name])


@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_layernorm_outside_domain_matches_torch(lib, impl_name):
    """Outside the validity domain no bound is claimed (docs/layernorm.md); for
    these documented examples KernelLens returns what PyTorch (CPU and CUDA)
    returns: d^2 underflows to 0 with eps = 0 -> +-inf; d^2 overflows -> +-0."""
    torch = pytest.importorskip("torch")
    for x, eps in ((np.array([[1e-30, -1e-30]], np.float32), 0.0), (np.array([[1e20, -1e20]], np.float32), 1e-5)):
        y = run(lib, x, None, None, eps, IMPLS[impl_name])
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            t = torch.nn.functional.layer_norm(torch.from_numpy(x).to(dev), (2,), None, None, eps).cpu().numpy()
            np.testing.assert_array_equal(y, t)


def test_layernorm_max_rows(lib):
    rows, cols = kl.LAYERNORM_MAX_ROWS, 2
    x = kl.fill_uniform(2400, rows * cols, -3.0, 3.0).reshape(rows, cols)
    for impl in IMPLS.values():
        kl.layernorm_check(x, None, None, EPS, run(lib, x, None, None, EPS, impl), impl)


@pytest.mark.parametrize("cols", [33, 257, 1000, 1024, 4097])
def test_layernorm_outlier_positions(lib, cols):
    positions = [j for j in (0, 31, 32, 63, 255, 256, cols - 1) if j < cols]
    x = kl.fill_uniform(1100 + cols, len(positions) * cols, -1.0, 1.0).reshape(len(positions), cols)
    for r, j in enumerate(positions):
        x[r, j] = 1000.0
    for impl in IMPLS.values():
        if supported(impl, cols):
            kl.layernorm_check(x, None, None, EPS, run(lib, x, None, None, EPS, impl), impl)


@pytest.mark.parametrize("cols,impl_name", cols_cases([2, 64]))
def test_layernorm_scale_path_rounding(lib, cols, impl_name):
    """Rows (+c, -c, ...) with eps = 0 and <= 2 elements per thread: every
    step but the scale path is exact, so y = fl(c * fl(1/c)) is +-1 or
    +-(1 - 2^-24) exactly (docs/layernorm.md; same test as the C++ one)."""
    rows = 4096
    m = kl.fill_uniform(1800 + cols, rows, 1.0, 2.0).astype(np.float64)
    e = np.floor(kl.fill_uniform(1900 + cols, rows, -20.0, 20.0)).astype(np.int64)
    c = np.ldexp(m, e).astype(np.float32)
    x = np.repeat(c[:, None], cols, axis=1)
    x[:, 1::2] *= -1
    y = run(lib, x, None, None, 0.0, IMPLS[impl_name])
    below_one = np.nextafter(np.float32(1), np.float32(0))
    assert np.all((np.abs(y) == 1) | (np.abs(y) == below_one))
    assert np.array_equal(np.signbit(y), np.signbit(x))


@pytest.mark.parametrize("cols,expected", [(1, kl.LAYERNORM_FUSED_WARP), (1024, kl.LAYERNORM_FUSED_WARP),
                                           (1025, kl.LAYERNORM_FUSED_SMEM), (8192, kl.LAYERNORM_FUSED_SMEM),
                                           (8193, kl.LAYERNORM_FUSED_BLOCK), (65536, kl.LAYERNORM_FUSED_BLOCK)])
def test_layernorm_auto_is_bitwise_the_resolved_impl(lib, cols, expected):
    assert kl.layernorm_resolve(cols, kl.LAYERNORM_AUTO) == expected
    rows = 3
    x = kl.fill_uniform(77 + cols, rows * cols, -8.0, 8.0).reshape(rows, cols)
    g, b = affine_params(cols, 78 + cols)
    assert kl.same_float(run(lib, x, g, b, EPS, kl.LAYERNORM_AUTO), run(lib, x, g, b, EPS, expected)).all()


def test_layernorm_rejects_bad_input_without_writing(lib):
    x = np.ones((4, 4), dtype=np.float32)
    y = np.full_like(x, SENTINEL)
    g, b = np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    xp, yp, gp, bp = kl.float_ptr(x), kl.float_ptr(y), kl.float_ptr(g), kl.float_ptr(b)
    host = lib.kl_layernorm_host
    assert host(-1, 4, xp, None, None, EPS, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, xp, None, None, EPS, yp, 9) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, xp, None, None, EPS, yp, -1) == kl.KL_ERR_INVALID_ARGUMENT
    for eps in (-1e-5, float("nan"), float("inf"), float("-inf")):
        assert host(4, 4, xp, None, None, eps, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, xp, gp, None, EPS, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, xp, None, bp, EPS, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(4, 4, None, None, None, EPS, yp, 0) == kl.KL_ERR_INVALID_ARGUMENT
    assert host(1, kl.LAYERNORM_MAX_COLS + 1, xp, None, None, EPS, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert host(kl.LAYERNORM_MAX_ROWS + 1, 1, xp, None, None, EPS, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert host(65, 1 << 20, xp, None, None, EPS, yp, 0) == kl.KL_ERR_UNSUPPORTED
    assert b"supported domain" in lib.kl_last_error_message()
    assert host(1, 1025, xp, None, None, EPS, yp, kl.LAYERNORM_FUSED_WARP) == kl.KL_ERR_UNSUPPORTED
    assert b"cols <= 1024" in lib.kl_last_error_message()
    assert host(1, 8193, xp, None, None, EPS, yp, kl.LAYERNORM_FUSED_SMEM) == kl.KL_ERR_UNSUPPORTED
    assert b"cols <= 8192" in lib.kl_last_error_message()
    assert host(4, 4, xp, None, None, EPS, yp, 5) == kl.KL_ERR_INVALID_ARGUMENT
    assert np.all(y == SENTINEL)
    assert lib.kl_device_live_bytes() == 0
    assert host(0, 4, None, None, None, EPS, None, 0) == kl.KL_OK
    assert lib.kl_layernorm_supported(64, 1 << 20, 0.0, 0) == kl.KL_OK
    assert lib.kl_layernorm_supported(65, 1 << 20, 0.0, 0) == kl.KL_ERR_UNSUPPORTED
    assert lib.kl_layernorm_supported(1, 1, -1.0, 0) == kl.KL_ERR_INVALID_ARGUMENT


def test_layernorm_workspace_bytes(lib):
    assert kl.layernorm_workspace_bytes(lib, 10, 2048, kl.LAYERNORM_MULTIPASS) == 80
    assert kl.layernorm_workspace_bytes(lib, 10, 2048, kl.LAYERNORM_AUTO) == 0
    assert kl.layernorm_workspace_bytes(lib, 10, 2048, kl.LAYERNORM_FUSED_BLOCK) == 0
    assert kl.layernorm_workspace_bytes(lib, 10, 2048, kl.LAYERNORM_FUSED_SMEM) == 0
    with pytest.raises(ValueError):
        kl.layernorm_workspace_bytes(lib, 1, 8193, kl.LAYERNORM_FUSED_SMEM)
    assert kl.layernorm_workspace_bytes(lib, 10, 100, kl.LAYERNORM_FUSED_WARP) == 0
    with pytest.raises(ValueError):
        kl.layernorm_workspace_bytes(lib, 1, 1025, kl.LAYERNORM_FUSED_WARP)


def test_layernorm_device_api_on_torch_tensors(lib):
    """kl_layernorm_device on PyTorch-owned memory and stream (the path the
    in-process benchmark uses) gives bit-identical results to the host API."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    stream = torch.cuda.Stream()
    for rows, cols in [(37, 100), (9, 2048), (100, 3000)]:
        x = kl.fill_uniform(2000 + cols, rows * cols, -8.0, 8.0).reshape(rows, cols)
        g, b = affine_params(cols, 2100 + cols)
        for impl in IMPLS.values():
            if not supported(impl, cols):
                continue
            xt, gt, bt = (torch.from_numpy(a).cuda() for a in (x, g, b))
            yt = torch.full_like(xt, float("nan"))
            ws = torch.empty(max(1, kl.layernorm_workspace_bytes(lib, rows, cols, impl) // 4), device="cuda")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                status = lib.kl_layernorm_device(rows, cols, xt.data_ptr(), gt.data_ptr(), bt.data_ptr(), EPS,
                                                 yt.data_ptr(), impl, ws.data_ptr(), stream.cuda_stream)
            assert status == kl.KL_OK, lib.kl_last_error_message()
            stream.synchronize()
            assert kl.same_float(yt.cpu().numpy(), run(lib, x, g, b, EPS, impl)).all()
    t = torch.zeros(512, device="cuda")  # 2048 bytes; offsets below are bytes
    p = t.data_ptr()
    dev = lib.kl_layernorm_device
    assert dev(4, 16, p, None, None, EPS, p, kl.LAYERNORM_FUSED_BLOCK, None, None) == kl.KL_ERR_INVALID_ARGUMENT
    assert dev(4, 16, p, p + 512, p + 576, EPS, p + 600, kl.LAYERNORM_FUSED_BLOCK, None,
               None) == kl.KL_ERR_INVALID_ARGUMENT  # y overlaps beta
    assert dev(4, 16, p, None, None, EPS, p + 256, kl.LAYERNORM_MULTIPASS, p + 280,
               None) == kl.KL_ERR_INVALID_ARGUMENT  # workspace overlaps y
    assert dev(4, 16, p, None, None, EPS, p + 258, kl.LAYERNORM_FUSED_BLOCK, None,
               None) == kl.KL_ERR_INVALID_ARGUMENT  # misaligned y
    # x = p+800.., gamma = p+512.., beta = p+576.., y = p+256.., workspace (32 B)
    for args, what in (((p + 802, p + 512, p + 576, p + 256, None), "misaligned x"),
                       ((p + 800, p + 514, p + 576, p + 256, None), "misaligned gamma"),
                       ((p + 800, p + 512, p + 578, p + 256, None), "misaligned beta"),
                       ((p + 800, p + 512, p + 576, p + 256, p + 642), "misaligned workspace"),
                       ((p + 800, p + 512, p + 576, p + 256, p + 1000), "workspace overlaps x")):
        xx, gg, bb, yy, ww = args
        impl = kl.LAYERNORM_MULTIPASS if ww is not None else kl.LAYERNORM_FUSED_BLOCK
        assert dev(4, 16, xx, gg, bb, EPS, yy, impl, ww, None) == kl.KL_ERR_INVALID_ARGUMENT, what
    assert dev(4, 16, p, None, None, EPS, p + 256, -1, None, None) == kl.KL_ERR_INVALID_ARGUMENT
    torch.cuda.synchronize()


def test_layernorm_inplace_host(lib):
    x = kl.fill_uniform(3, 5 * 777, -3.0, 3.0).reshape(5, 777)
    g, b = affine_params(777, 4)
    want = run(lib, x, g, b, EPS, kl.LAYERNORM_FUSED_BLOCK)
    xi = x.copy()
    assert lib.kl_layernorm_host(5, 777, kl.float_ptr(xi), kl.float_ptr(g), kl.float_ptr(b), EPS, kl.float_ptr(xi),
                                 kl.LAYERNORM_FUSED_BLOCK) == kl.KL_OK
    assert kl.same_float(xi, want).all()


@pytest.mark.parametrize("cols", [1, 256, 257, 1025, 4096, 8192])
def test_layernorm_smem_bitwise_equals_block(lib, cols):
    """FUSED_SMEM runs FUSED_BLOCK's arithmetic in the same order (docs/layernorm.md)."""
    x = kl.fill_uniform(3000 + cols, 9 * cols, -50.0, 50.0).reshape(9, cols)
    x[4] += 1000.0
    x[0, -1] = np.nan
    g, b = affine_params(cols, 3100 + cols)
    for gg, bb in ((g, b), (None, None)):
        assert kl.same_float(run(lib, x, gg, bb, EPS, kl.LAYERNORM_FUSED_SMEM),
                             run(lib, x, gg, bb, EPS, kl.LAYERNORM_FUSED_BLOCK)).all()


def test_layernorm_deterministic(lib):
    x = kl.fill_uniform(9, 64 * 3001, -20.0, 20.0).reshape(64, 3001)
    for impl in (kl.LAYERNORM_MULTIPASS, kl.LAYERNORM_FUSED_BLOCK, kl.LAYERNORM_FUSED_SMEM):
        assert kl.same_float(run(lib, x, None, None, EPS, impl), run(lib, x, None, None, EPS, impl)).all()
