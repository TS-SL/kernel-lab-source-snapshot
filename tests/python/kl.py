"""ctypes binding for kernellens.dll plus independent Python reference helpers.

Nothing here reuses the C++ implementation: the SplitMix64 generator and the
ULP comparison are re-implemented with numpy so the Python tests act as an
independent check of the library.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char, c_float, c_int, c_int64, c_size_t, c_uint64
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIB = REPO_ROOT / "build" / "release" / "kernellens.dll"

KL_OK = 0
KL_ERR_INVALID_ARGUMENT = 1
KL_ERR_UNSUPPORTED = 2
KL_ERR_RESOURCE_LIMIT = 3
KL_ERR_CUDA = 4
KL_ERR_NO_DEVICE = 5
KL_ERR_INTERNAL = 6


class DeviceInfo(ctypes.Structure):
    # Must mirror kl_device_info in include/kernellens/kernellens.h exactly.
    _fields_ = [
        ("name", c_char * 256),
        ("compute_major", c_int),
        ("compute_minor", c_int),
        ("multiprocessor_count", c_int),
        ("max_threads_per_multiprocessor", c_int),
        ("kernel_exec_timeout", c_int),
        ("driver_version", c_int),
        ("runtime_version", c_int),
        ("total_global_mem", c_size_t),
        ("free_global_mem", c_size_t),
    ]


_FLOAT_P = POINTER(c_float)


def load_library(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """Load kernellens.dll from $KERNELLENS_LIB or build/release. Fails loudly."""
    lib_path = Path(path or os.environ.get("KERNELLENS_LIB") or DEFAULT_LIB)
    if not lib_path.is_file():
        raise FileNotFoundError(
            f"kernellens library not found at {lib_path}. Build first "
            "(cmake --preset release; cmake --build --preset release) or set KERNELLENS_LIB."
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.kl_status_string.restype = ctypes.c_char_p
    lib.kl_status_string.argtypes = [c_int]
    lib.kl_last_error_message.restype = ctypes.c_char_p
    lib.kl_last_error_message.argtypes = []
    lib.kl_get_device_info.restype = c_int
    lib.kl_get_device_info.argtypes = [POINTER(DeviceInfo)]
    lib.kl_device_alloc_hard_cap.restype = c_size_t
    lib.kl_device_alloc_hard_cap.argtypes = []
    lib.kl_device_live_bytes.restype = c_size_t
    lib.kl_device_live_bytes.argtypes = []
    lib.kl_saxpy_host.restype = c_int
    lib.kl_saxpy_host.argtypes = [c_int64, c_float, _FLOAT_P, _FLOAT_P]
    lib.kl_fill_uniform.restype = c_int
    lib.kl_fill_uniform.argtypes = [c_uint64, c_int64, c_float, c_float, _FLOAT_P]
    lib.kl_softmax_supported.restype = c_int
    lib.kl_softmax_supported.argtypes = [c_int64, c_int64, c_int]
    lib.kl_softmax_host.restype = c_int
    lib.kl_softmax_host.argtypes = [c_int64, c_int64, _FLOAT_P, _FLOAT_P, c_int]
    lib.kl_softmax_workspace_bytes.restype = c_int
    lib.kl_softmax_workspace_bytes.argtypes = [c_int64, c_int64, c_int, POINTER(c_size_t)]
    # Device pointers and the stream are passed as raw addresses (void*).
    lib.kl_softmax_device.restype = c_int
    lib.kl_softmax_device.argtypes = [c_int64, c_int64, ctypes.c_void_p, ctypes.c_void_p, c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
    lib.kl_layernorm_supported.restype = c_int
    lib.kl_layernorm_supported.argtypes = [c_int64, c_int64, c_float, c_int]
    lib.kl_layernorm_workspace_bytes.restype = c_int
    lib.kl_layernorm_workspace_bytes.argtypes = [c_int64, c_int64, c_int, POINTER(c_size_t)]
    lib.kl_layernorm_host.restype = c_int
    lib.kl_layernorm_host.argtypes = [c_int64, c_int64, _FLOAT_P, _FLOAT_P, _FLOAT_P, c_float, _FLOAT_P, c_int]
    lib.kl_layernorm_device.restype = c_int
    lib.kl_layernorm_device.argtypes = [c_int64, c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        c_float, ctypes.c_void_p, c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.kl_gemm_supported.restype = c_int
    lib.kl_gemm_supported.argtypes = [c_int64, c_int64, c_int64, c_int]
    lib.kl_gemm_host.restype = c_int
    lib.kl_gemm_host.argtypes = [c_int64, c_int64, c_int64, _FLOAT_P, _FLOAT_P, _FLOAT_P, c_int]
    lib.kl_gemm_device.restype = c_int
    lib.kl_gemm_device.argtypes = [c_int64, c_int64, c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   c_int, ctypes.c_void_p]
    return lib


def softmax_workspace_bytes(lib, rows: int, cols: int, impl: int) -> int:
    out = c_size_t(0)
    status = lib.kl_softmax_workspace_bytes(rows, cols, impl, ctypes.byref(out))
    if status != KL_OK:
        raise ValueError(f"kl_softmax_workspace_bytes failed: {lib.kl_last_error_message()!r}")
    return out.value


# --- softmax (docs/softmax.md) ----------------------------------------------

SOFTMAX_AUTO, SOFTMAX_MULTIPASS, SOFTMAX_FUSED_ONLINE, SOFTMAX_FUSED_WARP = 0, 1, 2, 3
SOFTMAX_IMPLS = {"auto": 0, "multipass": 1, "fused_online": 2, "fused_warp": 3}
SOFTMAX_MAX_COLS = 1 << 20
SOFTMAX_MAX_ROWS = 1 << 20
SOFTMAX_MAX_ELEMENTS = 1 << 26
SOFTMAX_WARP_MAX_COLS = 1024
U = 2.0**-24
FLT_MIN = float(np.finfo(np.float32).tiny)
# Absolute slack for subnormal results (docs/softmax.md): 4 * 2^-149.
ABS_SLACK = 2.0**-147


def softmax_reference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise softmax of a float32 matrix in float64 (rows, cols).

    Returns (y, nan_rows): rows that the definition makes NaN (any NaN, any
    +inf, or all -inf) are flagged and their y is NaN.
    """
    x64 = x.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        m = np.max(np.where(np.isnan(x64), -np.inf, x64), axis=1, keepdims=True)
        nan_rows = np.isnan(x64).any(axis=1) | np.isposinf(x64).any(axis=1) | (m[:, 0] == -np.inf)
        e = np.exp(x64 - np.where(np.isfinite(m), m, 0.0))
        y = e / e.sum(axis=1, keepdims=True)
    y[nan_rows] = np.nan
    return y, nan_rows


SOFTMAX_ONLINE_MIN_ROWS = 64


def softmax_resolve(rows: int, cols: int, impl: int) -> int:
    """AUTO dispatch rule as documented in docs/softmax.md (independent copy)."""
    if impl != SOFTMAX_AUTO:
        return impl
    if cols <= SOFTMAX_WARP_MAX_COLS:
        return SOFTMAX_FUSED_WARP
    return SOFTMAX_FUSED_ONLINE if rows >= SOFTMAX_ONLINE_MIN_ROWS else SOFTMAX_MULTIPASS


def softmax_bound(x: np.ndarray, impl: int, exact_differences: bool = False) -> np.ndarray:
    """Per-element relative error bound B_i from docs/softmax.md (no 1.01 factor)."""
    rows, cols = x.shape
    resolved = softmax_resolve(rows, cols, impl)
    threads = 32 if resolved == SOFTMAX_FUSED_WARP else 256
    k = int(np.ceil(cols / threads))
    # Combining with an identity partial is exact, so only
    # ceil(log2(min(cols, T))) tree levels combine real data.
    levels = int(np.ceil(np.log2(min(cols, threads))))
    online = resolved == SOFTMAX_FUSED_ONLINE
    x64 = x.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.isfinite(x64)
        m = np.max(np.where(finite, x64, -np.inf), axis=1, keepdims=True)
        d = np.where(finite, m - x64, 0.0)
        e = np.where(finite, np.exp(-d), 0.0)
        # exp-weighted mean of d (what the derivation needs; <= max d)
        Dw = (e * d).sum(axis=1, keepdims=True) / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)
    base = np.zeros_like(d) if exact_differences else d + Dw
    const = 6 * (k - 1 + levels) + 9 if online else k + levels + 8
    return U * (base + const)


def softmax_check(x: np.ndarray, y: np.ndarray, impl: int, exact_differences: bool = False):
    """Asserts y satisfies the derived bound; returns worst error / allowed."""
    ref, nan_rows = softmax_reference(x)
    assert np.isnan(y[nan_rows]).all(), "rows the definition makes NaN must be NaN"
    ok = ~nan_rows
    allowed = 1.01 * softmax_bound(x, impl, exact_differences)[ok] * ref[ok] + ABS_SLACK
    err = np.abs(y[ok].astype(np.float64) - ref[ok])
    assert np.all(np.isfinite(y[ok])) and np.all(y[ok] >= 0) and np.all(y[ok] <= 1)
    ratio = float((err / allowed).max()) if err.size else 0.0
    assert ratio <= 1.0, f"error {ratio:.3f} x the derived bound"
    return ratio


# --- layer normalization (docs/layernorm.md) --------------------------------

LAYERNORM_AUTO, LAYERNORM_MULTIPASS, LAYERNORM_FUSED_BLOCK, LAYERNORM_FUSED_WARP, LAYERNORM_FUSED_SMEM = 0, 1, 2, 3, 4
LAYERNORM_IMPLS = {"auto": 0, "multipass": 1, "fused_block": 2, "fused_smem": 4, "fused_warp": 3}
LAYERNORM_MAX_COLS = 1 << 20
LAYERNORM_MAX_ROWS = 1 << 20
LAYERNORM_MAX_ELEMENTS = 1 << 26
LAYERNORM_WARP_MAX_COLS = 1024
LAYERNORM_SMEM_MAX_COLS = 8192


def layernorm_workspace_bytes(lib, rows: int, cols: int, impl: int) -> int:
    out = c_size_t(0)
    status = lib.kl_layernorm_workspace_bytes(rows, cols, impl, ctypes.byref(out))
    if status != KL_OK:
        raise ValueError(f"kl_layernorm_workspace_bytes failed: {lib.kl_last_error_message()!r}")
    return out.value


def layernorm_resolve(cols: int, impl: int) -> int:
    """AUTO dispatch rule as documented in docs/layernorm.md (independent copy)."""
    if impl != LAYERNORM_AUTO:
        return impl
    if cols <= LAYERNORM_WARP_MAX_COLS:
        return LAYERNORM_FUSED_WARP
    return LAYERNORM_FUSED_SMEM if cols <= LAYERNORM_SMEM_MAX_COLS else LAYERNORM_FUSED_BLOCK


# Validity domain of the bound (docs/layernorm.md): no intermediate overflow
# (sums of |x_j - x_0| and of squared deviations below 2^126) and
# var + eps >= 2^-100 (the scale stays far from the subnormal range).
LN_MAX_SUM = 2.0**126
LN_MIN_SCALE_SQUARED = 2.0**-100
HALF_MIN_SUBNORMAL = 2.0**-150


def layernorm_reference(x: np.ndarray, gamma: np.ndarray | None, beta: np.ndarray | None, eps: float):
    """Definition in float64. Returns a dict with y, z, nan_rows, in_domain and
    the per-row quantities the bound needs (mu, sig, a = mean |x_j - x_0|,
    dev = mean |x_j - mu|).

    nan_rows: rows the definition makes NaN (any non-finite x, or a zero
    scale: constant row with eps = 0). Their y is NaN.
    """
    x64 = x.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        nonfinite = ~np.isfinite(x64).all(axis=1)
        xs = np.where(nonfinite[:, None], 0.0, x64)
        mu = xs.mean(axis=1, keepdims=True)
        dev2 = ((xs - mu) ** 2).sum(axis=1, keepdims=True)
        var = dev2 / xs.shape[1]
        sig = np.sqrt(var + float(np.float32(eps)))
        nan_rows = nonfinite | (sig[:, 0] == 0)
        abs_shift = np.abs(xs - xs[:, :1]).sum(axis=1, keepdims=True)
        in_domain = ((abs_shift < LN_MAX_SUM) & (dev2 < LN_MAX_SUM) &
                     (var + float(np.float32(eps)) >= LN_MIN_SCALE_SQUARED))[:, 0]
        safe_sig = np.where(sig > 0, sig, 1.0)
        z = (xs - mu) / safe_sig
        y = z if gamma is None else z * gamma.astype(np.float64) + beta.astype(np.float64)
    y = np.array(y)
    y[nan_rows] = np.nan
    return {"y": y, "z": z, "nan_rows": nan_rows, "in_domain": in_domain, "mu": mu, "sig": safe_sig,
            "a": abs_shift / xs.shape[1], "dev": np.abs(xs - mu).mean(axis=1, keepdims=True)}


def layernorm_bound(x, gamma, beta, eps, impl: int) -> np.ndarray:
    """Per-element absolute error bound from docs/layernorm.md (no 1.01 factor,
    no absolute slack): e1 = (k+L+1) u A + u|mu| + 2^-150 for the pivot-shifted
    mean, e2 = (k+L+1) u (D + e1) + u|mu| + 2^-150 for the corrected mean."""
    _, cols = x.shape
    threads = 32 if layernorm_resolve(cols, impl) == LAYERNORM_FUSED_WARP else 256
    k = int(np.ceil(cols / threads))
    levels = int(np.ceil(np.log2(min(cols, threads))))
    r = layernorm_reference(x, gamma, beta, eps)
    e1 = (k + levels + 1) * U * r["a"] + U * np.abs(r["mu"]) + HALF_MIN_SUBNORMAL
    e2 = (k + levels + 1) * U * (r["dev"] + e1) + U * np.abs(r["mu"]) + HALF_MIN_SUBNORMAL
    c1 = e1 / r["sig"]
    E = e2 / r["sig"] + np.abs(r["z"]) * ((k + levels + 3) * U / 2 + 4.5 * U + c1 * c1 / 2)
    if gamma is None:
        return E
    with np.errstate(invalid="ignore"):
        return np.abs(gamma.astype(np.float64)) * E + U * np.abs(r["y"])


def layernorm_allowed(x, gamma, beta, eps, impl: int) -> np.ndarray:
    """1.01 * bound + (1 + |gamma_j|) 2^-147 (docs/layernorm.md)."""
    g = 0.0 if gamma is None else np.abs(gamma.astype(np.float64))
    return 1.01 * layernorm_bound(x, gamma, beta, eps, impl) + (1.0 + g) * ABS_SLACK


def layernorm_check(x, gamma, beta, eps, y, impl: int) -> float:
    """Asserts y satisfies the derived bound; returns worst error / allowed.

    Every row must be inside the bound's validity domain (or be a NaN row);
    NaN rows must be NaN; elements whose reference is NaN or +-inf (from a
    non-finite gamma/beta) must be that same value.
    """
    r = layernorm_reference(x, gamma, beta, eps)
    ref, nan_rows = r["y"], r["nan_rows"]
    assert (r["in_domain"] | nan_rows).all(), "row outside the bound's validity domain (docs/layernorm.md)"
    assert np.isnan(y[nan_rows]).all(), "rows the definition makes NaN must be NaN"
    ok = ~nan_rows[:, None] & np.isfinite(ref)
    special = ~nan_rows[:, None] & ~np.isfinite(ref)
    with np.errstate(invalid="ignore"):
        same = np.where(np.isnan(ref), np.isnan(y), y.astype(np.float64) == ref)
    assert same[special].all(), "non-finite gamma/beta must give the same special value"
    allowed = np.broadcast_to(layernorm_allowed(x, gamma, beta, eps, impl), ref.shape)[ok]
    got = y.astype(np.float64)[ok]
    err = np.where(np.isnan(got), np.inf, np.abs(got - ref[ok]))
    ratio = float((err / allowed).max()) if err.size else 0.0
    assert ratio <= 1.0, f"error {ratio:.3f} x the derived bound"
    return ratio


# --- matrix multiplication (docs/gemm.md) -------------------------------------

GEMM_AUTO, GEMM_NAIVE, GEMM_TILED, GEMM_REGBLOCK = 0, 1, 2, 3
GEMM_IMPLS = {"auto": 0, "naive": 1, "tiled": 2, "regblock": 3}
GEMM_MAX_DIM = 1 << 14
GEMM_MAX_ELEMENTS = 1 << 26
GEMM_MAX_S = 2.0**126


def gemm_check(a: np.ndarray, b: np.ndarray, c: np.ndarray, exact: bool = False) -> float:
    """Asserts C = A B within |C - C_ref| <= (gamma_K + K 2^-52) S + K 2^-150 (1 + gamma_K)
    (S = |A| |B|, gamma_K = K u / (1 - K u); independent implementation of
    docs/gemm.md). Non-finite references must be matched exactly; with
    exact=True every output must equal float32(reference). Returns the worst
    error / allowed."""
    k = a.shape[1]
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        ref = a64 @ b64
        s = np.abs(a64) @ np.abs(b64)
    special = ~np.isfinite(ref)
    with np.errstate(invalid="ignore"):
        same = np.where(np.isnan(ref), np.isnan(c), c.astype(np.float64) == ref)
    assert same[special].all(), "non-finite references must be matched exactly"
    ok = ~special
    assert (s[ok] < GEMM_MAX_S).all(), "outside the bound's validity domain"
    gamma = k * U / (1 - k * U)
    # float64 reference error <= ~K 2^-53 S; underflow roundings amplified by (1 + gamma) (docs/gemm.md).
    allowed = (gamma + k * 2.0**-52) * s[ok] + k * 2.0**-150 * (1 + gamma)
    got = c.astype(np.float64)[ok]
    err = np.where(np.isnan(got), np.inf, np.abs(got - ref[ok]))
    assert np.all(err <= allowed), f"error {float((err / np.maximum(allowed, 1e-300)).max()):.3f} x the bound"
    if exact:
        assert np.array_equal(c[ok], ref[ok].astype(np.float32)), "integer inputs must give exact results"
    return float((err / np.maximum(allowed, 1e-300)).max()) if err.size else 0.0


def float_ptr(arr: np.ndarray | None):
    """Pointer to a C-contiguous float32 array (None -> NULL). Never copies."""
    if arr is None:
        return None
    if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
        raise TypeError("expected a C-contiguous float32 array")
    return arr.ctypes.data_as(_FLOAT_P)


# --- independent reference helpers -----------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def splitmix64(seed: int, n: int) -> np.ndarray:
    """First n SplitMix64 outputs for `seed`, vectorized (uint64 wraps mod 2^64)."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed % 2**64) + k * _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def fill_uniform(seed: int, n: int, lo: float, hi: float) -> np.ndarray:
    """Python twin of kl_fill_uniform: same stream, same double->float rounding."""
    lo64 = float(np.float32(lo))  # the C API takes float and widens to double
    hi64 = float(np.float32(hi))
    if not (np.isfinite(lo64) and np.isfinite(hi64) and lo64 < hi64):
        raise ValueError("fill_uniform requires finite lo < hi (as float32)")
    u = (splitmix64(seed, n) >> np.uint64(40)).astype(np.float64) * (1.0 / 16777216.0)
    # Separate multiply and add (no FMA), matching the C++ expression.
    return (lo64 + (hi64 - lo64) * u).astype(np.float32)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise ULP distance between float32 arrays; +0 == -0, NaN == NaN."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**31)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**31)) - ib, ib)
    d = np.abs(ia - ib)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.where(nan_a & nan_b, 0, d)
    return np.where(nan_a ^ nan_b, np.iinfo(np.int64).max, d)


def same_float(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise identical bit patterns (+0 != -0); any NaN matches any NaN."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def saxpy_reference(a: float, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float64 reference for fmaf(a, x, y) plus a mask of provably exact elements.

    The product of two float32 values is exact in float64 (it cannot overflow
    or underflow there). The float64 sum is then rounded once; TwoSum recovers
    that rounding error e exactly. Where e == 0 the float64 sum IS the exact
    a*x+y, so its conversion to float32 is the correctly rounded result
    (including overflow to inf) and must match the GPU bit for bit. Where an
    input or the product is inf/NaN, no rounding is involved and IEEE rules
    give the same result in float64 and in fmaf, so those elements are exact
    too. Elsewhere the reference is within 1 ulp (double rounding).
    """
    with np.errstate(invalid="ignore", over="ignore"):
        p = np.float64(np.float32(a)) * x.astype(np.float64)
        yd = y.astype(np.float64)
        s = p + yd
        bp = s - yd
        e = (p - bp) + (yd - (s - bp))
        exact = (np.isfinite(s) & (e == 0)) | ~np.isfinite(p) | ~np.isfinite(yd)
        return s.astype(np.float32), exact


def saxpy_reference_f64(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """a*x + y in float64 (product exact), rounded once to float32."""
    return saxpy_reference(a, x, y)[0]
