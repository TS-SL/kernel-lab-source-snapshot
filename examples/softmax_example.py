"""Small example: KernelLens softmax and GEMM against float64 references.

Run after building (it uses build/release/kernellens.dll or $KERNELLENS_LIB):

    .venv\\Scripts\\python.exe scripts\\gpu_guard.py --log results\\raw\\latest-example-gpu.csv -- ^
        .venv\\Scripts\\python.exe examples\\softmax_example.py

The output is quoted in README.md.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
import kl  # noqa: E402  (ctypes bindings and float64 references used by the tests)

lib = kl.load_library()

# Softmax over a masked row: -inf entries must come out as exact zeros.
x = np.array([[1.0, 2.0, 3.0, -np.inf], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
y = np.empty_like(x)
status = lib.kl_softmax_host(2, 4, kl.float_ptr(x), kl.float_ptr(y), kl.SOFTMAX_AUTO)
print("softmax status:", lib.kl_status_string(status).decode())
print("softmax output:\n", y)
print("worst error / derived bound:", f"{kl.softmax_check(x, y, kl.SOFTMAX_AUTO):.3f}")

# GEMM: 96 x 80 x 70 (not a multiple of any tile size), seeded inputs.
m, n, k = 96, 80, 70
a = kl.fill_uniform(1, m * k, -1.0, 1.0).reshape(m, k)
b = kl.fill_uniform(2, k * n, -1.0, 1.0).reshape(k, n)
c = np.empty((m, n), dtype=np.float32)
status = lib.kl_gemm_host(m, n, k, kl.float_ptr(a), kl.float_ptr(b), kl.float_ptr(c), kl.GEMM_AUTO)
print("gemm status:", lib.kl_status_string(status).decode())
print("gemm worst error / derived bound:", f"{kl.gemm_check(a, b, c):.3f}")

# Unsupported input is rejected with a status, never computed wrongly.
status = lib.kl_gemm_host(m, n, 1 << 15, kl.float_ptr(a), kl.float_ptr(b), kl.float_ptr(c), kl.GEMM_AUTO)
print("gemm with K = 32768:", lib.kl_status_string(status).decode(), "-", lib.kl_last_error_message().decode())
