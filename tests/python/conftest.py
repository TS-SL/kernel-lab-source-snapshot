import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kl  # noqa: E402


@pytest.fixture(scope="session")
def lib():
    return kl.load_library()


def pytest_report_header(config):
    lines = []
    try:
        lib = kl.load_library()
        info = kl.DeviceInfo()
        if lib.kl_get_device_info(info) == kl.KL_OK:
            lines.append(
                f"kernellens device: {info.name.decode()} cc {info.compute_major}.{info.compute_minor}, "
                f"driver CUDA {info.driver_version}, runtime {info.runtime_version}"
            )
    except OSError as exc:  # header only; the lib fixture will fail the tests
        lines.append(f"kernellens library not loadable: {exc}")
    try:
        import torch

        lines.append(f"torch {torch.__version__} (CUDA {torch.version.cuda}, available={torch.cuda.is_available()})")
    except ImportError:
        lines.append("torch not installed: torch reference tests will be SKIPPED")
    return lines
