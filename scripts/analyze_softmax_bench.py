"""Softmax entry point: the softmax summary (scripts/analyze_bench.py --op softmax).

Usage: python scripts/analyze_softmax_bench.py [--name m2] [--boot 2000] [--note TEXT]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_bench  # noqa: E402

if __name__ == "__main__":
    analyze_bench.main(op="softmax", generator="scripts/analyze_softmax_bench.py")
