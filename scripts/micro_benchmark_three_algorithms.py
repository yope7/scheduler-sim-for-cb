#!/usr/bin/env python3
"""後方互換シム: 実体は scripts/benchmarks/micro_benchmark_three_algorithms.py。"""
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _target = Path(__file__).resolve().parent / "benchmarks" / "micro_benchmark_three_algorithms.py"
    os.execv(sys.executable, [sys.executable, str(_target)] + sys.argv[1:])
