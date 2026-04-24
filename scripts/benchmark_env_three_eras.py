#!/usr/bin/env python3
"""後方互換シム: 実体は scripts/benchmarks/benchmark_env_three_eras.py。"""
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _target = Path(__file__).resolve().parent / "benchmarks" / "benchmark_env_three_eras.py"
    os.execv(sys.executable, [sys.executable, str(_target)] + sys.argv[1:])
