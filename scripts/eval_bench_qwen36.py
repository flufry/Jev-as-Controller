"""Qwen3.6 cross-benchmark baseline: full response generation then option parsing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_bench_qwen36_native import main


if __name__ == '__main__':
    main()
