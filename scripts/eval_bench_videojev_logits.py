"""Compatibility alias; eval_bench_videojev.py now always saves raw logits."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_bench_videojev import main


if __name__ == '__main__':
    main()
