"""VideoJev benchmark screening, using the unchanged original model evaluator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_compat import run_model


def main(argv=None):
    return run_model('eval_longvideobench_videojev.py', argv)


if __name__ == '__main__':
    main()
