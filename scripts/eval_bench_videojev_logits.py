"""VideoJev screening: save original head candidate logits as well as probabilities."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_compat import install, load_legacy


def main(argv=None):
    path = Path(__file__).with_name('eval_longvideobench_videojev.py')
    with install(path):
        return load_legacy(path.name).main(argv, save_logits=True)


if __name__ == '__main__':
    main()
