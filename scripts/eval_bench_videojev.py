"""VideoJev benchmark evaluation: save raw head logits and score their argmax."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_compat import install, load_legacy


def main(argv=None):
    with install(Path(__file__).with_name('eval_longvideobench_videojev.py')):
        return load_legacy('eval_longvideobench_videojev.py').main(argv, save_logits=True, logits_only=True)


if __name__ == '__main__':
    main()
