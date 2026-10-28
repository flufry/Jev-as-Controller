"""Fuse same-protocol benchmark predictions with the unchanged beta=2 formula."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_compat import install, load_legacy


def evaluate(manifest, qwen, videojev, out, beta=2., qwen_method='qwen36'):
    if beta != 2 or qwen_method != 'qwen36':
        raise ValueError('Screening fixes beta=2 and Qwen3.6; sensitivity scans are diagnostics only')
    seeds = [json.loads((Path(p) / 'run.json').read_text(encoding='utf-8'))['identity'].get('seed')
             for p in (qwen, videojev)]
    if seeds[0] is None or seeds[0] != seeds[1]:
        raise ValueError('Both source runs must have the same explicit seed')
    legacy_path = Path(__file__).with_name('eval_longvideobench_controller.py')
    with install(legacy_path):
        legacy = load_legacy(legacy_path.name)
        original = legacy.control

        def control(raw, jev, value):
            result = original(raw, jev, value)
            result.update(stratum=raw.get('stratum', raw.get('duration_group')))
            return result

        legacy.control = control
        return legacy.evaluate(manifest, qwen, videojev, out, beta=beta, qwen_method=qwen_method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'qwen', 'videojev', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--qwen-method', choices=('qwen36',), default='qwen36')
    print(json.dumps(evaluate(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
