"""比例抽样清单的双 2 FPS Jev-as-Controller；仅适配元数据，复用原公式与逐题校验。"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_longvideobench_proportional import SAMPLING, read_sample
from timejev.lvb_experiment import sha256, write_json

ORIGINAL = ROOT/'scripts/eval_longvideobench_controller.py'
SCHEMA = 'longvideobench_proportional_controller_v1'


def load_original(path, name):
    """单独的模块命名空间；不修改 scripts.* 已导入模块的全局变量。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(manifest, qwen, videojev, out, beta=2., qwen_method='qwen36', seed=None):
    if type(beta) not in (int, float) or beta != 2 or qwen_method != 'qwen36':
        raise ValueError('比例抽样实验固定 Qwen3.6、beta=2')
    manifest = Path(manifest).resolve()
    if seed is None:
        seed = json.loads(manifest.with_suffix('.meta.json').read_text())['seed']
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError('需要 seed42、43、44 中的指定采样 seed')
    rows, sample = read_sample(manifest, seed=seed, n=100)
    legacy = load_original(ORIGINAL, '_lvb_proportional_controller_original')
    original_read = legacy.read_source

    def read_source(directory, targets, manifest_sha, method):
        predictions, source, metrics = original_read(directory, targets, manifest_sha, method)
        identity = source['identity']
        if (identity.get('seed') != seed or any(identity['protocol'].get(key) != value for key, value in
                dict(fps=2., max_frames=2048, context=262144, visual_token_budget=24576).items())):
            raise ValueError('Controller 来源需要同 seed、2FPS/2048 及指定上下文/token 预算')
        return predictions, source, metrics

    def save(path, value):
        value = deepcopy(value)
        if Path(path).name == 'run.json':
            value.update(schema=SCHEMA, sampling=SAMPLING, unit='question', seed=seed, sample=sample,
                original_controller_script_sha256=sha256(ORIGINAL), adapter_script_sha256=sha256(__file__))
            value['identity']['code_sha256']['scripts/eval_longvideobench_proportional_controller.py'] = sha256(__file__)
        elif Path(path).name == 'metrics.json':
            value.update(sampling=SAMPLING, unit='question', selected_counts=sample['selected_counts'])
        write_json(path, value)

    legacy.load_sample = lambda path: rows if Path(path).resolve() == manifest else read_sample(path, seed=seed, n=100)[0]
    legacy.read_source = read_source
    legacy.write_json = save
    return legacy.evaluate(manifest, qwen, videojev, out, beta=beta, qwen_method=qwen_method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--qwen', required=True)
    parser.add_argument('--videojev', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--qwen-method', choices=('qwen36',), default='qwen36')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44))
    print(json.dumps(evaluate(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
