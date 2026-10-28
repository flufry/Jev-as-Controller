"""Scoped adapters for legacy evaluators; never change the LongVideoBench files."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from scripts.bench_io import load_rows, sha256, stratum

ROOT = Path(__file__).resolve().parents[1]


def generic_load_sample(path):
    rows = load_rows(path)
    for row in rows:
        row['stratum'] = stratum(row)
        # Legacy append() and controller also index this key. This is an internal
        # alias, not a fabricated mapping to the four official LVB durations.
        row.setdefault('duration_group', row['stratum'])
    return rows


def validate_prediction(row, prediction):
    import math
    values = prediction.get('probabilities', [])
    if (prediction.get('id') != row['id'] or prediction.get('query') != row['query']
            or prediction.get('options') != row['options'] or prediction.get('task') != 'choice'
            or len(values) != len(row['options'])
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or not math.isclose(sum(values), 1., abs_tol=1e-5)
            or type(prediction.get('answer_index')) is not int
            or prediction['answer_index'] != max(range(len(values)), key=values.__getitem__)):
        raise ValueError(f'Invalid prediction: {row["id"]}')


def read_predictions(path, rows):
    predictions = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Missing, duplicate, or unknown prediction IDs')
    for row in rows:
        validate_prediction(row, by_id[row['id']])
    return by_id


def generic_score(rows, by_id):
    if not rows or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Scoring requires complete nonempty coverage')
    totals, hits, category_n, category_hits = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        prediction = by_id[row['id']]
        validate_prediction(row, prediction)
        hit = prediction['answer_index'] == row['answer_index']
        group = stratum(row)
        totals[group] += 1
        hits[group] += hit
        for category in row.get('question_type', []):
            category_n[category] += 1
            category_hits[category] += hit
    groups = {g: dict(n=totals[g], correct=hits[g], accuracy=hits[g] / totals[g]) for g in sorted(totals)}
    bench = rows[0].get('benchmark', 'longvideobench')
    report = dict(n=len(rows), videos=len({r['video_key'] for r in rows}), correct=sum(hits.values()),
        accuracy=sum(hits.values()) / len(rows), benchmark=bench, split='screening_subset',
        primary_metric='accuracy', by_stratum=groups, training_overlap_verified=False)
    if bench == 'mlvu':
        from timejev.mlvu_benchmark import TASK_COUNTS
        if set(totals) != set(TASK_COUNTS):
            raise ValueError('M-Avg requires all seven MLVU tasks in the sample')
        report.update(m_avg=sum(g['accuracy'] for g in groups.values()) / 7,
                      primary_metric='m_avg', by_task=groups)
    if category_n:
        report['by_question_type'] = {g: dict(n=category_n[g], correct=category_hits[g],
            accuracy=category_hits[g] / category_n[g]) for g in sorted(category_n)}
    return report


def adapter_hashes(legacy=None):
    paths = [ROOT / 'scripts' / name for name in ('bench_io.py', 'bench_compat.py')]
    if legacy:
        paths.append(Path(legacy))
        wrapper = ROOT / 'scripts' / Path(legacy).name.replace('eval_longvideobench_', 'eval_bench_')
        if wrapper.is_file():
            paths.append(wrapper)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


@contextmanager
def install(legacy=None):
    """Patch before importing an isolated entrypoint; restore everything afterwards."""
    from timejev import lvb_experiment as lvb
    names = ('load_sample', 'score', 'protocol', 'Experiment', 'write_json')
    saved = {name: getattr(lvb, name) for name in names}
    hashes = adapter_hashes(legacy)

    def protocol(args):
        return dict(saved['protocol'](args), name='video_bench_screen_rgb_v1')

    def write_json(path, value):
        value = deepcopy(value)
        if Path(path).name == 'run.json':
            value.update(schema='video_bench_screen_run_v1', split='screening_subset',
                         sampling='manifest_defined', training_overlap_verified=False,
                         note='视频 benchmark 粗筛子集；非官方全量指标')
            value['identity'].setdefault('code_sha256', {}).update(hashes)
        saved['write_json'](path, value)

    class Experiment(saved['Experiment']):
        def __init__(self, args, method, settings):
            meta_path = Path(args.manifest).with_suffix('.meta.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if meta.get('schema') == 'bench_subset_v1' and meta['seed'] != args.seed:
                    raise ValueError('Inference seed must match sample metadata seed')
            # Included before the legacy resume equality check, not added afterwards.
            super().__init__(args, method, dict(settings, bench_adapter_sha256=hashes))
            for name in hashes:
                target = self.out / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())

        def append(self, row, probabilities, **details):
            return super().append(row, probabilities, stratum=stratum(row),
                                  benchmark=row.get('benchmark', 'longvideobench'), **details)

    # Adding adapter hashes to identity.code_sha256 in write_json would break the
    # original resume comparison. Model runs already carry them in settings.
    def model_write_json(path, value):
        if Path(path).name == 'run.json' and value['identity']['method'] != 'jev_as_controller':
            value = deepcopy(value)
            value.update(schema='video_bench_screen_run_v1', split='screening_subset',
                         sampling='manifest_defined', training_overlap_verified=False)
            saved['write_json'](path, value)
        else:
            write_json(path, value)

    lvb.load_sample, lvb.score, lvb.protocol = generic_load_sample, generic_score, protocol
    lvb.Experiment, lvb.write_json = Experiment, model_write_json
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(lvb, name, value)


def load_legacy(filename):
    path = ROOT / 'scripts' / filename
    spec = importlib.util.spec_from_file_location('_bench_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_model(filename, argv=None):
    with install(ROOT / 'scripts' / filename):
        return load_legacy(filename).main(argv)
