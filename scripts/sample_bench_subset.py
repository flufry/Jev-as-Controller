"""Reproducible question-level, stratified screening subsets (no model scores)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import BENCHES, json_bytes, jsonl_bytes, load_rows, portable_rows, sha256, stratum, write_new


def allocate_counts(counts, n):
    # Generalization of sample_longvideobench_proportional.allocate_counts to arbitrary strata.
    if (not counts or any(type(c) is not int or c < 0 for c in counts.values())
            or type(n) is not int or not 1 <= n <= sum(counts.values())):
        raise ValueError('Invalid population or sample size')
    total = sum(counts.values())
    quotas = {g: n * counts[g] // total for g in counts}
    order = sorted(counts, key=lambda g: -(n * counts[g] % total))
    for g in order[:n - sum(quotas.values())]:
        quotas[g] += 1
    return quotas


def validate_population(bench, rows):
    counts = Counter(stratum(r) for r in rows)
    videos = len({r['video_key'] for r in rows})
    if bench == 'videomme':
        valid = counts == dict(short=900, medium=900, long=900) and videos == 900
    elif bench == 'mlvu':
        from timejev.mlvu_benchmark import TASK_COUNTS
        valid = counts == TASK_COUNTS and videos == 1122
    else:
        valid = len(rows) == 1549
    if not valid:
        raise ValueError('Sampling source must be the complete validated official population')


def sample(bench, out, manifest=None, seed=42, n=None, per_task=None, proportional=False,
           allow_incomplete_population=False):
    manifest = Path(manifest or f'data/benchmarks/{bench}/questions.jsonl').resolve()
    out = Path(out).resolve()
    if out.exists() or out.with_suffix('.meta.json').exists():
        raise FileExistsError('Choose new sample paths')
    source_sha = sha256(manifest)
    rows = load_rows(manifest)
    if any(r.get('benchmark') != bench for r in rows):
        raise ValueError('Manifest benchmark does not match --bench')
    source_meta = manifest.with_suffix('.meta.json')
    source_metadata = json.loads(source_meta.read_text(encoding='utf-8')) if source_meta.exists() else {}
    coverage = source_metadata.get('coverage', {})
    if allow_incomplete_population:
        excluded = [qid for video in coverage.get('excluded_videos', []) for qid in video['question_ids']]
        if (bench != 'lvbench' or source_metadata.get('schema') != 'bench_manifest_v1'
                or source_metadata.get('manifest_sha256') != source_sha
                or coverage.get('scope') != 'available_videos_subset'
                or coverage.get('official_full_benchmark') is not False
                or coverage.get('expected_questions') != 1549
                or coverage.get('available_questions') != len(rows)
                or coverage.get('available_videos') != len({r['video_key'] for r in rows})
                or coverage.get('excluded_questions') != len(excluded)
                or len(set(excluded)) != len(excluded)
                or set(excluded) & {r['id'] for r in rows}
                or len(rows) + len(excluded) != 1549):
            raise ValueError('Incomplete sampling requires an audited LVBench available-video manifest')
    else:
        if coverage.get('scope') == 'available_videos_subset':
            raise ValueError('Available-video population requires explicit --allow-incomplete-population')
        validate_population(bench, rows)
    groups = sorted({stratum(r) for r in rows})
    counts = {g: sum(stratum(r) == g for r in rows) for g in groups}
    if bench == 'mlvu':
        from timejev.mlvu_benchmark import TASK_COUNTS
        if set(groups) != set(TASK_COUNTS) or proportional or n is not None:
            raise ValueError('MLVU needs all seven tasks and --per-task (not --n/--proportional)')
        per_task = 20 if per_task is None else per_task
        if per_task < 1:
            raise ValueError('--per-task must be positive')
        quotas = dict.fromkeys(groups, per_task)
        strategy = 'equal_task'
    elif bench == 'videomme':
        n = 150 if n is None else n
        if set(groups) != {'short', 'medium', 'long'} or per_task is not None or proportional:
            raise ValueError('Video-MME needs three duration strata, equal allocation')
        if type(n) is not int or n < 1:
            raise ValueError('--n must be positive')
        quotas = {g: n // 3 + int(i < n % 3) for i, g in enumerate(groups)}
        strategy = 'equal_duration'
    else:
        if per_task is not None:
            raise ValueError('--per-task is only for MLVU')
        quotas = allocate_counts(counts, 150 if n is None else n)
        strategy = 'proportional_question_type_combinations'
    if any(quotas[g] > counts[g] for g in groups):
        raise ValueError('Sample quota exceeds a stratum population')
    rng, selected = random.Random(seed), set()
    for group in groups:
        selected.update(rng.sample([r['id'] for r in rows if stratum(r) == group], quotas[group]))
    chosen = [r for r in rows if r['id'] in selected]
    data = jsonl_bytes(portable_rows(chosen, out))
    meta = dict(schema='bench_subset_v1', benchmark=bench, unit='question', seed=seed, n=len(chosen),
        sampling=strategy, quota_method='integer_largest_remainder' if bench == 'lvbench' else 'equal',
        source_manifest=str(manifest), source_sha256=source_sha, source_counts=counts, quotas=quotas,
        tie_break_order=groups, selected_ids=[r['id'] for r in chosen],
        sample_sha256=hashlib.sha256(data).hexdigest(), sampler_sha256=sha256(__file__),
        videos=len({r['video_key'] for r in chosen}), training_overlap_verified=False)
    if source_metadata:
        meta['source_metadata'] = source_metadata
    if allow_incomplete_population:
        meta['population_scope'] = 'available_videos_subset'
        meta['official_full_benchmark'] = False
    if sha256(manifest) != source_sha:
        raise ValueError('Source manifest changed during sampling')
    write_new(out, data)
    write_new(out.with_suffix('.meta.json'), json_bytes(meta))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', choices=BENCHES, required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int)
    parser.add_argument('--per-task', type=int)
    parser.add_argument('--proportional', action='store_true', help='LVBench default; explicit for plan compatibility')
    parser.add_argument('--allow-incomplete-population', action='store_true',
                        help='LVBench only: sample an audited available-video subset instead of the full official population')
    print(json.dumps(sample(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
