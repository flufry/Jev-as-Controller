"""排除各 seed 的原 100 题，再按时长一次抽 50 题并分成两个互斥组。"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_longvideobench import sha256
from scripts.summarize_longvideobench_qwen36 import read_sample
from timejev.longvideobench_benchmark import DURATION_NAMES, formatted_options
from timejev.user_data import load_manifest

SEEDS = (42, 43, 44)
VERSION = 'exclude_group1_draw50_split25_v1'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_sources(manifest, reference):
    rows = load_manifest(manifest)
    targets = json.loads(reference.read_text(encoding='utf-8'))
    official = {row['id']: row for row in targets}
    if (len(official) != len(targets) or len(rows) != len(official)
            or {r.get('question_id') for r in rows} != set(official)):
        raise ValueError('源清单和官方标注必须完整覆盖相同且不重复的问题 ID')
    groups = {group: [] for group in DURATION_NAMES}
    for row in rows:
        target = official[row['question_id']]
        group, answer = target.get('duration_group'), target.get('correct_choice')
        if type(group) is not int or group not in groups:
            raise ValueError(f'未知官方时长组：{group}')
        expected_path = (manifest.parent/'videos'/target['video_path']).resolve()
        source = row.get('media_source', {})
        if (row['task'] != 'choice' or row['id'] != 'longvideobench/'+target['id']
                or row['query'] != target['question'] or row['options'] != formatted_options(target)
                or type(answer) is not int or not 0 <= answer < len(target['candidates'])
                or row['answer_index'] != answer or row['duration'] != target['duration']
                or row['video_key'] != 'longvideobench/'+target['video_id']
                or Path(row['video_path']) != expected_path
                or source and (source.get('kind') != 'video' or Path(source['path']) != expected_path)):
            raise ValueError(f'源问题与官方标注或原始视频不匹配：{row["id"]}')
        groups[group].append(row['id'])
    return rows, official, groups


def overlaps(samples):
    return [dict(first=a, second=b, questions=len(set(x['ids']) & set(y['ids'])),
        videos=len(set(x['video_keys']) & set(y['video_keys'])))
        for (a, x), (b, y) in combinations(samples.items(), 2)]


def sample_additional(manifest, reference, original_pattern, out):
    manifest, reference, out = [Path(path).resolve() for path in (manifest, reference, out)]
    destinations = [out/f'group{group}/seed{seed}/sample.jsonl' for group in (2, 3) for seed in SEEDS]
    targets = [*destinations, *(path.with_suffix('.meta.json') for path in destinations), out/'sampling.json']
    if len(set(targets)) != 13 or any(path.exists() for path in targets):
        raise FileExistsError('新增样本、元数据或 sampling.json 已存在，拒绝覆盖')
    source_manifest = dict(path=str(manifest), sha256=sha256(manifest))
    source_reference = dict(path=str(reference), sha256=sha256(reference))
    inputs = {manifest: source_manifest['sha256'], reference: source_reference['sha256']}
    rows, official, groups = read_sources(manifest, reference)
    outputs, all_samples, per_seed = {}, {}, {}
    for seed in SEEDS:
        original_path = Path(original_pattern.format(seed=seed)).resolve()
        original, lineage = read_sample(original_path, seed)
        if lineage['source_manifest'] != source_manifest or lineage['source_reference'] != source_reference:
            raise ValueError(f'seed{seed} 原样本不是来自指定源清单和官方标注')
        inputs[original_path] = lineage['sha256']
        inputs[original_path.with_suffix('.meta.json')] = lineage['meta_sha256']
        original_ids = [r['id'] for r in original]
        excluded = set(original_ids)
        rng, assignment, selected = random.Random(seed), {}, {2: set(), 3: set()}
        for group, ids in groups.items():
            remaining = [key for key in ids if key not in excluded]
            if len(remaining) < 50:
                raise ValueError(f'seed{seed} 时长组 {group} 排除原题后仅 {len(remaining)} 题，需要 50 题')
            draw = rng.sample(remaining, 50)
            assignment[str(group)] = dict(remaining=len(remaining), drawn_ids=draw,
                group2_ids=draw[:25], group3_ids=draw[25:])
            selected[2].update(draw[:25])
            selected[3].update(draw[25:])
        allocation_sha = hashlib.sha256(encoded(dict(seed=seed, excluded_ids=original_ids,
            by_duration_group=assignment))).hexdigest()
        local = {'group1': dict(ids=original_ids, video_keys=[r['video_key'] for r in original])}
        allocations = {}
        for group in (2, 3):
            path = out/f'group{group}/seed{seed}/sample.jsonl'
            sampled = []
            for row in rows:
                if row['id'] not in selected[group]:
                    continue
                member = dict(row)
                member.update({key: official[row['question_id']][key] for key in
                    ('duration_group', 'level', 'question_category', 'topic_category')
                    if key in official[row['question_id']]})
                member['video_path'] = os.path.relpath(row['video_path'], path.parent)
                if 'media_source' in member:
                    member['media_source'] = dict(member['media_source'],
                        path=os.path.relpath(member['media_source']['path'], path.parent))
                sampled.append(member)
            text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in sampled)
            allocations[f'group{group}'] = dict(path=str(path), sample_sha256=hashlib.sha256(text.encode()).hexdigest(),
                selected_ids=[r['id'] for r in sampled])
            local[f'group{group}'] = dict(ids=[r['id'] for r in sampled], video_keys=[r['video_key'] for r in sampled])
            outputs[path] = (text, sampled)
        same_seed_overlaps = overlaps(local)
        if any(item['questions'] for item in same_seed_overlaps):
            raise ValueError(f'seed{seed} 三组题目未互斥')
        for group in (2, 3):
            path = out/f'group{group}/seed{seed}/sample.jsonl'
            text, sampled = outputs[path]
            metadata = dict(schema='longvideobench_stratified_questions_v1', unit='question',
                sampling='equal_official_duration_groups', sampling_version=VERSION, seed=seed,
                group_index=group, per_group=25, n=len(sampled), videos=len({r['video_key'] for r in sampled}),
                source_manifest=source_manifest, source_reference=source_reference,
                source_counts={str(key): len(ids) for key, ids in groups.items()},
                selected_counts=dict(sorted(Counter(str(r['duration_group']) for r in sampled).items())),
                duration_names={str(key): name for key, name in DURATION_NAMES.items()},
                selected_ids=allocations[f'group{group}']['selected_ids'],
                sample_sha256=allocations[f'group{group}']['sample_sha256'], original_sample=lineage,
                excluded_ids=original_ids, allocation_sha256=allocation_sha, allocations=allocations,
                same_seed_overlaps=same_seed_overlaps)
            outputs[path] = (text, metadata)
        per_seed[str(seed)] = dict(original_sample=lineage, excluded_ids=original_ids,
            allocation_sha256=allocation_sha, by_duration_group=assignment,
            allocations=allocations, same_seed_overlaps=same_seed_overlaps)
        all_samples.update({f'{name}/seed{seed}': entry for name, entry in local.items()})
    summary = dict(schema='longvideobench_additional_groups_sampling_v1', sampling_version=VERSION,
        unit='question', seeds=list(SEEDS), groups=[1, 2, 3], questions_per_group_seed=100, per_duration_group=25,
        source_manifest=source_manifest, source_reference=source_reference, per_seed=per_seed,
        overlaps=overlaps(all_samples), total_question_occurrences=900,
        unique_questions=len(set.union(*(set(entry['ids']) for entry in all_samples.values()))),
        script_sha256=sha256(__file__),
        note='每个 seed 排除原100题，各时长组仅抽一次50题，按抽取顺序前后各25题；输出按源题序。'
             '同seed三组问题ID互斥；跨seed允许重叠，同视频可以出现在多组。')
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError('抽样期间源清单、官方标注或原样本发生变化')
    for path, (text, metadata) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
        with path.with_suffix('.meta.json').open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write('\n')
    with (out/'sampling.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/benchmarks/longvideobench/questions.jsonl')
    parser.add_argument('--reference', default='data/benchmarks/longvideobench/official.json')
    parser.add_argument('--original-pattern', default='runs/lvb100_equal_seed{seed}/sample.jsonl')
    parser.add_argument('--out', required=True)
    result = sample_additional(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('seeds', 'groups', 'total_question_occurrences', 'unique_questions')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
