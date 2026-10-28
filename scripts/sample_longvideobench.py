"""按 LongVideoBench 官方四个时长组等量抽题，保留官方问题与选项。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timejev.longvideobench_benchmark import DURATION_NAMES, formatted_options
from timejev.user_data import load_manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(manifest, reference, out, per_group=25, seed=42):
    """抽样单位为问题；同一视频可仅抽中部分问题，输出保持源 JSONL 题序。"""
    if type(per_group) is not int or per_group < 1:
        raise ValueError('per_group 必须是正整数')
    manifest, reference, out = map(lambda p: Path(p).resolve(), (manifest, reference, out))
    meta_path = out.with_suffix('.meta.json')
    if out == meta_path or out.exists() or meta_path.exists():
        raise FileExistsError(f'样本或元数据文件已存在，请选择新路径：{out}, {meta_path}')
    rows = load_manifest(manifest)
    targets = json.loads(reference.read_text())
    official = {row['id']: row for row in targets}
    if len(official) != len(targets):
        raise ValueError('官方问题 ID 重复')
    if len(rows) != len(official) or {row.get('question_id') for row in rows} != set(official):
        raise ValueError('输入清单和官方标注必须包含同一组完整问题')
    groups = {group: [] for group in DURATION_NAMES}
    for row in rows:
        target = official[row['question_id']]
        group = target['duration_group']
        if type(group) is not int or group not in groups:
            raise ValueError(f'未知的官方时长组 duration_group：{group}')
        expected_path = (manifest.parent/'videos'/target['video_path']).resolve()
        answer = target.get('correct_choice')
        if (row['task'] != 'choice' or row['id'] != 'longvideobench/'+target['id']
                or row['query'] != target['question'] or row['options'] != formatted_options(target)
                or type(answer) is not int or not 0 <= answer < len(target['candidates'])
                or row['answer_index'] != answer
                or row['video_key'] != 'longvideobench/'+target['video_id']
                or Path(row['video_path']) != expected_path or row['duration'] != target['duration']):
            raise ValueError(f'问题与官方标注不匹配：{row["id"]}')
        groups[group].append(row['id'])
    rng = random.Random(seed)
    selected = set()
    for group, ids in groups.items():
        if len(ids) < per_group:
            raise ValueError(f'duration_group={group} 只有 {len(ids)} 道题，需要 {per_group} 道题')
        selected.update(rng.sample(ids, per_group))
    sampled = []
    for row in rows:
        if row['id'] not in selected:
            continue
        member = dict(row)
        target = official[row['question_id']]
        for key in ('duration_group', 'level', 'question_category', 'topic_category'):
            if key in target:
                member[key] = target[key]
        member['video_path'] = os.path.relpath(row['video_path'], out.parent)
        if 'media_source' in member:
            member['media_source'] = dict(member['media_source'],
                path=os.path.relpath(member['media_source']['path'], out.parent))
        sampled.append(member)
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in sampled)
    metadata = dict(schema='longvideobench_stratified_questions_v1', unit='question',
        sampling='equal_official_duration_groups', seed=seed, per_group=per_group,
        n=len(sampled), videos=len({r['video_key'] for r in sampled}),
        source_manifest=dict(path=str(manifest), sha256=sha256(manifest)),
        source_reference=dict(path=str(reference), sha256=sha256(reference)),
        source_counts={str(group): len(ids) for group, ids in groups.items()},
        selected_counts=dict(sorted(Counter(str(r['duration_group']) for r in sampled).items())),
        duration_names={str(group): name for group, name in DURATION_NAMES.items()},
        selected_ids=[row['id'] for row in sampled],
        sample_sha256=hashlib.sha256(text.encode()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        handle.write(text)
    with meta_path.open('x') as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/benchmarks/longvideobench/questions.jsonl')
    parser.add_argument('--reference', default='data/benchmarks/longvideobench/official.json')
    parser.add_argument('--out', required=True, help='输出 JSONL；同名 .meta.json 保存来源和抽样记录')
    parser.add_argument('--per-group', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(sample(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
