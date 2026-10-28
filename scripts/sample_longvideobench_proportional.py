"""按完整 LongVideoBench val 四个时长组的题目数量比例独立抽样。"""
import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timejev.longvideobench_benchmark import DURATION_NAMES, formatted_options, validate_reference
from timejev.lvb_experiment import GROUPS, sha256
from timejev.user_data import load_manifest

SCHEMA = 'longvideobench_proportional_questions_v1'
SAMPLING = 'proportional_official_duration_groups'
QUOTA_METHOD = 'largest_remainder'


def allocate_counts(counts, n=100):
    """仅用整数最大余数法分配题数；同余数按官方 GROUPS 顺序。"""
    if (not isinstance(counts, Mapping) or set(counts) != set(GROUPS)
            or any(type(group) is not int or type(counts[group]) is not int or counts[group] < 0 for group in counts)):
        raise ValueError('counts 必须包含四个官方时长组及非负整数题数')
    total = sum(counts.values())
    if type(n) is not int or not 1 <= n <= total:
        raise ValueError('n 必须是正整数，且不能超过完整题目总数')
    quotas = {group: n*counts[group]//total for group in GROUPS}
    # sorted 稳定排序：余数并列时保持 GROUPS 的官方顺序。
    remainder_order = sorted(GROUPS, key=lambda group: -(n*counts[group] % total))
    for group in remainder_order[:n-sum(quotas.values())]:
        quotas[group] += 1
    return quotas


def read_sources(manifest, reference):
    """核验完整 val，返回已解析路径的原始题序和官方分组人口。"""
    manifest, reference = Path(manifest).resolve(), Path(reference).resolve()
    targets = json.loads(reference.read_text(encoding='utf-8'))
    official = validate_reference(targets)
    rows = load_manifest(manifest)
    if len(rows) != len(official) or {row.get('question_id') for row in rows} != set(official):
        raise ValueError('源清单与完整官方 val 必须覆盖同一组全部问题')
    groups = {group: [] for group in GROUPS}
    video_root = (manifest.parent/'videos').resolve()
    for row in rows:
        target = official[row['question_id']]
        group = target['duration_group']
        expected_path = (video_root/target['video_path']).resolve()
        media = row.get('media_source')
        if (Path(target['video_path']).is_absolute() or not expected_path.is_relative_to(video_root)
                or row['task'] != 'choice' or row['id'] != 'longvideobench/'+target['id']
                or row['query'] != target['question'] or row['options'] != formatted_options(target)
                or row['answer_index'] != target['correct_choice'] or row['duration'] != target['duration']
                or row['video_key'] != 'longvideobench/'+target['video_id']
                or Path(row['video_path']) != expected_path
                or media is not None and (media.get('kind') != 'video' or Path(media['path']) != expected_path)
                or any(key in row and row[key] != target.get(key) for key in
                    ('duration_group', 'level', 'question_category', 'topic_category'))):
            raise ValueError(f'源问题、标签或原始视频路径与官方标注不一致：{row["id"]}')
        groups[group].append(row['id'])
    return rows, official, groups


def payload(manifest, reference, out, n, seed):
    if type(seed) is not int:
        raise ValueError('seed 必须是整数')
    if type(n) is not int or n < 1:
        raise ValueError('n 必须是正整数')
    sources = dict(source_manifest=dict(path=str(manifest), sha256=sha256(manifest)),
                   source_reference=dict(path=str(reference), sha256=sha256(reference)))
    rows, official, groups = read_sources(manifest, reference)
    counts = {group: len(groups[group]) for group in GROUPS}
    quotas = allocate_counts(counts, n)
    rng, selected = random.Random(seed), set()
    for group in GROUPS:
        selected.update(rng.sample(groups[group], quotas[group]))
    sampled = []
    for row in rows:
        if row['id'] not in selected:
            continue
        member = dict(row)
        target = official[row['question_id']]
        member.update({key: target[key] for key in
            ('duration_group', 'level', 'question_category', 'topic_category') if key in target})
        member['video_path'] = os.path.relpath(row['video_path'], out.parent)
        if 'media_source' in member:
            member['media_source'] = dict(member['media_source'],
                path=os.path.relpath(member['media_source']['path'], out.parent))
        sampled.append(member)
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in sampled)
    selected_counts = Counter(row['duration_group'] for row in sampled)
    metadata = dict(schema=SCHEMA, unit='question', sampling=SAMPLING, quota_method=QUOTA_METHOD,
        seed=seed, n=n, videos=len({row['video_key'] for row in sampled}), **sources,
        source_counts={str(group): counts[group] for group in GROUPS},
        selected_counts={str(group): selected_counts[group] for group in GROUPS},
        quotas={str(group): quotas[group] for group in GROUPS}, tie_break_order=list(GROUPS),
        duration_names={str(group): DURATION_NAMES[group] for group in GROUPS},
        selected_ids=[row['id'] for row in sampled], sample_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    if any(sha256(source['path']) != source['sha256'] for source in sources.values()):
        raise ValueError('抽样期间源清单或官方标注发生变化')
    return text, metadata


def sample(manifest, reference, out, n=100, seed=42):
    """每个 seed 从完整 val 独立抽题；允许与历史样本及其它 seed 重叠。"""
    manifest, reference, out = [Path(path).resolve() for path in (manifest, reference, out)]
    meta_path = out.with_suffix('.meta.json')
    if out == meta_path or out.exists() or meta_path.exists():
        raise FileExistsError('样本或元数据已存在，请选择新的输出路径')
    text, metadata = payload(manifest, reference, out, n, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(text)
    with meta_path.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return metadata


def read_sample(manifest, seed=42, n=100):
    """重新核验完整人口、最大余数配额、官方内容、路径及精确 seed 抽题结果。"""
    manifest = Path(manifest).resolve()
    meta_path = manifest.with_suffix('.meta.json')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if (metadata.get('schema') != SCHEMA or metadata.get('unit') != 'question'
            or metadata.get('sampling') != SAMPLING or metadata.get('quota_method') != QUOTA_METHOD
            or type(metadata.get('seed')) is not int or metadata['seed'] != seed
            or type(metadata.get('n')) is not int or metadata['n'] != n):
        raise ValueError('需要指定 seed/n 的比例抽样元数据，不能使用等量抽样清单')
    paths = []
    for key in ('source_manifest', 'source_reference'):
        source = metadata.get(key, {})
        path = Path(source.get('path', ''))
        if not path.is_absolute() or not path.is_file() or sha256(path) != source.get('sha256'):
            raise ValueError(f'源清单或官方标注路径/SHA 不符：{key}')
        paths.append(path)
    text, expected = payload(*paths, manifest, n, seed)
    if (json.dumps(metadata, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False)
            or manifest.read_bytes() != text.encode('utf-8')):
        raise ValueError('样本或元数据与官方全体、比例配额及指定 seed 的精确抽样不一致')
    rows = load_manifest(manifest)
    return rows, dict(path=str(manifest), sha256=sha256(manifest), meta_sha256=sha256(meta_path),
        **{key: metadata[key] for key in ('schema', 'unit', 'sampling', 'quota_method', 'seed', 'n',
            'source_manifest', 'source_reference', 'source_counts', 'selected_counts', 'quotas')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/benchmarks/longvideobench/questions.jsonl')
    parser.add_argument('--reference', default='data/benchmarks/longvideobench/official.json')
    parser.add_argument('--out', required=True, help='输出 JSONL；同名 .meta.json 保存来源与配额')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    print(json.dumps(sample(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
