"""核验三次等时长分层抽样评估，汇总各方法准确率的均值和跨采样方差。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_longvideobench_controller import control, read_source
from timejev.longvideobench_benchmark import DURATION_NAMES
from timejev.lvb_experiment import GROUPS, load_sample, sha256, write_json

METHODS = {'qwen35': 'qwen', 'videojev': 'videojev', 'jev_as_controller': 'controller'}
NAMES = {'qwen35': 'Qwen3.5-4B', 'videojev': 'VideoJev-4B', 'jev_as_controller': 'Jev-as-Controller'}


def accuracy_statistics(values):
    if len(values) < 2:
        raise ValueError('计算样本方差至少需要两次准确率')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('准确率必须是 [0,1] 内的有限数值')
    return dict(values=values, mean=statistics.mean(values),
        sample_variance=statistics.variance(values), population_variance=statistics.pvariance(values),
        sample_std=statistics.stdev(values), population_std=statistics.pstdev(values))


def fingerprint(identity, method):
    keys = ('protocol', 'beta', 'code_sha256') if method == 'jev_as_controller' else (
        'checkpoint', 'settings', 'protocol', 'packages', 'code_sha256')
    if any(key not in identity or identity[key] is None for key in keys):
        raise ValueError(f'{method} 缺少模型、配置、代码或依赖指纹')
    if not identity['code_sha256']:
        raise ValueError(f'{method} 的代码指纹为空')
    if method != 'jev_as_controller' and (not identity['checkpoint'].get('files')
            or not identity['packages'] or not identity['code_sha256']):
        raise ValueError(f'{method} 的模型、代码或依赖指纹为空')
    return {key: identity[key] for key in keys}


def close_values(left, right):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(left, right))


def read_repeat(directory):
    directory = Path(directory).resolve()
    manifest = directory/'sample.jsonl'
    metadata_path = directory/'sample.meta.json'
    rows = load_sample(manifest)
    metadata = json.loads(metadata_path.read_text())
    sample_sha = sha256(manifest)
    counts = Counter(row['duration_group'] for row in rows)
    if (len(rows) != 100 or counts != {group: 25 for group in GROUPS}
            or metadata.get('sample_sha256') != sample_sha or metadata.get('n') != 100
            or metadata.get('per_group') != 25 or type(metadata.get('seed')) is not int
            or metadata.get('sampling') != 'equal_official_duration_groups'
            or metadata.get('selected_ids') != [row['id'] for row in rows]
            or metadata.get('selected_counts') != {str(group): 25 for group in GROUPS}):
        raise ValueError(f'每次必须是经元数据校验的四组各 25 题：{directory}')
    for key in ('source_manifest', 'source_reference'):
        source = metadata.get(key, {})
        path = Path(source.get('path', ''))
        if not path.is_file() or sha256(path) != source.get('sha256'):
            raise ValueError(f'原始标注文件缺失或 SHA 不匹配：{key}，{directory}')
    predictions, sources, metrics, identities = {}, {}, {}, {}
    for method, folder in METHODS.items():
        predictions[method], sources[folder], metrics[method] = read_source(
            directory/folder, rows, sample_sha, method)
        identities[method] = sources[folder]['identity']
        if method != 'jev_as_controller' and identities[method].get('seed') != metadata['seed']:
            raise ValueError(f'基线 seed 与抽样元数据不一致：{directory/folder}')
    controller = identities['jev_as_controller']
    protocol = identities['qwen35']['protocol']
    if any(identity['protocol'] != protocol for identity in identities.values()):
        raise ValueError(f'同一次三种方法的协议不一致：{directory}')
    media = identities['qwen35'].get('media')
    if (not isinstance(media, dict) or set(media) != {row['video_key'] for row in rows}
            or media != identities['videojev'].get('media')):
        raise ValueError(f'两基线媒体指纹不一致或未完整覆盖样本视频：{directory}')
    for video_key, identity in media.items():
        if (not all(key in identity for key in ('path', 'bytes', 'mtime_ns', 'sha256'))
                or type(identity['bytes']) is not int or identity['bytes'] < 0
                or not isinstance(identity['sha256'], str) or len(identity['sha256']) != 64):
            raise ValueError(f'视频媒体指纹不完整：{video_key}')
    if (controller.get('beta') != 2 or controller.get('sources') != {
            'qwen': sources['qwen'], 'videojev': sources['videojev']}):
        raise ValueError(f'控制器 beta 或对应基线来源指纹错误：{directory}')
    saved_metrics = json.loads((directory/'controller/metrics.json').read_text())
    if saved_metrics.get('beta') != 2:
        raise ValueError(f'控制器 metrics 中的 beta 错误：{directory}')
    for row in rows:
        raw, jev, actual = [predictions[method][row['id']] for method in METHODS]
        if (not isinstance(raw.get('preprocessing'), dict) or not raw['preprocessing']
                or raw['preprocessing'] != jev.get('preprocessing')):
            raise ValueError(f'基线预处理不一致：{row["id"]}')
        identity = media[row['video_key']]
        source = raw['preprocessing'].get('source', {})
        if (source.get('kind') != 'video' or source.get('path') != identity['path']
                or source.get('identity') != {key: identity[key] for key in ('path', 'bytes', 'mtime_ns')}
                or Path(identity['path']).resolve() != Path(row['video_path']).resolve()):
            raise ValueError(f'逐题预处理源身份与媒体指纹不一致：{row["id"]}')
        expected = control(raw, jev, 2)
        if (actual.get('beta') != 2 or actual.get('preprocessing') != raw['preprocessing']
                or actual.get('candidate_logits') != raw['candidate_logits']
                or any(not close_values(actual.get(key, []), expected[key]) for key in (
                    'probabilities', 'qwen_probabilities', 'videojev_probabilities'))):
            raise ValueError(f'控制器预测与源概率及既定公式不符：{row["id"]}')
    comparison_path = directory/'controller/comparison.json'
    if json.loads(comparison_path.read_text()) != metrics:
        raise ValueError(f'控制器 comparison 与逐题重算结果不一致：{directory}')
    files = [manifest, metadata_path, comparison_path]
    files += [directory/folder/name for folder in METHODS.values()
              for name in ('run.json', 'predictions.jsonl', 'metrics.json')]
    provenance = dict(directory=str(directory), seed=metadata['seed'], sample_sha256=sample_sha,
        files_sha256={str(path.relative_to(directory)): sha256(path) for path in files},
        media_content={key: {field: value[field] for field in ('bytes', 'sha256')} for key, value in media.items()},
        source_annotations={key: metadata[key]['sha256'] for key in ('source_manifest', 'source_reference')})
    return dict(provenance=provenance, metrics=metrics,
        fingerprints={method: fingerprint(identity, method) for method, identity in identities.items()},
        question_ids={row['id'] for row in rows}, video_keys={row['video_key'] for row in rows})


def summarize(runs, out):
    if len(runs) != 3:
        raise ValueError('请提供恰好三个完整运行根目录')
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'拒绝覆盖非空汇总目录：{out}')
    repeats = [read_repeat(directory) for directory in runs]
    if (len({r['provenance']['seed'] for r in repeats}) != 3
            or len({r['provenance']['sample_sha256'] for r in repeats}) != 3
            or len({frozenset(r['question_ids']) for r in repeats}) != 3):
        raise ValueError('三次运行必须使用不同种子和不同的抽样问题集合')
    for repeat in repeats[1:]:
        if (repeat['fingerprints'] != repeats[0]['fingerprints']
                or repeat['provenance']['source_annotations'] != repeats[0]['provenance']['source_annotations']):
            raise ValueError('跨次模型、协议、配置、代码、关键依赖或源标注指纹不一致')
    seen_media = {}
    for repeat in repeats:
        for video_key, content in repeat['provenance']['media_content'].items():
            if video_key in seen_media and seen_media[video_key] != content:
                raise ValueError(f'跨次重叠视频的内容 SHA 或字节数不同：{video_key}')
            seen_media[video_key] = content
    methods = {}
    for method in METHODS:
        reports = [repeat['metrics'][method] for repeat in repeats]
        methods[method] = dict(overall=accuracy_statistics([r['accuracy'] for r in reports]),
            by_duration_group={str(group): accuracy_statistics(
                [r['by_duration_group'][str(group)]['accuracy'] for r in reports]) for group in GROUPS})
    overlaps = [dict(seeds=[a['provenance']['seed'], b['provenance']['seed']],
        shared_questions=len(a['question_ids'] & b['question_ids']),
        shared_videos=len(a['video_keys'] & b['video_keys'])) for a, b in itertools.combinations(repeats, 2)]
    note = ('这是三次独立分层抽样的准确率方差，各次样本可以重叠，并非 300 道互异题目；'
            '也不是三种方法之间的方差。仅三次重复不足以支持有力的统计显著性结论。')
    limitation = ('未获得完整 QA 训练清单，无法按视频或问题确认训练与本验证集隔离；'
                  '不能称已验证无训练污染。')
    result = dict(schema='longvideobench_three_stratified_repeats_v1', repeats=3, questions_per_repeat=100,
        split='validation_subset', sampling='equal_official_duration_groups', beta=2,
        accuracy_unit='fraction', variance_unit='accuracy_fraction_squared', std_unit='accuracy_fraction',
        primary_variance='sample_variance', sample_variance_ddof=1, population_variance_ddof=0,
        methods=methods, provenance=[r['provenance'] for r in repeats], overlaps=overlaps,
        unique_questions=len(set.union(*(r['question_ids'] for r in repeats))),
        compatibility_fingerprints=repeats[0]['fingerprints'], note=note,
        training_overlap_verified=False, limitation=limitation,
        created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha256(__file__))
    seeds = [repeat['provenance']['seed'] for repeat in repeats]
    lines = ['# LongVideoBench 三次分层采样对比', '',
        '每次 100 道题，官方四个时长组各 25 道；固定模型、协议及控制器 β=2。', '',
        '| 方法 | 范围 | '+ ' | '.join(f'seed {seed}' for seed in seeds)
        +' | 均值 | 样本方差（百分点²） | 样本标准差（百分点） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for method, report in methods.items():
        for name, stats in [('总体', report['overall'])]+[
                (DURATION_NAMES[group], report['by_duration_group'][str(group)]) for group in GROUPS]:
            numbers = [f'{value*100:.2f}%' for value in stats['values']]
            lines.append('| '+ ' | '.join([NAMES[method], name, *numbers, f'{stats["mean"]*100:.2f}%',
                f'{stats["sample_variance"]*10000:.4f}', f'{stats["sample_std"]*100:.4f}'])+' |')
    lines += ['', '主报告样本方差使用 ddof=1；JSON 同时保存 ddof=0 总体方差及两种标准差。', '', note,
        '', limitation, '',
        f'三次合计涉及 {result["unique_questions"]} 道互异题目。', '']
    lines += [f'- seed {entry["seeds"][0]} / {entry["seeds"][1]}：重叠 '
              f'{entry["shared_questions"]} 道题、{entry["shared_videos"]} 个视频。' for entry in overlaps]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=3, required=True, help='三个完整运行根目录')
    parser.add_argument('--out', required=True, help='新的汇总目录')
    args = parser.parse_args()
    result = summarize(**vars(args))
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
