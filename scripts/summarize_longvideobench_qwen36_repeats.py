"""重新核验 seed42/43/44 的 Qwen3.6 单次产物，汇总跨采样准确率方差。"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_longvideobench_qwen36 import summarize as validate_single
from scripts.summarize_longvideobench_repeats import accuracy_statistics, fingerprint
from timejev.longvideobench_benchmark import DURATION_NAMES
from timejev.lvb_experiment import GROUPS, load_sample, sha256, write_json

METHODS = ('qwen36', 'videojev', 'jev_as_controller')
NAMES = dict(qwen36='Qwen3.6-35B-A3B（4 FPS）', videojev='VideoJev-4B（4 FPS）',
             jev_as_controller='Jev-as-Controller（双 2 FPS 源）')


def difference_statistics(values):
    if len(values) != 3 or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values):
        raise ValueError('融合增益需要三次有效的准确率差')
    return dict(values=values, mean=statistics.mean(values), sample_variance=statistics.variance(values),
        population_variance=statistics.pvariance(values), sample_std=statistics.stdev(values),
        population_std=statistics.pstdev(values))


def read_single(root):
    root = Path(root).resolve()
    path = root/'summary/results.json'
    saved = json.loads(path.read_text())
    if (saved.get('schema') != 'longvideobench_qwen36_single_seed_v1'
            or type(saved.get('seed')) is not int or saved['seed'] not in (42, 43, 44)):
        raise ValueError(f'需要 seed42/43/44 的完整 Qwen3.6 单次报告：{root}')
    provenance = saved['provenance']
    args = {role: source['directory'] for role, source in provenance['runs'].items()}
    fresh = validate_single(manifest=provenance['sample']['path'], out=None, seed=saved['seed'], **args)
    # 历史报告的生成时间和汇总脚本版本可不同，模型、来源和所有重算结果必须一致。
    stable = lambda report: {k: v for k, v in report.items() if k not in ('created_at', 'script_sha256')}
    if stable(saved) != stable(fresh):
        raise ValueError(f'单次报告与源预测及重算结果不一致：{root}')
    rows = load_sample(provenance['sample']['path'])
    return dict(report=fresh, question_ids={r['id'] for r in rows}, video_keys={r['video_key'] for r in rows},
        provenance=dict(seed=fresh['seed'], directory=str(root), summary_path=str(path), summary_sha256=sha256(path),
            protocols=fresh['protocols'], beta=fresh['beta'],
            summary_script_sha256=saved['script_sha256'], sources=fresh['provenance']))


def summarize(runs, out):
    if len(runs) != 3:
        raise ValueError('需要三个完整单次运行根目录')
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'拒绝覆盖非空汇总目录：{out}')
    checked = sorted((read_single(root) for root in runs), key=lambda r: r['report']['seed'])
    reports = [entry['report'] for entry in checked]
    if ([r['seed'] for r in reports] != [42, 43, 44]
            or len({r['provenance']['sample']['sha256'] for r in reports}) != 3
            or len({frozenset(entry['question_ids']) for entry in checked}) != 3):
        raise ValueError('必须恰好包含 seed42/43/44 及三个不同的原始抽样问题集合')
    models, annotations, seen_media = None, None, {}
    for report in reports:
        provenance = report['provenance']
        identities = {role: source['identity'] for role, source in provenance['runs'].items()}
        current = {role: fingerprint(identity, identity['method']) for role, identity in identities.items()}
        source_hashes = {key: provenance['sample'][key]['sha256'] for key in ('source_manifest', 'source_reference')}
        if models is not None and (current != models or source_hashes != annotations):
            raise ValueError('跨 seed 的模型、协议、配置、代码、依赖或原始标注 SHA 不一致')
        models, annotations = current, source_hashes
        for key, media in identities['qwen_controller']['media'].items():
            content = {field: media[field] for field in ('sha256', 'bytes')}
            if key in seen_media and seen_media[key] != content:
                raise ValueError(f'跨 seed 的重叠视频内容 SHA 或字节数不同：{key}')
            seen_media[key] = content
    methods = {method: dict(overall=accuracy_statistics([r['methods'][method]['accuracy'] for r in reports]),
        by_duration_group={str(g): accuracy_statistics([r['methods'][method]['by_duration_group'][str(g)]['accuracy']
            for r in reports]) for g in GROUPS}) for method in METHODS}
    gain = dict(overall=difference_statistics([r['controller2_minus_qwen2']['accuracy_difference'] for r in reports]),
        by_duration_group={str(g): difference_statistics([r['controller2_minus_qwen2']['by_duration_group'][str(g)]
            for r in reports]) for g in GROUPS}, comparison_scope='same_2fps_protocol')
    cross_protocol = {method: dict(overall=difference_statistics([
        r['controller_minus_baseline'][method]['accuracy_difference'] for r in reports]),
        comparison_scope='cross_protocol_requested_standalone_baseline') for method in ('qwen36', 'videojev')}
    overlaps = [dict(seeds=[a['report']['seed'], b['report']['seed']],
        shared_questions=len(a['question_ids'] & b['question_ids']),
        shared_videos=len(a['video_keys'] & b['video_keys'])) for a, b in itertools.combinations(checked, 2)]
    note = ('方差描述三次分层采样准确率的波动，不是三种方法之间的方差。三次题目可以重叠，'
            '不代表 300 道互异问题；三次重复不足以支持有力的统计显著性结论。')
    result = dict(schema='longvideobench_qwen36_three_repeats_v1', seeds=[42, 43, 44], repeats=3,
        questions_per_seed=100, split='validation_subset', methods=methods, controller2_minus_qwen2=gain,
        controller_minus_baseline=cross_protocol, protocols=reports[0]['protocols'], beta=2,
        accuracy_unit='fraction', gain_unit='accuracy_difference_fraction', variance_unit='accuracy_fraction_squared',
        sample_variance_ddof=1, population_variance_ddof=0, primary_variance='sample_variance',
        controller_source_metrics_by_seed=[dict(seed=r['seed'], **r['controller_source_metrics']) for r in reports],
        provenance=[entry['provenance'] for entry in checked], compatibility_fingerprints=models,
        overlaps=overlaps, unique_questions=len(set.union(*(entry['question_ids'] for entry in checked))),
        statistics_note=note, training_overlap_verified=False, limitation=reports[0]['limitation'],
        script_sha256=sha256(__file__), created_at=datetime.now(timezone.utc).isoformat())
    lines = ['# LongVideoBench：Qwen3.6 三次分层采样对照', '',
        '每个 seed 原样使用 100 道题，四组各 25 道。两条独立基线使用 4 FPS/4096 帧；'
        'Controller 的 Qwen3.6 与 VideoJev 两源均为 2 FPS/2048 帧，固定 β=2。', '',
        '| 方法 | 范围 | seed42 | seed43 | seed44 | 均值 | 样本方差（百分点²） | 标准差（百分点） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for method, report in methods.items():
        for scope, stats in [('总体', report['overall'])]+[(DURATION_NAMES[g], report['by_duration_group'][str(g)]) for g in GROUPS]:
            lines.append('| '+' | '.join([NAMES[method], scope, *[f'{v*100:.2f}%' for v in stats['values']],
                f'{stats["mean"]*100:.2f}%', f'{stats["sample_variance"]*10000:.4f}', f'{stats["sample_std"]*100:.4f}'])+' |')
    lines += ['', '| 同协议融合增益 Controller2 − Qwen2 | seed42（百分点） | seed43（百分点） | seed44（百分点） | 均值（百分点） | 样本方差（百分点²） |',
        '| --- | --- | --- | --- | --- | --- |']
    for scope, stats in [('总体', gain['overall'])]+[(DURATION_NAMES[g], gain['by_duration_group'][str(g)]) for g in GROUPS]:
        lines.append('| '+' | '.join([scope, *[f'{v*100:+.2f}' for v in stats['values']],
            f'{stats["mean"]*100:+.2f}', f'{stats["sample_variance"]*10000:.4f}'])+' |')
    lines += ['', 'Controller2 与指定 4 FPS 独立基线的跨协议差异另存 JSON，不能全部归因于融合。'
        '每个 seed 的帧数、token、计时与完整来源均已重新核验，资源审计保留在对应单次报告。', '',
        '主报告使用 ddof=1 样本方差；JSON 同时保存 ddof=0 总体方差和两种标准差。', '', note, '',
        f'三次共涉及 {result["unique_questions"]} 道互异问题。', '']
    lines += [f'- seed{entry["seeds"][0]} / seed{entry["seeds"][1]}：重叠 '
        f'{entry["shared_questions"]} 道题、{entry["shared_videos"]} 个视频。' for entry in overlaps]
    lines += ['', result['limitation']]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=3, required=True, help='三个包含 summary/results.json 的单次根目录')
    parser.add_argument('--out', required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
