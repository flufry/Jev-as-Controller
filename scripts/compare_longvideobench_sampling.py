"""比较指定采样设置的新基线与原 2 FPS Controller，复用同一批分层抽样题目。"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_longvideobench_controller import read_source, softmax
from scripts.summarize_longvideobench_repeats import (
    METHODS, NAMES, accuracy_statistics, close_values, fingerprint, read_repeat,
)
from timejev.longvideobench_benchmark import DURATION_NAMES
from timejev.lvb_experiment import GROUPS, load_sample, read_predictions, sampling_parameters, sha256, write_json
from timejev.user_data import load_manifest
from timejev.video import sampling_grid


def distribution(values):
    return dict(n=len(values), minimum=min(values), maximum=max(values), mean=statistics.mean(values),
                total=sum(values)) if values else dict(n=0, minimum=None, maximum=None, mean=None, total=None)


def validate_timings(values):
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for key, value in values.items() if key.endswith('_seconds')):
        raise ValueError('计时必须是有限非负秒数')


def resources(directory, rows, predictions):
    videos, input_tokens, timings = {}, [], {}
    for row in rows:
        prediction = predictions[row['id']]
        audit = prediction['preprocessing']
        videos.setdefault(row['video_key'], dict(frames=audit.get('frames'), visual_tokens=audit.get('tokens')))
        if prediction.get('input_tokens') is not None:
            input_tokens.append(prediction['input_tokens'])
        timing = prediction.get('timing', {})
        validate_timings(timing)
        for key in ('video_preprocessing_seconds', 'video_group_total_seconds'):
            if key in timing:
                previous = timings.setdefault(key, {}).setdefault(row['video_key'], timing[key])
                if previous != timing[key]:
                    raise ValueError(f'同视频重复记录的计时不一致：{row["video_key"]}')
        if 'question_inference_seconds' in timing:
            timings.setdefault('question_inference_seconds', {})[row['id']] = timing['question_inference_seconds']
    runtime_path = Path(directory)/'runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if runtime is not None:
        validate_timings(runtime)
    return dict(videos=videos, frames_per_video=distribution([v['frames'] for v in videos.values()
        if v['frames'] is not None]), visual_tokens_per_video=distribution([v['visual_tokens'] for v in videos.values()
        if v['visual_tokens'] is not None]), input_tokens_per_question=distribution(input_tokens),
        timing_seconds={key: distribution(list(values.values())) for key, values in timings.items()},
        runtime=runtime)


def read_baseline(directory, rows, old, seed, method, reference_predictions, fps=1., max_frames=4096):
    fps, max_frames = sampling_parameters(fps, max_frames)
    predictions, source, metrics = read_source(directory, rows, old['provenance']['sample_sha256'], method)
    identity = source['identity']
    old_identity = old['fingerprints'][method]
    protocol = old_identity['protocol']
    if any(protocol.get(key) != value for key, value in dict(
            fps=2., max_frames=2048, context=262144, visual_token_budget=24576).items()):
        raise ValueError('参照结果必须采用原 2 FPS / 2048 帧和既定上下文、视觉预算')
    expected = dict(protocol, fps=fps, max_frames=max_frames)
    if identity['protocol'] != expected or identity.get('seed') != seed:
        raise ValueError(f'新基线必须仅改变 FPS 和最大帧数，且 seed 匹配：{directory}')
    if any(identity.get(key) != old_identity[key] for key in ('checkpoint', 'settings', 'packages')):
        raise ValueError(f'新旧基线模型、设置或依赖不一致：{directory}')
    media = identity.get('media', {})
    if set(media) != {row['video_key'] for row in rows}:
        raise ValueError(f'新基线媒体指纹未完整覆盖样本：{directory}')
    video_audits = {}
    for row in rows:
        prediction = predictions[row['id']]
        audit = prediction.get('preprocessing', {})
        previous = video_audits.setdefault(row['video_key'], audit)
        if previous != audit:
            raise ValueError(f'同视频不同题目的新基线预处理不一致：{row["video_key"]}')
        content = media[row['video_key']]
        if ({key: content.get(key) for key in ('bytes', 'sha256')}
                != old['provenance']['media_content'][row['video_key']]):
            raise ValueError(f'新旧视频媒体内容不一致：{row["video_key"]}')
        source_audit = audit.get('source', {})
        if (source_audit.get('kind') != 'video' or source_audit.get('path') != content.get('path')
                or source_audit.get('identity') != {key: content.get(key) for key in ('path', 'bytes', 'mtime_ns')}
                or Path(content['path']).resolve() != Path(row['video_path']).resolve()):
            raise ValueError(f'新基线预处理源身份不匹配：{row["id"]}')
        duration = audit.get('sampled_duration', 0)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= row['duration']:
            raise ValueError(f'新基线采样时长无效：{row["id"]}')
        if duration != reference_predictions[row['id']]['preprocessing'].get('sampled_duration'):
            raise ValueError(f'新旧同方法的完整视频采样时长不一致：{row["id"]}')
        times, effective, mode = sampling_grid(duration, fps, max_frames)
        grid = audit.get('grid_thw', [])
        if (audit.get('fps') != fps or audit.get('frames') != len(times)
                or audit.get('effective_fps') != effective or audit.get('sampling_mode') != mode
                or len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid)
                or grid[0] != (len(times)+1)//2 or grid[1]%2 or grid[2]%2
                or audit.get('tokens') != grid[0]*(grid[1]//2)*(grid[2]//2)
                or not 0 < audit['tokens'] <= 24576
                or audit.get('processed_hw') != [grid[1]*16, grid[2]*16]
                or grid[1]*grid[2]*256 > 409600):
            raise ValueError(f'新基线实际帧或视觉 token 记录与 {fps:g} FPS / {max_frames} 帧协议不符：{row["id"]}')
        if method == 'qwen35':
            logits = prediction.get('candidate_logits', [])
            if (len(logits) != len(row['options']) or any(not math.isfinite(v) for v in logits)
                    or not close_values(softmax(logits), prediction['probabilities'])):
                raise ValueError(f'新 Qwen 候选 logits 与概率不一致：{row["id"]}')
    runtime = Path(directory)/'runtime.json'
    if runtime.is_file():
        source['runtime_sha256'] = sha256(runtime)
    return predictions, source, metrics


def compare(baseline_root, reference_root_pattern, out, seeds=(42, 43, 44), fps=1., max_frames=4096):
    fps, max_frames = sampling_parameters(fps, max_frames)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('需要三个不同的抽样 seed')
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'拒绝覆盖非空汇总目录：{out}')
    reports, provenance, audits, old_repeats, new_fingerprints = [], [], [], [], []
    for seed in seeds:
        reference = Path(reference_root_pattern.format(seed=seed)).resolve()
        old = read_repeat(reference)
        if old['provenance']['seed'] != seed:
            raise ValueError(f'参照目录的采样 seed 错误：{reference}')
        rows = load_sample(reference/'sample.jsonl')
        meta = json.loads((reference/'sample.meta.json').read_text())
        originals = {row['id']: row for row in load_manifest(meta['source_manifest']['path'])}
        official = {row['id']: row for row in json.loads(Path(meta['source_reference']['path']).read_text())}
        for row in rows:
            original = originals.get(row['id'], {})
            target = official.get(row.get('question_id'), {})
            if (any(row.get(key) != original.get(key) for key in (
                    'question_id', 'task', 'video_key', 'video_path', 'duration', 'query', 'options', 'answer_index'))
                    or row['duration_group'] != target.get('duration_group')
                    or row['answer_index'] != target.get('correct_choice')):
                raise ValueError(f'参照样本的问题或标签与源标注不一致：{row["id"]}')
        current, sources, predictions, model_fingerprints = {}, {}, {}, {}
        seed_audits = {}
        reference_predictions = {method: read_predictions(reference/METHODS[method]/'predictions.jsonl', rows)
                                 for method in ('qwen35', 'videojev')}
        for method in ('qwen35', 'videojev'):
            directory = Path(baseline_root)/f'seed{seed}'/METHODS[method]
            predictions[method], sources[method], current[method] = read_baseline(
                directory, rows, old, seed, method, reference_predictions[method], fps, max_frames)
            model_fingerprints[method] = fingerprint(sources[method]['identity'], method)
            seed_audits[method] = resources(directory, rows, predictions[method])
        if sources['qwen35']['identity']['media'] != sources['videojev']['identity']['media']:
            raise ValueError('新两基线的媒体指纹不一致')
        if any(predictions['qwen35'][row['id']]['preprocessing'] != predictions['videojev'][row['id']]['preprocessing']
               for row in rows):
            raise ValueError('新两基线的逐题预处理不一致')
        old_controller = read_predictions(reference/'controller/predictions.jsonl', rows)
        seed_audits['jev_as_controller'] = resources(reference/'controller', rows, old_controller)
        seed_audits['jev_as_controller']['source_baselines'] = {method: resources(
            reference/METHODS[method], rows, reference_predictions[method])
            for method in ('qwen35', 'videojev')}
        current['jev_as_controller'] = old['metrics']['jev_as_controller']
        reports.append(current)
        provenance.append(dict(seed=seed, reference=old['provenance'], new_baselines=sources))
        audits.append(dict(seed=seed, methods=seed_audits))
        old_repeats.append(old)
        new_fingerprints.append(model_fingerprints)
    if (len({old['provenance']['sample_sha256'] for old in old_repeats}) != 3
            or len({frozenset(old['question_ids']) for old in old_repeats}) != 3):
        raise ValueError('三个 seed 必须对应不同的原始样本')
    seen_media = {}
    for old, current in zip(old_repeats, new_fingerprints):
        if old['fingerprints'] != old_repeats[0]['fingerprints'] or current != new_fingerprints[0]:
            raise ValueError('同方法跨三个 seed 的模型、设置、协议、代码或依赖指纹不一致')
        if old['provenance']['source_annotations'] != old_repeats[0]['provenance']['source_annotations']:
            raise ValueError('三个 seed 的源标注不一致')
        for key, content in old['provenance']['media_content'].items():
            if key in seen_media and seen_media[key] != content:
                raise ValueError(f'跨 seed 的重叠视频内容不一致：{key}')
            seen_media[key] = content
    methods, paired = {}, {}
    for method in METHODS:
        methods[method] = dict(overall=accuracy_statistics([r[method]['accuracy'] for r in reports]),
            by_duration_group={str(group): accuracy_statistics([r[method]['by_duration_group'][str(group)]['accuracy']
                for r in reports]) for group in GROUPS})
        if method != 'jev_as_controller':
            delta = [r['jev_as_controller']['accuracy']-r[method]['accuracy'] for r in reports]
            paired[method] = dict(values=delta, mean=statistics.mean(delta),
                sample_variance=statistics.variance(delta), sample_std=statistics.stdev(delta),
                better_each_seed=[v > 1e-12 for v in delta], better_all_seeds=all(v > 1e-12 for v in delta),
                better_mean=statistics.mean(delta) > 1e-12)
    limitation = ('未获得完整 QA 训练清单，无法按视频或问题确认训练与本验证集隔离；'
                  '不能称已验证无训练污染。')
    compute_note = (f'两条新基线为 {fps:g} FPS、最多 {max_frames} 帧；Controller 直接复用旧 2 FPS、最多 2048 帧、β=2 结果，'
        '未使用新基线重新融合。相同视觉 token 上限不代表严格算力对齐；旧 Controller 未保存同口径阶段计时。')
    timing_limitation = ('逐视频计时去重适用于同视频各题共享一次连续处理的记录；'
        '若在同视频多题之间恢复运行，产生不同的重复计时，本汇总会拒绝，需单独审计恢复记录。')
    result = dict(schema='longvideobench_changed_sampling_comparison_v1', seeds=list(seeds),
        baseline_sampling=dict(fps=fps, max_frames=max_frames),
        controller_sampling=dict(fps=2., max_frames=2048, beta=2),
        questions_per_seed=100, split='validation_subset', methods=methods, controller_minus_baseline=paired,
        accuracy_unit='fraction', variance_unit='accuracy_fraction_squared', sample_variance_ddof=1,
        provenance=provenance, resource_audits=audits, new_fingerprints=new_fingerprints[0],
        reference_fingerprints=old_repeats[0]['fingerprints'],
        old_baselines=[dict(seed=seed, metrics=old['metrics']) for seed, old in zip(seeds, old_repeats)],
        training_overlap_verified=False, limitation=limitation, compute_note=compute_note,
        resource_timing_limitation=timing_limitation,
        statistics_note='三次抽样可重叠；方差衡量跨采样波动，不能凭三次结果宣称显著性。',
        script_sha256=sha256(__file__), created_at=datetime.now(timezone.utc).isoformat())
    lines = ['# LongVideoBench 新基线与旧 Controller 对比', '', compute_note, '',
        '| 方法 | 时长组 | '+' | '.join(f'seed {s}' for s in seeds)+' | 均值 | 样本方差（百分点²） | 标准差（百分点） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for method, report in methods.items():
        for scope, stats in [('总体', report['overall'])]+[(DURATION_NAMES[g], report['by_duration_group'][str(g)]) for g in GROUPS]:
            lines.append('| '+' | '.join([NAMES[method], scope, *[f'{v*100:.2f}%' for v in stats['values']],
                f'{stats["mean"]*100:.2f}%', f'{stats["sample_variance"]*10000:.4f}', f'{stats["sample_std"]*100:.4f}'])+' |')
    lines += ['', '| Controller 减基线 | '+' | '.join(f'seed {s}（百分点）' for s in seeds)
        +' | 均值差（百分点） | 每次均更优 | 均值更优 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for method, stats in paired.items():
        lines.append('| '+' | '.join([NAMES[method], *[f'{v*100:+.2f}' for v in stats['values']],
            f'{stats["mean"]*100:+.2f}', '是' if stats['better_all_seeds'] else '否',
            '是' if stats['better_mean'] else '否'])+' |')
    lines += ['', '| seed | 方法 | 帧数均值/最大 | 视觉 token 均值/最大 | 峰值已分配显存（GiB） | wall（秒） |',
        '| --- | --- | --- | --- | --- | --- |']
    for audit in audits:
        for method in METHODS:
            values = audit['methods'][method]
            ranges = [f'{values[key]["mean"]:.1f}/{values[key]["maximum"]}'
                      if values[key]['n'] else '—' for key in ('frames_per_video', 'visual_tokens_per_video')]
            runtime = values['runtime'] or {}
            peak = runtime.get('peak_cuda_allocated_bytes')
            wall = runtime.get('wall_seconds')
            lines.append('| '+' | '.join([str(audit['seed']), NAMES[method], *ranges,
                f'{peak/2**30:.2f}' if peak is not None else '—',
                f'{wall:.2f}' if wall is not None else '—'])+' |')
    lines += ['', 'Controller 的帧数与视觉 token 表示其原两条基线各自使用的相同输入，'
              '仅供输入规模比较，不是两模型计算量之和；缺失显存或耗时不填估计值。']
    lines += ['', '样本方差采用 ddof=1；逐视频帧数、视觉 token、逐题输入 token 及现有计时保存在 JSON 的 resource_audits。',
        '视频预处理和 VideoJev 整组计时按视频去重；Qwen 单题推理计时按题相加。缺失计时保留为空，不推算旧 Controller 总耗时。',
        '', timing_limitation, '', result['statistics_note'], '', limitation]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--reference-root-pattern', default='runs/lvb100_equal_seed{seed}')
    parser.add_argument('--seeds', type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument('--fps', type=float, default=1., help='新基线的预期 FPS；默认保留原 1 FPS 比较')
    parser.add_argument('--max-frames', type=int, default=4096, help='新基线的预期最大帧数')
    parser.add_argument('--out', required=True)
    result = compare(**vars(parser.parse_args()))
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
