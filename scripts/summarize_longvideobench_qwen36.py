"""核验指定 seed 的 Qwen3.6、VideoJev 和双 2 FPS 源 Controller 单次对照。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_longvideobench_sampling import resources
from scripts.eval_longvideobench_controller import control, read_source, softmax
from scripts.summarize_longvideobench_repeats import close_values, fingerprint
from timejev.longvideobench_benchmark import DURATION_NAMES
from timejev.lvb_experiment import GROUPS, load_sample, sha256, write_json
from timejev.user_data import load_manifest
from timejev.video import sampling_grid


def read_sample(manifest, seed=42):
    manifest = Path(manifest).resolve()
    meta_path = manifest.with_suffix('.meta.json')
    rows = load_sample(manifest)
    meta = json.loads(meta_path.read_text())
    digest = sha256(manifest)
    if (len(rows) != 100 or Counter(r['duration_group'] for r in rows) != {g: 25 for g in GROUPS}
            or meta.get('seed') != seed or type(meta['seed']) is not int
            or meta.get('sample_sha256') != digest or meta.get('n') != 100
            or meta.get('per_group') != 25 or meta.get('sampling') != 'equal_official_duration_groups'
            or meta.get('selected_ids') != [r['id'] for r in rows]
            or meta.get('selected_counts') != {str(g): 25 for g in GROUPS}):
        raise ValueError(f'需要原 seed{seed} 的四个时长组各 25 题及匹配的抽样元数据')
    for key in ('source_manifest', 'source_reference'):
        source = meta.get(key, {})
        path = Path(source.get('path', ''))
        if not path.is_file() or sha256(path) != source.get('sha256'):
            raise ValueError(f'源标注文件缺失或 SHA 不匹配：{key}')
    originals = {r['id']: r for r in load_manifest(meta['source_manifest']['path'])}
    official = {r['id']: r for r in json.loads(Path(meta['source_reference']['path']).read_text())}
    for row in rows:
        original, target = originals.get(row['id'], {}), official.get(row.get('question_id'), {})
        if (any(row.get(key) != original.get(key) for key in ('question_id', 'task', 'video_key',
                'video_path', 'duration', 'query', 'options', 'answer_index'))
                or row['duration_group'] != target.get('duration_group')
                or row['answer_index'] != target.get('correct_choice')):
            raise ValueError(f'抽样问题或标签与原始标注不一致：{row["id"]}')
    return rows, dict(path=str(manifest), sha256=digest, meta_sha256=sha256(meta_path),
                     seed=seed, source_manifest=meta['source_manifest'], source_reference=meta['source_reference'])


def validate_audits(rows, predictions, identity):
    media, protocol = identity.get('media'), identity['protocol']
    if not isinstance(media, dict) or set(media) != {r['video_key'] for r in rows}:
        raise ValueError('媒体指纹必须完整覆盖本次样本视频')
    by_video = {}
    for row in rows:
        prediction = predictions[row['id']]
        audit = prediction.get('preprocessing', {})
        content = media[row['video_key']]
        if (not all(key in content for key in ('path', 'bytes', 'mtime_ns', 'sha256'))
                or type(content['bytes']) is not int or content['bytes'] < 0
                or not isinstance(content['sha256'], str) or len(content['sha256']) != 64):
            raise ValueError(f'媒体指纹不完整：{row["video_key"]}')
        source = audit.get('source', {})
        if (source.get('kind') != 'video' or source.get('path') != content['path']
                or source.get('identity') != {k: content[k] for k in ('path', 'bytes', 'mtime_ns')}
                or Path(content['path']).resolve() != Path(row['video_path']).resolve()
                or by_video.setdefault(row['video_key'], audit) != audit):
            raise ValueError(f'视频预处理或媒体源身份不一致：{row["id"]}')
        duration = audit.get('sampled_duration', 0)
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= row['duration']:
            raise ValueError(f'视频采样时长无效：{row["id"]}')
        times, effective, mode = sampling_grid(duration, protocol['fps'], protocol['max_frames'])
        grid = audit.get('grid_thw', [])
        if (audit.get('fps') != protocol['fps'] or audit.get('frames') != len(times)
                or audit.get('effective_fps') != effective or audit.get('sampling_mode') != mode
                or len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid)
                or grid[0] != (len(times)+1)//2 or grid[1]%2 or grid[2]%2
                or audit.get('tokens') != grid[0]*(grid[1]//2)*(grid[2]//2)
                or not 0 < audit['tokens'] <= 24576 or grid[1]*grid[2]*256 > 409600
                or audit.get('processed_hw') != [grid[1]*16, grid[2]*16]):
            raise ValueError(f'实际帧时间线或视觉 token 与声明协议不符：{row["id"]}')
        if identity['method'] == 'qwen36':
            values = prediction.get('candidate_logits', [])
            if (len(values) != len(row['options'])
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or not close_values(softmax(values), prediction['probabilities'])):
                raise ValueError(f'Qwen3.6 首步候选 logits 与概率不符：{row["id"]}')


def summarize(manifest, qwen_baseline, qwen_controller, videojev_baseline, videojev_controller, controller, out, seed=42):
    # out=None 仅重验并返回结果，供重复实验汇总复用，不写入任何单次产物。
    if out is not None:
        out = Path(out).resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f'拒绝覆盖非空输出目录：{out}')
    rows, sample = read_sample(manifest, seed)
    paths = dict(qwen_baseline=qwen_baseline, qwen_controller=qwen_controller,
                 videojev_baseline=videojev_baseline, videojev_controller=videojev_controller, controller=controller)
    predictions, sources, metrics = {}, {}, {}
    for role, directory in paths.items():
        method = 'jev_as_controller' if role == 'controller' else 'qwen36' if role.startswith('qwen') else 'videojev'
        predictions[role], sources[role], metrics[role] = read_source(directory, rows, sample['sha256'], method)
        identity = sources[role]['identity']
        fingerprint(identity, method)
        fps, cap = (4., 4096) if role.endswith('baseline') else (2., 2048)
        if (any(identity['protocol'].get(k) != v for k, v in dict(fps=fps, max_frames=cap,
                context=262144, visual_token_budget=24576).items())
                or role != 'controller' and identity.get('seed') != seed):
            raise ValueError(f'方法 seed 或 4 FPS / 2 FPS 协议错误：{role}')
        if role != 'controller':
            validate_audits(rows, predictions[role], identity)
    two = sources['qwen_controller']['identity']['protocol']
    four = sources['qwen_baseline']['identity']['protocol']
    if (dict(two, fps=4., max_frames=4096) != four
            or sources['videojev_baseline']['identity']['protocol'] != four
            or sources['videojev_controller']['identity']['protocol'] != two
            or sources['controller']['identity']['protocol'] != two):
        raise ValueError('两模型同协议设置不一致，或不同协议还改变了 FPS/帧上限之外的设置')
    for prefix in ('qwen', 'videojev'):
        baseline, auxiliary = [sources[f'{prefix}_{suffix}']['identity'] for suffix in ('baseline', 'controller')]
        keys = ('checkpoint', 'settings', 'packages', 'code_sha256') if prefix == 'qwen' else ('checkpoint', 'settings', 'packages')
        if any(baseline[k] != auxiliary[k] for k in keys):
            raise ValueError(f'{prefix} 两种协议的模型、设置、依赖或模型实现代码不一致')
    config_identity = sources['qwen_baseline']['identity']['checkpoint']
    config_path = Path(config_identity['path'])/'config.json'
    expected_config = config_identity['files'].get('config.json', {})
    if (not config_path.is_file() or sha256(config_path) != expected_config.get('sha256')
            or json.loads(config_path.read_text()).get('model_type') != 'qwen3_5_moe'):
        raise ValueError('Qwen3.6 必须对应已记录 config SHA 的本地 MoE checkpoint，不能冒充原 4B 模型')
    media = sources['qwen_controller']['identity']['media']
    if any(sources[role]['identity']['media'] != media for role in paths if role != 'controller'):
        raise ValueError('四份源的媒体路径及内容指纹不一致')
    for row in rows:
        audits = {role: predictions[role][row['id']]['preprocessing'] for role in paths}
        if (audits['qwen_baseline'] != audits['videojev_baseline']
                or audits['qwen_controller'] != audits['videojev_controller']
                or any(a['sampled_duration'] != audits['qwen_controller']['sampled_duration'] for a in audits.values())):
            raise ValueError(f'同协议预处理或跨协议完整采样时长不一致：{row["id"]}')
    identity = sources['controller']['identity']
    if (identity.get('beta') != 2 or identity.get('sources') != {
            'qwen': sources['qwen_controller'], 'videojev': sources['videojev_controller']}
            or json.loads((Path(controller)/'metrics.json').read_text()).get('beta') != 2):
        raise ValueError('Controller 必须真实使用 Qwen3.6 与 VideoJev 的两份 2 FPS 源和 beta=2')
    for row in rows:
        raw, jev, actual = [predictions[role][row['id']] for role in ('qwen_controller', 'videojev_controller', 'controller')]
        expected = control(raw, jev, 2.)
        if (actual.get('beta') != 2 or actual.get('candidate_logits') != raw['candidate_logits']
                or actual.get('preprocessing') != raw['preprocessing']
                or any(not close_values(actual.get(key, []), expected[key]) for key in (
                    'probabilities', 'qwen_probabilities', 'videojev_probabilities'))):
            raise ValueError(f'Controller 逐题概率与两份 2 FPS 源及既定公式不符：{row["id"]}')
    comparison_path = Path(controller)/'comparison.json'
    if json.loads(comparison_path.read_text()) != dict(qwen36=metrics['qwen_controller'],
            videojev=metrics['videojev_controller'], jev_as_controller=metrics['controller']):
        raise ValueError('Controller comparison 与逐题重算的 Qwen3.6 来源结果不符')
    methods = dict(qwen36=metrics['qwen_baseline'], videojev=metrics['videojev_baseline'],
                   jev_as_controller=metrics['controller'])
    differences = {method: dict(accuracy_difference=methods['jev_as_controller']['accuracy']-methods[method]['accuracy'],
        comparison_scope='cross_protocol_requested_standalone_baseline',
        better=methods['jev_as_controller']['accuracy'] > methods[method]['accuracy'],
        by_duration_group={str(g): methods['jev_as_controller']['by_duration_group'][str(g)]['accuracy']
            - methods[method]['by_duration_group'][str(g)]['accuracy'] for g in GROUPS}) for method in ('qwen36', 'videojev')}
    fusion_gain = dict(accuracy_difference=metrics['controller']['accuracy']-metrics['qwen_controller']['accuracy'],
        comparison_scope='same_2fps_protocol', by_duration_group={str(g):
            metrics['controller']['by_duration_group'][str(g)]['accuracy']
            - metrics['qwen_controller']['by_duration_group'][str(g)]['accuracy'] for g in GROUPS})
    audits = {role: resources(paths[role], rows, predictions[role]) for role in paths}
    extra_files = {role: dict(path=str((Path(directory)/'runtime.json').resolve()),
        sha256=sha256(Path(directory)/'runtime.json')) for role, directory in paths.items()
        if (Path(directory)/'runtime.json').is_file()}
    note = (f'仅运行 seed{seed} 一次，不估计重复实验均值、方差或显著性。基线各自使用 4 FPS/4096 帧；'
            'Controller 的 Qwen3.6 和 VideoJev 两源均为 2 FPS/2048 帧，β=2。相同视觉预算不代表严格算力对齐。')
    limitation = '未获得完整 QA 训练清单，无法按视频或问题确认训练与验证集隔离，不能称已验证无训练污染。'
    result = dict(schema='longvideobench_qwen36_single_seed_v1', seed=seed, repetitions=1, n=100,
        split='validation_subset', methods=methods, controller_minus_baseline=differences,
        controller2_minus_qwen2=fusion_gain,
        protocols=dict(baseline=four, controller=two), beta=2, accuracy_unit='fraction',
        controller_source_metrics=dict(qwen36=metrics['qwen_controller'], videojev=metrics['videojev_controller']),
        provenance=dict(sample=sample, runs=sources, runtime_files=extra_files,
                        controller_comparison_sha256=sha256(comparison_path)), resource_audits=audits,
        statistics_note=note, training_overlap_verified=False, limitation=limitation,
        script_sha256=sha256(__file__), created_at=datetime.now(timezone.utc).isoformat())
    names = dict(qwen36='Qwen3.6-35B-A3B', videojev='VideoJev-4B', jev_as_controller='Jev-as-Controller（Qwen3.6 + VideoJev）')
    lines = ['# LongVideoBench：Qwen3.6 单次对照', '', note, '',
        '| 方法 | 输入协议 | 答对/100 | 准确率 | '+' | '.join(DURATION_NAMES[g] for g in GROUPS)+' |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for method, report in methods.items():
        lines.append('| '+' | '.join([names[method], '2 FPS/2048' if method == 'jev_as_controller' else '4 FPS/4096',
            str(report['correct']), f'{report["accuracy"]*100:.2f}%',
            *[f'{report["by_duration_group"][str(g)]["accuracy"]*100:.2f}%' for g in GROUPS]])+' |')
    lines += ['', 'Controller 使用候选内 π₀=softmax(Qwen3.6 首步裸字母 logits)，'
        'π₁=softmax(log π₀+2×VideoJev候选概率)；FS 不取 log 或再次 softmax。', '',
        '| Controller2 减指定独立基线4（跨协议） | 准确率差（百分点） | 是否更优 |', '| --- | --- | --- |']
    lines += ['| '+names[method]+f' | {report["accuracy_difference"]*100:+.2f} | '
        +('是' if report['better'] else '否')+' |' for method, report in differences.items()]
    lines += ['', f'同协议融合增益 Controller2 − Qwen3.6源2：{fusion_gain["accuracy_difference"]*100:+.2f} 个百分点。'
        '上表与 4 FPS 独立基线的差异同时包含协议变化，不能全部归因于融合。']
    lines += ['', '| 来源 | 视频帧数均值/最大 | 视觉 token 均值/最大 | 单题输入 token 均值 | wall秒 |',
        '| --- | --- | --- | --- | --- |']
    role_names = dict(qwen_baseline='Qwen3.6 独立基线（4 FPS）', qwen_controller='Qwen3.6 Controller源（2 FPS）',
        videojev_baseline='VideoJev 独立基线（4 FPS）', videojev_controller='VideoJev Controller源（2 FPS）',
        controller='Controller 融合产物（2 FPS）')
    for role, audit in audits.items():
        ranges = [f'{audit[key]["mean"]:.1f}/{audit[key]["maximum"]}' if audit[key]['n'] else '—'
                  for key in ('frames_per_video', 'visual_tokens_per_video')]
        inputs = audit['input_tokens_per_question']
        wall = (audit['runtime'] or {}).get('wall_seconds')
        lines.append('| '+' | '.join([role_names[role], *ranges, f'{inputs["mean"]:.1f}' if inputs['n'] else '—',
                                      f'{wall:.2f}' if wall is not None else '—'])+' |')
    lines += ['', '资源表记录四份模型源及融合产物；Controller 的帧数不是两模型计算量之和。'
        '缺失计时不推算，原始显存/阶段计时保存在 JSON。共享视频计时按视频去重；'
        '同视频中途恢复产生不同重复计时的记录需单独审计。', '', limitation]
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'results.json', result)
        (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--manifest')
    parser.add_argument('--qwen-baseline', required=True)
    parser.add_argument('--qwen-controller', required=True)
    parser.add_argument('--videojev-baseline')
    parser.add_argument('--videojev-controller')
    parser.add_argument('--controller', required=True)
    parser.add_argument('--out', required=True)
    args = vars(parser.parse_args())
    seed = args['seed']
    args['manifest'] = args['manifest'] or f'runs/lvb100_equal_seed{seed}/sample.jsonl'
    args['videojev_baseline'] = args['videojev_baseline'] or f'runs/lvb100_baselines_fps4_frames4096_seed42_43_44/seed{seed}/videojev'
    args['videojev_controller'] = args['videojev_controller'] or f'runs/lvb100_equal_seed{seed}/videojev'
    result = summarize(**args)
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
