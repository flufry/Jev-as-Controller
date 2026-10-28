"""严格重验三组×三 seed 的 LongVideoBench 结果，汇总随机选边和 oracle 上界。"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.simulate_longvideobench_random_choice import select_answers
from scripts.summarize_longvideobench_qwen36 import read_sample
from scripts.summarize_longvideobench_qwen36_repeats import difference_statistics, read_single
from scripts.summarize_longvideobench_repeats import accuracy_statistics, fingerprint
from timejev.lvb_experiment import GROUPS, read_predictions, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT/'runs/lvb100_expanded3groups_seed42_43_44'
LEGACY_RANDOM = ROOT/'reports/lvb_qwen36_random_choice50_seed42_43_44'
SEEDS = (42, 43, 44)
METHODS = dict(qwen36='Qwen3.6-35B-A3B', videojev='VideoJev-4B',
    random_choice='Qwen3.6-35B-A3B + VideoJev-4B', jev_as_controller='Jev-as-Controller',
    oracle_or='Qwen3.6-35B-A3B or VideoJev-4B')
REPEATS = 50
RNG_RULE = '((group_index - 1) << 48) + (sample_seed << 32) + repeat；repeat=1..50，仅分歧题消耗 randrange(2)'
NOTE = ('每张表的样本方差是 seed42/43/44 三个准确率的 ddof=1 方差；'
        '随机选边一行先在每个 seed 内对 50 次准确率取均值，再计算三个均值的方差。'
        '随机选边的 seed 内方差另存 JSON，不混作跨题集方差。')
LIMITATION = ('两条基线使用 4 FPS/4096 帧；Controller 两源使用 2 FPS/2048 帧、β=2，'
    'context=262144、视觉 token 预算=24576，无字幕或音频。跨协议差异不能全部归因于融合；'
    '相同视觉 token 预算不代表严格计算量对齐。OR 使用真实标签，是 oracle 上界。'
    '未获得完整 QA 训练清单，训练与验证集隔离仍未核实。三个 seed 可重叠，仅三次重复不足以证明统计显著性。')


def locations(base, group, seed):
    if group == 1:
        return (ROOT/f'runs/lvb100_equal_seed{seed}/sample.jsonl',
                ROOT/f'runs/lvb100_qwen36_35b_controller_seed{seed}')
    directory = Path(base)/f'group{group}'/f'seed{seed}'
    return directory/'sample.jsonl', directory


def check_samples(base, groups):
    """仅校验清单、原始标签及同 seed 组间互斥，不加载模型或要求预测。"""
    selected = sorted(set(groups))
    if not selected or any(type(group) is not int or group not in (1, 2, 3) for group in selected):
        raise ValueError('组编号只能为 1、2、3')
    checked, source_identity = {}, None
    # 单独处理 group3 时也检查前两组，防止忽略组间重复。
    for group in range(1, max(selected)+1):
        for seed in SEEDS:
            manifest, directory = locations(base, group, seed)
            rows, sample = read_sample(manifest, seed)
            ids = {row['id'] for row in rows}
            if len(ids) != 100:
                raise ValueError('每格需要 100 道互异问题')
            annotations = {key: sample[key]['sha256'] for key in ('source_manifest', 'source_reference')}
            if source_identity is not None and annotations != source_identity:
                raise ValueError('各组及 seed 的原始标注 SHA 不一致')
            source_identity = annotations
            for earlier in range(1, group):
                if ids & checked[earlier, seed]['question_ids']:
                    raise ValueError(f'seed{seed} 的 group{earlier} 与 group{group} 问题重复')
            checked[group, seed] = dict(rows=rows, sample=sample, directory=directory,
                question_ids=ids, video_keys={row['video_key'] for row in rows})
    return checked


def random_trial(rows, qwen, videojev, group, seed, repeat):
    rng_seed = ((group-1) << 48)+(seed << 32)+repeat
    answers, choices = select_answers(qwen, videojev, random.Random(rng_seed))
    correct = sum(answer == row['answer_index'] for answer, row in zip(answers, rows))
    return dict(sample_seed=seed, repeat=repeat, rng_seed=rng_seed, n=len(rows), correct=correct,
                accuracy=correct/len(rows), answer_indices=answers, choices=choices)


def random_statistics(rows, qwen, videojev, trials, protocol):
    gold, n = [row['answer_index'] for row in rows], len(rows)
    q_correct = sum(answer == target for answer, target in zip(qwen, gold))
    f_correct = sum(answer == target for answer, target in zip(videojev, gold))
    sensitive = sum((q == g) != (f == g) for q, f, g in zip(qwen, videojev, gold))
    values = [trial['accuracy'] for trial in trials]
    return dict(n=n, repeats=REPEATS, question_ids=[row['id'] for row in rows], protocol=protocol,
        qwen_accuracy=q_correct/n, videojev_accuracy=f_correct/n,
        disagreement_questions=sum(q != f for q, f in zip(qwen, videojev)),
        exactly_one_correct_questions=sensitive, accuracies=values, mean=statistics.mean(values),
        sample_variance=statistics.variance(values), sample_std=statistics.stdev(values),
        population_variance=statistics.pvariance(values), minimum=min(values), maximum=max(values),
        theoretical_mean=(q_correct+f_correct)/(2*n), theoretical_variance=sensitive/(4*n*n),
        theoretical_std=sensitive**.5/(2*n))


def read_legacy_random(directory):
    """旧组的 150 次选边必须逐字节核验来源，再独立按问题回放。"""
    directory = Path(directory).resolve()
    artifact_path = directory/'artifact_manifest.json'
    artifact = json.loads(artifact_path.read_text())
    expected = {'exitcode', 'results.json', 'run.log', 'source.py', 'trials.jsonl', 'validation.json'}
    if artifact.get('schema') != 'lvb_random_choice_artifacts_v1' or set(artifact.get('files', {})) != expected:
        raise ValueError('原随机选边产物清单 schema 或六个文件不完整')
    verified = {str(artifact_path): sha256(artifact_path)}
    for name, content in artifact['files'].items():
        path = directory/name
        if not path.is_file() or sha256(path) != content['sha256'] or path.stat().st_size != content['bytes']:
            raise ValueError(f'原随机选边产物 SHA 或字节数不符：{path}')
        verified[str(path)] = content['sha256']
    report = json.loads((directory/'results.json').read_text())
    if ((directory/'exitcode').read_text().strip() != '0'
            or json.loads((directory/'validation.json').read_text()).get('passed') is not True
            or report.get('schema') != 'lvb_qwen36_videojev_random_choice_v1'
            or report.get('repeats_per_sample') != REPEATS or report.get('total_trials') != 150
            or set(report.get('seeds', {})) != set(map(str, SEEDS))
            or report.get('script_sha256') != sha256(directory/'source.py')):
        raise ValueError('原随机选边运行未通过验收或协议不一致')
    for path, content in report['verified_sources'].items():
        if sha256(path) != content['sha256']:
            raise ValueError(f'原随机选边源文件 SHA 不符：{path}')
        verified[path] = content['sha256']
    source = report['source_artifact_manifest']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('原完整实验产物清单 SHA 不符')
    verified[source['path']] = source['sha256']
    trials = [json.loads(line) for line in (directory/'trials.jsonl').read_text().splitlines()]
    if len(trials) != 150 or len({(t['sample_seed'], t['repeat']) for t in trials}) != 150:
        raise ValueError('原随机选边必须包含 150 个互异 seed/repeat')
    return dict(report=report, trials={(t['sample_seed'], t['repeat']): t for t in trials}, verified=verified)


def collect_group(group, samples, legacy=None):
    entries, all_trials, sources = [], [], {}
    if group == 1:
        legacy = legacy if legacy is not None else read_legacy_random(LEGACY_RANDOM)
        sources.update(legacy['verified'])
    for seed in SEEDS:
        entry = samples[group, seed]
        checked = read_single(entry['directory'])
        report, rows = checked['report'], entry['rows']
        if report['seed'] != seed or report['provenance']['sample'] != entry['sample']:
            raise ValueError('单次报告不是当前组、seed 或清单的结果')
        qwen, videojev, controller = [], [], []
        sources[str(entry['sample']['path'])] = entry['sample']['sha256']
        sources[str(Path(entry['sample']['path']).with_suffix('.meta.json'))] = entry['sample']['meta_sha256']
        sources[checked['provenance']['summary_path']] = checked['provenance']['summary_sha256']
        predictions = {}
        for role, source in report['provenance']['runs'].items():
            directory = Path(source['directory'])
            for name, field in (('run.json', 'run_sha256'), ('predictions.jsonl', 'predictions_sha256'),
                                ('metrics.json', 'metrics_sha256')):
                sources[str(directory/name)] = source[field]
            if role in ('qwen_baseline', 'videojev_baseline', 'controller'):
                by_id = read_predictions(directory/'predictions.jsonl', rows)
                predictions[role] = [by_id[row['id']]['answer_index'] for row in rows]
        qwen, videojev, controller = [predictions[role] for role in ('qwen_baseline', 'videojev_baseline', 'controller')]
        trials = [random_trial(rows, qwen, videojev, group, seed, repeat) for repeat in range(1, REPEATS+1)]
        random_stats = random_statistics(rows, qwen, videojev, trials, report['protocols']['baseline'])
        if group == 1:
            if (any(trial != legacy['trials'].get((seed, trial['repeat'])) for trial in trials)
                    or random_stats != legacy['report']['seeds'][str(seed)]):
                raise ValueError(f'原 seed{seed} 随机选边问题、答案或 50 次回放结果不一致')
        gold = [row['answer_index'] for row in rows]
        both = sum(q == target and f == target for q, f, target in zip(qwen, videojev, gold))
        oracle = sum(q == target or f == target for q, f, target in zip(qwen, videojev, gold))
        accuracies = {key: report['methods'][key]['accuracy'] for key in ('qwen36', 'videojev', 'jev_as_controller')}
        accuracies.update(random_choice=random_stats['mean'], oracle_or=oracle/len(rows))
        entries.append(dict(group_index=group, seed=seed, n=len(rows), accuracies=accuracies,
            random_choice=random_stats, oracle=dict(correct=oracle, both_correct=both,
                exactly_one_correct=oracle-both, both_wrong=len(rows)-oracle),
            same_protocol_fusion_gain=report['controller2_minus_qwen2']['accuracy_difference'],
            question_ids=[row['id'] for row in rows], video_keys=sorted(entry['video_keys']),
            provenance=checked['provenance']))
        all_trials += [dict(group_index=group, **trial) for trial in trials]
    return dict(group_index=group, entries=entries, trials=all_trials, sources=sources)


def verify_compatibility(groups):
    expected, annotations, seen_media = None, None, {}
    for group in groups:
        for entry in group['entries']:
            provenance = entry['provenance']['sources']
            identities = {role: source['identity'] for role, source in provenance['runs'].items()}
            current = {role: fingerprint(identity, identity['method']) for role, identity in identities.items()}
            source_hashes = {key: provenance['sample'][key]['sha256'] for key in ('source_manifest', 'source_reference')}
            if expected is not None and (current != expected or annotations != source_hashes):
                raise ValueError('跨组或 seed 的模型、协议、代码、配置、依赖或原始标注不一致')
            expected, annotations = current, source_hashes
            for key, media in identities['qwen_controller']['media'].items():
                content = {field: media[field] for field in ('sha256', 'bytes')}
                if key in seen_media and seen_media[key] != content:
                    raise ValueError(f'重叠视频内容不一致：{key}')
                seen_media[key] = content
    return expected


def overlap_statistics(entries):
    overlaps = []
    for a, b in itertools.combinations(entries, 2):
        overlaps.append(dict(left=[a['group_index'], a['seed']], right=[b['group_index'], b['seed']],
            shared_questions=len(set(a['question_ids']) & set(b['question_ids'])),
            shared_videos=len(set(a['video_keys']) & set(b['video_keys']))))
    return dict(unique_questions=len(set().union(*(set(e['question_ids']) for e in entries))),
        unique_videos=len(set().union(*(set(e['video_keys']) for e in entries))), pairs=overlaps)


def method_statistics(entries):
    if [entry['seed'] for entry in entries] != list(SEEDS):
        raise ValueError('统计恰好需要依次排列的 seed42、43、44')
    return {method: accuracy_statistics([entry['accuracies'][method] for entry in entries]) for method in METHODS}


def table(methods):
    lines = ['| 方法 | seed42 | seed43 | seed44 | 平均准确率 | 样本方差（百分点²） |',
             '| --- | --- | --- | --- | --- | --- |']
    for method, stats in methods.items():
        lines.append('| '+' | '.join([METHODS[method], *[f'{value*100:.2f}%' for value in stats['values']],
            f'{stats["mean"]*100:.2f}%', f'{stats["sample_variance"]*10000:.4f}'])+' |')
    return lines


def report_common():
    return dict(seeds=list(SEEDS), repeats_per_random_selection=REPEATS, accuracy_unit='fraction',
        variance_unit='accuracy_fraction_squared', std_unit='accuracy_fraction', sample_variance_ddof=1,
        population_variance_ddof=0, variance_scope=NOTE, rng_rule=RNG_RULE,
        method_labels=METHODS, training_overlap_verified=False, limitation=LIMITATION,
        script_sha256=sha256(__file__), created_at=datetime.now(timezone.utc).isoformat())


def group_report(group, fingerprints):
    trial_text = ''.join(json.dumps(t, ensure_ascii=False, allow_nan=False)+'\n' for t in group['trials'])
    result = dict(schema='lvb_expanded_group_summary_v1', group_index=group['group_index'], questions_per_seed=100,
        methods=method_statistics(group['entries']), by_seed=group['entries'],
        same_protocol_fusion_gain=difference_statistics([e['same_protocol_fusion_gain'] for e in group['entries']]),
        overlap=overlap_statistics(group['entries']), compatibility_fingerprints=fingerprints,
        verified_sources=group['sources'], random_trials_sha256=hashlib.sha256(trial_text.encode()).hexdigest(), **report_common())
    markdown = '\n'.join([f'# LongVideoBench 第 {group["group_index"]} 组', '',
        '每个 seed 100 道题，官方四个时长组各 25 道。', '', *table(result['methods']), '', NOTE, '', LIMITATION, ''])
    return result, markdown, trial_text


def save_group(out, payload, reuse=False):
    out = Path(out)
    result, markdown, trials = payload
    if out.exists():
        if not reuse:
            raise FileExistsError(f'拒绝覆盖已有组汇总：{out}')
        if not out.is_dir() or {p.name for p in out.iterdir()} != {'results.json', 'results.md', 'random_trials.jsonl'}:
            raise ValueError(f'已有组汇总目录内容不完整或多出文件：{out}')
        saved = json.loads((out/'results.json').read_text())
        stable = lambda value: {key: val for key, val in value.items() if key != 'created_at'}
        if (stable(saved) != stable(result) or (out/'results.md').read_text() != markdown
                or (out/'random_trials.jsonl').read_text() != trials):
            raise ValueError(f'已有组汇总与源预测、来源 SHA 或随机回放重算不符：{out}')
        return saved
    out.mkdir(parents=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text(markdown, encoding='utf-8')
    (out/'random_trials.jsonl').write_text(trials, encoding='utf-8')
    return result


def summarize(base=DEFAULT_BASE, group=None, all_groups=False, check_only=False):
    base = Path(base).resolve()
    selected = [1, 2, 3] if all_groups else [group]
    if all_groups and group is not None or not all_groups and group not in (1, 2, 3):
        raise ValueError('必须选择 --group 1/2/3 或 --all，不能同时选择')
    samples = check_samples(base, selected)
    if check_only:
        return dict(passed=True, check_scope='manifest_only_no_model_results_required',
            groups=selected, checked_cells=len(samples), question_ids_disjoint_within_each_seed=True,
            manifests=[dict(group_index=g, **entry['sample']) for (g, s), entry in samples.items()])
    target = base/'summary' if all_groups else base/f'group{group}_summary'
    if target.exists():
        raise FileExistsError(f'拒绝覆盖已有汇总：{target}')
    collected = [collect_group(index, samples) for index in selected]
    fingerprints = verify_compatibility(collected)
    group_reports = [save_group(base/f'group{value["group_index"]}_summary', group_report(value, fingerprints),
                                reuse=all_groups) for value in collected]
    if not all_groups:
        return group_reports[0]
    entries = [dict(seed=seed, n=300, accuracies={method: statistics.mean(
        report['by_seed'][offset]['accuracies'][method] for report in group_reports) for method in METHODS})
        for offset, seed in enumerate(SEEDS)]
    result = dict(schema='lvb_expanded_three_groups_summary_v1', groups=[1, 2, 3], questions_per_seed_per_group=100,
        cumulative_questions_per_seed=300, group_reports=group_reports, cumulative=dict(methods=method_statistics(entries),
        by_seed=entries, variance_scope='每个 seed 合并三组互斥问题后的 300 题准确率，取三个 seed 的 ddof=1 方差'),
        overlap=overlap_statistics([entry for value in collected for entry in value['entries']]),
        compatibility_fingerprints=fingerprints,
        group_summary_sources={str(base/f'group{index}_summary/results.json'): sha256(base/f'group{index}_summary/results.json')
            for index in selected}, **report_common())
    lines = ['# LongVideoBench：三组各三 seed 的扩展实验', '', NOTE, '']
    for value in group_reports:
        lines += [f'## 第 {value["group_index"]} 组：每个 seed 100 题', '', *table(value['methods']), '']
    lines += ['## 合并三组：每个 seed 300 道互斥问题', '', *table(result['cumulative']['methods']), '',
        '以上方差描述三个 seed 结果的波动；跨 seed 问题/视频重叠明细见 JSON。', '', LIMITATION, '']
    target.mkdir(parents=True)
    write_json(target/'results.json', result)
    (target/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=str(DEFAULT_BASE))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--group', type=int, choices=(1, 2, 3))
    selection.add_argument('--all', action='store_true', dest='all_groups')
    parser.add_argument('--check-only', action='store_true', help='仅校验清单和互斥关系，不要求模型预测')
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(result.get('methods', result.get('cumulative', result)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
