"""比例抽样五方法汇总。validate_all(base) 只读重验三份单次来源、随机回放及最终三文件。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_longvideobench_proportional_controller import SCHEMA as CONTROLLER_SCHEMA, load_original
from scripts.sample_longvideobench_proportional import SAMPLING, read_sample
from scripts.simulate_longvideobench_random_choice import select_answers
from scripts.summarize_longvideobench_qwen36_repeats import difference_statistics
from scripts.summarize_longvideobench_repeats import accuracy_statistics, fingerprint
from timejev.lvb_experiment import GROUPS, read_predictions, sha256, write_json

ORIGINAL = ROOT/'scripts/summarize_longvideobench_qwen36.py'
CONTROLLER = ROOT/'scripts/eval_longvideobench_proportional_controller.py'
SINGLE_SCHEMA = 'longvideobench_proportional_single_seed_v1'
ALL_SCHEMA = 'longvideobench_proportional_three_seeds_v1'
SEEDS = (42, 43, 44)
ROLES = ('qwen_baseline', 'qwen_controller', 'videojev_baseline', 'videojev_controller', 'controller')
METHODS = dict(qwen36='Qwen3.6-35B-A3B', videojev='VideoJev-4B',
    random_choice='Qwen3.6-35B-A3B + VideoJev-4B', jev_as_controller='Jev-as-Controller',
    oracle_or='Qwen3.6-35B-A3B or VideoJev-4B')
NOTE = ('每个 seed 从完整 val 按四时长组题目数量比例独立抽 100 题，采用最大余数配额；'
    '不排除历史题目，跨 seed 允许重叠。随机选边不使用标签，每个 seed 独立重复 50 次；'
    '先取 50 次准确率均值，再计算三个 seed 均值的 ddof=1 样本方差，显示单位为百分点²。')
LIMITATION = ('两条基线为 4 FPS/4096 帧；Controller 双源为 2 FPS/2048 帧、β=2，'
    'context=262144、视觉 token 预算=24576，无字幕或音频。跨协议差异不能全部归因于融合，'
    '固定视觉预算不等于严格计算量对齐。OR 使用真实标签，是 oracle 上界。'
    '训练与验证集隔离仍未核实；三次重复不足以证明统计显著性。')


def stable(value):
    return {key: item for key, item in value.items() if key != 'created_at'}


def new_directory(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'拒绝覆盖已有输出：{out}')
    return out


def sampling_fields(rows, sample):
    return dict(sampling=SAMPLING, unit='question', quota_method=sample['quota_method'],
        source_counts=sample['source_counts'], quotas=sample['quotas'],
        selected_counts={str(group): Counter(row['duration_group'] for row in rows)[group] for group in GROUPS})


def source_code():
    paths = [Path(__file__), ORIGINAL, CONTROLLER, ROOT/'scripts/sample_longvideobench_proportional.py',
        ROOT/'scripts/eval_longvideobench_controller.py', ROOT/'scripts/simulate_longvideobench_random_choice.py',
        ROOT/'scripts/summarize_longvideobench_repeats.py', ROOT/'scripts/summarize_longvideobench_qwen36_repeats.py']
    return {str(path): sha256(path) for path in paths}


def single_markdown(report):
    lines = [f'# LongVideoBench 比例抽样：seed{report["seed"]}', '',
        '四个时长组实际题数：'+json.dumps(report['selected_counts'], ensure_ascii=False)+'。', '',
        '| 方法 | 答对/100 | 准确率 |', '| --- | --- | --- |']
    for method, metrics in report['methods'].items():
        lines.append(f'| {METHODS[method]} | {metrics["correct"]} | {metrics["accuracy"]*100:.2f}% |')
    lines += ['', f'同协议 Controller2 − Qwen2：{report["controller2_minus_qwen2"]["accuracy_difference"]*100:+.2f} 个百分点。',
        '', '完整时长分组、四份预处理/概率/公式校验及资源审计保留在 results.json。', '', LIMITATION, '']
    return '\n'.join(lines)


def summarize_single(manifest, qwen_baseline, qwen_controller, videojev_baseline, videojev_controller, controller, out=None, seed=42):
    if seed not in SEEDS or type(seed) is not int:
        raise ValueError('单次报告需要 seed42、43、44')
    output = new_directory(out) if out is not None else None
    rows, sample = read_sample(manifest, seed=seed, n=100)
    metadata = json.loads((Path(controller)/'run.json').read_text())
    if (metadata.get('schema') != CONTROLLER_SCHEMA or metadata.get('sampling') != SAMPLING
            or metadata.get('unit') != 'question' or metadata.get('seed') != seed or metadata.get('sample') != sample
            or metadata.get('adapter_script_sha256') != sha256(CONTROLLER)
            or metadata['identity']['code_sha256'].get('scripts/eval_longvideobench_proportional_controller.py') != sha256(CONTROLLER)):
        raise ValueError('Controller 不是该比例抽样清单及指定版本适配器产生的结果')
    legacy = load_original(ORIGINAL, '_lvb_proportional_single_original')
    legacy.read_sample = lambda path, current_seed=42: read_sample(path, seed=current_seed, n=100)
    result = legacy.summarize(manifest, qwen_baseline, qwen_controller, videojev_baseline,
        videojev_controller, controller, out=None, seed=seed)
    sources = result['provenance']['runs']
    if sources['videojev_baseline']['identity']['code_sha256'] != sources['videojev_controller']['identity']['code_sha256']:
        raise ValueError('本次比例方案要求 VideoJev 两协议使用统一源码，不接受旧版本等价白名单')
    result.update(schema=SINGLE_SCHEMA, **sampling_fields(rows, sample), script_sha256=sha256(__file__),
        original_summary_script_sha256=sha256(ORIGINAL), source_code_sha256=source_code())
    if output is not None:
        output.mkdir(parents=True)
        write_json(output/'results.json', result)
        (output/'results.md').write_text(single_markdown(result), encoding='utf-8')
    return result


def read_single(base, seed):
    cell = Path(base).resolve()/f'seed{seed}'
    paths = {role: cell/role for role in ROLES}
    fresh = summarize_single(cell/'sample.jsonl', out=None, seed=seed, **paths)
    report_path = cell/'summary/results.json'
    saved = json.loads(report_path.read_text())
    if stable(saved) != stable(fresh) or (cell/'summary/results.md').read_text() != single_markdown(fresh):
        raise ValueError(f'单次报告与完整源、比例抽样或重算结果不一致：seed{seed}')
    rows, sample = read_sample(cell/'sample.jsonl', seed=seed, n=100)
    return dict(report=saved, rows=rows, sample=sample, report_path=report_path)


def random_trials(rows, qwen, videojev, seed):
    gold, n, trials = [row['answer_index'] for row in rows], len(rows), []
    for repeat in range(1, 51):
        rng_seed = (seed << 32)+repeat
        answers, choices = select_answers(qwen, videojev, random.Random(rng_seed))
        correct = sum(answer == target for answer, target in zip(answers, gold))
        trials.append(dict(sample_seed=seed, repeat=repeat, rng_seed=rng_seed, n=n, correct=correct,
            accuracy=correct/n, answer_indices=answers, choices=choices))
    stats = accuracy_statistics([trial['accuracy'] for trial in trials])
    q_correct = sum(answer == target for answer, target in zip(qwen, gold))
    f_correct = sum(answer == target for answer, target in zip(videojev, gold))
    sensitive = sum((q == target) != (f == target) for q, f, target in zip(qwen, videojev, gold))
    stats.update(repeats=50, theoretical_mean=(q_correct+f_correct)/(2*n), theoretical_variance=sensitive/(4*n*n),
        variance_scope='固定该 seed 的 100 题，仅随机选边产生的方差；不作为总表跨 seed 方差')
    return trials, stats


def table(methods):
    lines = ['| 方法 | seed42 | seed43 | seed44 | 平均准确率 | 样本方差（百分点²） |',
             '| --- | --- | --- | --- | --- | --- |']
    for method, stats in methods.items():
        lines.append('| '+' | '.join([METHODS[method], *[f'{v*100:.2f}%' for v in stats['values']],
            f'{stats["mean"]*100:.2f}%', f'{stats["sample_variance"]*10000:.4f}'])+' |')
    return lines


def all_markdown(report):
    return '\n'.join(['# LongVideoBench：按完整 val 时长分布抽样', '', NOTE, '',
        '每个 seed 的四组配额：'+json.dumps(report['quotas'], ensure_ascii=False)+'。', '',
        *table(report['methods']), '', LIMITATION, ''])


def build_all(base):
    checked = [read_single(base, seed) for seed in SEEDS]
    entries, trials, sources, media = [], [], source_code(), {}
    fingerprints, sampling = None, None
    for item in checked:
        report, rows, sample = item['report'], item['rows'], item['sample']
        identities = {role: source['identity'] for role, source in report['provenance']['runs'].items()}
        current = {role: fingerprint(identity, identity['method']) for role, identity in identities.items()}
        strata = {key: report[key] for key in ('sampling', 'unit', 'quota_method', 'source_counts', 'quotas', 'selected_counts')}
        annotations = {key: sample[key]['sha256'] for key in ('source_manifest', 'source_reference')}
        if fingerprints is not None and (current != fingerprints or sampling != dict(strata=strata, annotations=annotations)):
            raise ValueError('跨 seed 模型/协议/源码/依赖或比例抽样来源/配额不一致')
        fingerprints, sampling = current, dict(strata=strata, annotations=annotations)
        for key, content in identities['qwen_controller']['media'].items():
            value = {field: content[field] for field in ('bytes', 'sha256')}
            if key in media and value != media[key]:
                raise ValueError('跨 seed 重叠视频内容不一致')
            media[key] = value
        predictions = {}
        for role, source in report['provenance']['runs'].items():
            directory = Path(source['directory'])
            for name, field in (('run.json', 'run_sha256'), ('predictions.jsonl', 'predictions_sha256'), ('metrics.json', 'metrics_sha256')):
                sources[str(directory/name)] = source[field]
            if role in ('qwen_baseline', 'videojev_baseline'):
                by_id = read_predictions(directory/'predictions.jsonl', rows)
                predictions[role] = [by_id[row['id']]['answer_index'] for row in rows]
        for content in report['provenance']['runtime_files'].values():
            sources[content['path']] = content['sha256']
        sources[str(Path(identities['controller']['sources']['qwen']['directory']).parent/'controller/comparison.json')] = report['provenance']['controller_comparison_sha256']
        sources[str(item['report_path'])] = sha256(item['report_path'])
        sources[str(item['report_path'].with_suffix('.md'))] = sha256(item['report_path'].with_suffix('.md'))
        sources[sample['path']] = sample['sha256']
        sources[str(Path(sample['path']).with_suffix('.meta.json'))] = sample['meta_sha256']
        for key in ('source_manifest', 'source_reference'):
            sources[sample[key]['path']] = sample[key]['sha256']
        qwen, jev = predictions['qwen_baseline'], predictions['videojev_baseline']
        current_trials, random_stats = random_trials(rows, qwen, jev, report['seed'])
        trials += current_trials
        oracle = sum(q == row['answer_index'] or f == row['answer_index'] for q, f, row in zip(qwen, jev, rows))
        accuracies = {method: report['methods'][method]['accuracy'] for method in ('qwen36', 'videojev', 'jev_as_controller')}
        accuracies.update(random_choice=random_stats['mean'], oracle_or=oracle/len(rows))
        entries.append(dict(seed=report['seed'], n=len(rows), question_ids=[row['id'] for row in rows],
            video_keys=sorted({row['video_key'] for row in rows}), accuracies=accuracies,
            random_choice=random_stats, oracle_correct=oracle, sample=sample,
            controller2_minus_qwen2=report['controller2_minus_qwen2']['accuracy_difference']))
    if len({frozenset(entry['question_ids']) for entry in entries}) != 3:
        raise ValueError('三个 seed 应产生三个不同的问题集合')
    overlaps = [dict(seeds=[a['seed'], b['seed']], shared_questions=len(set(a['question_ids']) & set(b['question_ids'])),
        shared_videos=len(set(a['video_keys']) & set(b['video_keys']))) for a, b in itertools.combinations(entries, 2)]
    trial_text = ''.join(json.dumps(trial, ensure_ascii=False, allow_nan=False)+'\n' for trial in trials)
    result = dict(schema=ALL_SCHEMA, seeds=list(SEEDS), questions_per_seed=100, repeats_per_random_selection=50,
        **sampling['strata'], methods={method: accuracy_statistics([entry['accuracies'][method] for entry in entries]) for method in METHODS},
        by_seed=entries, same_protocol_fusion_gain=difference_statistics([entry['controller2_minus_qwen2'] for entry in entries]),
        overlaps=overlaps, unique_questions=len(set().union(*(set(entry['question_ids']) for entry in entries))),
        unique_videos=len(set().union(*(set(entry['video_keys']) for entry in entries))),
        compatibility_fingerprints=fingerprints, verified_sources=sources,
        random_trials_sha256=hashlib.sha256(trial_text.encode()).hexdigest(),
        rng_rule='random.Random((sample_seed << 32) + repeat)，repeat=1..50；仅分歧题消耗randrange(2)',
        accuracy_unit='fraction', variance_unit='accuracy_fraction_squared', std_unit='accuracy_fraction',
        sample_variance_ddof=1, population_variance_ddof=0, variance_scope=NOTE,
        training_overlap_verified=False, limitation=LIMITATION, script_sha256=sha256(__file__),
        original_summary_script_sha256=sha256(ORIGINAL), created_at=datetime.now(timezone.utc).isoformat())
    return result, all_markdown(result), trial_text


def summarize_all(base, out=None):
    base = Path(base).resolve()
    output = new_directory(out if out is not None else base/'summary')
    if output != base/'summary':
        raise ValueError('总报告固定写入 --base/summary，供 validate_all(base) 只读重验')
    result, markdown, trials = build_all(base)
    output.mkdir(parents=True)
    write_json(output/'results.json', result)
    (output/'results.md').write_text(markdown, encoding='utf-8')
    (output/'random_trials.jsonl').write_text(trials, encoding='utf-8')
    return result


def validate_all(base):
    """重新读取三份单次来源和完整采样；重算并逐字节核对总报告及150次逐题随机记录，不写文件。"""
    base = Path(base).resolve()
    output = base/'summary'
    if not output.is_dir() or {path.name for path in output.iterdir()} != {'results.json', 'results.md', 'random_trials.jsonl'}:
        raise ValueError('总报告需要 results.json/results.md/random_trials.jsonl 三个文件')
    result, markdown, trials = build_all(base)
    saved = json.loads((output/'results.json').read_text())
    if (stable(saved) != stable(result) or (output/'results.md').read_text() != markdown
            or (output/'random_trials.jsonl').read_text() != trials):
        raise ValueError('总报告、单位、来源 SHA 或逐题随机回放与严格重算不一致')
    return dict(passed=True, report=saved, source_hashes={**result['verified_sources'],
        **{str(path): sha256(path) for path in output.iterdir()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single', 'all'), default='single')
    parser.add_argument('--base')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    for name in ('manifest', 'qwen-baseline', 'qwen-controller', 'videojev-baseline', 'videojev-controller', 'controller'):
        parser.add_argument('--'+name)
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    if mode == 'all':
        if not args['base'] or any(args[name] is not None for name in ('manifest', *ROLES)):
            parser.error('--mode all 需要 --base，且不接受单次来源目录')
        result = summarize_all(args['base'], args['out'])
    else:
        args.pop('base')
        if any(args[name] is None for name in ('manifest', *ROLES)):
            parser.error('--mode single 需要 --manifest 及五个来源目录')
        result = summarize_single(**args)
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
