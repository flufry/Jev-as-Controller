"""固定 LongVideoBench 题集，按已保存候选概率重复采样并报告均值和样本方差。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_longvideobench_proportional import read_single
from timejev.lvb_experiment import read_predictions, sha256, write_json

METHOD_ROLES = dict(qwen36='qwen_baseline', videojev='videojev_baseline',
                    jev_as_controller='controller')
NAMES = dict(qwen36='Qwen3.6-35B-A3B', videojev='VideoJev-4B',
             jev_as_controller='Jev-as-Controller')
RNG = ("每方法/每次重复使用 random.Random(f'{inference_seed}:{method}')；"
       '按固定题集的原始顺序，每题恰好消耗一次 random()，使用归一化概率的累积分布采样。')
NOTE = ('固定题目、视频输入与已保存的首步候选概率，temperature=1，不使用 top-k/top-p；'
        '复用已校验模型分数进行候选内条件采样，没有重新执行 GPU 前向或完整词表生成。'
        '方差仅反映答案采样，使用 ddof=1；基线4 FPS/4096帧、Controller双源2 FPS/2048帧、β=2。')


def valid_probabilities(values):
    if (not values or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                          for p in values) or not math.isclose(sum(values), 1., abs_tol=1e-5)):
        raise ValueError('候选概率必须非空、有限、非负且归一化，不能包含 bool')
    return sum(values)


def sample_index(probabilities, rng):
    """仅接收概率，不接收标签；零概率选项不会被选中。"""
    total = valid_probabilities(probabilities)
    point, cumulative = rng.random()*total, 0.
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if point < cumulative:
            return index
    # 只处理浮点累加的尾部舍入，仍只返回非零概率选项。
    return max(index for index, probability in enumerate(probabilities) if probability > 0)


def simulate(rows, predictions, seeds=(42, 43, 44)):
    """固定分布三次采样；真实标签仅用于采样完成后的计分。"""
    seeds = list(seeds)
    if (len(seeds) < 2 or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('至少需要两个互不重复的整数推理种子，不能包含 bool')
    ids = [row['id'] for row in rows]
    if not rows or len(set(ids)) != len(ids) or set(predictions) != set(METHOD_ROLES):
        raise ValueError('题集必须非空、ID唯一，且包含三个指定方法')
    for row in rows:
        if (type(row.get('answer_index')) is not int
                or not 0 <= row['answer_index'] < len(row['options'])):
            raise ValueError(f'真实标签超出选项范围：{row["id"]}')
    for method, source in predictions.items():
        if set(source) != set(ids):
            raise ValueError(f'方法预测未完整对齐固定题集：{method}')
        for row in rows:
            prediction = source[row['id']]
            values = prediction.get('probabilities', [])
            valid_probabilities(values)
            if (prediction.get('id') != row['id'] or prediction.get('query') != row['query']
                    or prediction.get('options') != row['options'] or len(values) != len(row['options'])):
                raise ValueError(f'预测问题或选项与固定题集不一致：{row["id"]}')
    trials, sampled, methods = [], [], {}
    for method in METHOD_ROLES:
        source, accuracies = predictions[method], []
        for seed in seeds:
            rng = random.Random(f'{seed}:{method}')
            answers = [sample_index(source[row['id']]['probabilities'], rng) for row in rows]
            totals, hits = Counter(), Counter()
            for row, index in zip(rows, answers):
                correct = index == row['answer_index']
                group = str(row['duration_group'])
                totals[group] += 1
                hits[group] += correct
                sampled.append(dict(method=method, inference_seed=seed, id=row['id'],
                    query=row['query'], options=row['options'], video_key=row.get('video_key'),
                    duration_group=row['duration_group'], probabilities=source[row['id']]['probabilities'],
                    answer_index=index, answer=row['options'][index],
                    ground_truth_index=row['answer_index'], correct=correct))
            correct = sum(hits.values())
            accuracy = correct/len(rows)
            accuracies.append(accuracy)
            trials.append(dict(method=method, inference_seed=seed, n=len(rows), correct=correct,
                accuracy=accuracy, by_duration_group={g: dict(n=totals[g], correct=hits[g],
                    accuracy=hits[g]/totals[g]) for g in totals}))
        gold_probs = [source[row['id']]['probabilities'][row['answer_index']]
                      /sum(source[row['id']]['probabilities']) for row in rows]
        greedy = sum(max(range(len(row['options'])),
            key=source[row['id']]['probabilities'].__getitem__) == row['answer_index'] for row in rows)
        methods[method] = dict(accuracies=accuracies, mean_accuracy=statistics.mean(accuracies),
            sample_variance_fraction_squared=statistics.variance(accuracies),
            sample_variance_percentage_points_squared=statistics.variance(accuracies)*10000,
            theoretical_mean_accuracy=sum(gold_probs)/len(rows),
            theoretical_variance_percentage_points_squared=sum(p*(1-p) for p in gold_probs)/len(rows)**2*10000,
            greedy_accuracy=greedy/len(rows))
    return dict(schema='longvideobench_fixed_questions_sampling_v1', n=len(rows),
        inference_seeds=seeds, repetitions=len(seeds), question_ids=ids, methods=methods,
        temperature=1., top_k=None, top_p=None, probability_scope='original_question_candidates',
        probability_normalization='逐题除以源概率总和；逐题输出保留源概率，理论统计使用同一归一化分布。',
        rng=RNG, accuracy_unit='fraction', sample_variance_ddof=1,
        variance_scope='fixed_questions_answer_sampling', note=NOTE), trials, sampled


def markdown(result):
    seeds = result['inference_seeds']
    lines = ['# LongVideoBench：固定题集的答案采样', '',
        f'固定选题 seed={result["sample_seed"]}，{result["n"]} 道题；推理采样种子为 {seeds}。', '', NOTE, '',
        '| 方法 | '+' | '.join(f'采样seed{seed}' for seed in seeds)+' | 平均准确率 | 样本方差（百分点²） |',
        '| --- | '+' | '.join('---:' for _ in seeds)+' | ---: | ---: |']
    for method, stats in result['methods'].items():
        lines.append('| '+NAMES[method]+' | '+' | '.join(f'{x*100:.2f}%' for x in stats['accuracies'])
            +f' | {stats["mean_accuracy"]*100:.2f}% | {stats["sample_variance_percentage_points_squared"]:.4f} |')
    lines += ['', '解析期望与方差按每题正确答案概率计算，不替代上面的三次实测。', '',
        '| 方法 | 原贪心准确率 | 采样准确率理论期望 | 理论方差（百分点²） |',
        '| --- | ---: | ---: | ---: |']
    for method, stats in result['methods'].items():
        lines.append(f'| {NAMES[method]} | {stats["greedy_accuracy"]*100:.2f}%'
            +f' | {stats["theoretical_mean_accuracy"]*100:.2f}%'
            +f' | {stats["theoretical_variance_percentage_points_squared"]:.4f} |')
    return '\n'.join(lines)+f'\n\n{RNG}\n'


def source_hashes(entry):
    report, sample = entry['report'], entry['sample']
    sources = {sample['path']: sample['sha256'],
        str(Path(sample['path']).with_suffix('.meta.json')): sample['meta_sha256'],
        str(entry['report_path']): sha256(entry['report_path']),
        str(entry['report_path'].with_suffix('.md')): sha256(entry['report_path'].with_suffix('.md')),
        str(Path(__file__).resolve()): sha256(__file__), **report['source_code_sha256']}
    for name in ('source_manifest', 'source_reference'):
        sources[sample[name]['path']] = sample[name]['sha256']
    for source in report['provenance']['runs'].values():
        for name, key in (('run.json', 'run_sha256'), ('predictions.jsonl', 'predictions_sha256'),
                          ('metrics.json', 'metrics_sha256')):
            sources[str(Path(source['directory'])/name)] = source[key]
    for content in report['provenance']['runtime_files'].values():
        sources[content['path']] = content['sha256']
    controller = report['provenance']['runs']['controller']['directory']
    sources[str(Path(controller)/'comparison.json')] = report['provenance']['controller_comparison_sha256']
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.split('.')[0] in ('scripts', 'timejev'):
            path = Path(path).resolve()
            if path.is_file() and path.is_relative_to(ROOT):
                sources[str(path)] = sha256(path)
    return sources


def verify_sources(sources):
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError(f'本次实验依赖源文件发生变化：{path}')


def run(base, out, sample_seed=42, inference_seeds=(42, 43, 44)):
    out, base = Path(out).resolve(), Path(base).resolve()
    if out.exists():
        raise FileExistsError(f'拒绝覆盖已有输出目录：{out}')
    started_at = datetime.now(timezone.utc).isoformat()
    entry = read_single(base, sample_seed)
    sources = source_hashes(entry)
    verify_sources(sources)
    rows = entry['rows']
    predictions = {method: read_predictions(base/f'seed{sample_seed}'/role/'predictions.jsonl', rows)
                   for method, role in METHOD_ROLES.items()}
    result, trials, sampled = simulate(rows, predictions, inference_seeds)
    result.update(sample_seed=sample_seed, sample=entry['sample'],
        original_greedy_metrics=entry['report']['methods'], protocols=entry['report']['protocols'],
        beta=entry['report']['beta'], verified_sources=sources,
        source_runs=entry['report']['provenance']['runs'])
    verify_sources(sources)
    out.mkdir(parents=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text(markdown(result), encoding='utf-8')
    for name, records in (('trials.jsonl', trials), ('predictions.jsonl', sampled)):
        with (out/name).open('x', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    snapshot = out/'source'/Path(__file__).name
    snapshot.parent.mkdir()
    snapshot.write_bytes(Path(__file__).read_bytes())
    verify_sources(sources)
    write_json(out/'run.json', dict(status='completed', started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        base=str(base), out=str(out), hostname=socket.gethostname(), pid=os.getpid(),
        python=sys.version, sample_seed=sample_seed, inference_seeds=list(inference_seeds),
        temperature=1., probability_scope='original_question_candidates', execution='cpu_probability_reuse',
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        verified_sources=sources, note=NOTE))
    write_json(out/'artifact_manifest.json', dict(schema='longvideobench_fixed_sampling_artifacts_v1',
        files={str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob('*')) if path.is_file()},
        verified_sources=sources))
    verify_sources(sources)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='runs/lvb100_proportional_seed42_43_44')
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--inference-seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--out', required=True)
    result = run(**vars(parser.parse_args()))
    print(markdown(result))


if __name__ == '__main__':
    main()
