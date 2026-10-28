"""用已校验的两条 4 FPS 基线答案，逐题等概率随机选择并重复评估。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT/'runs/lvb100_qwen36_35b_repeat3_seed42_43_44'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_answers(qwen, videojev, rng):
    """选择过程只接收预测答案，不接收真实标签。"""
    if len(qwen) != len(videojev):
        raise ValueError('两条基线的问题数必须一致')
    choices = ['shared' if q == f else ('qwen36' if rng.randrange(2) == 0 else 'videojev')
               for q, f in zip(qwen, videojev)]
    answers = [f if choice == 'videojev' else q
               for q, f, choice in zip(qwen, videojev, choices)]
    return answers, choices


def run(out, repeats=50):
    if repeats < 2:
        raise ValueError('计算样本方差至少需要重复两次')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'拒绝覆盖已有输出：{out}')
    started_at = datetime.now(timezone.utc).isoformat()
    artifact_path = MASTER/'artifact_manifest.json'
    artifact = json.loads(artifact_path.read_text())
    sources = {}

    def read_verified(path, jsonl=False):
        path = path.resolve()
        expected = artifact['files'][str(path)]
        sha = digest(path)
        if sha != (expected['sha256'] if isinstance(expected, dict) else expected):
            raise ValueError(f'源文件与完整实验验收的 SHA256 不一致：{path}')
        sources[str(path)] = dict(sha256=sha)
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)

    reports, trials = {}, []
    for seed in (42, 43, 44):
        sample_path = ROOT/f'runs/lvb100_equal_seed{seed}/sample.jsonl'
        sample = read_verified(sample_path, jsonl=True)
        ids = [r['id'] for r in sample]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f'seed{seed} 必须包含原始 100 道互异题')
        directories = dict(
            qwen36=ROOT/f'runs/lvb100_qwen36_35b_controller_seed{seed}/qwen_baseline',
            videojev=ROOT/f'runs/lvb100_baselines_fps4_frames4096_seed42_43_44/seed{seed}/videojev')
        predictions, protocol = {}, None
        for method, directory in directories.items():
            identity = read_verified(directory/'run.json')['identity']
            current = identity['protocol']
            if (identity['method'] != method or identity['seed'] != seed
                    or identity['manifest_sha256'] != digest(sample_path)
                    or current['fps'] != 4 or current['max_frames'] != 4096
                    or (protocol is not None and protocol != current)):
                raise ValueError(f'模型、样本、seed 或 4 FPS 基线协议不匹配：{directory}')
            protocol = current
            rows = read_verified(directory/'predictions.jsonl', jsonl=True)
            by_id = {r['id']: r for r in rows}
            if len(rows) != 100 or set(by_id) != set(ids):
                raise ValueError(f'预测问题集合不匹配：{directory}')
            for gold in sample:
                pred = by_id[gold['id']]
                if (pred['query'] != gold['query'] or pred['options'] != gold['options']
                        or pred['answer_index'] != max(range(len(pred['probabilities'])),
                                                       key=pred['probabilities'].__getitem__)):
                    raise ValueError(f'问题、选项或预测索引不一致：{gold["id"]}')
            predictions[method] = [by_id[qid]['answer_index'] for qid in ids]

        qwen, videojev = predictions['qwen36'], predictions['videojev']
        gold = [r['answer_index'] for r in sample]
        n = len(gold)
        q_correct = sum(p == g for p, g in zip(qwen, gold))
        f_correct = sum(p == g for p, g in zip(videojev, gold))
        sensitive = sum((q == g) != (f == g) for q, f, g in zip(qwen, videojev, gold))
        values = []
        for repeat in range(1, repeats + 1):
            rng_seed = (seed << 32) + repeat
            answers, choices = select_answers(qwen, videojev, random.Random(rng_seed))
            correct = sum(a == g for a, g in zip(answers, gold))
            values.append(correct/n)
            trials.append(dict(sample_seed=seed, repeat=repeat, rng_seed=rng_seed,
                n=n, correct=correct, accuracy=correct/n, answer_indices=answers, choices=choices))
        reports[str(seed)] = dict(n=n, repeats=repeats, question_ids=ids, protocol=protocol,
            qwen_accuracy=q_correct/n, videojev_accuracy=f_correct/n,
            disagreement_questions=sum(q != f for q, f in zip(qwen, videojev)),
            exactly_one_correct_questions=sensitive, accuracies=values,
            mean=statistics.mean(values), sample_variance=statistics.variance(values),
            sample_std=statistics.stdev(values), population_variance=statistics.pvariance(values),
            minimum=min(values), maximum=max(values),
            theoretical_mean=(q_correct + f_correct)/(2*n),
            theoretical_variance=sensitive/(4*n*n), theoretical_std=math.sqrt(sensitive)/(2*n))

    result = dict(schema='lvb_qwen36_videojev_random_choice_v1',
        label='Qwen3.6-35B-A3B / VideoJev-4B random 50:50',
        selection='答案相同时保留；答案不同时逐题独立、等概率选择一个模型的答案；选择不使用真实标签。',
        rng='Python random.Random((sample_seed << 32) + repeat_index)，repeat_index 从 1 开始；仅分歧题消耗 rng.randrange(2)。',
        seeds=reports, repeats_per_sample=repeats, total_trials=len(trials),
        mean_across_sample_seeds=statistics.mean(v['mean'] for v in reports.values()),
        theoretical_mean_across_sample_seeds=statistics.mean(v['theoretical_mean'] for v in reports.values()),
        accuracy_unit='fraction', variance_unit='accuracy_fraction_squared', sample_variance_ddof=1,
        variance_scope='固定各 seed 的原始 100 道题，仅随机选择答案；不同于三次重新抽题的方差。',
        theoretical_distribution='两者均正确的题数 + Binomial(只有一方正确的题数, 0.5)；双错题始终错误。',
        source_artifact_manifest=dict(path=str(artifact_path), sha256=digest(artifact_path)),
        verified_sources=sources, script_sha256=digest(__file__), started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        hostname=socket.gethostname(), pid=os.getpid(), python=platform.python_version(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        training_overlap_verified=False)
    out.mkdir(parents=True)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (out/'trials.jsonl').open('x') as handle:
        for trial in trials:
            handle.write(json.dumps(trial, ensure_ascii=False, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='新的结果目录，拒绝覆盖已有目录')
    parser.add_argument('--repeats', type=int, default=50)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({seed: {k: v for k, v in stats.items()
        if k in ('mean', 'sample_variance', 'sample_std', 'minimum', 'maximum',
                 'theoretical_mean', 'disagreement_questions', 'exactly_one_correct_questions')}
        for seed, stats in result['seeds'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
