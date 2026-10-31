"""固定100题：原生Qwen生成与保留的VideoJev/Controller候选采样三次结果汇总。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_longvideobench_fixed_questions import METHOD_ROLES, simulate, verify_sources
from scripts.summarize_longvideobench_proportional import read_single
from timejev.lvb_experiment import read_predictions, sha256, write_json
from timejev.lvb_native_generation import (
    audit_generated_tokens, generation_parameters, question_seed, read_generated, score_generated,
)

SEEDS = (42, 43, 44)
LABELS = dict(qwen36_native_instruct='Qwen3.6-35B-A3B（Instruct生成）',
              videojev='VideoJev-4B（候选采样）', jev_as_controller='Jev-as-Controller（候选采样）')


def statistics_for(values):
    return dict(accuracies=values, mean_accuracy=statistics.mean(values),
        sample_variance_fraction_squared=statistics.variance(values),
        sample_variance_percentage_points_squared=statistics.variance(values)*10000)


def summarize(native_base, fixed_sampling, out):
    native_base, fixed_sampling, out = [Path(path).resolve() for path in (native_base, fixed_sampling, out)]
    if out.exists():
        raise FileExistsError(f'拒绝覆盖已有汇总：{out}')
    fixed = json.loads((fixed_sampling/'results.json').read_text())
    if (fixed.get('schema') != 'longvideobench_fixed_questions_sampling_v1'
            or fixed.get('sample_seed') != 42 or fixed.get('inference_seeds') != list(SEEDS)
            or fixed.get('temperature') != 1 or fixed.get('n') != 100):
        raise ValueError('需要原固定seed42题集、temperature=1的三次候选采样结果')
    verify_sources(fixed['verified_sources'])
    entry = read_single(Path(fixed['sample']['path']).parent.parent, 42)
    rows, sample = entry['rows'], entry['sample']
    if fixed['sample'] != sample or fixed['question_ids'] != [row['id'] for row in rows]:
        raise ValueError('固定题集及来源不一致')
    artifact = json.loads((fixed_sampling/'artifact_manifest.json').read_text())
    sources = {str(fixed_sampling/'artifact_manifest.json'): sha256(fixed_sampling/'artifact_manifest.json')}
    for name, digest in artifact['files'].items():
        path = fixed_sampling/name
        if sha256(path) != digest:
            raise ValueError(f'已有采样产物SHA不匹配：{path}')
        sources[str(path)] = digest
    original_predictions = {method: read_predictions(
        Path(fixed['source_runs'][role]['directory'])/'predictions.jsonl', rows)
        for method, role in METHOD_ROLES.items()}
    replay, trials, predictions = simulate(rows, original_predictions, SEEDS)
    for name, expected in (('trials.jsonl', trials), ('predictions.jsonl', predictions)):
        actual = [json.loads(line) for line in (fixed_sampling/name).read_text().splitlines()]
        if actual != expected:
            raise ValueError(f'候选采样结果不能按原种子完整回放：{name}')
    if replay['methods'] != fixed['methods']:
        raise ValueError('候选采样统计不能从原预测重算')
    qwen_values, invalid_counts, native_sources, fingerprint = [], [], {}, None
    tokenizer = None
    for seed in SEEDS:
        directory = native_base/f'seed{seed}'
        metadata = json.loads((directory/'run.json').read_text())
        identity = metadata['identity']
        settings = identity['settings']
        if (metadata.get('status') != 'completed' or identity.get('method') != 'qwen36_native_instruct'
                or identity.get('inference_seed') != seed or identity.get('sample_seed') != 42
                or identity.get('manifest_sha256') != sample['sha256'] or identity.get('limit') is not None
                or identity.get('selected_ids') != fixed['question_ids']
                or identity.get('protocol') != entry['report']['protocols']['baseline']
                or identity.get('checkpoint') != fixed['source_runs']['qwen_baseline']['identity']['checkpoint']
                or settings.get('decoding') != generation_parameters()
                or settings.get('thinking') is not False or settings.get('presence_penalty') != 1.5
                or settings.get('scoring') != 'native_generation_then_option_parsing'):
            raise ValueError(f'原生生成的题集、checkpoint、采样配置、协议或完成状态不符：seed{seed}')
        current = {key: value for key, value in identity.items() if key != 'inference_seed'}
        if fingerprint is not None and current != fingerprint:
            raise ValueError('三次原生生成除推理seed外的身份指纹不一致')
        fingerprint = current
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(identity['checkpoint']['path'], local_files_only=True)
        by_id = read_generated(directory/'predictions.jsonl', rows)
        metrics = score_generated(rows, by_id)
        saved = json.loads((directory/'metrics.json').read_text())
        if (any(saved.get(key) != value for key, value in metrics.items())
                or saved.get('inference_seed') != seed or saved.get('method') != identity['method']
                or saved.get('protocol') != identity['protocol']):
            raise ValueError(f'原生生成计分与保存指标不一致：seed{seed}')
        for row in rows:
            prediction = by_id[row['id']]
            audit_generated_tokens(prediction, tokenizer)
            tokens = prediction.get('generated_token_ids', [])
            if (not tokens or any(type(token) is not int or token < 0 for token in tokens)
                    or len(tokens) != prediction.get('generated_tokens')
                    or prediction.get('question_seed') != question_seed(seed, row['id'])
                    or prediction.get('preprocessing') != original_predictions['qwen36'][row['id']]['preprocessing']):
                raise ValueError(f'缺少真实生成token、正确种子或同协议视频输入：{row["id"]}')
        qwen_values.append(metrics['accuracy'])
        invalid_counts.append(metrics['invalid'])
        native_sources[str(seed)] = dict(directory=str(directory), identity=identity)
        for name in ('run.json', 'predictions.jsonl', 'metrics.json'):
            sources[str(directory/name)] = sha256(directory/name)
    methods = dict(qwen36_native_instruct=statistics_for(qwen_values),
        videojev=statistics_for(fixed['methods']['videojev']['accuracies']),
        jev_as_controller=statistics_for(fixed['methods']['jev_as_controller']['accuracies']))
    result = dict(schema='longvideobench_native_baseline_comparison_v1', sample_seed=42,
        inference_seeds=list(SEEDS), n=100, question_ids=fixed['question_ids'], methods=methods,
        qwen_invalid_answers=invalid_counts, protocols=entry['report']['protocols'],
        qwen_sampling=dict(**generation_parameters(), presence_penalty=1.5, enable_thinking=False),
        other_sampling=dict(temperature=1., source='saved_candidate_probabilities'),
        sample=sample, native_runs=native_sources, fixed_sampling_directory=str(fixed_sampling),
        verified_sources=sources, script_sha256=sha256(__file__), sample_variance_ddof=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        note='固定100题；Qwen实际自回归采样后解析，无效回答计错；另外两方法保留原候选采样。'
             '基线4FPS，Controller2FPS；Qwen官方Instruct采样配置与候选temperature=1不同。')
    lines = ['# LongVideoBench：原生生成基线', '', result['note'], '',
        '| 方法 | 采样42 | 采样43 | 采样44 | 平均准确率 | 样本方差（百分点²） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, stats in methods.items():
        lines.append('| '+LABELS[method]+' | '+' | '.join(f'{value*100:.2f}%' for value in stats['accuracies'])
            +f' | {stats["mean_accuracy"]*100:.2f}% | {stats["sample_variance_percentage_points_squared"]:.4f} |')
    lines += ['', f'Qwen三次无效/截断回答数：{invalid_counts}；均保留在100题分母中。', '']
    verify_sources(fixed['verified_sources'])
    verify_sources(sources)
    out.mkdir(parents=True)
    write_json(out/'results.json', result)
    (out/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-base', required=True)
    parser.add_argument('--fixed-sampling', default='runs/lvb100_proportional_seed42_fixed_questions_sampling3_T1')
    parser.add_argument('--out', required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(result['methods'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
