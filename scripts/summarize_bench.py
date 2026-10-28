"""Audit complete screening runs and report paired, stratified diagnostics on CPU."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import json_bytes, load_rows, sha256, stratum, write_new
from scripts.bench_compat import generic_score, read_predictions

LIMITATIONS = ('粗筛子集，非官方全量口径；两模型未做算力对齐；训练重叠未核实。'
               '置信区间为按分层、问题级配对 bootstrap，同视频多题的相关性未建模。'
               'OR 是两个 top-1 答案的 oracle 参照，融合可能选第三个选项，因此不是严格数学上界。'
               'β 扫描仅诊断，不能用来选择该子集上的最佳 β。')


def controlled(raw, jev, beta):
    logits = raw.get('candidate_logits', [])
    if (len(logits) != len(raw['options'])
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)
            or not math.isfinite(beta)):
        raise ValueError('Invalid Qwen candidate logits or beta')
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    pi0 = [x / sum(weights) for x in weights]
    if any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6) for p, q in zip(pi0, raw['probabilities'])):
        raise ValueError('Qwen logits do not match source probabilities')
    scores = [z + beta * p for z, p in zip(logits, jev['probabilities'])]
    weights = [math.exp(x - max(scores)) for x in scores]
    probabilities = [x / sum(weights) for x in weights]
    return probabilities, max(range(len(scores)), key=scores.__getitem__)


def read_run(directory, rows, manifest_sha, method):
    directory = Path(directory).resolve()
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    identity = run['identity']
    if identity['method'] != method or identity['manifest_sha256'] != manifest_sha:
        raise ValueError('Run method/manifest SHA mismatch')
    predictions = read_predictions(directory / 'predictions.jsonl', rows)
    metrics = generic_score(rows, predictions)
    saved = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if (any(saved.get(k) != v for k, v in metrics.items()) or saved.get('method') != method
            or saved.get('protocol') != identity['protocol']
            or method == 'jev_as_controller' and saved.get('beta') != identity.get('beta')):
        raise ValueError(f'Saved metrics differ from predictions: {directory}')
    source = dict(directory=str(directory), identity=identity, **{
        name + '_sha256': sha256(directory / filename) for name, filename in
        (('run', 'run.json'), ('predictions', 'predictions.jsonl'), ('metrics', 'metrics.json'))})
    return predictions, source, metrics


def summarize(manifest, qwen, videojev, controller, out, beta_sweep='1,2,3', bootstrap=10000,
              random_repeats=50, seed=42):
    import numpy as np
    if bootstrap < 1 or random_repeats < 2:
        raise ValueError('Require positive bootstrap count and at least two random repeats')
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new summary directory')
    rows = load_rows(manifest)
    digest = sha256(manifest)
    runs = [read_run(p, rows, digest, m) for p, m in
            ((qwen, 'qwen36'), (videojev, 'videojev'), (controller, 'jev_as_controller'))]
    (qp, qs, qm), (jp, js, jm), (cp, cs, cm) = runs
    protocols = [s['identity']['protocol'] for _, s, _ in runs]
    protocol = protocols[0]
    expected = dict(fps=2., max_frames=2048, context=262144, visual_token_budget=24576,
                    use_subtitles=False, use_audio=False)
    if any(p != protocol for p in protocols) or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Screening requires the same fixed 2FPS/2048/262144/24576 video-only protocol')
    if qs['identity'].get('seed') is None or qs['identity'].get('seed') != js['identity'].get('seed'):
        raise ValueError('Source run seeds differ or are missing')
    sample_meta_path = Path(manifest).with_suffix('.meta.json')
    if sample_meta_path.exists():
        sample_meta_value = json.loads(sample_meta_path.read_text(encoding='utf-8'))
        if (sample_meta_value.get('schema') == 'bench_subset_v1'
                and sample_meta_value['seed'] != qs['identity']['seed']):
            raise ValueError('Source inference seed differs from sample seed')
    if cs['identity'].get('beta') != 2 or cm.get('beta', 2) != 2:
        raise ValueError('Primary controller must use beta=2')
    for name, source in (('qwen', qs), ('videojev', js)):
        if cs['identity']['sources'].get(name) != source:
            raise ValueError('Controller source fingerprints no longer match input runs')
    for row in rows:
        key = row['id']
        audit = qp[key].get('preprocessing')
        if not isinstance(audit, dict) or not audit or audit != jp[key].get('preprocessing') or audit != cp[key].get('preprocessing'):
            raise ValueError(f'Video preprocessing mismatch: {key}')
        probabilities, index = controlled(qp[key], jp[key], 2.)
        if index != cp[key]['answer_index'] or any(not math.isclose(p, q, abs_tol=1e-7)
                                                  for p, q in zip(probabilities, cp[key]['probabilities'])):
            raise ValueError(f'Controller formula does not match source predictions: {key}')
    gold = np.array([r['answer_index'] for r in rows])
    choices = {name: np.array([p[r['id']]['answer_index'] for r in rows])
               for name, p in (('qwen36', qp), ('videojev', jp), ('controller', cp))}
    hits = {name: (values == gold).astype(float) for name, values in choices.items()}
    hits['or_reference'] = np.maximum(hits['qwen36'], hits['videojev'])
    strata = {g: np.array([i for i, r in enumerate(rows) if stratum(r) == g])
              for g in sorted({stratum(r) for r in rows})}
    macro = qm['primary_metric'] == 'm_avg'

    def metric(values):
        return float(np.mean([values[idx].mean() for idx in strata.values()])) if macro else float(values.mean())

    # Resample the SAME question indices for every method and for the paired delta.
    # Fixed per-stratum sample sizes preserve MLVU M-Avg and the screening design.
    rng = np.random.default_rng(seed)
    draws = {name: np.zeros(bootstrap) for name in hits}
    for idx in strata.values():
        selected = rng.choice(idx, size=(bootstrap, len(idx)), replace=True)
        weight = 1 / len(strata) if macro else len(idx) / len(rows)
        for name, values in hits.items():
            draws[name] += weight * values[selected].mean(axis=1)
    ci = lambda values: [float(v) for v in np.quantile(values, [.025, .975])]
    methods = {name: dict(metric=metric(values), correct=int(values.sum()), n=len(rows),
                         ci95=ci(draws[name])) for name, values in hits.items()}
    rng = np.random.default_rng(seed + 1)
    random_values = [metric(np.where(rng.random(len(rows)) < .5, hits['qwen36'], hits['videojev']))
                     for _ in range(random_repeats)]
    methods['random_choice'] = dict(metric=float(np.mean(random_values)),
        sample_variance=float(np.var(random_values, ddof=1)), repeats=random_repeats,
        expected_metric=(methods['qwen36']['metric'] + methods['videojev']['metric']) / 2,
        min=float(min(random_values)), max=float(max(random_values)))
    qhit, chit = hits['qwen36'].astype(bool), hits['controller'].astype(bool)
    rescued, harmed = int((~qhit & chit).sum()), int((qhit & ~chit).sum())
    discordant = rescued + harmed
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(rescued, harmed) + 1)) / 2**discordant) if discordant else 1.
    disagree = choices['qwen36'] != choices['videojev']
    delta = methods['controller']['metric'] - methods['qwen36']['metric']
    room = methods['or_reference']['metric'] - methods['qwen36']['metric']
    sweep = {}
    for beta in [float(v) for v in beta_sweep.split(',')]:
        values = np.array([controlled(qp[r['id']], jp[r['id']], beta)[1] == r['answer_index'] for r in rows])
        sweep[str(beta)] = dict(metric=metric(values), delta_vs_qwen=metric(values) - methods['qwen36']['metric'])
    same_direction = all(v['delta_vs_qwen'] > 0 for v in sweep.values()) if delta > 0 else all(v['delta_vs_qwen'] < 0 for v in sweep.values()) if delta < 0 else False
    decision = ('no-go: 排查清单/协议后，两个 top-1 参照空间不足' if room < .01 or disagree.mean() < .05
                else 'go: 正方向，需后续独立样本确认' if delta >= .02 and same_direction
                else 'negative: 如实记录反向证据' if delta <= -.02 and same_direction
                else 'uncertain: 查看分层，考虑独立 seed43')
    by_stratum = {g: dict(n=len(idx), **{name: float(values[idx].mean()) for name, values in hits.items()},
                          delta_vs_qwen=float((hits['controller'][idx] - hits['qwen36'][idx]).mean()))
                  for g, idx in strata.items()}
    categories = {}
    for category in sorted({c for r in rows for c in r.get('question_type', [])}):
        idx = [i for i, r in enumerate(rows) if category in r['question_type']]
        categories[category] = dict(n=len(idx), **{name: float(v[idx].mean()) for name, v in hits.items()},
            delta_vs_qwen=float((hits['controller'][idx] - hits['qwen36'][idx]).mean()))
    sample_meta = Path(manifest).with_suffix('.meta.json')
    result = dict(schema='bench_screen_summary_v1', benchmark=rows[0].get('benchmark'), n=len(rows),
        seed=qs['identity']['seed'], manifest_sha256=digest,
        sample_meta=json.loads(sample_meta.read_text(encoding='utf-8')) if sample_meta.exists() else None,
        primary_metric=qm['primary_metric'], protocol=protocol, beta=2., methods=methods,
        delta_vs_qwen=delta, delta_ci95=ci(draws['controller'] - draws['qwen36']), or_minus_qwen=room,
        disagreement=dict(n=int(disagree.sum()), rate=float(disagree.mean()),
            controller_changes_on_agreement=int(((~disagree) & (choices['controller'] != choices['qwen36'])).sum()),
            third_choice_count=int(((choices['controller'] != choices['qwen36']) & (choices['controller'] != choices['videojev'])).sum()),
            controller_correct_both_baselines_wrong=int(((hits['or_reference'] == 0) & chit).sum())),
        paired=dict(rescued=rescued, harmed=harmed, mcnemar_exact_two_sided_p=p,
                    mcnemar_scope='pooled question correctness; not a test of macro M-Avg'),
        by_stratum=by_stratum, by_question_type=categories, beta_sweep=sweep,
        bootstrap=dict(repeats=bootstrap, seed=seed, unit='question', paired=True, stratified=True),
        decision=decision, limitations=LIMITATIONS, training_overlap_verified=False,
        sources=dict(qwen=qs, videojev=js, controller=cs), summarizer_sha256=sha256(__file__))
    lines = [f'# {result["benchmark"]} 粗筛（n={len(rows)}, seed={result["seed"]}）', '',
             f'主指标：{result["primary_metric"]}；2 FPS / 2048 帧 / context 262144 / 视觉 token 24576；无字幕、无音频。', '',
             '| 方法 | 主指标 % | 答对/总数 | 95% bootstrap CI % |', '|---|---:|---:|---|']
    for name, row in methods.items():
        interval = ' / '.join(f'{100*v:.2f}' for v in row['ci95']) if 'ci95' in row else '—（50%随机选边模拟）'
        count = f'{row["correct"]}/{row["n"]}' if 'correct' in row else '—'
        lines.append(f'| {name} | {100*row["metric"]:.2f} | {count} | {interval} |')
    lines += ['', f'Controller − Qwen：{100*delta:+.2f} pp；配对 95% CI：{[round(100*v, 2) for v in result["delta_ci95"]]} pp。',
        f'OR 参照 − Qwen：{100*room:+.2f} pp；异答率：{100*disagree.mean():.2f}%（{int(disagree.sum())}题）。',
        f'纠正 {rescued} / 损害 {harmed}；McNemar 精确双侧 p={p:.6g}（问题微平均正确性，非宏平均差的检验）。',
        f'两基线均错但 Controller 答对：{result["disagreement"]["controller_correct_both_baselines_wrong"]}；选第三个选项：{result["disagreement"]["third_choice_count"]}。', '']
    for title, table in (('互斥分层', by_stratum), ('官方多标签类别（计数可重叠）', categories)):
        if table:
            lines += [f'## {title}', '', '| 分组 | n | Qwen % | VideoJev % | Controller % | Δ pp |', '|---|---:|---:|---:|---:|---:|']
            for group, row in table.items():
                lines.append(f'| {group} | {row["n"]} | {100*row["qwen36"]:.2f} | {100*row["videojev"]:.2f} | {100*row["controller"]:.2f} | {100*row["delta_vs_qwen"]:+.2f} |')
            lines.append('')
    lines += ['## β 敏感性（诊断）', '', '| β | 主指标 % | Δ pp |', '|---|---:|---:|']
    for beta, row in sweep.items():
        lines.append(f'| {beta} | {100*row["metric"]:.2f} | {100*row["delta_vs_qwen"]:+.2f} |')
    lines += ['', f'决策建议：{decision}。', '', LIMITATIONS, '']
    write_new(out / 'results.json', json_bytes(result))
    write_new(out / 'results.md', '\n'.join(lines).encode('utf-8'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'qwen', 'videojev', 'controller', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--beta-sweep', default='1,2,3')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--random-repeats', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42, help='Statistics RNG seed, independent of sampling seed')
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ('benchmark', 'n', 'delta_vs_qwen', 'decision')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
