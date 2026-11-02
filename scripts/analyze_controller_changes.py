"""Audit Controller answer changes and export per-option probabilities on CPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import json_bytes, jsonl_bytes, sha256, stratum, write_new
from scripts.summarize_bench import controlled, read_run


FILTERS = {
    'all': ('全部题目', lambda r: True),
    'disagreements': ('Qwen 与 VideoJev 异答', lambda r: r['disagreement']),
    'changed': ('Controller 修改 Qwen 答案', lambda r: r['changed_vs_qwen']),
    'rescued': ('改对：Qwen 错 → Controller 对', lambda r: r['change_type'] == 'rescued'),
    'harmed': ('改错：Qwen 对 → Controller 错', lambda r: r['change_type'] == 'harmed'),
    'changed_still_wrong': ('答案修改后仍错', lambda r: r['change_type'] == 'changed_still_wrong'),
    'third_choice': ('Controller 选择两专家 top-1 之外的选项', lambda r: r['third_choice']),
}
NOTE = ('改答、改对、改错均相对 Controller 输入的 Qwen 候选评分结果，非 qwen_native。'
    'π₀=softmax(z)，Fₛ=VideoJev候选概率，π₁=softmax(z+βFₛ)。'
    'z+βFₛ 与 log(π₀)+βFₛ 相差题内公共常数，softmax 后相同。'
    '所有概率均为该题原始选项内归一化概率，不是全词表概率。')


def read_manifest(path):
    """Validate saved annotations without requiring the original videos/models."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('清单须非空且题目 ID 不重复')
    for row in rows:
        options = row.get('options')
        if (row.get('task') != 'choice' or not isinstance(options, list) or not 2 <= len(options) <= 8
                or not all(isinstance(v, str) and v.strip() for v in options)
                or not isinstance(row.get('query'), str)
                or type(row.get('answer_index')) is not int
                or not 0 <= row['answer_index'] < len(options)):
            raise ValueError(f'清单题目或真实答案无效：{row["id"]}')
        stratum(row)
    if len({r.get('benchmark') for r in rows}) != 1 or not rows[0].get('benchmark'):
        raise ValueError('需要含明确 benchmark 的单数据集清单')
    meta_path = path.with_suffix('.meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('sample_sha256', meta.get('manifest_sha256')) != sha256(path):
            raise ValueError('清单 SHA 与 sample meta 不一致')
        if meta.get('schema') == 'bench_subset_v1' and (
                meta.get('selected_ids') != [r['id'] for r in rows]
                or meta.get('n') != len(rows)
                or {k: v for k, v in meta['quotas'].items() if v} != dict(Counter(stratum(r) for r in rows))):
            raise ValueError('清单题目/分层数量与 sample meta 不一致')
    return rows


def source_matches(saved, actual):
    # A copied run retains Linux paths in metadata; only its location may differ.
    return all(saved.get(key) == actual[key] for key in
        ('identity', 'run_sha256', 'predictions_sha256', 'metrics_sha256'))


def analyze(manifest, qwen, videojev, controller):
    rows = read_manifest(manifest)
    digest = sha256(manifest)
    (qp, qs, _), (jp, js, _), (cp, cs, _) = [read_run(p, rows, digest, method)
        for p, method in ((qwen, 'qwen36'), (videojev, 'videojev'), (controller, 'jev_as_controller'))]
    identity = cs['identity']
    beta = identity.get('beta')
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('Controller beta 必须是非负有限数值')
    protocol = qs['identity']['protocol']
    if protocol != js['identity']['protocol'] or protocol != identity['protocol']:
        raise ValueError('Qwen、VideoJev 和 Controller 协议不一致')
    seed = qs['identity'].get('seed')
    if type(seed) is not int or seed != js['identity'].get('seed'):
        raise ValueError('两个专家的推理 seed 不一致或缺失')
    meta_path = Path(manifest).with_suffix('.meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('schema') == 'bench_subset_v1' and meta.get('seed') != seed:
            raise ValueError('sample seed 与输入运行的 seed 不一致')
    for name, source in (('qwen', qs), ('videojev', js)):
        if not source_matches(identity.get('sources', {}).get(name, {}), source):
            raise ValueError(f'Controller 来源指纹不匹配：{name}')
    records = []
    for row in rows:
        key = row['id']
        q, j, c = qp[key], jp[key], cp[key]
        audit = q.get('preprocessing')
        if not isinstance(audit, dict) or not audit or audit != j.get('preprocessing') or audit != c.get('preprocessing'):
            raise ValueError(f'三份结果预处理不一致：{key}')
        probabilities, index = controlled(q, j, beta)
        if index != c['answer_index'] or any(not math.isclose(p, v, rel_tol=1e-7, abs_tol=1e-7)
                for p, v in zip(probabilities, c['probabilities'])):
            raise ValueError(f'Controller 概率无法由输入按公式重算：{key}')
        for field, values in (('candidate_logits', q['candidate_logits']),
                ('qwen_probabilities', q['probabilities']), ('videojev_probabilities', j['probabilities'])):
            saved = c.get(field)
            if not isinstance(saved, list) or len(saved) != len(values) or any(
                    type(x) not in (int, float) or not math.isfinite(x)
                    or not math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6) for x, y in zip(saved, values)):
                raise ValueError(f'Controller 内嵌 {field} 与来源不一致：{key}')
        if c.get('beta') != beta:
            raise ValueError(f'逐题 beta 与 run.json 不一致：{key}')
        qi, ji, ci, gold = q['answer_index'], j['answer_index'], index, row['answer_index']
        qhit, jhit, chit = qi == gold, ji == gold, ci == gold
        changed = ci != qi
        change_type = ('unchanged' if not changed else 'rescued' if chit else
                       'harmed' if qhit else 'changed_still_wrong')

        def choice(i):
            return dict(index=i, letter=chr(65+i), text=row['options'][i])

        records.append(dict(id=key, benchmark=row['benchmark'], seed=seed, stratum=stratum(row),
            video_key=row['video_key'], query=row['query'], options=row['options'], gold=choice(gold),
            qwen=choice(qi), videojev=choice(ji), controller=choice(ci),
            qwen_correct=qhit, videojev_correct=jhit, controller_correct=chit,
            disagreement=qi != ji, changed_vs_qwen=changed, change_type=change_type,
            changed_vs_videojev=ci != ji, rescued_vs_videojev=not jhit and chit,
            harmed_vs_videojev=jhit and not chit, third_choice=ci not in (qi, ji), beta=beta,
            per_option=[dict(index=i, letter=chr(65+i), text=option,
                qwen_logit=q['candidate_logits'][i], qwen_probability=q['probabilities'][i],
                videojev_probability=j['probabilities'][i], correction=beta*j['probabilities'][i],
                fused_score=q['candidate_logits'][i]+beta*j['probabilities'][i],
                controller_probability=c['probabilities'][i]) for i, option in enumerate(row['options'])]))
    return records, dict(manifest=str(Path(manifest).resolve()), manifest_sha256=digest,
        seed=seed, benchmark=rows[0]['benchmark'], beta=beta, protocol=protocol,
        sources=dict(qwen=qs, videojev=js, controller=cs))


def counts(records):
    result = {name: sum(select(r) for r in records) for name, (_, select) in FILTERS.items()}
    result.update(qwen_correct=sum(r['qwen_correct'] for r in records),
        videojev_correct=sum(r['videojev_correct'] for r in records),
        controller_correct=sum(r['controller_correct'] for r in records),
        changed_vs_videojev=sum(r['changed_vs_videojev'] for r in records),
        rescued_vs_videojev=sum(r['rescued_vs_videojev'] for r in records),
        harmed_vs_videojev=sum(r['harmed_vs_videojev'] for r in records))
    result['net_correct_gain_vs_qwen'] = result['rescued']-result['harmed']
    return result


def escape(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\r', '').replace('\n', '<br>')


def detail_markdown(records, title):
    lines = [f'# {title}（{len(records)} 题次）', '', NOTE, '',
        '表中概率为 0–1 小数；显示 9 位小数，JSONL 保留原始精度。', '']
    for r in records:
        lines += [f'## seed {r["seed"]} · {escape(r["id"])}', '',
            f'分层：{escape(r["stratum"])}；类型：{r["change_type"]}；β={r["beta"]}。', '',
            escape(r['query']), '',
            f'正确答案：{r["gold"]["letter"]}；Qwen：{r["qwen"]["letter"]}；'
            f'VideoJev：{r["videojev"]["letter"]}；Controller：{r["controller"]["letter"]}。', '',
            '| 选项 | 内容 | Qwen z | Qwen π₀ | VideoJev Fₛ | βFₛ | 融合分数 z+βFₛ | Controller π₁ |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for p in r['per_option']:
            values = ' | '.join(f'{p[k]:.9f}' for k in ('qwen_logit', 'qwen_probability',
                'videojev_probability', 'correction', 'fused_score', 'controller_probability'))
            lines.append(f'| {p["letter"]} | {escape(p["text"])} | {values} |')
        lines.append('')
    return '\n'.join(lines)


def export(jobs, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'请选择全新输出目录：{out}')
    records, runs, seen = [], [], set()
    for job in jobs:
        items, source = analyze(**job)
        identity = source['benchmark'], source['seed']
        if identity in seen:
            raise ValueError(f'重复数据集/seed：{identity}')
        seen.add(identity)
        records.extend(items)
        runs.append(dict(source, counts=counts(items)))
    if not records or len({r['benchmark'] for r in records}) != 1:
        raise ValueError('每次导出需要同一数据集的非空运行列表')
    summary = dict(schema='controller_answer_changes_v1', benchmark=records[0]['benchmark'],
        counts=counts(records), runs=runs, definitions=NOTE, analyzer_sha256=sha256(__file__))
    lines = ['# Controller 逐题变化统计', '', NOTE, '',
        '不同 seed 合计为题次，可能重复抽到同一道题；计数包括全部题目，不只限于异答题。', '',
        '| seed | n | 异答 | 改答 | 改对 | 改错 | 改后仍错 | 第三选项 | Qwen 对 | VideoJev 对 | Controller 对 | 净增正确 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, c in [(r['seed'], r['counts']) for r in runs]+[('合计', summary['counts'])]:
        keys = ('all', 'disagreements', 'changed', 'rescued', 'harmed', 'changed_still_wrong',
                'third_choice', 'qwen_correct', 'videojev_correct', 'controller_correct', 'net_correct_gain_vs_qwen')
        lines.append(f'| {seed} | '+' | '.join(str(c[k]) for k in keys)+' |')
    lines += ['', '改答 = 改对 + 改错 + 改后仍错；净增正确 = 改对 − 改错。', '',
        '## 逐题文件', '', '每组同时输出 Markdown 明细与 JSONL；每条包含全部选项的分数和概率。', '']
    # All inputs are validated before any output is created.
    out.mkdir(parents=True)
    for name, (title, select) in FILTERS.items():
        selected = [r for r in records if select(r)]
        write_new(out/f'{name}.jsonl', jsonl_bytes(selected))
        write_new(out/f'{name}.md', detail_markdown(selected, title).encode('utf-8'))
        lines.append(f'- [{title}](./{name}.md)：{len(selected)} 题次。')
    write_new(out/'summary.json', json_bytes(summary))
    write_new(out/'summary.md', ('\n'.join(lines)+'\n').encode('utf-8'))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', help='批量模式：含 sample_42.jsonl、qwen_logits、controller 等的目录')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44], help='批量模式 seeds，默认 42 43 44')
    for name in ('manifest', 'qwen', 'videojev', 'controller'):
        parser.add_argument('--'+name, help='单次模式输入；不能与 --runs-root 同时使用')
    parser.add_argument('--out', required=True, help='全新输出目录；不覆盖已有文件')
    args = parser.parse_args(argv)
    names = ('manifest', 'qwen', 'videojev', 'controller')
    explicit = {k: getattr(args, k) for k in names}
    if args.runs_root:
        if any(explicit.values()):
            parser.error('--runs-root 不能与单次模式输入同时使用')
        root = Path(args.runs_root)
        jobs = []
        for seed in args.seeds:
            suffix = '' if seed == 42 else f'_{seed}'
            manifest = root/f'sample_{seed}.jsonl'
            if seed == 42 and not manifest.is_file():
                manifest = root/'sample.jsonl'
            jobs.append(dict(manifest=manifest, qwen=root/f'qwen_logits{suffix}',
                videojev=root/f'videojev{suffix}', controller=root/f'controller{suffix}'))
    else:
        if not all(explicit.values()):
            parser.error('请指定 --runs-root，或同时指定 --manifest --qwen --videojev --controller')
        jobs = [explicit]
    result = export(jobs, args.out)
    print(json.dumps(dict(out=str(Path(args.out).resolve()), **result['counts']), ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
