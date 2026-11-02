"""Log-linear fusion: softmax(Qwen candidate logits + beta * VideoJev head scores)."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_controller_changes import read_manifest
from scripts.bench_compat import generic_score
from scripts.bench_io import json_bytes, jsonl_bytes, sha256, stratum, write_new
from scripts.summarize_bench import read_run
from scripts.videojev_candidate_logits import validate_logits

METHOD = 'jev_as_controller_loglinear'
FORMULA = 'softmax(qwen_candidate_logits + beta * videojev_candidate_logits)'


def softmax(values):
    peak = max(values)
    weights = [math.exp(z-peak) for z in values]
    return [v/sum(weights) for v in weights]


def control(qwen, videojev, beta=2., *, score_source='raw_logits'):
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('beta必须为非负有限数值')
    count = len(qwen['options'])
    z = validate_logits(qwen, count)
    if videojev.get('options') != qwen['options'] or videojev.get('id') != qwen['id']:
        raise ValueError('专家题目ID或选项不一致')
    if score_source == 'raw_logits':
        fs = validate_logits(videojev, count)
    elif score_source == 'log_probabilities':
        p = videojev.get('probabilities', [])
        if (len(p) != count or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in p)
                or not math.isclose(sum(p), 1., abs_tol=1e-5)):
            raise ValueError('log_probabilities要求所有候选概率严格大于0；含0时请重新导出原始logits，禁止裁剪替代')
        fs = [math.log(v) for v in p]
    else:
        raise ValueError('未知score_source')
    fused = [q+beta*f for q, f in zip(z, fs)]
    if any(not math.isfinite(v) for v in fused):
        raise ValueError('融合分数溢出')
    probabilities = softmax(fused)
    index = max(range(count), key=fused.__getitem__)
    return dict(id=qwen['id'], task='choice', query=qwen['query'], options=qwen['options'],
        candidate_logits=z, qwen_probabilities=qwen['probabilities'],
        videojev_scores=fs, videojev_score_source=score_source,
        videojev_probabilities=videojev['probabilities'], fused_logits=fused,
        beta=beta, fusion='loglinear', probabilities=probabilities,
        answer_index=index, answer=qwen['options'][index], preprocessing=qwen['preprocessing'])


def evaluate(manifest, qwen, videojev, out, beta=2., score_source='raw_logits'):
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('beta必须为非负有限数值')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'请选择新输出目录：{out}')
    rows, digest = read_manifest(manifest), sha256(manifest)
    (qp, qs, qm), (jp, js, jm) = [read_run(p, rows, digest, method) for p, method in
        ((qwen, 'qwen36'), (videojev, 'videojev'))]
    protocol = qs['identity']['protocol']
    seed = qs['identity'].get('seed')
    if protocol != js['identity']['protocol'] or protocol.get('use_subtitles') is not False or protocol.get('use_audio') is not False:
        raise ValueError('需要完全一致的无字幕无音频输入协议')
    if type(seed) is not int or seed != js['identity'].get('seed'):
        raise ValueError('两个来源必须具有相同seed')
    meta_path = Path(manifest).with_suffix('.meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('schema') == 'bench_subset_v1' and meta.get('seed') != seed:
            raise ValueError('sample seed与输入结果seed不一致')
    predictions = {}
    for row in rows:
        q, j = qp[row['id']], jp[row['id']]
        if not isinstance(q.get('preprocessing'), dict) or not q['preprocessing'] or q['preprocessing'] != j.get('preprocessing'):
            raise ValueError(f'两模型预处理不一致：{row["id"]}')
        result = control(q, j, beta, score_source=score_source)
        result.update(video_key=row['video_key'], benchmark=row['benchmark'], stratum=stratum(row))
        predictions[row['id']] = result
    metrics = generic_score(rows, predictions)
    metrics.update(method=METHOD, protocol=protocol, beta=beta, fusion='loglinear',
        videojev_score_source=score_source, finished_at=datetime.now(timezone.utc).isoformat())
    code_names = ['scripts/eval_bench_controller_loglinear.py', 'scripts/videojev_candidate_logits.py',
                  'scripts/analyze_controller_changes.py', 'scripts/bench_io.py', 'scripts/bench_compat.py',
                  'scripts/summarize_bench.py']
    root = Path(__file__).resolve().parents[1]
    metadata = dict(schema='video_bench_loglinear_run_v1', benchmark=rows[0]['benchmark'], split='screening_subset',
        identity=dict(method=METHOD, manifest_sha256=digest, seed=seed, protocol=protocol, beta=beta,
            fusion='loglinear', videojev_score_source=score_source, sources=dict(qwen=qs, videojev=js),
            code_sha256={name: sha256(root/name) for name in code_names}),
        formula=FORMULA, equivalent_formula='pi1_i proportional to pi0_i * ps_i**beta',
        score_note=('保存实际VideoJev head logits' if score_source == 'raw_logits' else
            '使用log(ps)=Fs-logsumexp(Fs)，与原始Fs相差公共常数；源概率有限精度可能造成舍入误差'),
        training_overlap_verified=False, manifest=str(Path(manifest).resolve()), command=sys.argv)
    qhit = [qp[r['id']]['answer_index'] == r['answer_index'] for r in rows]
    chit = [predictions[r['id']]['answer_index'] == r['answer_index'] for r in rows]
    metrics['paired_vs_qwen'] = dict(rescued=sum(not q and c for q, c in zip(qhit, chit)),
        harmed=sum(q and not c for q, c in zip(qhit, chit)),
        changed=sum(qp[r['id']]['answer_index'] != predictions[r['id']]['answer_index'] for r in rows))
    comparison = dict(qwen36=qm, videojev=jm, **{METHOD: metrics})
    out.mkdir(parents=True)
    for name, value in [('run.json', metadata), ('metrics.json', metrics), ('comparison.json', comparison)]:
        write_new(out/name, json_bytes(value))
    write_new(out/'predictions.jsonl', jsonl_bytes(list(predictions.values())))
    for name in code_names:
        write_new(out/'source'/name, (root/name).read_bytes())
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'qwen', 'videojev', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--score-source', choices=('raw_logits', 'log_probabilities'), default='raw_logits',
        help='默认用新VideoJev接口的原始head logits；旧结果可显式用log_probabilities（要求无零概率）')
    result = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
