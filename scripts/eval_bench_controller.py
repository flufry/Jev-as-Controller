"""Benchmark Controller: upstream log-linear fusion by default; explicit legacy probability mode."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_controller_changes import read_manifest
from scripts.bench_compat import benchmark_scope, generic_score
from scripts.bench_io import json_bytes, jsonl_bytes, sample_seed, sha256, source_sample_seed, stratum, write_new
from scripts.eval_longvideobench_controller import control as upstream_control, FUSION_FORMULAS
from scripts.summarize_bench import read_run
from scripts.videojev_candidate_logits import candidate_probabilities, softmax, validate_logits


def control(qwen, videojev, beta=2., fusion='loglinear', *, score_source='auto'):
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('beta 必须为非负有限数值')
    if fusion not in FUSION_FORMULAS or score_source not in ('auto', 'raw_logits', 'log_probabilities'):
        raise ValueError('未知融合模式或分数来源')
    jev = dict(videojev, probabilities=candidate_probabilities(videojev))
    if 'logits' in videojev or 'candidate_logits' in videojev:
        jev['logits'] = validate_logits(videojev, len(qwen['options']))
    if fusion == 'loglinear' and score_source == 'raw_logits' and 'logits' not in jev:
        raise ValueError('raw_logits 要求 VideoJev 来源保存原始 logits')
    if fusion == 'loglinear' and score_source == 'log_probabilities':
        jev.pop('logits', None)
    # Share upstream's implementation, including zero-probability and consistency checks.
    result = upstream_control(qwen, jev, beta=beta, fusion=fusion)
    scores = result.get('videojev_logits', jev['probabilities'])
    result.update(fused_logits=[q+beta*j for q, j in zip(qwen['candidate_logits'], scores)],
                  videojev_scores=list(scores),
                  videojev_score_source=result.get('videojev_logits_source', 'softmax_probabilities'))
    return result


def evaluate(manifest, qwen, videojev, out, beta=2., qwen_method='qwen36', fusion='loglinear', score_source='auto'):
    if qwen_method != 'qwen36':
        raise ValueError('Controller requires Qwen3.6 candidate logits, not native text generation')
    if fusion not in FUSION_FORMULAS or type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError('Invalid fusion or beta')
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Choose a new Controller output directory')
    rows, digest = read_manifest(manifest), sha256(manifest)
    (qp, qs, qm), (jp, js, jm) = [read_run(p, rows, digest, method) for p, method in
                                 ((qwen, 'qwen36'), (videojev, 'videojev'))]
    protocol = qs['identity']['protocol']
    seed = qs['identity'].get('seed')
    if protocol != js['identity']['protocol'] or protocol.get('use_subtitles') is not False or protocol.get('use_audio') is not False:
        raise ValueError('Both inputs require the same video-only protocol')
    if type(seed) is not int or seed != js['identity'].get('seed'):
        raise ValueError('Both source runs must have the same explicit seed')
    fixed_seed = sample_seed(manifest)
    if fixed_seed is not None and any(source_sample_seed(source['identity']) != fixed_seed for source in (qs, js)):
        raise ValueError('Model runs must use the manifest question-sampling seed')
    predictions = {}
    for row in rows:
        q, j = qp[row['id']], jp[row['id']]
        if not isinstance(q.get('preprocessing'), dict) or not q['preprocessing'] or q['preprocessing'] != j.get('preprocessing'):
            raise ValueError(f'Video preprocessing mismatch: {row["id"]}')
        value = control(q, j, beta, fusion, score_source=score_source)
        value.update(video_key=row['video_key'], benchmark=row['benchmark'], stratum=stratum(row))
        predictions[row['id']] = value
    metrics = generic_score(rows, predictions)
    qhit = [qp[r['id']]['answer_index'] == r['answer_index'] for r in rows]
    chit = [predictions[r['id']]['answer_index'] == r['answer_index'] for r in rows]
    metrics.update(method='jev_as_controller', protocol=protocol, beta=beta, fusion=fusion,
                   finished_at=datetime.now(timezone.utc).isoformat(),
                   paired_vs_qwen=dict(rescued=sum(not q and c for q, c in zip(qhit, chit)),
                       harmed=sum(q and not c for q, c in zip(qhit, chit)),
                       changed=sum(qp[r['id']]['answer_index'] != predictions[r['id']]['answer_index'] for r in rows)))
    code_names = ['scripts/eval_bench_controller.py', 'scripts/eval_longvideobench_controller.py',
                  'scripts/videojev_candidate_logits.py', 'scripts/analyze_controller_changes.py',
                  'scripts/bench_io.py', 'scripts/bench_compat.py', 'scripts/summarize_bench.py']
    root = Path(__file__).resolve().parents[1]
    metadata = dict(schema='video_bench_screen_run_v1', benchmark=rows[0]['benchmark'], split=benchmark_scope(rows),
        identity=dict(method='jev_as_controller', manifest_sha256=digest, seed=seed, sample_seed=fixed_seed,
            protocol=protocol,
            beta=beta, fusion=fusion, score_source=score_source, sources=dict(qwen=qs, videojev=js),
            code_sha256={name: sha256(root/name) for name in code_names}),
        formula=FUSION_FORMULAS[fusion], manifest=str(Path(manifest).resolve()), command=sys.argv,
        note='视频 benchmark；raw logits 默认直接相加，旧概率公式必须显式指定 probability')
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('run.json', metadata), ('metrics.json', metrics),
                        ('comparison.json', dict(qwen36=qm, videojev=jm, jev_as_controller=metrics))]:
        write_new(out/name, json_bytes(value))
    write_new(out/'predictions.jsonl', jsonl_bytes(list(predictions.values())))
    for name in code_names:
        write_new(out/'source'/name, (root/name).read_bytes())
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'qwen', 'videojev', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--fusion', choices=tuple(FUSION_FORMULAS), default='loglinear')
    parser.add_argument('--score-source', choices=('auto', 'raw_logits', 'log_probabilities'), default='auto',
                        help='auto: raw logits first; positive historical probabilities may be logged')
    parser.add_argument('--qwen-method', choices=('qwen36',), default='qwen36')
    result = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
