"""用 Qwen 首步候选字母 logits 和 VideoJev 原始分数进行 log-linear 融合。"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timejev.lvb_experiment import ROOT, load_sample, read_predictions, score, sha256, write_json


FUSION_FORMULAS = {
    'loglinear': 'softmax(qwen_candidate_logits + beta * videojev_logits)',
    'probability': 'softmax(log_softmax(qwen_candidate_logits) + beta * videojev_probabilities)',
}


def controller_fusion(identity):
    """没有模式标识的历史产物只按原概率公式解释。"""
    fusion = identity.get('fusion', 'probability')
    if not isinstance(fusion, str) or fusion not in FUSION_FORMULAS:
        raise ValueError(f'未知 Controller 融合方式：{fusion}')
    return fusion


def softmax(values):
    peak = max(values)
    weights = [math.exp(value-peak) for value in values]
    total = sum(weights)
    return [weight/total for weight in weights]


def control(qwen, videojev, beta=2., fusion='loglinear'):
    """默认相加原始 logits；概率模式仅用于复现历史公式。"""
    controller_fusion(dict(fusion=fusion))
    if type(beta) not in (int, float) or not math.isfinite(beta):
        raise ValueError('beta 必须是有限数值')
    if any(qwen.get(key) != videojev.get(key) for key in ('id', 'query', 'options')):
        raise ValueError('两个模型的问题、选项或顺序不一致')
    logits = qwen.get('candidate_logits', [])
    if (not isinstance(logits, (list, tuple)) or len(logits) != len(qwen['options'])
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits)):
        raise ValueError(f'缺少有效的 Qwen 候选字母 logits：{qwen["id"]}')
    pi0 = softmax(logits)
    q_probs = qwen.get('probabilities', [])
    if (not isinstance(q_probs, (list, tuple)) or len(q_probs) != len(logits)
            or any(type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1 for q in q_probs)
            or any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6) for p, q in zip(pi0, q_probs))):
        raise ValueError(f'Qwen logits 与候选内 softmax 概率不一致：{qwen["id"]}')
    fs = videojev.get('probabilities', [])
    if (not isinstance(fs, (list, tuple)) or len(fs) != len(logits) or any(type(p) not in (int, float)
            or not math.isfinite(p) or not 0 <= p <= 1 for p in fs)
            or not math.isclose(sum(fs), 1., rel_tol=1e-5, abs_tol=1e-6)):
        raise ValueError('VideoJev 候选概率无效或未归一化')
    details = {}
    if fusion == 'loglinear':
        if 'logits' in videojev:
            scores = videojev['logits']
            if (not isinstance(scores, (list, tuple)) or len(scores) != len(logits) or any(type(v) not in (int, float)
                    or not math.isfinite(v) for v in scores)
                    or any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6)
                           for p, q in zip(softmax(scores), fs))):
                raise ValueError('VideoJev 原始 logits 与候选概率不一致')
            source = 'saved_logits'
        else:
            if any(p == 0 for p in fs):
                raise ValueError('VideoJev 概率含0，无法恢复原始分差；请使用保存 logits 的新来源')
            # log softmax(s_J) 与原始 s_J 只差候选公共常数，融合分布相同。
            scores = [math.log(p) for p in fs]
            source = 'log_probabilities'
        details.update(videojev_logits=list(scores), videojev_logits_source=source)
    else:
        scores = fs
    combined = [value+beta*score for value, score in zip(logits, scores)]
    if any(not math.isfinite(value) for value in combined):
        raise ValueError('融合 logits 不是有限数值')
    probabilities = softmax(combined)
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return dict(id=qwen['id'], task='choice', query=qwen['query'], options=qwen['options'],
        candidate_logits=logits, qwen_probabilities=pi0,
        videojev_probabilities=fs, beta=beta, fusion=fusion, **details,
        probabilities=probabilities, answer_index=index, answer=qwen['options'][index],
        preprocessing=qwen['preprocessing'])


def read_source(directory, rows, manifest_sha256, method):
    directory = Path(directory).resolve()
    run_path, metrics_path = directory/'run.json', directory/'metrics.json'
    if not run_path.is_file() or not metrics_path.is_file():
        raise ValueError(f'基线尚未完整完成：{directory}')
    metadata = json.loads(run_path.read_text())
    identity = metadata['identity']
    if identity['method'] != method or identity['manifest_sha256'] != manifest_sha256:
        raise ValueError(f'基线方法或样本 SHA 不匹配：{directory}')
    protocol = identity['protocol']
    if protocol.get('use_subtitles') is not False or protocol.get('use_audio') is not False:
        raise ValueError('控制器比较要求两基线均不使用字幕或音频')
    predictions_path = directory/'predictions.jsonl'
    predictions = read_predictions(predictions_path, rows)
    actual = score(rows, predictions)
    saved = json.loads(metrics_path.read_text())
    if (any(saved.get(key) != value for key, value in actual.items())
            or saved.get('method') != method or saved.get('protocol') != protocol):
        raise ValueError(f'基线 metrics 与完整预测不一致：{directory}')
    if method == 'jev_as_controller':
        fusion = controller_fusion(identity)
        if (metadata.get('formula') != FUSION_FORMULAS[fusion]
                or controller_fusion(saved) != fusion):
            raise ValueError(f'Controller 公式或融合模式与运行 identity 不一致：{directory}')
    source = dict(directory=str(directory), run_sha256=sha256(run_path),
        predictions_sha256=sha256(predictions_path), metrics_sha256=sha256(metrics_path),
        identity=identity)
    return predictions, source, actual


def evaluate(manifest, qwen, videojev, out, beta=2., qwen_method='qwen35', fusion='loglinear'):
    started_at = datetime.now(timezone.utc).isoformat()
    controller_fusion(dict(fusion=fusion))
    if qwen_method not in ('qwen35', 'qwen36'):
        raise ValueError('需要明确指定 Qwen 基线方法：qwen35 或 qwen36')
    if type(beta) not in (int, float) or not math.isfinite(beta):
        raise ValueError('beta 必须是有限数值')
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'拒绝覆盖非空输出目录：{out}')
    rows = load_sample(manifest)
    manifest_sha256 = sha256(manifest)
    qwen_predictions, qwen_source, qwen_metrics = read_source(qwen, rows, manifest_sha256, qwen_method)
    jev_predictions, jev_source, jev_metrics = read_source(videojev, rows, manifest_sha256, 'videojev')
    protocol = qwen_source['identity']['protocol']
    if protocol != jev_source['identity']['protocol']:
        raise ValueError('两基线的评估协议不完全一致')
    predictions = {}
    for row in rows:
        raw, jev = qwen_predictions[row['id']], jev_predictions[row['id']]
        if (not isinstance(raw.get('preprocessing'), dict) or not raw['preprocessing']
                or raw['preprocessing'] != jev.get('preprocessing')):
            raise ValueError(f'两基线的视频预处理不完全一致：{row["id"]}')
        prediction = control(raw, jev, beta, fusion=fusion)
        prediction.update(video_key=row['video_key'], duration_group=row['duration_group'])
        predictions[row['id']] = prediction
    metrics = score(rows, predictions)
    metrics.update(method='jev_as_controller', protocol=protocol, beta=beta, fusion=fusion,
                   finished_at=datetime.now(timezone.utc).isoformat())
    metadata = dict(identity=dict(method='jev_as_controller', manifest_sha256=manifest_sha256,
        protocol=protocol, beta=beta, fusion=fusion, sources=dict(qwen=qwen_source, videojev=jev_source),
        code_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__),
                     'timejev/lvb_experiment.py': sha256(ROOT/'timejev/lvb_experiment.py')}),
        manifest=str(Path(manifest).resolve()), command=sys.argv, started_at=started_at,
        hostname=socket.gethostname(), python=sys.version,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        formula=FUSION_FORMULAS[fusion],
        probability_scope='original_question_candidates', split='validation_subset',
        sampling='equal_official_duration_groups',
        note='分层抽样验证子集，不能作为完整官方 1337 题指标')
    comparison = {qwen_method: qwen_metrics, 'videojev': jev_metrics,
                  'jev_as_controller': score(rows, predictions)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'run.json', metadata)
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for prediction in predictions.values():
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+'\n')
    read_predictions(out/'predictions.jsonl', rows)
    write_json(out/'comparison.json', comparison)
    write_json(out/'metrics.json', metrics)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--qwen', required=True, help='已完整完成的 Qwen 评估目录')
    parser.add_argument('--videojev', required=True, help='已完整完成的 VideoJev 评估目录')
    parser.add_argument('--out', required=True)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--fusion', choices=tuple(FUSION_FORMULAS), default='loglinear',
                        help='loglinear 使用原始 logits；probability 仅复现历史概率公式')
    parser.add_argument('--qwen-method', choices=('qwen35', 'qwen36'), default='qwen35')
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
