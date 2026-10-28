"""用 Qwen 首步候选字母 logits 和 VideoJev 概率评估 Jev-as-Controller。"""
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


def softmax(values):
    peak = max(values)
    weights = [math.exp(value-peak) for value in values]
    total = sum(weights)
    return [weight/total for weight in weights]


def control(qwen, videojev, beta):
    """FS 直接使用 VideoJev 概率，不取 log 或再次 softmax。"""
    logits = qwen.get('candidate_logits', [])
    if (len(logits) != len(qwen['options'])
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits)):
        raise ValueError(f'缺少有效的 Qwen 候选字母 logits：{qwen["id"]}')
    pi0 = softmax(logits)
    if any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6)
           for p, q in zip(pi0, qwen['probabilities'])):
        raise ValueError(f'Qwen logits 与候选内 softmax 概率不一致：{qwen["id"]}')
    # softmax(log_softmax(logits) + beta*FS) 中的公共归一化常数可消去。
    probabilities = softmax([value+beta*fs for value, fs in zip(logits, videojev['probabilities'])])
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return dict(id=qwen['id'], task='choice', query=qwen['query'], options=qwen['options'],
        candidate_logits=logits, qwen_probabilities=pi0,
        videojev_probabilities=videojev['probabilities'], beta=beta,
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
    source = dict(directory=str(directory), run_sha256=sha256(run_path),
        predictions_sha256=sha256(predictions_path), metrics_sha256=sha256(metrics_path),
        identity=identity)
    return predictions, source, actual


def evaluate(manifest, qwen, videojev, out, beta=2., qwen_method='qwen35'):
    started_at = datetime.now(timezone.utc).isoformat()
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
        prediction = control(raw, jev, beta)
        prediction.update(video_key=row['video_key'], duration_group=row['duration_group'])
        predictions[row['id']] = prediction
    metrics = score(rows, predictions)
    metrics.update(method='jev_as_controller', protocol=protocol, beta=beta,
                   finished_at=datetime.now(timezone.utc).isoformat())
    metadata = dict(identity=dict(method='jev_as_controller', manifest_sha256=manifest_sha256,
        protocol=protocol, beta=beta, sources=dict(qwen=qwen_source, videojev=jev_source),
        code_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__),
                     'timejev/lvb_experiment.py': sha256(ROOT/'timejev/lvb_experiment.py')}),
        manifest=str(Path(manifest).resolve()), command=sys.argv, started_at=started_at,
        hostname=socket.gethostname(), python=sys.version,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        formula='softmax(log_softmax(qwen_candidate_logits) + beta * videojev_probabilities)',
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
    parser.add_argument('--qwen-method', choices=('qwen35', 'qwen36'), default='qwen35')
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
