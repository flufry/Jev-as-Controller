import copy
import json
import math

import pytest

from scripts.eval_longvideobench_controller import evaluate, softmax
from timejev.lvb_experiment import load_sample, read_predictions, score, sha256, write_json
from timejev.user_data import write_jsonl


@pytest.fixture
def experiment(tmp_path):
    video = tmp_path/'video.mp4'
    video.touch()
    manifest = tmp_path/'sample.jsonl'
    rows = [dict(id=f'q{i}', question_id=f'q{i}', video_key='v', video_path='video.mp4',
        duration=15., task='choice', query=f'question {i}',
        options=['A. one', 'B. two', 'C. three'], answer_index=i%3, duration_group=group)
        for i, group in enumerate((15, 60, 600, 3600))]
    write_jsonl(rows, manifest)
    protocol = dict(fps=2., max_frames=2048, context=262144, visual_token_budget=24576,
                    use_subtitles=False, use_audio=False)
    sources = {}
    for method in ('qwen35', 'videojev'):
        directory = tmp_path/method
        directory.mkdir()
        predictions = []
        for row in rows:
            values = [1., -.5, 0.] if method == 'qwen35' else [-2., 3., 0.]
            probabilities = softmax(values)
            prediction = dict(id=row['id'], query=row['query'], options=row['options'],
                task='choice', answer_index=max(range(3), key=probabilities.__getitem__),
                probabilities=probabilities, preprocessing=dict(frames=30, grid_thw=[15, 4, 4]))
            if method == 'qwen35':
                prediction['candidate_logits'] = values
            predictions.append(prediction)
        # 特意打乱源输出顺序，以 ID 对齐而非 zip。
        if method == 'videojev':
            predictions.reverse()
        write_jsonl(predictions, directory/'predictions.jsonl')
        identity = dict(method=method, manifest_sha256=sha256(manifest), protocol=protocol,
            checkpoint=dict(path=f'models/{method}', files={'model.safetensors':dict(sha256='mock', bytes=1)}),
            settings=dict(mock=True))
        write_json(directory/'run.json', dict(identity=identity))
        metrics = score(load_sample(manifest), read_predictions(directory/'predictions.jsonl', rows))
        metrics.update(method=method, protocol=protocol)
        write_json(directory/'metrics.json', metrics)
        sources[method] = directory
    return manifest, rows, sources


def invoke(experiment, tmp_path, **kwargs):
    manifest, _, sources = experiment
    return evaluate(manifest, sources['qwen35'], sources['videojev'], tmp_path/'controller', **kwargs)


def test_formula_id_alignment_and_comparison(experiment, tmp_path):
    comparison = invoke(experiment, tmp_path)
    manifest, rows, sources = experiment
    qwen = read_predictions(sources['qwen35']/'predictions.jsonl', rows)
    jev = read_predictions(sources['videojev']/'predictions.jsonl', rows)
    controlled = read_predictions(tmp_path/'controller/predictions.jsonl', rows)
    assert set(comparison) == {'qwen35', 'videojev', 'jev_as_controller'}
    assert comparison['jev_as_controller']['n'] == 4
    for row in rows:
        raw, fs, result = qwen[row['id']], jev[row['id']], controlled[row['id']]
        # 独立按 pi0*exp(beta*FS) 算，避免复刻实现。
        weights = [p*math.exp(2*f) for p, f in zip(raw['probabilities'], fs['probabilities'])]
        expected = [value/sum(weights) for value in weights]
        assert result['probabilities'] == pytest.approx(expected)
        assert result['answer_index'] == 1
        assert result['qwen_probabilities'] == pytest.approx(raw['probabilities'])
        assert result['videojev_probabilities'] == fs['probabilities']
    metadata = json.loads((tmp_path/'controller/run.json').read_text())
    assert metadata['identity']['manifest_sha256'] == sha256(manifest)
    assert metadata['identity']['sources']['qwen']['identity']['checkpoint']['files']
    assert metadata['split'] == 'validation_subset'
    assert json.loads((tmp_path/'controller/metrics.json').read_text())['n'] == 4


def test_beta_zero_matches_qwen(experiment, tmp_path):
    invoke(experiment, tmp_path, beta=0.)
    _, rows, sources = experiment
    qwen = read_predictions(sources['qwen35']/'predictions.jsonl', rows)
    result = read_predictions(tmp_path/'controller/predictions.jsonl', rows)
    for identity in qwen:
        assert result[identity]['probabilities'] == pytest.approx(qwen[identity]['probabilities'])


def test_qwen36_is_explicit_and_preserves_source_identity(experiment, tmp_path):
    manifest, rows, sources = experiment
    directory = sources['qwen35']
    metadata = json.loads((directory/'run.json').read_text())
    metadata['identity']['method'] = 'qwen36'
    metadata['identity']['checkpoint']['path'] = 'models/Qwen3.6-35B-A3B'
    write_json(directory/'run.json', metadata)
    metrics = json.loads((directory/'metrics.json').read_text())
    metrics['method'] = 'qwen36'
    write_json(directory/'metrics.json', metrics)
    with pytest.raises(ValueError, match='方法或样本'):
        invoke(experiment, tmp_path)
    comparison = invoke(experiment, tmp_path, qwen_method='qwen36')
    assert set(comparison) == {'qwen36', 'videojev', 'jev_as_controller'}
    saved = json.loads((tmp_path/'controller/run.json').read_text())
    assert saved['identity']['sources']['qwen']['identity']['method'] == 'qwen36'
    assert saved['identity']['sources']['qwen']['identity']['checkpoint']['path'] == 'models/Qwen3.6-35B-A3B'
    assert read_predictions(tmp_path/'controller/predictions.jsonl', rows)


def test_unknown_qwen_method_rejected(experiment, tmp_path):
    with pytest.raises(ValueError, match='明确指定'):
        invoke(experiment, tmp_path, qwen_method='unknown')


def test_labels_do_not_enter_controller_distribution(experiment, tmp_path):
    invoke(experiment, tmp_path)
    original = [json.loads(line) for line in (tmp_path/'controller/predictions.jsonl').read_text().splitlines()]
    manifest, rows, sources = experiment
    changed = copy.deepcopy(rows)
    for row in changed:
        row['answer_index'] = (row['answer_index']+1)%len(row['options'])
    write_jsonl(changed, manifest)
    for method, directory in sources.items():
        metadata = json.loads((directory/'run.json').read_text())
        metadata['identity']['manifest_sha256'] = sha256(manifest)
        write_json(directory/'run.json', metadata)
        metrics = score(changed, read_predictions(directory/'predictions.jsonl', changed))
        metrics.update(method=method, protocol=metadata['identity']['protocol'])
        write_json(directory/'metrics.json', metrics)
    evaluate(manifest, sources['qwen35'], sources['videojev'], tmp_path/'changed')
    updated = [json.loads(line) for line in (tmp_path/'changed/predictions.jsonl').read_text().splitlines()]
    assert [r['probabilities'] for r in original] == [r['probabilities'] for r in updated]


@pytest.mark.parametrize('kind', ['method', 'sample', 'protocol', 'subtitles', 'audio',
    'preprocessing', 'options', 'missing', 'incomplete_metrics', 'metrics', 'logits'])
def test_incompatible_sources_rejected(experiment, tmp_path, kind):
    _, _, sources = experiment
    directory = sources['qwen35']
    if kind in ('method', 'sample', 'protocol', 'subtitles', 'audio'):
        metadata = json.loads((directory/'run.json').read_text())
        identity = metadata['identity']
        if kind == 'method':
            identity['method'] = 'wrong'
        elif kind == 'sample':
            identity['manifest_sha256'] = 'wrong'
        else:
            identity['protocol'][dict(protocol='context', subtitles='use_subtitles', audio='use_audio')[kind]] = 1
            metrics = json.loads((directory/'metrics.json').read_text())
            metrics['protocol'] = identity['protocol']
            write_json(directory/'metrics.json', metrics)
        write_json(directory/'run.json', metadata)
    elif kind == 'incomplete_metrics':
        (directory/'metrics.json').unlink()
    elif kind == 'metrics':
        metrics = json.loads((directory/'metrics.json').read_text())
        metrics['n'] = 3
        write_json(directory/'metrics.json', metrics)
    else:
        predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        if kind == 'preprocessing':
            predictions[0]['preprocessing']['frames'] = 29
        elif kind == 'options':
            predictions[0]['options'].reverse()
        elif kind == 'missing':
            predictions.pop()
        else:
            predictions[0]['candidate_logits'] = [0., 1., 2.]
        write_jsonl(predictions, directory/'predictions.jsonl')
    with pytest.raises(ValueError):
        invoke(experiment, tmp_path)
    assert not (tmp_path/'controller/metrics.json').exists()


def test_no_overwrite(experiment, tmp_path):
    invoke(experiment, tmp_path)
    before = (tmp_path/'controller/metrics.json').read_bytes()
    with pytest.raises(FileExistsError):
        invoke(experiment, tmp_path)
    assert (tmp_path/'controller/metrics.json').read_bytes() == before


@pytest.mark.parametrize('beta', [float('nan'), float('inf')])
def test_nonfinite_beta_rejected(experiment, tmp_path, beta):
    with pytest.raises(ValueError):
        invoke(experiment, tmp_path, beta=beta)
