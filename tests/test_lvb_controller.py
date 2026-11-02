import copy
import json
import math

import pytest

from scripts.eval_longvideobench_controller import FUSION_FORMULAS, evaluate, read_source, softmax
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
        for index, row in enumerate(rows):
            values = [1., -.5+index*.05, 0.] if method == 'qwen35' else [-2., 3.+index*.1, 0.]
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
        # 独立按 pQ*pJ**2 归一化，验证默认 log-linear 公式及固定 beta。
        weights = [p*f**2 for p, f in zip(raw['probabilities'], fs['probabilities'])]
        expected = [value/sum(weights) for value in weights]
        assert result['probabilities'] == pytest.approx(expected)
        assert result['answer_index'] == 1
        assert result['qwen_probabilities'] == pytest.approx(raw['probabilities'])
        assert result['videojev_probabilities'] == fs['probabilities']
        assert result['videojev_logits'] == pytest.approx([math.log(p) for p in fs['probabilities']])
        assert result['videojev_logits_source'] == 'log_probabilities'
        assert result['fusion'] == 'loglinear'
        assert result['beta'] == 2.
    metadata = json.loads((tmp_path/'controller/run.json').read_text())
    assert metadata['identity']['manifest_sha256'] == sha256(manifest)
    assert metadata['identity']['sources']['qwen']['identity']['checkpoint']['files']
    assert metadata['identity']['fusion'] == 'loglinear'
    assert metadata['formula'] == FUSION_FORMULAS['loglinear']
    assert metadata['split'] == 'validation_subset'
    metrics = json.loads((tmp_path/'controller/metrics.json').read_text())
    assert metrics['n'] == 4 and metrics['fusion'] == 'loglinear'


def test_beta_zero_matches_qwen(experiment, tmp_path):
    invoke(experiment, tmp_path, beta=0.)
    _, rows, sources = experiment
    qwen = read_predictions(sources['qwen35']/'predictions.jsonl', rows)
    result = read_predictions(tmp_path/'controller/predictions.jsonl', rows)
    for identity in qwen:
        assert result[identity]['probabilities'] == pytest.approx(qwen[identity]['probabilities'])


def test_probability_mode_keeps_original_formula_and_metadata(experiment, tmp_path):
    invoke(experiment, tmp_path, fusion='probability')
    _, rows, sources = experiment
    qwen = read_predictions(sources['qwen35']/'predictions.jsonl', rows)
    jev = read_predictions(sources['videojev']/'predictions.jsonl', rows)
    result = read_predictions(tmp_path/'controller/predictions.jsonl', rows)
    for row in rows:
        weights = [p*math.exp(2*f) for p, f in zip(qwen[row['id']]['probabilities'], jev[row['id']]['probabilities'])]
        assert result[row['id']]['probabilities'] == pytest.approx([w/sum(weights) for w in weights])
        assert result[row['id']]['fusion'] == 'probability'
    run = json.loads((tmp_path/'controller/run.json').read_text())
    metrics = json.loads((tmp_path/'controller/metrics.json').read_text())
    assert run['identity']['fusion'] == metrics['fusion'] == 'probability'
    assert run['formula'] == FUSION_FORMULAS['probability']


def test_saved_raw_logits_are_preserved_in_evaluation(experiment, tmp_path):
    _, rows, sources = experiment
    path = sources['videojev']/'predictions.jsonl'
    predictions = list(read_predictions(path, rows).values())
    for prediction in predictions:
        prediction['logits'] = [math.log(p)+100. for p in prediction['probabilities']]
    write_jsonl(predictions, path)
    invoke(experiment, tmp_path)
    result = read_predictions(tmp_path/'controller/predictions.jsonl', rows)
    for prediction in predictions:
        actual = result[prediction['id']]
        assert actual['videojev_logits'] == prediction['logits']
        assert actual['videojev_logits_source'] == 'saved_logits'


@pytest.mark.parametrize('kind', ['unknown_fusion', 'wrong_formula', 'metric_fusion'])
def test_controller_source_rejects_formula_or_fusion_mislabel(experiment, tmp_path, kind):
    invoke(experiment, tmp_path)
    manifest, rows, _ = experiment
    directory = tmp_path/'controller'
    path = directory/('metrics.json' if kind == 'metric_fusion' else 'run.json')
    value = json.loads(path.read_text())
    if kind == 'unknown_fusion':
        value['identity']['fusion'] = 'unknown'
    elif kind == 'wrong_formula':
        value['formula'] = FUSION_FORMULAS['probability']
    else:
        value['fusion'] = 'probability'
    write_json(path, value)
    with pytest.raises(ValueError):
        read_source(directory, rows, sha256(manifest), 'jev_as_controller')


def test_historical_probability_source_without_fusion_is_readable(experiment, tmp_path):
    invoke(experiment, tmp_path, fusion='probability')
    manifest, rows, _ = experiment
    directory = tmp_path/'controller'
    for name in ('run.json', 'metrics.json'):
        path = directory/name
        value = json.loads(path.read_text())
        (value['identity'] if name == 'run.json' else value).pop('fusion')
        write_json(path, value)
    predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    for prediction in predictions:
        prediction.pop('fusion')
    write_jsonl(predictions, directory/'predictions.jsonl')
    actual, source, metrics = read_source(directory, rows, sha256(manifest), 'jev_as_controller')
    assert len(actual) == metrics['n'] == 4
    assert 'fusion' not in source['identity']


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
