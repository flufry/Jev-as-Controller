import json
from types import SimpleNamespace

import pytest
import torch

from scripts.eval_bench_controller_loglinear import control, evaluate, softmax
from scripts.videojev_candidate_logits import predict_features_with_logits, validate_logits
from test_bench_screen import make_runs


def experts():
    base = dict(id='q', query='Which?', task='choice', options=['A. one', 'B. two', 'C. three'],
        preprocessing=dict(frames=30))
    return dict(base, candidate_logits=[3., 0., -1.], probabilities=softmax([3., 0., -1.])), \
        dict(base, candidate_logits=[-2., 3., 0.], probabilities=softmax([-2., 3., 0.]))


def test_product_of_experts_formula_and_raw_logits_can_flip_large_qwen_margin():
    q, j = experts()
    result = control(q, j)
    weights = [p*s**2 for p, s in zip(q['probabilities'], j['probabilities'])]
    assert result['probabilities'] == pytest.approx([v/sum(weights) for v in weights])
    assert q['candidate_logits'][0] > q['candidate_logits'][1]+2
    assert result['answer_index'] == 1
    assert result['fused_logits'] == [-1., 6., -1.]


def test_raw_and_log_probability_fusion_are_invariant_to_common_constant():
    q, j = experts()
    raw = control(q, j)
    logp = control(q, j, score_source='log_probabilities')
    assert raw['probabilities'] == pytest.approx(logp['probabilities'])
    shifted = dict(j, candidate_logits=[v+100 for v in j['candidate_logits']])
    assert control(q, shifted)['probabilities'] == pytest.approx(raw['probabilities'])
    assert control(q, j, beta=0)['probabilities'] == pytest.approx(q['probabilities'])


@pytest.mark.parametrize('damage', ['missing', 'nan', 'mismatch', 'options', 'zero', 'beta'])
def test_fusion_rejects_invalid_scores(damage):
    q, j = experts()
    mode, beta = 'raw_logits', 2.
    if damage == 'missing':
        j.pop('candidate_logits')
    elif damage == 'nan':
        j['candidate_logits'][0] = float('nan')
    elif damage == 'mismatch':
        j['candidate_logits'] = [0., 0., 0.]
    elif damage == 'options':
        j['options'] = ['other']*3
    elif damage == 'zero':
        j['probabilities'] = [0., 1., 0.]
        mode = 'log_probabilities'
    else:
        beta = -1
    with pytest.raises(ValueError):
        control(q, j, beta, score_source=mode)


def test_hf_formatter_exposes_unpadded_head_logits_only_when_requested():
    from videojev_hf.processing_videojev import format_predictions
    output = SimpleNamespace(logits=torch.tensor([[-2., 3., 999.]]),
        probabilities=torch.tensor([[.006692851, .993307149, 0.]]), score_mask=torch.tensor([[True, True, False]]))
    row = dict(id='q', task='choice', options=['A', 'B'])
    old = format_predictions(output, [row], {}, {})[0]
    new = format_predictions(output, [row], {}, {}, return_logits=True)[0]
    assert 'candidate_logits' not in old
    assert new['candidate_logits'] == [-2., 3.]
    assert new['probabilities'] == old['probabilities']
    validate_logits(new, 2)


def test_old_checkpoint_forward_adapter_preserves_scores_and_hides_labels(monkeypatch):
    from videojev_hf import processing_videojev as processing
    row = dict(id='q', task='choice', query='Which?', options=['A', 'B'])
    monkeypatch.setattr(processing, 'prepare_query_groups', lambda p, r, q, c: iter([({'mock': 1}, q)]))
    class Model:
        config = SimpleNamespace(decoder={})
        def __call__(self, **kwargs):
            assert set(kwargs) == {'mock', 'visual_features'}
            return SimpleNamespace(logits=torch.tensor([[-2., 3.]]),
                probabilities=torch.tensor([[-2., 3.]]).softmax(-1), score_mask=torch.ones((1, 2), dtype=torch.bool))
    result = predict_features_with_logits(Model(), None, torch.zeros(1), {}, [row], 1024)
    assert result[0]['candidate_logits'] == [-2., 3.]
    assert result[0]['answer_index'] == 1


def test_new_evaluator_entrypoint_uses_logits_mode(monkeypatch):
    from scripts import eval_bench_videojev_logits as entry
    calls = []
    monkeypatch.setattr(entry, 'load_legacy', lambda name: SimpleNamespace(
        main=lambda argv, **kwargs: calls.append((name, argv, kwargs))))
    entry.main(['--help'])
    assert calls == [('eval_longvideobench_videojev.py', ['--help'], {'save_logits': True})]


def test_new_video_group_keeps_raw_scores_and_never_passes_gold(monkeypatch):
    from scripts.eval_longvideobench_videojev import predict_group
    import timejev.rgb_media
    import scripts.videojev_candidate_logits as adapter
    monkeypatch.setattr(timejev.rgb_media, 'preprocess', lambda *a: [torch.zeros((8, 3))])
    observed = []
    def predict(model, processor, features, record, questions, context):
        observed.extend(questions)
        assert features.shape == (2, 3)
        return [dict(id='q', candidate_logits=[-2., 3.], probabilities=softmax([-2., 3.]))]
    monkeypatch.setattr(adapter, 'predict_features_with_logits', predict)
    row = dict(id='q', task='choice', query='Which?', options=['A', 'B'], answer_index=1)
    model = SimpleNamespace(encode_video=lambda *a: torch.zeros((2, 3)))
    result = predict_group(model, None, [row], dict(grid_thw=[2, 2, 2]), 1024, save_logits=True)
    assert result[0]['candidate_logits'] == [-2., 3.]
    assert observed == [{k: row[k] for k in ('id', 'task', 'query', 'options')}]


@pytest.mark.parametrize('api', ['predict_video', 'predict_features'])
def test_public_model_api_returns_logits_without_changing_existing_answers(monkeypatch, api):
    from videojev_hf import modeling_videojev as module
    rows = [dict(id='q', task='choice', options=['A', 'B'])]
    monkeypatch.setattr(module, 'prepare_query_groups', lambda *a: iter([({}, rows)]))
    monkeypatch.setattr(module, 'prepare_video', lambda *a, **kw: (None, dict(grid_thw=[1, 2, 2])))
    class Model:
        predict_video = module.VideoJevForVideoUnderstanding.predict_video
        predict_features = module.VideoJevForVideoUnderstanding.predict_features
        config = SimpleNamespace(decoder={'threshold': .25})
        def encode_video(self, *a):
            return None
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[-2., 3.]]),
                probabilities=torch.tensor([[-2., 3.]]).softmax(-1), score_mask=torch.ones((1, 2), dtype=torch.bool))
    model = Model()
    args = (None, 'video.mp4', rows) if api == 'predict_video' else (None, None, {}, rows)
    old = getattr(model, api)(*args)
    new = getattr(model, api)(*args, return_logits=True)
    assert new[0].pop('candidate_logits') == [-2., 3.]
    assert new == old


def test_fusion_existing_runs_without_media_and_strict_raw_requirement(tmp_path):
    manifest, rows = make_runs(tmp_path)
    # Post-hoc scoring must also work after the videos have been removed/copied elsewhere.
    (tmp_path/'v.mp4').unlink()
    q, j = tmp_path/'qwen36', tmp_path/'videojev'
    with pytest.raises(ValueError, match='logits'):
        evaluate(manifest, q, j, tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    metrics = evaluate(manifest, q, j, tmp_path/'logp', score_source='log_probabilities')
    assert metrics['method'] == 'jev_as_controller_loglinear'
    path = j/'predictions.jsonl'
    values = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    for value in values:
        value['candidate_logits'] = [0., 2., 1.8]
    path.write_text(''.join(json.dumps(v)+'\n' for v in values), encoding='utf-8')
    raw = evaluate(manifest, q, j, tmp_path/'raw')
    assert raw['accuracy'] == metrics['accuracy']
    saved = json.loads((tmp_path/'raw/run.json').read_text(encoding='utf-8'))
    assert saved['identity']['videojev_score_source'] == 'raw_logits'
    assert saved['identity']['sources']['qwen']['predictions_sha256']
    with pytest.raises(FileExistsError):
        evaluate(manifest, q, j, tmp_path/'raw')
    # Inputs must remain immutable throughout scoring.
    assert all(v['candidate_logits'] == [0., 2., 1.8] for v in values)
