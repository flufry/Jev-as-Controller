"""Log-linear 融合与本地 VideoJev 原始候选分数的 CPU 回归。"""
import copy
import importlib
import json
import math
import sys
from types import SimpleNamespace

import pytest

from scripts.eval_longvideobench_controller import control, controller_fusion, softmax


def sources(qwen_logits=(3., 0.), videojev_logits=(-2., 2.), saved=True):
    common = dict(id='q', query='Question?', options=['A. one', 'B. two'],
                  preprocessing=dict(frames=2))
    qwen = dict(common, candidate_logits=list(qwen_logits), probabilities=softmax(qwen_logits))
    jev = dict(common, probabilities=softmax(videojev_logits))
    if saved:
        jev['logits'] = list(videojev_logits)
    return qwen, jev


def test_default_matches_normalized_product_and_differs_from_legacy():
    qwen, jev = sources()
    result = control(qwen, jev)
    weights = [p*q**2 for p, q in zip(qwen['probabilities'], jev['probabilities'])]
    assert result['probabilities'] == pytest.approx([w/sum(weights) for w in weights])
    assert result['answer_index'] == 1
    assert result['beta'] == 2. and result['fusion'] == 'loglinear'
    assert result['videojev_logits'] == jev['logits']
    assert result['videojev_logits_source'] == 'saved_logits'
    legacy = control(qwen, jev, fusion='probability')
    weights = [p*math.exp(2*q) for p, q in zip(qwen['probabilities'], jev['probabilities'])]
    assert legacy['probabilities'] == pytest.approx([w/sum(weights) for w in weights])
    assert legacy['answer_index'] == 0 and legacy['fusion'] == 'probability'


def test_positive_probabilities_recover_equivalent_logits_without_clipping():
    qwen, jev = sources(videojev_logits=(100., 104.))
    original = control(qwen, jev)
    del jev['logits']
    recovered = control(qwen, jev)
    assert recovered['probabilities'] == pytest.approx(original['probabilities'])
    assert recovered['videojev_logits'] == pytest.approx([math.log(p) for p in jev['probabilities']])
    assert recovered['videojev_logits_source'] == 'log_probabilities'
    qwen, jev = sources(qwen_logits=(0., 1400.), videojev_logits=(0., -700.), saved=False)
    assert control(qwen, jev)['probabilities'] == pytest.approx([.5, .5])


def test_raw_logits_preserve_underflowed_score_gap_and_beta_zero():
    qwen, jev = sources(qwen_logits=(0., 1900.), videojev_logits=(0., -1000.))
    assert jev['probabilities'] == [1., 0.]
    result = control(qwen, jev)
    assert result['probabilities'][1] == pytest.approx(math.exp(-100), rel=1e-12, abs=0.)
    assert result['answer_index'] == 0
    assert result['videojev_logits'] == [0., -1000.]
    assert control(qwen, jev, beta=0.)['probabilities'] == qwen['probabilities']
    assert control(qwen, jev, fusion='probability')['answer_index'] == 1


@pytest.mark.parametrize('beta', [0., 2.])
def test_missing_logits_with_zero_probability_requires_new_source(beta):
    qwen, jev = sources(videojev_logits=(0., -1000.), saved=False)
    with pytest.raises(ValueError, match='概率含0'):
        control(qwen, jev, beta=beta)
    assert control(qwen, jev, beta=beta, fusion='probability')['fusion'] == 'probability'


def test_common_logit_shifts_leave_fusion_unchanged_and_inputs_intact():
    qwen, jev = sources(qwen_logits=(1., -.5), videojev_logits=(-2., 3.))
    before = copy.deepcopy((qwen, jev))
    expected = control(qwen, jev)['probabilities']
    qwen_shifted = dict(qwen, candidate_logits=[v+4096 for v in qwen['candidate_logits']])
    jev_shifted = dict(jev, logits=[v-2048 for v in jev['logits']])
    assert control(qwen_shifted, jev_shifted)['probabilities'] == pytest.approx(expected)
    assert (qwen, jev) == before


@pytest.mark.parametrize('side,field,value', [
    ('qwen', 'candidate_logits', [0.]),
    ('qwen', 'candidate_logits', [True, 0.]),
    ('qwen', 'candidate_logits', [float('nan'), 0.]),
    ('qwen', 'candidate_logits', [float('inf'), 0.]),
    ('qwen', 'candidate_logits', [0., 0.]),
    ('qwen', 'probabilities', [1.]),
    ('qwen', 'probabilities', [.5, .5]),
    ('qwen', 'probabilities', [float('nan'), 0.]),
    ('jev', 'logits', None),
    ('jev', 'logits', [0.]),
    ('jev', 'logits', [False, 0.]),
    ('jev', 'logits', [float('nan'), 0.]),
    ('jev', 'logits', [float('-inf'), 0.]),
    ('jev', 'logits', [0., 0.]),
    ('jev', 'probabilities', [1.]),
    ('jev', 'probabilities', [.2, .2]),
    ('jev', 'probabilities', [float('nan'), 0.]),
    ('jev', 'probabilities', [-.1, 1.1]),
    ('jev', 'probabilities', [.5, .5]),
])
def test_corrupt_raw_scores_and_probabilities_rejected(side, field, value):
    qwen, jev = sources()
    (qwen if side == 'qwen' else jev)[field] = value
    with pytest.raises(ValueError):
        control(qwen, jev)


@pytest.mark.parametrize('field,value', [
    ('id', 'another-question'), ('query', 'Another question?'), ('options', ['B. two', 'A. one']),
])
def test_different_question_or_option_order_rejected(field, value):
    qwen, jev = sources()
    jev[field] = value
    with pytest.raises(ValueError, match='问题、选项或顺序'):
        control(qwen, jev)


@pytest.mark.parametrize('beta', [float('nan'), float('inf'), True])
def test_invalid_beta_rejected_directly(beta):
    with pytest.raises(ValueError, match='beta'):
        control(*sources(), beta=beta)


def test_history_without_fusion_keeps_probability_formula():
    assert controller_fusion({}) == 'probability'
    assert controller_fusion(dict(fusion='loglinear')) == 'loglinear'
    assert controller_fusion(dict(fusion='probability')) == 'probability'


@pytest.mark.parametrize('fusion', ['unknown', None, [], True])
def test_unknown_fusion_rejected(fusion):
    with pytest.raises(ValueError, match='未知'):
        controller_fusion(dict(fusion=fusion))
    with pytest.raises(ValueError, match='未知'):
        control(*sources(), fusion=fusion)


@pytest.mark.parametrize('name', ['eval_longvideobench_controller', 'eval_longvideobench_proportional_controller'])
def test_cli_defaults_forward_loglinear_and_beta_two(monkeypatch, capsys, name):
    module = importlib.import_module('scripts.'+name)
    received = {}

    def evaluate(**kwargs):
        received.update(kwargs)
        return dict(mock=True)

    monkeypatch.setattr(module, 'evaluate', evaluate)
    monkeypatch.setattr(sys, 'argv', [name, '--manifest', 'sample.jsonl', '--qwen', 'qwen',
                                    '--videojev', 'videojev', '--out', 'out'])
    module.main()
    assert received['fusion'] == 'loglinear' and received['beta'] == 2.
    assert json.loads(capsys.readouterr().out) == dict(mock=True)


@pytest.fixture
def packed_model(monkeypatch):
    import torch
    import timejev.rgb_media
    monkeypatch.setattr(timejev.rgb_media, 'preprocess', lambda *_: [torch.zeros(8, 3)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fail = False
            self.reversed = False
            self.outputs = []

        def encode_video(self, *_):
            return torch.zeros(2, 3)

        def forward(self, batch):
            return self.outputs[batch]

        def predict_features(self, processor, features, record, questions, **kwargs):
            self.questions = questions
            assert features.shape == (2, 3)
            predictions = []
            for batch, output in enumerate(self.outputs):
                self(batch)
                if self.fail:
                    raise RuntimeError('mock packed inference failed')
                for values, mask in zip(output.logits, output.score_mask):
                    count = int(mask.sum())
                    question = questions[len(predictions)]
                    predictions.append(dict(id=question['id'], probabilities=values[:count].float().softmax(-1).tolist()))
            return predictions[::-1] if self.reversed else predictions

    model = Model()
    model.outputs = [
        SimpleNamespace(logits=torch.tensor([[1., -2., 500.], [0., 2., 3.]], dtype=torch.float16),
                        score_mask=torch.tensor([[True, True, False], [True, True, True]])),
        SimpleNamespace(logits=torch.tensor([[-1000., 0., 999.]], dtype=torch.float16),
                        score_mask=torch.tensor([[True, True, False]])),
    ]
    rows = [dict(id=f'q{i}', task='choice', query='Question?', options=['A. one', 'B. two']+(['C. three'] if i == 1 else []),
                 answer_index=i%2) for i in range(3)]
    return model, rows


def test_videojev_hook_saves_raw_logits_across_packed_forwards_without_padding(packed_model):
    from scripts.eval_longvideobench_videojev import predict_group
    model, rows = packed_model
    external_calls = []
    external = model.register_forward_hook(lambda *args: external_calls.append(True))
    before = set(model._forward_hooks)
    predictions = predict_group(model, object(), rows, dict(grid_thw=[2, 2, 2]), 262144)
    assert [p['id'] for p in predictions] == [row['id'] for row in rows]
    assert [p['logits'] for p in predictions] == [[1., -2.], [0., 2., 3.], [-1000., 0.]]
    assert predictions[-1]['probabilities'] == [0., 1.]
    assert len(external_calls) == 2 and set(model._forward_hooks) == before
    assert all(set(q) == {'id', 'task', 'query', 'options'} for q in model.questions)
    assert json.loads(json.dumps(predictions, allow_nan=False))[0]['logits'] == [1., -2.]
    external.remove()


def test_videojev_hook_is_removed_when_native_predict_features_raises(packed_model):
    from scripts.eval_longvideobench_videojev import predict_group
    model, rows = packed_model
    model.fail = True
    with pytest.raises(RuntimeError, match='mock packed inference failed'):
        predict_group(model, object(), rows, dict(grid_thw=[2, 2, 2]), 262144)
    assert not model._forward_hooks


def test_videojev_hook_rejects_reordered_native_outputs_and_is_removed(packed_model):
    from scripts.eval_longvideobench_videojev import predict_group
    model, rows = packed_model
    model.reversed = True
    with pytest.raises(ValueError, match='未完整对齐'):
        predict_group(model, object(), rows, dict(grid_thw=[2, 2, 2]), 262144)
    assert not model._forward_hooks
