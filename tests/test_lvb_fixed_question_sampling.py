import copy
import json
import math
import statistics

import pytest

from scripts.sample_longvideobench_fixed_questions import (
    METHOD_ROLES, run, sample_index, simulate, verify_sources,
)
from timejev.lvb_experiment import sha256


class Uniform:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


@pytest.fixture
def inputs():
    rows = [dict(id=f'q{i}', task='choice', video_key=f'v{i}', query=f'question {i}',
        options=['A. one', 'B. two', 'C. three'], answer_index=i,
        duration_group=group) for i, group in enumerate((15, 3600))]
    predictions = {}
    for method in METHOD_ROLES:
        predictions[method] = {}
        for row, probabilities in zip(rows, ([.25, .75, 0.], [.1, .4, .5])):
            index = max(range(3), key=probabilities.__getitem__)
            predictions[method][row['id']] = dict(id=row['id'], query=row['query'],
                options=row['options'], probabilities=list(probabilities),
                answer_index=index, answer=row['options'][index])
    return rows, predictions


@pytest.mark.parametrize('uniform, expected', [(0., 1), (.249, 1), (.25, 2), (.999999, 2)])
def test_categorical_boundaries_and_zero_probability(uniform, expected):
    rng = Uniform(uniform)
    assert sample_index([0., .25, .75], rng) == expected
    assert rng.calls == 1


@pytest.mark.parametrize('probabilities', [[], [-.1, 1.1], [math.nan, 1.],
    [math.inf, 0.], [.2, .3], [False, 1.], [0., 0.]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        sample_index(probabilities, Uniform(.5))


def test_replay_saved_samples_and_statistics(inputs):
    rows, predictions = inputs
    before = copy.deepcopy((rows, predictions))
    report, trials, samples = simulate(rows, predictions)
    assert simulate(rows, predictions) == (report, trials, samples)
    assert (rows, predictions) == before
    assert len(trials) == 9
    assert len(samples) == 18
    assert {(row['method'], row['inference_seed'], row['id']) for row in samples} == {
        (method, seed, row['id']) for method in METHOD_ROLES
        for seed in (42, 43, 44) for row in rows}
    gold = {row['id']: row for row in rows}
    for method in METHOD_ROLES:
        values = []
        for seed in (42, 43, 44):
            current = [row for row in samples if row['method'] == method and row['inference_seed'] == seed]
            trial = next(row for row in trials if row['method'] == method and row['inference_seed'] == seed)
            correct = 0
            for sampled in current:
                target = gold[sampled['id']]
                index = sampled['answer_index']
                assert sampled['probabilities'][index] > 0
                assert sampled['answer'] == target['options'][index]
                assert sampled['ground_truth_index'] == target['answer_index']
                assert sampled['correct'] == (index == target['answer_index'])
                correct += sampled['correct']
            assert trial['n'] == 2
            assert trial['correct'] == correct
            assert trial['accuracy'] == correct / 2
            assert sum(group['correct'] for group in trial['by_duration_group'].values()) == correct
            values.append(correct / 2)
        summary = report['methods'][method]
        assert summary['accuracies'] == values
        assert summary['mean_accuracy'] == statistics.mean(values)
        assert summary['sample_variance_fraction_squared'] == statistics.variance(values)
        assert summary['sample_variance_percentage_points_squared'] == pytest.approx(statistics.variance(values) * 10000)
        assert summary['theoretical_mean_accuracy'] == pytest.approx(.325)
        assert summary['theoretical_variance_percentage_points_squared'] == pytest.approx(1068.75)
        assert summary['greedy_accuracy'] == 0
    # 所有输出都能保存为严格 JSON，不能包含 NaN 或非标准对象。
    json.dumps([report, trials, samples], allow_nan=False)


def test_labels_only_affect_scoring(inputs):
    rows, predictions = inputs
    _, _, original = simulate(rows, predictions)
    changed = copy.deepcopy(rows)
    for row in changed:
        row['answer_index'] = (row['answer_index'] + 1) % 3
    _, _, updated = simulate(changed, predictions)
    assert [row['answer_index'] for row in original] == [row['answer_index'] for row in updated]
    assert [row['probabilities'] for row in original] == [row['probabilities'] for row in updated]


def test_degenerate_distribution_has_zero_sampling_variance(inputs):
    rows, predictions = inputs
    for by_id in predictions.values():
        for row in rows:
            prediction = by_id[row['id']]
            index = row['answer_index']
            prediction.update(probabilities=[float(i == index) for i in range(3)],
                answer_index=index, answer=row['options'][index])
    report, _, samples = simulate(rows, predictions)
    assert all(row['correct'] for row in samples)
    for summary in report['methods'].values():
        assert summary['accuracies'] == [1., 1., 1.]
        assert summary['sample_variance_fraction_squared'] == 0
        assert summary['theoretical_variance_percentage_points_squared'] == 0


@pytest.mark.parametrize('seeds', [(42,), (42, 42, 44), (42, True, 44), (42, '43', 44)])
def test_invalid_inference_seeds_rejected(inputs, seeds):
    with pytest.raises(ValueError):
        simulate(*inputs, seeds=seeds)


@pytest.mark.parametrize('kind', ['missing_question', 'unknown_question', 'wrong_options',
    'invalid_gold', 'bad_probabilities', 'missing_method', 'duplicate_question'])
def test_misaligned_or_invalid_inputs_rejected(inputs, kind):
    rows, predictions = inputs
    by_id = predictions['qwen36']
    if kind == 'missing_question':
        by_id.pop('q0')
    elif kind == 'unknown_question':
        by_id['unknown'] = copy.deepcopy(by_id['q0'])
    elif kind == 'wrong_options':
        by_id['q0']['options'] = list(reversed(by_id['q0']['options']))
    elif kind == 'invalid_gold':
        rows[0]['answer_index'] = 3
    elif kind == 'bad_probabilities':
        by_id['q0']['probabilities'] = [.5, .5]
    elif kind == 'missing_method':
        predictions.pop('videojev')
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        simulate(rows, predictions)


def test_existing_output_rejected_before_reading_sources(tmp_path, monkeypatch):
    out = tmp_path/'experiment'
    out.mkdir()
    marker = out/'original.txt'
    marker.write_text('original result')

    def unexpected_read(*args):
        pytest.fail('已有输出时不应读取源实验')

    monkeypatch.setattr('scripts.sample_longvideobench_fixed_questions.read_single', unexpected_read)
    with pytest.raises(FileExistsError):
        run(tmp_path/'unavailable_source', out)
    assert marker.read_text() == 'original result'
    assert list(out.iterdir()) == [marker]


def test_changed_source_hash_rejected(tmp_path):
    source = tmp_path/'probabilities.jsonl'
    source.write_text('original source')
    hashes = {str(source): sha256(source)}
    verify_sources(hashes)
    source.write_text('changed source')
    with pytest.raises(ValueError, match='源文件发生变化'):
        verify_sources(hashes)
