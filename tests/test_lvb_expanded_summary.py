"""扩展组汇总的纯 CPU 检查；不联网、不加载权重。"""
import copy
import hashlib
import json
from pathlib import Path
import statistics

import pytest

from scripts import summarize_longvideobench_expanded_groups as expanded
from timejev.lvb_experiment import sha256, write_json


def tiny_rows(prefix='q'):
    return [dict(id=f'{prefix}{index}', answer_index=0, video_key=f'v{index}',
                 query=f'question {index}', options=['A', 'B', 'C', 'D']) for index in range(4)]


def source_report(tmp_path, rows, seed):
    sample = dict(path=str(tmp_path/'sample.jsonl'), sha256='sample', meta_sha256='meta', seed=seed)
    sources, answers = {}, dict(qwen_baseline=[0, 1, 0, 2], videojev_baseline=[1, 0, 0, 3], controller=[0, 0, 0, 2])
    for role, values in answers.items():
        directory = tmp_path/role
        directory.mkdir(parents=True)
        predictions = [dict(id=row['id'], query=row['query'], options=row['options'],
            answer_index=answer, probabilities=[float(i == answer) for i in range(4)])
            for row, answer in zip(rows, values)]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        sources[role] = dict(directory=str(directory), run_sha256='run', metrics_sha256='metrics',
                             predictions_sha256=sha256(directory/'predictions.jsonl'))
    report = dict(seed=seed, provenance=dict(sample=sample, runs=sources), protocols=dict(baseline=dict(fps=4)),
        methods={key: dict(accuracy=value) for key, value in dict(qwen36=.5, videojev=.5, jev_as_controller=.75).items()},
        controller2_minus_qwen2=dict(accuracy_difference=.25))
    checked = dict(report=report, provenance=dict(summary_path=str(tmp_path/'summary.json'), summary_sha256='summary'))
    return dict(rows=rows, sample=sample, directory=tmp_path,
                question_ids={r['id'] for r in rows}, video_keys={r['video_key'] for r in rows}), checked


def make_legacy(tmp_path, entries):
    directory = tmp_path/'legacy'
    directory.mkdir()
    source = directory/'original_source.json'
    source.write_text('source')
    original_manifest = directory/'original_artifacts.json'
    original_manifest.write_text('{}')
    trials, seeds = [], {}
    for seed in expanded.SEEDS:
        rows = entries[seed]['rows']
        qwen, videojev = [0, 1, 0, 2], [1, 0, 0, 3]
        current = [expanded.random_trial(rows, qwen, videojev, 1, seed, repeat) for repeat in range(1, 51)]
        trials += current
        seeds[str(seed)] = expanded.random_statistics(rows, qwen, videojev, current, dict(fps=4))
    (directory/'source.py').write_text('# original implementation\n')
    report = dict(schema='lvb_qwen36_videojev_random_choice_v1', repeats_per_sample=50, total_trials=150,
        seeds=seeds, script_sha256=sha256(directory/'source.py'),
        verified_sources={str(source): dict(sha256=sha256(source))},
        source_artifact_manifest=dict(path=str(original_manifest), sha256=sha256(original_manifest)))
    write_json(directory/'results.json', report)
    write_json(directory/'validation.json', dict(passed=True))
    (directory/'exitcode').write_text('0\n')
    (directory/'run.log').write_text('completed\n')
    (directory/'trials.jsonl').write_text(''.join(json.dumps(trial)+'\n' for trial in trials))
    refresh_legacy_artifact(directory)
    return directory


def refresh_legacy_artifact(directory):
    files = {name: dict(sha256=sha256(directory/name), bytes=(directory/name).stat().st_size)
        for name in ('exitcode', 'results.json', 'run.log', 'source.py', 'trials.jsonl', 'validation.json')}
    write_json(directory/'artifact_manifest.json', dict(schema='lvb_random_choice_artifacts_v1', files=files))


@pytest.fixture
def group_fixture(tmp_path, monkeypatch):
    samples, checked = {}, {}
    for seed in expanded.SEEDS:
        entry, report = source_report(tmp_path/f'seed{seed}', tiny_rows(f's{seed}q'), seed)
        samples[1, seed] = entry
        checked[entry['directory']] = report
    monkeypatch.setattr(expanded, 'read_single', lambda root: checked[root])
    return samples


def test_random_is_label_free_and_differs_from_oracle(group_fixture):
    group = expanded.collect_group(2, {(2, seed): entry for (_, seed), entry in group_fixture.items()})
    for entry in group['entries']:
        assert entry['accuracies']['oracle_or'] == .75
        assert entry['oracle'] == dict(correct=3, both_correct=1, exactly_one_correct=2, both_wrong=1)
        assert entry['accuracies']['random_choice'] < .75
        assert entry['random_choice']['theoretical_mean'] == .5
        assert entry['random_choice']['theoretical_variance'] == 2/(4*16)
    trials = group['trials']
    assert len(trials) == 150
    assert trials[0]['rng_seed'] == (1 << 48)+(42 << 32)+1
    rows = tiny_rows()
    original = expanded.random_trial(rows, [0, 1, 0, 2], [1, 0, 0, 3], 2, 42, 1)
    for row in rows:
        row['answer_index'] = 3
    changed = expanded.random_trial(rows, [0, 1, 0, 2], [1, 0, 0, 3], 2, 42, 1)
    assert changed['answer_indices'] == original['answer_indices'] and changed['choices'] == original['choices']
    assert original['choices'][2] == 'shared'


def test_sample_variance_uses_three_seed_means_not_50_inner_trials():
    values = [.7438, .7794, .7182]
    entries = [dict(seed=seed, accuracies={key: value for key in expanded.METHODS})
               for seed, value in zip(expanded.SEEDS, values)]
    result = expanded.method_statistics(entries)['random_choice']
    assert result['mean'] == pytest.approx(.7471333333333333)
    assert result['sample_variance']*10000 == pytest.approx(9.44693333333333)
    assert result['population_variance'] == pytest.approx(result['sample_variance']*2/3)
    assert result['sample_std'] == pytest.approx(statistics.stdev(values))
    assert '9.4469' in '\n'.join(expanded.table({'random_choice': result}))
    with pytest.raises(ValueError):
        expanded.method_statistics(entries[::-1])


def test_group1_preserves_all_150_original_trials_and_provenance(group_fixture, tmp_path):
    directory = make_legacy(tmp_path, {seed: group_fixture[1, seed] for seed in expanded.SEEDS})
    original_hash = sha256(directory/'trials.jsonl')
    legacy = expanded.read_legacy_random(directory)
    group = expanded.collect_group(1, group_fixture, legacy=legacy)
    assert group['sources'][str(directory/'trials.jsonl')] == original_hash
    assert sha256(directory/'trials.jsonl') == original_hash
    for trial in group['trials']:
        assert {key: value for key, value in trial.items() if key != 'group_index'} == legacy['trials'][trial['sample_seed'], trial['repeat']]
    assert group['entries'][0]['random_choice'] == legacy['report']['seeds']['42']


@pytest.mark.parametrize('kind', ['artifact_file', 'source_file', 'missing_artifact', 'bad_replay', 'bad_ids'])
def test_group1_rejects_sha_or_replay_tampering(group_fixture, tmp_path, kind):
    directory = make_legacy(tmp_path, {seed: group_fixture[1, seed] for seed in expanded.SEEDS})
    if kind == 'artifact_file':
        (directory/'run.log').write_text('different')
    elif kind == 'source_file':
        (directory/'original_source.json').write_text('different')
    elif kind == 'missing_artifact':
        data = json.loads((directory/'artifact_manifest.json').read_text())
        del data['files']['validation.json']
        write_json(directory/'artifact_manifest.json', data)
    elif kind == 'bad_replay':
        lines = [json.loads(line) for line in (directory/'trials.jsonl').read_text().splitlines()]
        lines[0]['answer_indices'][0] = 3
        (directory/'trials.jsonl').write_text(''.join(json.dumps(line)+'\n' for line in lines))
        refresh_legacy_artifact(directory)
    else:
        data = json.loads((directory/'results.json').read_text())
        data['seeds']['42']['question_ids'][0] = 'forged_id'
        write_json(directory/'results.json', data)
        refresh_legacy_artifact(directory)
    with pytest.raises(ValueError):
        expanded.collect_group(1, group_fixture, legacy=expanded.read_legacy_random(directory))


def test_check_only_needs_no_predictions_and_checks_same_seed_disjointness(monkeypatch, tmp_path):
    def fake_sample(path, seed):
        group = int(path.parts[-3].replace('group', '')) if 'expanded' in str(path) else 1
        rows = [dict(id=f'g{group}s{seed}q{i}', video_key=f'v{i}') for i in range(100)]
        return rows, dict(path=str(path), sha256=f'g{group}s{seed}', seed=seed,
            source_manifest=dict(sha256='original'), source_reference=dict(sha256='official'))
    monkeypatch.setattr(expanded, 'read_sample', fake_sample)
    monkeypatch.setattr(expanded, 'read_single', lambda root: pytest.fail('不应加载模型结果'))
    result = expanded.summarize(tmp_path/'expanded', all_groups=True, check_only=True)
    assert result['passed'] and result['checked_cells'] == 9
    assert [entry['seed'] for entry in result['manifests']] == list(expanded.SEEDS)*3
    assert not (tmp_path/'expanded').exists()
    original = expanded.read_sample
    def overlapping(path, seed):
        rows, sample = original(path, seed)
        if 'group3' in str(path):
            rows[0]['id'] = f'g2s{seed}q0'
        return rows, sample
    monkeypatch.setattr(expanded, 'read_sample', overlapping)
    with pytest.raises(ValueError, match='问题重复'):
        expanded.summarize(tmp_path/'expanded', group=3, check_only=True)


def test_output_refuses_overwrite_and_all_reuse_requires_exact_recalculation(group_fixture, tmp_path):
    group = expanded.collect_group(2, {(2, seed): entry for (_, seed), entry in group_fixture.items()})
    payload = expanded.group_report(group, {'fingerprint': 'same'})
    out = tmp_path/'summary'
    written = expanded.save_group(out, payload)
    assert (out/'results.json').is_file() and (out/'random_trials.jsonl').is_file()
    assert hashlib.sha256((out/'random_trials.jsonl').read_bytes()).hexdigest() == written['random_trials_sha256']
    with pytest.raises(FileExistsError):
        expanded.save_group(out, payload)
    reused = expanded.save_group(out, payload, reuse=True)
    assert reused == written
    (out/'random_trials.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='重算不符'):
        expanded.save_group(out, payload, reuse=True)


def test_overlap_counts_question_and_video_separately():
    entries = [dict(group_index=1, seed=42, question_ids=['q1', 'q2'], video_keys=['v1']),
               dict(group_index=2, seed=42, question_ids=['q3', 'q4'], video_keys=['v1', 'v2'])]
    stats = expanded.overlap_statistics(entries)
    assert stats['unique_questions'] == 4 and stats['unique_videos'] == 2
    assert stats['pairs'][0]['shared_questions'] == 0 and stats['pairs'][0]['shared_videos'] == 1


def test_compatibility_rejects_changed_model_protocol_or_media(monkeypatch):
    monkeypatch.setattr(expanded, 'fingerprint', lambda identity, method: identity['fingerprint'])
    provenance = dict(sources=dict(sample={key: dict(sha256='annotation') for key in ('source_manifest', 'source_reference')},
        runs=dict(qwen_controller=dict(identity=dict(method='qwen36', fingerprint='same',
            media={'v': dict(sha256='content', bytes=1)})))))
    groups = [dict(entries=[dict(provenance=copy.deepcopy(provenance))]),
              dict(entries=[dict(provenance=copy.deepcopy(provenance))])]
    assert expanded.verify_compatibility(groups) == {'qwen_controller': 'same'}
    second = groups[1]['entries'][0]['provenance']['sources']['runs']['qwen_controller']['identity']
    second['fingerprint'] = 'changed'
    with pytest.raises(ValueError, match='跨组'):
        expanded.verify_compatibility(groups)
    second['fingerprint'] = 'same'
    second['media']['v']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='重叠视频'):
        expanded.verify_compatibility(groups)


def test_all_three_groups_cumulative_table_and_strict_group_reuse(group_fixture, monkeypatch, tmp_path):
    base = tmp_path/'expanded'
    groups = {}
    for index in (1, 2, 3):
        remapped = {(index, seed): entry for (_, seed), entry in group_fixture.items()}
        # group1 的原始回放已由专用测试覆盖；此处隔离累计汇总与既有组只读复用。
        group = expanded.collect_group(2, {(2, seed): entry for (_, seed), entry in remapped.items()})
        group['group_index'] = index
        for entry in group['entries']:
            entry['group_index'] = index
            entry['question_ids'] = [f'g{index}_{qid}' for qid in entry['question_ids']]
            entry['accuracies']['qwen36'] = index/4
        for trial in group['trials']:
            trial['group_index'] = index
        groups[index] = group
    monkeypatch.setattr(expanded, 'check_samples', lambda base, groups: group_fixture)
    monkeypatch.setattr(expanded, 'collect_group', lambda index, samples: groups[index])
    monkeypatch.setattr(expanded, 'verify_compatibility', lambda groups: {'compatible': True})
    first = expanded.summarize(base, group=1)
    before = {path.name: sha256(path) for path in (base/'group1_summary').iterdir()}
    result = expanded.summarize(base, all_groups=True)
    assert result['cumulative']['methods']['qwen36']['values'] == [.5, .5, .5]
    assert result['cumulative']['methods']['qwen36']['sample_variance'] == 0
    assert result['cumulative']['by_seed'][0]['n'] == 300
    assert len(result['group_reports']) == 3 and result['group_reports'][0] == first
    assert before == {path.name: sha256(path) for path in (base/'group1_summary').iterdir()}
    markdown = (base/'summary/results.md').read_text()
    assert markdown.count('| 方法 | seed42') == 4
    assert json.loads((base/'summary/results.json').read_text()) == result
    with pytest.raises(FileExistsError):
        expanded.summarize(base, all_groups=True)
