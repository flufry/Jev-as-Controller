"""Screening protocol tests use synthetic annotations/scores, never model weights."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bench_io import json_bytes, jsonl_bytes, load_rows, sha256, write_new
from scripts.bench_compat import generic_load_sample, generic_score
from scripts.build_bench_manifest import adapt, build, mlvu_reference, validate_lvbench
from scripts.sample_bench_subset import allocate_counts, sample, validate_population
from scripts.summarize_bench import controlled, summarize
from scripts.report_bench_screen import report
from timejev.mlvu_benchmark import TASK_COUNTS


def save_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def fixture_rows(tmp_path, bench='videomme'):
    video = tmp_path / 'v.mp4'
    video.touch()
    groups = list(TASK_COUNTS) if bench == 'mlvu' else ['short', 'medium', 'long']
    rows = []
    for group in groups:
        for i in range(4 if group == groups[0] else 2):
            rows.append(dict(id=f'{bench}/{group}/{i}', question_id=f'{group}/{i}', benchmark=bench,
                video_key='v', video_path=str(video), media_source=dict(kind='video', path=str(video)),
                duration=60., task='choice', query=f'Question {group} {i}',
                options=['A. one', 'B. two', 'C. three'], answer_index=i % 3, stratum=group))
    return rows


@pytest.mark.parametrize('counts,n,expected', [
    ({'a': 189, 'b': 172, 'c': 412, 'd': 564}, 100, [14, 13, 31, 42]),
    ({'b': 1, 'a': 1, 'c': 1}, 2, [1, 1, 0]),
    ({'a': 0, 'b': 3}, 2, [0, 2]),
])
def test_integer_allocation(counts, n, expected):
    assert list(allocate_counts(counts, n).values()) == expected


@pytest.mark.parametrize('n', [0, -1, 4])
def test_invalid_allocation(n):
    with pytest.raises(ValueError):
        allocate_counts({'a': 3}, n)


def test_sample_preserves_order_options_paths_and_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.sample_bench_subset.validate_population', lambda *_: None)
    rows = fixture_rows(tmp_path)
    manifest = tmp_path / 'questions.jsonl'
    manifest.write_bytes(jsonl_bytes(rows))
    outputs = [tmp_path / d / 'sample.jsonl' for d in ('one', 'two')]
    metas = [sample('videomme', p, manifest, n=6) for p in outputs]
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert metas[0] == metas[1]
    assert metas[0]['quotas'] == dict(long=2, medium=2, short=2)
    selected = load_rows(outputs[0])
    assert selected == [r for r in rows if r['id'] in metas[0]['selected_ids']]
    assert metas[0]['source_sha256'] == sha256(manifest)
    assert metas[0]['sample_sha256'] == sha256(outputs[0])
    with pytest.raises(FileExistsError):
        sample('videomme', outputs[0], manifest, n=6)
    outputs[0].write_bytes(outputs[0].read_bytes().replace(b'Question', b'Changed!'))
    with pytest.raises(ValueError, match='SHA'):
        load_rows(outputs[0])


def test_mlvu_equal_tasks_macro_differs_from_micro(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.sample_bench_subset.validate_population', lambda *_: None)
    rows = fixture_rows(tmp_path, 'mlvu')
    manifest = tmp_path / 'questions.jsonl'
    manifest.write_bytes(jsonl_bytes(rows))
    meta = sample('mlvu', tmp_path / 'sample.jsonl', manifest, per_task=2)
    assert meta['n'] == 14 and set(meta['quotas'].values()) == {2}
    predictions = {}
    for row in rows:
        choice = row['answer_index'] if row['stratum'] == 'plotQA' else (row['answer_index'] + 1) % 3
        predictions[row['id']] = dict(row, answer_index=choice, probabilities=[float(j == choice) for j in range(3)])
    metrics = generic_score(rows, predictions)
    assert metrics['m_avg'] == pytest.approx(1 / 7)
    assert metrics['accuracy'] == pytest.approx(4 / 16)
    assert metrics['primary_metric'] == 'm_avg'
    with pytest.raises(ValueError):
        sample('mlvu', tmp_path / 'bad.jsonl', manifest, n=14)


@pytest.mark.parametrize('damage', ['duplicate', 'duration', 'source', 'group', 'label'])
def test_manifest_rejects_unsafe_or_inconsistent_inputs(tmp_path, damage):
    rows = fixture_rows(tmp_path)
    if damage == 'duplicate':
        rows.append(rows[0])
    elif damage == 'duration':
        rows[0]['duration'] = float('nan')
    elif damage == 'source':
        rows[0]['media_source']['kind'] = 'frames_sidecar'
    elif damage == 'group':
        rows[0]['duration'] = 20.
    else:
        rows[0]['answer_index'] = True
    path = tmp_path / 'bad.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        load_rows(path)


def test_build_full_videomme_probes_each_video_once_preserves_official_text(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    reference = []
    for i in range(900):
        key = f'v{i}'
        (videos / (key + '.mp4')).touch()
        for j in range(3):
            reference.append(dict(question_id=f'{i}-{j}', videoID=key, video_id=str(i),
                duration=('short', 'medium', 'long')[i // 300], domain='d', sub_category='s', task_type='t',
                question=f'Unchanged {i}-{j}?', options=['A. 1', 'B. 2', 'C. 3', 'D. 4'], answer='C'))
    annotation = tmp_path / 'official.json'
    save_json(annotation, reference)
    calls = []

    def probe(path):
        calls.append(path)
        return 320, 180, 123.5

    meta = build('videomme', tmp_path, probe=probe)
    assert len(calls) == 900 and meta['n'] == 2700
    rows = load_rows(tmp_path / 'questions.jsonl')
    assert rows[0]['duration'] == 123.5 and rows[0]['official_duration'] == 'short'
    assert rows[0]['options'] == reference[0]['options']
    assert rows[0]['answer_index'] == 2
    assert rows[0]['media_source']['path'] == rows[0]['video_path']
    sampled = sample('videomme', tmp_path / 'subset/sample.jsonl', tmp_path / 'questions.jsonl', n=150)
    assert sampled['n'] == 150 and set(sampled['quotas'].values()) == {50}
    assert sampled['source_metadata']['sources'] == meta['sources']


def test_mlvu_original_task_files_ids_repeated_answer_and_prefix(tmp_path):
    annotations, videos = tmp_path / 'json', tmp_path / 'videos'
    annotations.mkdir()
    videos.mkdir()
    index = 0
    originals = []
    for number, (task, count) in enumerate(TASK_COUNTS.items(), 1):
        values = []
        for i in range(count):
            video = f'{index % 1122}.mp4'
            (videos / video).touch()
            row = dict(question=f'{task} {i}?', candidates=['x', 'y', 'x', 'z'], answer='x', video=video, duration=100.)
            values.append(row)
            index += 1
        save_json(annotations / f'{number}_{task}.json', values)
        originals.extend(values)
    ref, _ = mlvu_reference(annotations)
    assert len(ref) == 2174
    rows = adapt('mlvu', ref, videos)
    assert rows[0]['answer_index'] == 0
    assert rows[0]['options'] == ['A. x', 'B. y', 'C. x', 'D. z']
    assert rows[0]['question_id'] == 'plotQA/0'
    assert len({r['id'] for r in rows}) == 2174


def test_lvbench_multilabel_combination_is_exclusive_and_categories_overlap(tmp_path):
    (tmp_path / 'video.mp4').touch()
    reference = [dict(key='video', qa=[dict(uid=str(i), question='Who?\n(A) one\n(B) two\n(C) three\n(D) four',
        answer='B', question_type=['reasoning', 'event understanding'] if i % 2 else ['reasoning']) for i in range(1549)])]
    validate_lvbench(reference)
    rows = adapt('lvbench', reference, tmp_path)
    for row in rows:
        row['duration'] = 60.
    assert len({r['stratum'] for r in rows}) == 2
    predictions = {r['id']: dict(r, probabilities=[0., 1., 0., 0.]) for r in rows}
    metrics = generic_score(rows, predictions)
    assert metrics['by_question_type']['reasoning']['n'] == 1549
    assert sum(g['n'] for g in metrics['by_question_type'].values()) > 1549
    reference[0]['qa'][-1]['uid'] = '0'
    with pytest.raises(ValueError, match='Duplicate'):
        validate_lvbench(reference)


def make_runs(tmp_path, bench='videomme'):
    rows = fixture_rows(tmp_path, bench)
    manifest = tmp_path / 'sample.jsonl'
    manifest.write_bytes(jsonl_bytes(rows))
    protocol = dict(fps=2., max_frames=2048, context=262144, visual_token_budget=24576, use_subtitles=False, use_audio=False)
    for method in ('qwen36', 'videojev'):
        out = tmp_path / method
        out.mkdir()
        predictions = []
        for i, row in enumerate(rows):
            # Includes third-choice rescue: Q top-1 A, J top-1 B, fusion top-1 C.
            values = [1., 0., .9] if method == 'qwen36' else [0., 2., 1.8]
            weights = [math.exp(v - max(values)) for v in values]
            probabilities = [w / sum(weights) for w in weights]
            prediction = {k: row[k] for k in ('id', 'task', 'query', 'options', 'stratum')}
            prediction.update(probabilities=probabilities, answer_index=max(range(3), key=probabilities.__getitem__),
                              preprocessing=dict(frames=120, tokens=100))
            if method == 'qwen36':
                prediction['candidate_logits'] = values
            predictions.append(prediction)
        (out / 'predictions.jsonl').write_bytes(jsonl_bytes(reversed(predictions)))
        save_json(out / 'run.json', dict(identity=dict(method=method, manifest_sha256=sha256(manifest), protocol=protocol, seed=42)))
        save_json(out / 'metrics.json', dict(generic_score(rows, {r['id']: r for r in predictions}), method=method, protocol=protocol))
    return manifest, rows


def test_controller_fusion_summary_report_and_scope_restoration(tmp_path):
    from scripts.eval_bench_controller import evaluate
    from timejev import lvb_experiment as lvb
    manifest, rows = make_runs(tmp_path)
    original = (lvb.load_sample, lvb.score, lvb.Experiment)
    evaluate(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller')
    assert (lvb.load_sample, lvb.score, lvb.Experiment) == original
    meta = json.loads((tmp_path / 'controller/run.json').read_text(encoding='utf-8'))
    assert meta['split'] == 'screening_subset'
    assert 'scripts/bench_compat.py' in meta['identity']['code_sha256']
    assert '1337' not in meta['note']
    result = summarize(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller',
                       tmp_path / 'summary', bootstrap=500)
    assert result['disagreement']['third_choice_count'] == len(rows)
    assert result['disagreement']['controller_correct_both_baselines_wrong'] > 0
    assert result['methods']['controller']['metric'] == pytest.approx(1 / len(rows))
    assert result['paired']['rescued'] == 1
    assert result['paired']['harmed'] == 4
    assert result['paired']['mcnemar_exact_two_sided_p'] == .375
    assert result['delta_ci95'][0] <= result['delta_vs_qwen'] <= result['delta_ci95'][1]
    report([tmp_path / 'summary/results.json'], tmp_path / 'report')
    assert 'videomme' in (tmp_path / 'report/results.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        evaluate(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller')


@pytest.mark.parametrize('change', ['preprocessing', 'protocol', 'seed', 'logits', 'metrics', 'missing'])
def test_fusion_rejects_bad_sources(tmp_path, change):
    from scripts.eval_bench_controller import evaluate
    manifest, rows = make_runs(tmp_path)
    q = tmp_path / 'qwen36'
    if change in ('preprocessing', 'logits', 'missing'):
        path = q / 'predictions.jsonl'
        predictions = [json.loads(s) for s in path.read_text().splitlines()]
        if change == 'preprocessing':
            predictions[0]['preprocessing'] = {}
        elif change == 'logits':
            predictions[0]['candidate_logits'] = [3., 1., 0.]
        else:
            predictions.pop()
        path.write_bytes(jsonl_bytes(predictions))
    else:
        path = q / ('metrics.json' if change == 'metrics' else 'run.json')
        data = json.loads(path.read_text())
        if change == 'metrics':
            data['correct'] += 1
        elif change == 'seed':
            data['identity']['seed'] = 43
        else:
            data['identity']['protocol']['fps'] = 4.
        save_json(path, data)
    with pytest.raises(ValueError):
        evaluate(manifest, q, tmp_path / 'videojev', tmp_path / 'controller')


def test_summary_rejects_controller_tampering(tmp_path):
    from scripts.eval_bench_controller import evaluate
    manifest, _ = make_runs(tmp_path)
    evaluate(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller')
    path = tmp_path / 'controller/run.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['identity']['sources']['qwen']['predictions_sha256'] = 'changed'
    save_json(path, data)
    with pytest.raises(ValueError, match='fingerprints'):
        summarize(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller', tmp_path / 'summary', bootstrap=20)


def test_experiment_append_resume_and_adapter_fingerprint(tmp_path, monkeypatch):
    from scripts import bench_compat
    from timejev import lvb_experiment as lvb
    rows = fixture_rows(tmp_path)
    manifest = tmp_path / 'sample.jsonl'
    manifest.write_bytes(jsonl_bytes(rows))
    model = tmp_path / 'model'
    model.mkdir()
    save_json(model / 'config.json', {})
    args = SimpleNamespace(manifest=str(manifest), out=str(tmp_path / 'run'), checkpoint=str(model),
        context=262144, visual_token_budget=24576, fps=2., max_frames=2048, memory_fraction=.9,
        seed=42, device='cpu', resume=False)
    monkeypatch.setattr('sys.argv', [str(Path(bench_compat.__file__).with_name('eval_bench_videojev.py'))])
    with bench_compat.install():
        run = lvb.Experiment(args, 'videojev', {})
        run.append(run.rows[0], [1., 0., 0.])
    args.resume = True
    with bench_compat.install():
        resumed = lvb.Experiment(args, 'videojev', {})
        assert resumed.done == {rows[0]['id']}
    monkeypatch.setattr(bench_compat, 'adapter_hashes', lambda legacy: {'changed.py': 'new-code'})
    with bench_compat.install(), pytest.raises(ValueError, match='指纹'):
        lvb.Experiment(args, 'videojev', {})


def test_beta_zero_diagnostic_matches_source():
    raw = dict(id='q', options=['a', 'b', 'c'], candidate_logits=[1., 0., .9])
    weights = [math.exp(v) for v in raw['candidate_logits']]
    raw['probabilities'] = [v / sum(weights) for v in weights]
    probabilities, choice = controlled(raw, dict(probabilities=[0., 1., 0.]), 0.)
    assert probabilities == pytest.approx(raw['probabilities']) and choice == 0


def test_sampling_rejects_incomplete_population(tmp_path):
    rows = fixture_rows(tmp_path)
    with pytest.raises(ValueError, match='complete'):
        validate_population('videomme', rows)


def test_mlvu_summary_uses_macro_metric_for_all_diagnostics(tmp_path):
    from scripts.eval_bench_controller import evaluate
    manifest, _ = make_runs(tmp_path, 'mlvu')
    evaluate(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller')
    result = summarize(manifest, tmp_path / 'qwen36', tmp_path / 'videojev', tmp_path / 'controller',
                       tmp_path / 'summary', bootstrap=100)
    assert result['primary_metric'] == 'm_avg'
    assert result['methods']['controller']['metric'] == pytest.approx(1 / 28)
    assert result['methods']['controller']['correct'] / result['n'] == pytest.approx(1 / 16)
    assert result['methods']['random_choice']['expected_metric'] == pytest.approx(
        (result['methods']['qwen36']['metric'] + result['methods']['videojev']['metric']) / 2)
    assert result['beta_sweep']['2.0']['metric'] == result['methods']['controller']['metric']
