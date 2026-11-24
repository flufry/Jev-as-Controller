"""Unavailable videos must remain auditable and never bypass full-inventory checks."""
import json

import pytest

from scripts.bench_io import load_rows
from scripts.build_bench_manifest import build
from scripts.sample_bench_subset import sample


def inputs(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    (videos / 'available.mp4').write_bytes(b'video')
    (videos / 'empty.mp4').touch()
    reference = []
    for key, start, stop in [('available', 0, 1500), ('missing', 1500, 1530), ('empty', 1530, 1549)]:
        reference.append(dict(key=key, qa=[dict(uid=str(i),
            question='Who?\n(A) one\n(B) two\n(C) three\n(D) four', answer='B',
            question_type=['reasoning']) for i in range(start, stop)]))
    annotation = tmp_path / 'video_info.meta.jsonl'
    annotation.write_text(''.join(json.dumps(v) + '\n' for v in reference), encoding='utf-8')
    return annotation, reference


def available(tmp_path):
    annotation, _ = inputs(tmp_path)
    manifest = tmp_path / 'questions.available.jsonl'
    meta = build('lvbench', tmp_path, annotation=annotation, out=manifest,
                 allow_missing_videos=True, probe=lambda _: (320, 180, 60.))
    return manifest, meta


def test_available_build_and_sampling_are_audited_and_opt_in(tmp_path):
    manifest, meta = available(tmp_path)
    coverage = meta['coverage']
    assert coverage['available_questions'] == 1500
    assert coverage['excluded_questions'] == 49
    assert coverage['available_videos'] == 1
    assert coverage['question_coverage'] == 1500 / 1549
    assert [len(v['question_ids']) for v in coverage['excluded_videos']] == [30, 19]
    rows = load_rows(manifest)
    assert [r['uid'] for r in rows] == [str(i) for i in range(1500)]
    assert all(r['answer_index'] == 1 for r in rows)
    with pytest.raises(ValueError, match='allow-incomplete-population'):
        sample('lvbench', tmp_path / 'strict.jsonl', manifest, n=150)
    outputs = []
    for name in ['one', 'two']:
        out = tmp_path / name / 'sample.jsonl'
        sampled = sample('lvbench', out, manifest, n=150, allow_incomplete_population=True)
        assert sampled['source_metadata']['coverage'] == coverage
        assert sampled['population_scope'] == 'available_videos_subset'
        assert len(load_rows(out)) == 150
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]
    with pytest.raises(ValueError, match='sample size'):
        sample('lvbench', tmp_path / 'oversize.jsonl', manifest, n=1501, allow_incomplete_population=True)


def test_strict_build_still_requires_missing_videos(tmp_path):
    annotation, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='Missing video'):
        build('lvbench', tmp_path, annotation=annotation)
    assert not (tmp_path / 'questions.jsonl').exists()


def test_available_build_still_validates_official_uids(tmp_path):
    annotation, reference = inputs(tmp_path)
    reference[1]['qa'][0]['uid'] = '0'
    annotation.write_text(''.join(json.dumps(v) + '\n' for v in reference), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate LVBench UID'):
        build('lvbench', tmp_path, annotation=annotation, allow_missing_videos=True)


def test_available_build_rejects_paths_outside_root(tmp_path):
    annotation, _ = inputs(tmp_path)
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'missing': '../outside.mp4'}), encoding='utf-8')
    with pytest.raises(ValueError, match='outside video root'):
        build('lvbench', tmp_path, annotation=annotation, video_map=mapping, allow_missing_videos=True)


def test_available_build_does_not_silently_skip_corrupt_videos(tmp_path):
    annotation, _ = inputs(tmp_path)
    def failed_probe(_):
        raise ValueError('decode failed')
    with pytest.raises(ValueError, match='decode failed'):
        build('lvbench', tmp_path, annotation=annotation, allow_missing_videos=True, probe=failed_probe)
    assert not (tmp_path / 'questions.jsonl').exists()


def test_available_sampling_rejects_inconsistent_exclusions(tmp_path):
    manifest, meta = available(tmp_path)
    meta['coverage']['excluded_videos'][0]['question_ids'][0] = 'lvbench/0'
    manifest.with_suffix('.meta.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='audited LVBench'):
        sample('lvbench', tmp_path / 'bad.jsonl', manifest, allow_incomplete_population=True)


def test_build_default_matches_downloaded_annotation_name(tmp_path):
    inputs(tmp_path)
    meta = build('lvbench', tmp_path, allow_missing_videos=True, probe=lambda _: (320, 180, 60.))
    assert meta['n'] == 1500
    assert str(tmp_path / 'video_info.meta.jsonl') in meta['sources']


def full_available(tmp_path):
    from timejev import bench_full as full
    source, _ = available(tmp_path)
    manifest = tmp_path / 'full/questions.jsonl'
    full.prepare('lvbench', source, tmp_path / 'video_info.meta.jsonl', manifest,
                 allow_incomplete_population=True)
    rows, metadata = full.read_full(manifest)
    return manifest, rows, metadata


def test_prepare_available_roundtrip_and_default_remains_strict(tmp_path):
    from timejev import bench_full as full
    from scripts import run_bench_full as runner
    manifest, rows, meta = full_available(tmp_path)
    assert len(rows) == meta['n'] == 1500
    assert meta['split'] == 'test_available_videos'
    assert meta['sampling'] == 'all_available_questions_no_selection'
    assert meta['coverage']['excluded_questions'] == 49
    assert meta['coverage']['official_full_benchmark'] is False
    with pytest.raises(ValueError, match='allow-incomplete-population'):
        full.prepare('lvbench', meta['source_manifest'], meta['reference'], tmp_path / 'strict.jsonl')
    runner.run(runner.parser().parse_args(['--bench', 'lvbench', '--base', str(manifest.parent), '--dry-run']))
    assert not (manifest.parent / 'experiment.json').exists()
    source_meta = full.read_json(meta['source_manifest'].replace('.jsonl', '.meta.json'))
    source_meta['coverage']['excluded_questions'] = 48
    from pathlib import Path
    Path(meta['source_manifest']).with_suffix('.meta.json').write_text(json.dumps(source_meta), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA'):
        full.read_full(manifest)


@pytest.mark.parametrize('mutation', ['excluded_id', 'partial_video', 'reference_sha', 'manifest_sha'])
def test_prepare_available_rejects_forged_population(tmp_path, mutation):
    from timejev import bench_full as full
    from scripts.bench_io import jsonl_bytes, sha256
    source, meta = available(tmp_path)
    if mutation == 'excluded_id':
        meta['coverage']['excluded_videos'][0]['question_ids'][0] = 'lvbench/0'
    elif mutation == 'partial_video':
        rows = full.read_jsonl(source)
        removed = rows.pop()
        source.write_bytes(jsonl_bytes(rows))
        meta.update(n=len(rows), manifest_sha256=sha256(source))
        meta['coverage'].update(available_questions=len(rows), excluded_questions=50, question_coverage=len(rows)/1549)
        meta['coverage']['excluded_videos'].append(dict(video_key=removed['video_key'], question_ids=[removed['id']]))
    elif mutation == 'reference_sha':
        meta['sources'] = {str(tmp_path / 'video_info.meta.jsonl'): 'bad'}
    else:
        meta['manifest_sha256'] = 'bad'
    source.with_suffix('.meta.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError):
        full.prepare('lvbench', source, tmp_path / 'video_info.meta.jsonl', tmp_path / 'bad.jsonl',
                     allow_incomplete_population=True)
    assert not (tmp_path / 'bad.jsonl').exists()


def test_missing_reference_explains_downloaded_filename(tmp_path):
    from timejev import bench_full as full
    with pytest.raises(FileNotFoundError, match='video_info.meta.jsonl'):
        full._official_rows('lvbench', tmp_path / 'official.jsonl')
    with pytest.raises(ValueError, match='仅用于LVBench'):
        full.prepare('mlvu', 'unused', 'unused', tmp_path / 'unused.jsonl', allow_incomplete_population=True)


def test_available_scope_propagates_to_candidate_and_native_metrics(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from timejev import bench_full as full, lvb_experiment as lvb, lvb_native_generation as native
    manifest, rows, meta = full_available(tmp_path)
    predictions = {r['id']: dict(r, probabilities=[0., 1., 0., 0.]) for r in rows}
    with full.install_sources(rows, meta):
        candidate = lvb.score(rows, predictions)
    assert candidate['correct'] == candidate['n'] == 1500
    assert candidate['coverage'] == meta['coverage']
    assert candidate['split'] == 'test_available_videos'
    monkeypatch.setattr(native, 'checkpoint_identity', lambda _: dict(fake=True))
    args = SimpleNamespace(manifest=str(manifest), checkpoint='fake', out=str(tmp_path / 'native'),
                           limit=None, resume=False, seed=43)
    run = native.NativeExperiment(args, {}, rows=rows, code_paths=[Path(native.__file__)],
        benchmark='lvbench', full_dataset=meta, input_protocol=dict(name='video_bench_full_rgb_v1'))
    monkeypatch.setattr(native, 'read_generated', lambda *args, **kw: predictions)
    generated = run.finish()
    assert generated['coverage'] == candidate['coverage']
    assert generated['n'] == 1500 and generated['accuracy'] == 1.
    assert run.metadata['sampling'] == meta['sampling']


def test_available_six_method_summary_retains_coverage(tmp_path, monkeypatch):
    from timejev import bench_full as full
    from scripts.videojev_candidate_logits import softmax
    manifest, rows, meta = full_available(tmp_path)
    def source(directory, targets, info, role, expected_seed):
        native = role.startswith('native')
        baseline = role == 'videojev_baseline'
        qwen = role == 'qwen_controller' or native
        fps = 4. if native or baseline else 2.
        protocol = dict(name='video_bench_full_rgb_v1', fps=fps, max_frames=4096 if fps==4 else 2048)
        values = {}
        for row in targets:
            logits = [0., 1., .5, 0.]
            pred = dict(row, preprocessing={'fps':fps}, probabilities=softmax(logits), answer_index=1)
            if native:
                pred.update(valid=True, truncated=False)
            elif qwen:
                pred['candidate_logits'] = logits
            else:
                pred['logits'] = logits
            values[row['id']] = pred
        identity = dict(protocol=protocol, media={'fixture':True}, checkpoint={'model':'qwen' if qwen else 'jev'}, settings={})
        return values, dict(identity=identity, files={}), {}
    monkeypatch.setattr(full, 'read_source', source)
    report = full.summarize(manifest, manifest.parent)
    assert report['n'] == 1500 and report['coverage'] == meta['coverage']
    assert len(report['methods']) == 6
    for method in report['methods'].values():
        assert len(method['trials']) == 3
        assert all(trial['coverage'] == meta['coverage'] for trial in method['trials'])
    for metrics in report['argmax_diagnostics'].values():
        assert metrics['coverage'] == meta['coverage']
    text = full.markdown(report)
    assert '1500/1549' in text and '49 道题' in text and '不是官方1549题全量结果' in text
