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
