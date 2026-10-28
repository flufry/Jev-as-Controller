import copy
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts.sample_longvideobench import sample
from timejev.longvideobench_benchmark import formatted_options
from timejev.user_data import load_manifest, write_jsonl


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'source'
    (root/'videos').mkdir(parents=True)
    reference, rows = [], []
    for group in (15, 60, 600, 3600):
        for index in range(30):
            # 官方 TikTok ID 和实际媒体文件名不同，不能通过 ID 拼接路径。
            video_id = f'@user-{group}-{index//2}'
            video_path = f'{group}-{index//2}.mp4'
            (root/'videos'/video_path).touch()
            target = dict(id=f'{video_id}_{index%2}', video_id=video_id, video_path=video_path,
                question=f'question {group} {index}', candidates=['first', 'second', 'third'],
                correct_choice=index%3, duration_group=group, duration=float(group),
                level='L1-Perception', question_category='test', topic_category='test')
            reference.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+video_id, video_path='videos/'+video_path,
                task='choice', query=target['question'], options=formatted_options(target),
                answer_index=target['correct_choice'], duration=target['duration']))
    manifest, official = root/'questions.jsonl', root/'official.json'
    write_jsonl(rows, manifest)
    official.write_text(json.dumps(reference))
    return manifest, official, rows, reference


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_equal_sampling_reproducible_and_portable(source, tmp_path):
    manifest, official, rows, reference = source
    first, second, third = [tmp_path/name/'sample.jsonl' for name in ('a', 'b', 'c')]
    meta = sample(manifest, official, first)
    sample(manifest, official, second)
    sample(manifest, official, third, seed=43)
    selected, repeat, different = [read_rows(p) for p in (first, second, third)]
    assert len(selected) == 100
    assert Counter(r['duration_group'] for r in selected) == {15: 25, 60: 25, 600: 25, 3600: 25}
    assert [r['id'] for r in selected] == [r['id'] for r in repeat]
    assert {r['id'] for r in selected} != {r['id'] for r in different}
    selected_ids = {r['id'] for r in selected}
    assert [r['id'] for r in selected] == [r['id'] for r in rows if r['id'] in selected_ids]
    originals = {r['id']: r for r in rows}
    for raw, resolved in zip(selected, load_manifest(first)):
        original = originals[raw['id']]
        assert not Path(raw['video_path']).is_absolute()
        assert Path(resolved['video_path']) == (manifest.parent/original['video_path']).resolve()
        for key in ('id', 'question_id', 'video_key', 'query', 'options', 'answer_index', 'duration'):
            assert raw[key] == original[key]
    assert meta == json.loads(first.with_suffix('.meta.json').read_text())
    assert meta['sample_sha256'] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert meta['source_manifest']['sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_custom_count_and_no_overwrite(source, tmp_path):
    manifest, official, _, _ = source
    out = tmp_path/'sample.jsonl'
    meta = sample(manifest, official, out, per_group=2)
    assert meta['n'] == 8
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        sample(manifest, official, out)
    assert out.read_bytes() == before
    out.unlink()
    with pytest.raises(FileExistsError):
        sample(manifest, official, out)


@pytest.mark.parametrize('count', [0, -1, 31])
def test_insufficient_or_invalid_count_rejected(source, tmp_path, count):
    manifest, official, _, _ = source
    with pytest.raises(ValueError):
        sample(manifest, official, tmp_path/'sample.jsonl', per_group=count)
    assert not (tmp_path/'sample.jsonl').exists()


@pytest.mark.parametrize('key,value', [
    ('query', 'changed'), ('options', ['changed']*3), ('answer_index', 2),
    ('video_key', 'longvideobench/changed'), ('duration', 14),
    ('video_path', 'videos/15-1.mp4'), ('id', 'longvideobench/changed'),
    ('question_id', 'unknown'),
])
def test_manifest_mismatch_rejected(source, tmp_path, key, value):
    manifest, official, rows, _ = source
    modified = copy.deepcopy(rows)
    modified[0][key] = value
    write_jsonl(modified, manifest)
    with pytest.raises(ValueError):
        sample(manifest, official, tmp_path/'sample.jsonl')


@pytest.mark.parametrize('kind', ['manifest_duplicate', 'reference_duplicate', 'unknown_group'])
def test_duplicates_and_unknown_groups_rejected(source, tmp_path, kind):
    manifest, official, rows, reference = source
    if kind == 'manifest_duplicate':
        write_jsonl(rows+[rows[0]], manifest)
    elif kind == 'reference_duplicate':
        official.write_text(json.dumps(reference+[reference[0]]))
    else:
        reference[0]['duration_group'] = 999
        official.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        sample(manifest, official, tmp_path/'sample.jsonl')
