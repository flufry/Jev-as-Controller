from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pytest

from scripts.sample_longvideobench import sample
from scripts.sample_longvideobench_additional_groups import encoded, sample_additional
from scripts.summarize_longvideobench_qwen36 import read_sample
from timejev.longvideobench_benchmark import formatted_options
from timejev.lvb_experiment import load_sample
from timejev.user_data import write_jsonl


def make_source(tmp_path, count=76):
    source = tmp_path/'source'
    (source/'videos').mkdir(parents=True)
    rows, targets = [], []
    for group in (15, 60, 600, 3600):
        for index in range(count):
            video_id, filename = f'@user-{group}-{index//2}', f'actual-{group}-{index//2}.mp4'
            (source/'videos'/filename).touch()
            target = dict(id=f'{video_id}_{index%2}', video_id=video_id, video_path=filename,
                question=f'问题 {group} {index}', candidates=['第一项', '第二项', '第三项'],
                correct_choice=index%3, duration_group=group, duration=float(group),
                level='L1-Perception', question_category='test', topic_category='test')
            targets.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+video_id, video_path='videos/'+filename,
                media_source=dict(kind='video', path='videos/'+filename), task='choice',
                query=target['question'], options=formatted_options(target),
                answer_index=target['correct_choice'], duration=target['duration']))
    manifest, reference = source/'questions.jsonl', source/'official.json'
    write_jsonl(rows, manifest)
    reference.write_text(json.dumps(targets, ensure_ascii=False))
    pattern = str(tmp_path/'original/seed{seed}/sample.jsonl')
    for seed in (42, 43, 44):
        sample(manifest, reference, pattern.format(seed=seed), seed=seed)
    return manifest, reference, pattern, rows


@pytest.fixture
def inputs(tmp_path):
    return make_source(tmp_path)


def test_six_samples_disjoint_portable_and_compatible(inputs, tmp_path):
    manifest, reference, pattern, original_rows = inputs
    out = tmp_path/'expanded'
    out.mkdir()
    other = out/'launcher-note.txt'
    other.write_text('其他 agent 的文件')
    result = sample_additional(manifest, reference, pattern, out)
    assert other.read_text() == '其他 agent 的文件'
    assert result['total_question_occurrences'] == 900
    assert result == json.loads((out/'sampling.json').read_text())
    by_id = {r['id']: r for r in original_rows}
    for seed in (42, 43, 44):
        ids = [set(r['id'] for r in load_sample(pattern.format(seed=seed)))]
        for group in (2, 3):
            path = out/f'group{group}/seed{seed}/sample.jsonl'
            rows = load_sample(path)
            read_sample(path, seed)  # 既有严格评估校验可直接读取新增清单。
            meta = json.loads(path.with_suffix('.meta.json').read_text())
            raw = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 100 and Counter(r['duration_group'] for r in rows) == {g: 25 for g in (15, 60, 600, 3600)}
            selected = {r['id'] for r in rows}
            assert [r['id'] for r in rows] == [r['id'] for r in original_rows if r['id'] in selected]
            for item, resolved in zip(raw, rows):
                original = by_id[item['id']]
                assert not Path(item['video_path']).is_absolute()
                assert not Path(item['media_source']['path']).is_absolute()
                assert Path(resolved['video_path']) == (manifest.parent/original['video_path']).resolve()
                for key in ('id', 'question_id', 'video_key', 'query', 'options', 'answer_index', 'duration'):
                    assert item[key] == original[key]
            assert meta['seed'] == seed and meta['group_index'] == group
            assert meta['sample_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert meta['excluded_ids'] == result['per_seed'][str(seed)]['excluded_ids']
            assert all(entry['questions'] == 0 for entry in meta['same_seed_overlaps'])
            ids.append(selected)
        assert len(set.union(*ids)) == 300 and all(not a & b for a, b in ((ids[0],ids[1]),(ids[0],ids[2]),(ids[1],ids[2])))


def test_reproducibility_and_single_draw_split(inputs, tmp_path):
    manifest, reference, pattern, _ = inputs
    first = sample_additional(manifest, reference, pattern, tmp_path/'a')
    second = sample_additional(manifest, reference, pattern, tmp_path/'b')
    source = load_sample(pattern.format(seed=42))
    excluded = {r['id'] for r in source}
    reference_rows = {r['id']: r for r in json.loads(reference.read_text())}
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rng = random.Random(42)
    for group in (15, 60, 600, 3600):
        candidates = [r['id'] for r in all_rows if r['id'] not in excluded and reference_rows[r['question_id']]['duration_group'] == group]
        drawn = rng.sample(candidates, 50)
        assignment = first['per_seed']['42']['by_duration_group'][str(group)]
        assert assignment['drawn_ids'] == drawn
        assert assignment['group2_ids'] == drawn[:25] and assignment['group3_ids'] == drawn[25:]
    for seed in (42, 43, 44):
        a, b = first['per_seed'][str(seed)], second['per_seed'][str(seed)]
        assert a['allocation_sha256'] == b['allocation_sha256']
        assert a['allocation_sha256'] == hashlib.sha256(encoded(dict(seed=seed, excluded_ids=a['excluded_ids'],
            by_duration_group=a['by_duration_group']))).hexdigest()
        for group in (2, 3):
            assert (tmp_path/f'a/group{group}/seed{seed}/sample.jsonl').read_bytes() == (tmp_path/f'b/group{group}/seed{seed}/sample.jsonl').read_bytes()


@pytest.mark.parametrize('existing', ['group2/seed42/sample.jsonl', 'group3/seed44/sample.meta.json', 'sampling.json'])
def test_no_partial_write_or_overwrite(inputs, tmp_path, existing):
    manifest, reference, pattern, _ = inputs
    out = tmp_path/'expanded'
    path = out/existing
    path.parent.mkdir(parents=True)
    path.write_text('保留')
    with pytest.raises(FileExistsError):
        sample_additional(manifest, reference, pattern, out)
    assert path.read_text() == '保留'
    assert [p for p in out.rglob('*') if p.is_file()] == [path]


def test_insufficient_remaining_rejected_before_writes(tmp_path):
    manifest, reference, pattern, _ = make_source(tmp_path, count=74)
    with pytest.raises(ValueError, match='需要 50'):
        sample_additional(manifest, reference, pattern, tmp_path/'expanded')
    assert not (tmp_path/'expanded').exists()


@pytest.mark.parametrize('key,value', [('options',['bad']*3),('answer_index',2),('video_path','videos/actual-15-1.mp4')])
def test_official_mismatch_rejected(inputs, tmp_path, key, value):
    manifest, reference, pattern, rows = inputs
    rows[0][key] = value
    write_jsonl(rows, manifest)
    with pytest.raises(ValueError):
        sample_additional(manifest, reference, pattern, tmp_path/'expanded')
    assert not (tmp_path/'expanded').exists()


def test_original_metadata_seed_or_sha_rejected(inputs, tmp_path):
    manifest, reference, pattern, _ = inputs
    path = Path(pattern.format(seed=42)).with_suffix('.meta.json')
    original = json.loads(path.read_text())
    for key, value in [('seed',43),('sample_sha256','0'*64)]:
        path.write_text(json.dumps(original | {key:value}))
        with pytest.raises(ValueError):
            sample_additional(manifest, reference, pattern, tmp_path/'expanded')
        assert not (tmp_path/'expanded').exists()
