from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest

from scripts import sample_longvideobench_proportional as proportional
from timejev.longvideobench_benchmark import formatted_options
from timejev.user_data import load_manifest, write_jsonl

COUNTS = {15: 189, 60: 172, 600: 412, 3600: 564}
QUOTAS = {15: 14, 60: 13, 600: 31, 3600: 42}


@pytest.fixture
def source(tmp_path):
    directory = tmp_path/'source'
    videos = directory/'videos'
    videos.mkdir(parents=True)
    reference, rows = [], []
    video_counts = {15: 100, 60: 100, 600: 230, 3600: 323}
    for group, count in COUNTS.items():
        for index in range(count):
            video_number = index % video_counts[group]
            video_id = f'@user-{group}-{video_number}'
            filename = f'media-{group}-{video_number}.mp4'
            (videos/filename).touch(exist_ok=True)
            number = len(reference)
            choices = 3 if number == 0 else 4 if number < 354 else 5
            target = dict(id=f'{video_id}_{index//video_counts[group]}', video_id=video_id,
                video_path=filename, question=f'Question {group} {index}',
                candidates=[f'choice {i}' for i in range(choices)], correct_choice=number % choices,
                duration_group=group, duration=float(group), level='L1-Perception',
                question_category='test', topic_category='test')
            reference.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+video_id, video_path='videos/'+filename,
                media_source=dict(kind='video', path='videos/'+filename), task='choice',
                query=target['question'], options=formatted_options(target), answer_index=target['correct_choice'],
                duration=target['duration']))
    random.Random(17).shuffle(rows)
    manifest, official = directory/'questions.jsonl', directory/'official.json'
    write_jsonl(rows, manifest)
    official.write_text(json.dumps(reference, ensure_ascii=False), encoding='utf-8')
    return manifest, official, rows, reference


def raw_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def metadata(path):
    return json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))


def update_meta(path, value):
    path.with_suffix('.meta.json').write_text(json.dumps(value, ensure_ascii=False)+'\n', encoding='utf-8')


def test_actual_inventory_allocates_fourteen_thirteen_thirty_one_forty_two():
    assert proportional.allocate_counts(COUNTS) == QUOTAS
    assert proportional.allocate_counts(COUNTS, 1337) == COUNTS
    assert sum(proportional.allocate_counts(COUNTS, 101).values()) == 101


def test_ties_follow_official_group_order_using_integer_arithmetic():
    counts = dict(reversed([(group, 2**60+1) for group in proportional.GROUPS]))
    assert proportional.allocate_counts(counts, 2) == {15: 1, 60: 1, 600: 0, 3600: 0}
    assert proportional.allocate_counts({15: 0, 60: 0, 600: 0, 3600: 5}, 1) == {15: 0, 60: 0, 600: 0, 3600: 1}


@pytest.mark.parametrize('n', [0, -1, 1338, 100.0, True, None])
def test_invalid_or_overpopulation_sample_size_rejected(n):
    with pytest.raises(ValueError):
        proportional.allocate_counts(COUNTS, n)


@pytest.mark.parametrize('counts', [None, {}, {15: 1}, dict(COUNTS, extra=1),
    {15: -1, 60: 1, 600: 1, 3600: 1}, {15: True, 60: 1, 600: 1, 3600: 1},
    {15: 1.0, 60: 1, 600: 1, 3600: 1}, {15: 0, 60: 0, 600: 0, 3600: 0}])
def test_invalid_group_populations_rejected(counts):
    with pytest.raises(ValueError):
        proportional.allocate_counts(counts)


def test_reproducible_independent_question_sampling_preserves_source_order_and_paths(source, tmp_path):
    manifest, official, original, _ = source
    paths = [tmp_path/name/'sample.jsonl' for name in ('first', 'repeat', 'seed43', 'seed44')]
    outputs = [proportional.sample(manifest, official, path, seed=seed) for path, seed in zip(paths, (42, 42, 43, 44))]
    selections = [[row['id'] for row in raw_rows(path)] for path in paths]
    assert selections[0] == selections[1] and selections[0] != selections[2] and selections[2] != selections[3]
    assert set(selections[0]) & set(selections[2])  # 不排除其它 seed 的题目。
    assert selections[0] == [row['id'] for row in original if row['id'] in set(selections[0])]
    rows, provenance = proportional.read_sample(paths[0])
    assert len(rows) == 100 and Counter(row['duration_group'] for row in rows) == QUOTAS
    assert outputs[0]['schema'] == proportional.SCHEMA and outputs[0]['unit'] == 'question'
    assert outputs[0]['sampling'] == proportional.SAMPLING and outputs[0]['quota_method'] == 'largest_remainder'
    assert outputs[0]['source_counts'] == {str(key): value for key, value in COUNTS.items()}
    assert outputs[0]['quotas'] == outputs[0]['selected_counts'] == {str(key): value for key, value in QUOTAS.items()}
    assert provenance['path'] == str(paths[0]) and provenance['sha256'] == proportional.sha256(paths[0])
    assert provenance['meta_sha256'] == proportional.sha256(paths[0].with_suffix('.meta.json'))
    assert provenance['source_manifest']['path'] == str(manifest)
    assert provenance['source_reference']['path'] == str(official)
    assert provenance['seed'] == 42 and provenance['quotas'] == outputs[0]['quotas']
    originals = {row['id']: row for row in load_manifest(manifest)}
    for raw, row in zip(raw_rows(paths[0]), rows):
        assert not Path(raw['video_path']).is_absolute() and not Path(raw['media_source']['path']).is_absolute()
        assert row['video_path'] == row['media_source']['path'] == originals[row['id']]['video_path']
        assert all(row[key] == originals[row['id']][key] for key in
            ('question_id', 'task', 'video_key', 'query', 'options', 'answer_index', 'duration'))


def test_small_quota_keeps_zero_count_groups_and_reader_accepts_requested_n(source, tmp_path):
    path = tmp_path/'sample.jsonl'
    info = proportional.sample(*source[:2], path, n=1)
    assert info['quotas'] == info['selected_counts'] == {'15': 0, '60': 0, '600': 0, '3600': 1}
    rows, _ = proportional.read_sample(path, n=1)
    assert len(rows) == 1
    with pytest.raises(ValueError, match='指定 seed/n'):
        proportional.read_sample(path)


def test_selection_does_not_use_labels_or_historical_exclusions(source, tmp_path):
    manifest, official, rows, reference = source
    first, second = tmp_path/'first.jsonl', tmp_path/'second.jsonl'
    proportional.sample(manifest, official, first)
    changed = deepcopy(reference)
    labels = {}
    for target in changed:
        target['correct_choice'] = (target['correct_choice']+1) % len(target['candidates'])
        labels[target['id']] = target['correct_choice']
    modified = [dict(row, answer_index=labels[row['question_id']]) for row in rows]
    write_jsonl(modified, manifest)
    official.write_text(json.dumps(changed), encoding='utf-8')
    proportional.sample(manifest, official, second)
    assert [row['id'] for row in raw_rows(first)] == [row['id'] for row in raw_rows(second)]


@pytest.mark.parametrize('field,value', [('schema', 'longvideobench_stratified_questions_v1'),
    ('sampling', 'equal_official_duration_groups'), ('quota_method', 'round'), ('unit', 'video'),
    ('seed', 43), ('seed', 42.0), ('n', 99), ('selected_ids', ['forged']),
    ('quotas', {'15': 25, '60': 25, '600': 25, '3600': 25}),
    ('source_counts', {'15': 1, '60': 1, '600': 1, '3600': 1}),
    ('selected_counts', {'15': 25, '60': 25, '600': 25, '3600': 25}),
    ('tie_break_order', [3600, 600, 60, 15]), ('videos', 999), ('sample_sha256', '0'*64)])
def test_reader_rejects_tampered_metadata_and_equal_sampler_disguises(source, tmp_path, field, value):
    path = tmp_path/'sample.jsonl'
    proportional.sample(*source[:2], path)
    info = metadata(path)
    info[field] = value
    update_meta(path, info)
    with pytest.raises(ValueError):
        proportional.read_sample(path)


def test_metadata_integer_counts_cannot_be_replaced_by_equal_float_values(source, tmp_path):
    path = tmp_path/'sample.jsonl'
    proportional.sample(*source[:2], path)
    info = metadata(path)
    info['quotas']['15'] = 14.0
    update_meta(path, info)
    with pytest.raises(ValueError):
        proportional.read_sample(path)


@pytest.mark.parametrize('change', ['different_seed', 'duplicate_id', 'query', 'options', 'label', 'absolute_video_path', 'media_path', 'reorder'])
def test_reader_rejects_changed_questions_even_with_updated_sample_sha(source, tmp_path, change):
    path = tmp_path/'sample.jsonl'
    proportional.sample(*source[:2], path)
    rows, info = raw_rows(path), metadata(path)
    if change == 'different_seed':
        other = tmp_path/'other.jsonl'
        proportional.sample(*source[:2], other, seed=43)
        rows = raw_rows(other)
    elif change == 'duplicate_id':
        rows[-1] = deepcopy(rows[0])
    elif change == 'query':
        rows[0]['query'] = 'altered question'
    elif change == 'options':
        rows[0]['options'] = rows[0]['options'][::-1]
    elif change == 'label':
        rows[0]['answer_index'] = (rows[0]['answer_index']+1) % len(rows[0]['options'])
    elif change == 'absolute_video_path':
        rows[0]['video_path'] = str((path.parent/rows[0]['video_path']).resolve())
    elif change == 'media_path':
        rows[0]['media_source']['path'] = rows[1]['media_source']['path']
    else:
        rows.reverse()
    write_jsonl(rows, path)
    info['sample_sha256'] = proportional.sha256(path)
    info['selected_ids'] = [row['id'] for row in rows]
    update_meta(path, info)
    with pytest.raises(ValueError):
        proportional.read_sample(path)


@pytest.mark.parametrize('change', ['source_sha', 'official_sha', 'duplicate_manifest', 'duplicate_official',
    'incomplete_official', 'changed_question', 'wrong_label', 'wrong_video_path', 'wrong_group', 'frames_source'])
def test_source_population_official_content_labels_and_media_are_strict(source, tmp_path, change):
    manifest, official, rows, reference = source
    path = tmp_path/'sample.jsonl'
    proportional.sample(manifest, official, path)
    if change == 'source_sha':
        manifest.write_text(manifest.read_text()+'\n', encoding='utf-8')
    elif change == 'official_sha':
        official.write_text(official.read_text()+'\n', encoding='utf-8')
    elif change == 'duplicate_manifest':
        write_jsonl(rows+[rows[0]], manifest)
    elif change == 'duplicate_official':
        official.write_text(json.dumps(reference+[reference[0]]), encoding='utf-8')
    elif change == 'incomplete_official':
        official.write_text(json.dumps(reference[:-1]), encoding='utf-8')
    elif change == 'changed_question':
        rows[0]['query'] = 'changed'
        write_jsonl(rows, manifest)
    elif change == 'wrong_label':
        rows[0]['answer_index'] = (rows[0]['answer_index']+1) % len(rows[0]['options'])
        write_jsonl(rows, manifest)
    elif change == 'wrong_video_path':
        rows[0]['video_path'] = rows[1]['video_path']
        rows[0]['media_source']['path'] = rows[1]['video_path']
        write_jsonl(rows, manifest)
    elif change == 'wrong_group':
        reference[0]['duration_group'] = 60
        official.write_text(json.dumps(reference), encoding='utf-8')
    else:
        for row in rows:
            row['media_source']['kind'] = 'frames_sidecar'
        write_jsonl(rows, manifest)
    with pytest.raises(ValueError):
        proportional.read_sample(path)
    if change not in ('source_sha', 'official_sha'):
        with pytest.raises(ValueError):
            proportional.sample(manifest, official, tmp_path/'new.jsonl')


def test_no_overwrite_if_either_output_already_exists(source, tmp_path):
    path = tmp_path/'sample.jsonl'
    proportional.sample(*source[:2], path)
    previous = path.read_bytes(), path.with_suffix('.meta.json').read_bytes()
    with pytest.raises(FileExistsError):
        proportional.sample(*source[:2], path)
    assert previous == (path.read_bytes(), path.with_suffix('.meta.json').read_bytes())
    path.unlink()
    with pytest.raises(FileExistsError):
        proportional.sample(*source[:2], path)
    assert not path.exists()


@pytest.mark.parametrize('n,seed', [(0, 42), (1338, 42), (100, True)])
def test_invalid_inputs_leave_no_new_output(source, tmp_path, n, seed):
    path = tmp_path/'new/sample.jsonl'
    with pytest.raises(ValueError):
        proportional.sample(*source[:2], path, n=n, seed=seed)
    assert not path.parent.exists()


def test_cli_defaults_and_out_are_forwarded_without_executing_sampler(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(proportional, 'sample', lambda **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(sys, 'argv', ['sample_longvideobench_proportional.py', '--out', '/new/sample.jsonl'])
    proportional.main()
    assert seen == [dict(manifest='data/benchmarks/longvideobench/questions.jsonl',
        reference='data/benchmarks/longvideobench/official.json', out='/new/sample.jsonl', n=100, seed=42)]
    assert capsys.readouterr().out.strip() == '{}'
