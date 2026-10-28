"""比例抽样 Controller/五方法汇总的 CPU 回归；只用临时媒体与模型元数据。"""
from collections import Counter
import json
from pathlib import Path
import statistics

import pytest

from scripts import eval_longvideobench_controller as original_controller
from scripts import eval_longvideobench_proportional_controller as controller
from scripts import sample_longvideobench_proportional as sampler
from scripts import summarize_longvideobench_proportional as summary
from scripts import summarize_longvideobench_qwen36 as original_summary
from test_lvb_qwen36_summary import make_source
from timejev.lvb_experiment import load_sample, sha256, write_json
from timejev.user_data import write_jsonl

COUNTS = {15: 189, 60: 172, 600: 412, 3600: 564}
QUOTAS = {'15': 14, '60': 13, '600': 31, '3600': 42}


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    # 官方人口计数保留；候选和视频数用小型 mock，独立 sampler 测试负责完整官方 schema。
    monkeypatch.setattr(sampler, 'validate_reference', lambda rows: {row['id']: row for row in rows})
    source = tmp_path/'source'
    (source/'videos').mkdir(parents=True)
    rows, official = [], []
    for group, count in COUNTS.items():
        (source/'videos'/f'{group}.mp4').write_bytes(b'mock local video')
        for index in range(count):
            target = dict(id=f'{group}_{index}', video_id=str(group), video_path=f'{group}.mp4',
                duration=float(group), duration_group=group, question=f'question {group} {index}',
                candidates=['right', 'wrong'], correct_choice=0)
            official.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+str(group), video_path='videos/'+target['video_path'],
                duration=float(group), task='choice', query=target['question'],
                options=['A. right', 'B. wrong'], answer_index=0))
    write_jsonl(rows, source/'questions.jsonl')
    write_json(source/'official.json', official)
    checkpoints = {}
    for method, kind in (('qwen36', 'qwen3_5_moe'), ('videojev', 'videojev')):
        path = tmp_path/'models'/method
        path.mkdir(parents=True)
        write_json(path/'config.json', dict(model_type=kind))
        checkpoints[method] = path
    base = tmp_path/'proportional'
    for offset, seed in enumerate(summary.SEEDS):
        cell = base/f'seed{seed}'
        manifest = cell/'sample.jsonl'
        sampler.sample(source/'questions.jsonl', source/'official.json', manifest, seed=seed)
        for role, method, fps, hits in (('qwen_baseline', 'qwen36', 4., 40+10*offset),
            ('qwen_controller', 'qwen36', 2., 50+5*offset),
            ('videojev_baseline', 'videojev', 4., 75-5*offset),
            ('videojev_controller', 'videojev', 2., 72+2*offset)):
            directory = cell/role
            make_source(directory, manifest, checkpoints[method], method, fps, hits, seed=seed)
            # 原 helper 模拟旧等量协议；本轮统一源码并记录正确的视频问题组大小。
            metadata = json.loads((directory/'run.json').read_text())
            metadata['identity']['code_sha256'] = {'mock.py': ('b' if method == 'qwen36' else 'c')*64}
            write_json(directory/'run.json', metadata)
            if method == 'videojev':
                targets = load_sample(manifest)
                sizes = Counter(row['video_key'] for row in targets)
                predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
                for target, prediction in zip(targets, predictions):
                    prediction['timing']['question_count'] = sizes[target['video_key']]
                (directory/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
        controller.evaluate(manifest, cell/'qwen_controller', cell/'videojev_controller', cell/'controller', seed=seed)
        summary.summarize_single(manifest, **{role: cell/role for role in summary.ROLES}, out=cell/'summary', seed=seed)
    return base


def test_proportional_metadata_quotas_and_original_modules_unchanged(complete_run):
    before = (original_controller.load_sample, original_controller.write_json, original_controller.read_source,
              original_summary.read_sample)
    cell = complete_run/'seed42'
    metadata = json.loads((cell/'controller/run.json').read_text())
    assert metadata['schema'] == controller.SCHEMA
    assert metadata['sampling'] == sampler.SAMPLING and metadata['unit'] == 'question'
    assert metadata['sample']['quotas'] == QUOTAS
    assert metadata['sample']['source_counts'] == {str(key): value for key, value in COUNTS.items()}
    assert metadata['adapter_script_sha256'] == sha256(controller.__file__)
    assert metadata['identity']['code_sha256']['scripts/eval_longvideobench_proportional_controller.py'] == sha256(controller.__file__)
    # 省略 seed 时读取同一真实比例元数据，不能默认为 42。
    other = complete_run/'seed43'
    controller.evaluate(other/'sample.jsonl', other/'qwen_controller', other/'videojev_controller', other/'controller_inferred')
    assert json.loads((other/'controller_inferred/run.json').read_text())['seed'] == 43
    report = summary.summarize_single(cell/'sample.jsonl', **{role: cell/role for role in summary.ROLES}, seed=42)
    assert report['schema'] == summary.SINGLE_SCHEMA and report['selected_counts'] == QUOTAS
    assert report['script_sha256'] == sha256(summary.__file__)
    assert report['methods']['qwen36']['by_duration_group']['3600']['n'] == 42
    assert before == (original_controller.load_sample, original_controller.write_json, original_controller.read_source,
                      original_summary.read_sample)


def test_five_rows_variance_replay_and_read_only_validation(complete_run):
    result = summary.summarize_all(complete_run)
    assert list(result['methods']) == list(summary.METHODS)
    qwen = result['methods']['qwen36']
    assert qwen['values'] == [.4, .5, .6]
    assert qwen['mean'] == pytest.approx(.5) and qwen['sample_variance'] == pytest.approx(.01)
    random_means = [entry['random_choice']['mean'] for entry in result['by_seed']]
    assert result['methods']['random_choice']['values'] == random_means
    assert result['methods']['random_choice']['sample_variance'] == statistics.variance(random_means)
    assert result['methods']['random_choice']['sample_variance'] != result['by_seed'][0]['random_choice']['sample_variance']
    assert result['methods']['oracle_or']['values'] == [.75, .7, .65]
    assert result['methods']['oracle_or']['mean'] > result['methods']['random_choice']['mean']
    assert result['sample_variance_ddof'] == 1 and result['variance_unit'] == 'accuracy_fraction_squared'
    assert result['selected_counts'] == QUOTAS and result['training_overlap_verified'] is False
    trials_path = complete_run/'summary/random_trials.jsonl'
    trials = [json.loads(line) for line in trials_path.read_text().splitlines()]
    assert len(trials) == 150
    for seed in summary.SEEDS:
        cell = complete_run/f'seed{seed}'
        rows = load_sample(cell/'sample.jsonl')
        answers = {}
        for role in ('qwen_baseline', 'videojev_baseline'):
            predictions = {row['id']: row for row in map(json.loads, (cell/role/'predictions.jsonl').read_text().splitlines())}
            answers[role] = [predictions[row['id']]['answer_index'] for row in rows]
        for trial in [t for t in trials if t['sample_seed'] == seed]:
            import random
            rng = random.Random((seed << 32)+trial['repeat'])
            expected = [q if q == f or rng.choice([0, 1]) == 0 else f
                for q, f in zip(answers['qwen_baseline'], answers['videojev_baseline'])]
            assert trial['answer_indices'] == expected
            assert trial['correct'] == sum(answer == row['answer_index'] for answer, row in zip(expected, rows))
    hashes = {str(path): sha256(path) for path in complete_run.rglob('*') if path.is_file()}
    checked = summary.validate_all(complete_run)
    assert checked['passed'] and checked['report'] == result
    assert hashes == {str(path): sha256(path) for path in complete_run.rglob('*') if path.is_file()}
    assert sha256(trials_path) == result['random_trials_sha256']
    assert '100.0000' in (complete_run/'summary/results.md').read_text()
    with pytest.raises(FileExistsError):
        summary.summarize_all(complete_run)


def test_random_selection_blind_to_labels_and_distinct_from_oracle():
    rows = [dict(id=str(i), answer_index=0) for i in range(4)]
    qwen, fs = [0, 1, 0, 2], [1, 0, 0, 3]
    trials, stats = summary.random_trials(rows, qwen, fs, 42)
    changed, _ = summary.random_trials([dict(row, answer_index=3) for row in rows], qwen, fs, 42)
    assert [t['answer_indices'] for t in trials] == [t['answer_indices'] for t in changed]
    assert [t['choices'] for t in trials] == [t['choices'] for t in changed]
    assert all(t['choices'][2] == 'shared' for t in trials)
    assert stats['theoretical_mean'] == .5 and stats['theoretical_variance'] == 2/(4*16)
    assert stats['mean'] < .75  # OR 为3/4，但选边不能利用标签。


@pytest.mark.parametrize('kind', ['equal_schema', 'single_report', 'trial', 'units', 'controller_metadata', 'model_prediction'])
def test_tampered_sources_reports_or_random_replay_rejected(complete_run, kind):
    summary.summarize_all(complete_run)
    cell = complete_run/'seed42'
    if kind == 'equal_schema':
        path = cell/'sample.meta.json'
        data = json.loads(path.read_text()); data['sampling'] = 'equal_official_duration_groups'
    elif kind == 'single_report':
        path = cell/'summary/results.json'
        data = json.loads(path.read_text()); data['methods']['qwen36']['accuracy'] = .99
    elif kind == 'trial':
        path = complete_run/'summary/random_trials.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]['answer_indices'][0] = 99
        path.write_text(''.join(json.dumps(record)+'\n' for record in records))
    elif kind == 'units':
        path = complete_run/'summary/results.json'
        data = json.loads(path.read_text()); data['variance_unit'] = 'percentage_points_squared'
    elif kind == 'controller_metadata':
        path = cell/'controller/run.json'
        data = json.loads(path.read_text()); data['sampling'] = 'equal_official_duration_groups'
    else:
        path = cell/'qwen_baseline/predictions.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]['candidate_logits'] = [10., -10.]
        path.write_text(''.join(json.dumps(record)+'\n' for record in records))
    if kind not in ('trial', 'model_prediction'):
        write_json(path, data)
    with pytest.raises(ValueError):
        summary.validate_all(complete_run)


def test_controller_rejects_equal_sample_and_wrong_source_seed_protocol(complete_run):
    cell = complete_run/'seed42'
    manifest = cell/'sample.jsonl'
    fs_path = cell/'videojev_controller/run.json'
    metadata = json.loads(fs_path.read_text())
    for key, value in (('seed', 43), ('fps', 4.)):
        modified = json.loads(json.dumps(metadata))
        if key == 'seed':modified['identity']['seed'] = value
        else:modified['identity']['protocol'][key] = value
        write_json(fs_path, modified)
        with pytest.raises(ValueError):
            controller.evaluate(manifest, cell/'qwen_controller', cell/'videojev_controller', cell/'new_controller', seed=42)
        assert not (cell/'new_controller').exists()
    write_json(fs_path, metadata)
    meta = json.loads(manifest.with_suffix('.meta.json').read_text())
    meta['sampling'] = 'equal_official_duration_groups'
    write_json(manifest.with_suffix('.meta.json'), meta)
    with pytest.raises(ValueError):
        controller.evaluate(manifest, cell/'qwen_controller', cell/'videojev_controller', cell/'new_controller', seed=42)


def test_no_legacy_fs2_code_whitelist_and_output_protection(complete_run):
    cell = complete_run/'seed42'
    with pytest.raises(FileExistsError):
        summary.summarize_single(cell/'sample.jsonl', **{role: cell/role for role in summary.ROLES}, out=cell/'summary', seed=42)
    path = cell/'videojev_baseline/run.json'
    metadata = json.loads(path.read_text())
    metadata['identity']['code_sha256']['mock.py'] = 'd'*64
    write_json(path, metadata)
    with pytest.raises(ValueError, match='统一源码'):
        summary.summarize_single(cell/'sample.jsonl', **{role: cell/role for role in summary.ROLES}, seed=42)
