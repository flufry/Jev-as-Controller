import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from scripts import run_longvideobench_proportional as runner


def plan(tmp_path):
    return dict(cells=[dict(group=1, seed=seed, directory=str(tmp_path/f'seed{seed}'),
        manifest=str(tmp_path/f'seed{seed}/sample.jsonl')) for seed in runner.SEEDS],
        devices='0,1,2,3,4,6', videojev_devices=['7'],
        admission=dict(min_free_mib=23000, interval_seconds=.001, timeout_seconds=30))


def args(tmp_path, phase='full', check_only=False):
    return SimpleNamespace(out=tmp_path, phase=phase, check_only=check_only,
                           devices='0,1,2,3,4,6', videojev_devices='7')


def test_three_seeds_twelve_models_and_seven_cpu_jobs(tmp_path):
    frozen = plan(tmp_path)
    lanes = runner.model_lanes(frozen)
    assert {key: len(value) for key, value in lanes.items()} == dict(qwen=6, videojev=6)
    models, cpu = runner.all_jobs(tmp_path, frozen)
    assert len(models) == 12 and len(cpu) == 7
    assert len({item['name'] for item in models+cpu}) == 19
    assert len({item['directory'] for item in models+cpu}) == 19
    for prefix in ('qwen', 'videojev'):
        assert [job['role'] for job in lanes[prefix]] == [prefix+'_baseline', prefix+'_controller']*3
    for job in models:
        command = job['command']
        assert '--resume' not in command
        assert command[command.index('--context')+1] == '262144'
        assert command[command.index('--visual-token-budget')+1] == '24576'
        assert (job['fps'], job['max_frames']) == ((4, 4096) if job['role'].endswith('baseline') else (2, 2048))
        assert job['devices'] == ('0,1,2,3,4,6' if job['method'] == 'qwen36' else '7')
        if job['method'] == 'qwen36':
            assert command[command.index('--gpu-count')+1] == '6'
            assert command[command.index('--max-memory-gib')+1] == '18.5'
    fusion, single = runner.cpu_jobs(frozen['cells'][0])
    assert fusion['command'][fusion['command'].index('--qwen')+1].endswith('/qwen_controller')
    assert fusion['command'][fusion['command'].index('--videojev')+1].endswith('/videojev_controller')
    assert fusion['command'][fusion['command'].index('--seed')+1] == '42'
    assert fusion['command'][fusion['command'].index('--beta')+1] == '2'
    assert single['command'][single['command'].index('--mode')+1] == 'single'
    assert cpu[-1]['command'][-5:] == ['all', '--base', str(tmp_path), '--out', str(tmp_path/'summary')]


def test_sources_include_new_entrypoints_and_protocol_doc():
    names = {str(path.relative_to(runner.ROOT)) for path in runner.source_paths()}
    assert {f'scripts/{name}' for name in runner.NEW_SOURCES} <= names
    assert 'docs/LVB_PROPORTIONAL.md' in names
    assert 'scripts/eval_longvideobench_qwen36.py' in names


def test_admission_records_busy_then_free_and_refuses_retry(tmp_path, monkeypatch):
    (tmp_path/'resource_admission').mkdir()
    settings = dict(min_free_mib=23000, interval_seconds=.001, timeout_seconds=30)
    values = iter([{'7': 1000}, {'7': 24000}])
    monkeypatch.setattr(runner, 'free_memory', lambda visible: next(values))
    job = dict(name='seed42_videojev_baseline', devices='7')
    result = runner.admission(tmp_path, job, threading.Event(), settings)
    assert result['status'] == 'admitted'
    assert [sample['free_mib']['7'] for sample in result['samples']] == [1000, 24000]
    assert runner.read(tmp_path/'resource_admission/seed42_videojev_baseline.json') == result
    with pytest.raises(ValueError, match='禁止重试'):
        runner.admission(tmp_path, job, threading.Event(), settings)


@pytest.mark.parametrize('cancelled', [False, True])
def test_admission_timeout_or_cancel_preserves_failure(tmp_path, monkeypatch, cancelled):
    (tmp_path/'resource_admission').mkdir()
    stopped = threading.Event()
    if cancelled:
        stopped.set()
    monkeypatch.setattr(runner, 'free_memory', lambda visible: {'7': 100})
    with pytest.raises(ValueError, match='队列已停止|等待超时'):
        runner.admission(tmp_path, dict(name='blocked', devices='7'), stopped,
                         dict(min_free_mib=23000, interval_seconds=0, timeout_seconds=0))
    assert runner.read(tmp_path/'resource_admission/blocked.json')['status'] == 'failed'
    assert not (tmp_path/'steps').exists()


def test_free_memory_queries_only_requested_cards_and_rejects_missing(monkeypatch):
    seen = []
    def query(command, **kwargs):
        seen.append(command)
        return '0, 24000\n6, 23001\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', query)
    assert runner.free_memory('0,6') == {'0': 24000, '6': 23001}
    assert '--id=0,6' in seen[0]
    with pytest.raises(ValueError, match='未覆盖'):
        runner.free_memory('0,1')


@pytest.mark.parametrize('instants', [[100, 100, 106], [100, 100, 104, 105]])
def test_query_returns_free_after_deadline_never_starts_model(tmp_path, monkeypatch, instants):
    (tmp_path/'resource_admission').mkdir()
    frozen = plan(tmp_path)
    frozen['admission']['timeout_seconds'] = 5
    job = runner.model_lanes(frozen)['videojev'][0]
    monkeypatch.setattr(runner, 'model_lanes', lambda _: {'videojev': [job]})
    times = iter(instants)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(runner, 'free_memory', lambda _: {'7': 24000})
    monkeypatch.setattr(runner.core, 'run_step', lambda *a: pytest.fail('超过 deadline 不能启动模型'))
    stopped = threading.Event()
    with pytest.raises(ValueError, match='等待超时'):
        runner.run_lanes(tmp_path, frozen, stopped, lambda job: pytest.fail('不能标记完成'))
    gate = runner.read(tmp_path/'resource_admission'/f'{job["name"]}.json')
    assert stopped.is_set() and gate['status'] == 'failed'
    assert gate['samples'][-1]['free_mib'] == {'7': 24000}
    assert not (tmp_path/'steps').exists()


def test_each_lane_is_serial_and_two_lanes_run_concurrently(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    (tmp_path/'steps').mkdir()
    active, maximum, order = {}, {}, {}
    barrier, lock = threading.Barrier(2), threading.Lock()
    def step(master, job, phase, stopped):
        with lock:
            key = job['devices']
            active[key] = active.get(key, 0)+1
            maximum[key] = max(maximum.get(key, 0), active[key])
            order.setdefault(key, []).append(job['role'])
        if job['seed'] == 42 and job['role'].endswith('baseline'):
            barrier.wait(timeout=5)
        time.sleep(.001)
        with lock:
            active[key] -= 1
        runner.write(master/'steps'/f'{job["name"]}.json', dict(status='completed'))
    monkeypatch.setattr(runner, 'admission', lambda *a: None)
    monkeypatch.setattr(runner.core, 'run_step', step)
    monkeypatch.setattr(runner.core, 'verify_model', lambda job: {})
    completed = []
    runner.run_lanes(tmp_path, frozen, threading.Event(), lambda job: completed.append(job['name']))
    assert len(completed) == 12 and maximum == {'0,1,2,3,4,6': 1, '7': 1}
    assert order['7'] == ['videojev_baseline', 'videojev_controller']*3
    assert all(runner.read(path)['output_validation']['passed'] for path in (tmp_path/'steps').glob('*.json'))


def test_model_validation_failure_stops_peer_without_retry(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    (tmp_path/'steps').mkdir()
    monkeypatch.setattr(runner, 'admission', lambda *a: None)
    monkeypatch.setattr(runner.core, 'run_step', lambda master, job, *_:
                        runner.write(master/'steps'/f'{job["name"]}.json', dict(exit_code=0)))
    def fail(job):
        raise ValueError('bad output')
    monkeypatch.setattr(runner.core, 'verify_model', fail)
    stopped = threading.Event()
    with pytest.raises(ValueError, match='队列失败'):
        runner.run_lanes(tmp_path, frozen, stopped, lambda job: pytest.fail('不能标为成功'))
    assert stopped.is_set()
    assert len(list((tmp_path/'steps').glob('*.json'))) <= 2
    assert all(not runner.read(path)['output_validation']['passed'] for path in (tmp_path/'steps').glob('*.json'))


def test_prepare_refuses_existing_state_before_inspection(tmp_path, monkeypatch):
    (tmp_path/'source').mkdir()
    monkeypatch.setattr(runner, 'inspect_inputs', lambda *a: pytest.fail('应先拒绝覆盖'))
    with pytest.raises(ValueError, match='拒绝覆盖'):
        runner.prepare(args(tmp_path, 'prepare'))


def test_full_check_only_has_no_gpu_query_subprocess_or_writes(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'check_environment', lambda *_: None)
    monkeypatch.setattr(runner, 'free_memory', lambda *_: pytest.fail('check-only 不查询 GPU'))
    monkeypatch.setattr(runner.core, 'run_step', lambda *_: pytest.fail('check-only 不启动任务'))
    runner.write(tmp_path/'experiment.json', dict(status='prepared'))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    result = runner.execute(args(tmp_path, check_only=True))
    assert result['model_steps'] == 12 and result['cpu_steps'] == 7
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert not (tmp_path/'queue_execution.json').exists()
    assert not (tmp_path/'steps').exists()


def test_full_failure_records_failed_state_without_fabricated_full_exitcode(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'check_environment', lambda *_: None)
    (tmp_path/'run_plan.json').write_text('{}\n')
    runner.write(tmp_path/'experiment.json', dict(status='prepared'))
    def fail(*a):
        raise ValueError('actual child failure')
    monkeypatch.setattr(runner, 'run_lanes', fail)
    with pytest.raises(ValueError, match='actual child failure'):
        runner.execute(args(tmp_path))
    assert runner.read(tmp_path/'queue_execution.json')['status'] == 'failed'
    assert runner.read(tmp_path/'experiment.json')['status'] == 'failed'
    assert not (tmp_path/'full.exitcode').exists()
    assert not (tmp_path/'validation.json').exists()
    with pytest.raises(ValueError, match='已有执行现场|禁止自动重试'):
        runner.execute(args(tmp_path))


def test_frozen_proportional_plan_rejects_equal_or_wrong_seed(tmp_path, monkeypatch):
    frozen = dict(plan(tmp_path), schema='lvb_proportional_runner_v1', sampling=runner.SAMPLING,
        seeds=[42, 43, 44], n_per_seed=100, model_steps=12, cpu_steps=7,
        new_gpu_pilot_runs=0, reused_pilot_master=str(runner.PRIOR))
    for cell in frozen['cells']:
        cell['manifest_sha256'] = 'good'
    monkeypatch.setattr(runner.core, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'sample_reader', lambda: lambda *a, **kw: ([], dict(sha256='good')))
    assert runner.validate_frozen(tmp_path) == frozen
    frozen['sampling'] = 'equal_official_duration_groups'
    with pytest.raises(ValueError, match='比例方案'):
        runner.validate_frozen(tmp_path)
    frozen['sampling'] = runner.SAMPLING
    frozen['cells'][1]['seed'] = 42
    with pytest.raises(ValueError, match='seed'):
        runner.validate_frozen(tmp_path)


def final_state(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'check_environment', lambda *_: None)
    checked = dict(passed=True, model_steps=12, cpu_steps=7, step_sha256={},
                   summary_source_hashes={}, report_sha256='report')
    monkeypatch.setattr(runner, 'validate_outputs', lambda *_: checked)
    runner.write(tmp_path/'run_plan.json', {})
    runner.write(tmp_path/'validation.json', dict(checked, run_plan_sha256=runner.digest(tmp_path/'run_plan.json')))
    runner.write(tmp_path/'queue_execution.json', dict(status='completed', completed_model_steps=12, completed_cpu_steps=7))
    runner.write(tmp_path/'experiment.json', dict(status='awaiting_finalization'))
    (tmp_path/'full.exitcode').write_text('0\n')
    (tmp_path/'monitor').mkdir()
    runner.write(tmp_path/'monitor/status.json', dict(monitor_status='completed'))
    return checked


def test_finalize_needs_real_full_exit_and_monitor_then_check_only_is_read_only(tmp_path, monkeypatch):
    final_state(tmp_path, monkeypatch)
    (tmp_path/'full.exitcode').write_text('17\n')
    with pytest.raises(ValueError, match='退出码'):
        runner.finalize(args(tmp_path, 'finalize'))
    (tmp_path/'full.exitcode').write_text('0\n')
    runner.write(tmp_path/'monitor/status.json', dict(monitor_status='running'))
    with pytest.raises(ValueError, match='监控尚未完成'):
        runner.finalize(args(tmp_path, 'finalize'))
    runner.write(tmp_path/'monitor/status.json', dict(monitor_status='completed'))
    before = runner.artifact_files(tmp_path)
    result = runner.finalize(args(tmp_path, 'finalize', True))
    assert result['sealed'] is False and runner.artifact_files(tmp_path) == before
    assert not (tmp_path/'artifact_manifest.json').exists()


def test_finalization_seal_rejects_content_tamper_and_new_files(tmp_path, monkeypatch):
    final_state(tmp_path, monkeypatch)
    result = runner.finalize(args(tmp_path, 'finalize'))
    assert result['completed'] and runner.read(tmp_path/'experiment.json')['status'] == 'completed'
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    checked = runner.finalize(args(tmp_path, 'finalize', True))
    assert checked['sealed'] and before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='已有封存'):
        runner.finalize(args(tmp_path, 'finalize'))
    (tmp_path/'extra.txt').write_text('late write')
    with pytest.raises(ValueError, match='覆盖范围'):
        runner.finalize(args(tmp_path, 'finalize', True))
    (tmp_path/'extra.txt').unlink()
    (tmp_path/'finalization.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA'):
        runner.finalize(args(tmp_path, 'finalize', True))


def test_atomic_state_update_preserves_json_and_has_no_temporary_file(tmp_path):
    path = tmp_path/'state.json'
    runner.write(path, dict(state=1), new=True)
    runner.write(path, dict(state=2))
    assert json.loads(path.read_text()) == dict(state=2)
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(FileExistsError):
        runner.write(path, dict(state=3), new=True)


def complete_steps(tmp_path, monkeypatch):
    frozen = plan(tmp_path)
    for name in ('steps', 'logs', 'resource_admission', 'summary'):
        (tmp_path/name).mkdir()
    models, cpu = runner.all_jobs(tmp_path, frozen)
    for job in models+cpu:
        log = tmp_path/'logs'/f'{job["name"]}.log'
        log.write_text('actual child output\n')
        env = runner.core.environment(job['devices'])
        runner.write(tmp_path/'steps'/f'{job["name"]}.json', dict(status='completed', exit_code=0,
            command=job['command'], cuda_visible_devices=job['devices'],
            spawn_command=runner.core.shell_command(job['command']), pid=321, process_group=321,
            lane=job['lane'], phase='full', log=str(log), log_sha256=runner.digest(log),
            environment={key: env[key] for key in runner.core.ENV_KEYS},
            output_validation=dict(passed=True)))
        (tmp_path/'steps'/f'{job["name"]}.exitcode').write_text('0\n')
        if job in models:
            runner.write(tmp_path/'resource_admission'/f'{job["name"]}.json', dict(
                status='admitted', devices=job['devices'], settings=frozen['admission'],
                samples=[dict(free_mib={index: 24000 for index in job['devices'].split(',')})]))
    (tmp_path/'summary/results.json').write_text('{}\n')
    monkeypatch.setattr(runner.core, 'verify_model', lambda job: {})
    monkeypatch.setattr(runner, 'summary_validation', lambda master: dict(passed=True, source_hashes={'test': 'hash'}))
    return frozen, models, cpu


@pytest.mark.parametrize('failure', ['exitcode', 'log', 'spawn', 'extra_step', 'gate', 'environment'])
def test_output_validation_requires_real_step_evidence_and_admission(tmp_path, monkeypatch, failure):
    frozen, models, cpu = complete_steps(tmp_path, monkeypatch)
    checked = runner.validate_outputs(tmp_path, frozen)
    assert checked['model_steps'] == 12 and checked['cpu_steps'] == 7
    assert len(checked['step_sha256']) == 19
    job = models[0]
    if failure == 'exitcode':
        (tmp_path/'steps'/f'{job["name"]}.exitcode').write_text('17\n')
    elif failure == 'log':
        (tmp_path/'logs'/f'{job["name"]}.log').write_text('changed log\n')
    elif failure == 'extra_step':
        (tmp_path/'steps/failed_unexpected.json').write_text('{}\n')
    elif failure == 'gate':
        path = tmp_path/'resource_admission'/f'{job["name"]}.json'
        value = runner.read(path)
        value['samples'][-1]['free_mib']['0'] = 1000
        runner.write(path, value)
    else:
        path = tmp_path/'steps'/f'{job["name"]}.json'
        value = runner.read(path)
        if failure == 'spawn':
            value['spawn_command'] = ['false']
        else:
            value['environment']['HTTPS_PROXY'] = ''
        runner.write(path, value)
    with pytest.raises(ValueError):
        runner.validate_outputs(tmp_path, frozen)
