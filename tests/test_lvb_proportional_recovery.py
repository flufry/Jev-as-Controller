from pathlib import Path
import subprocess
import threading

import pytest

from scripts import recover_longvideobench_proportional as recovery

runner = recovery.runner


def hashes(base):
    return {str(path): runner.digest(path) for path in base.rglob('*') if path.is_file()}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    plan = dict(cells=[dict(group=1, seed=seed, directory=str(tmp_path/f'seed{seed}'),
        manifest=str(tmp_path/f'seed{seed}/sample.jsonl')) for seed in (42, 43, 44)],
        devices='0,1,2,3,4,6', videojev_devices=['7'],
        admission=dict(min_free_mib=23000, interval_seconds=60, timeout_seconds=21600))
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: plan)
    monkeypatch.setattr(runner, 'check_environment', lambda *a: None)
    monkeypatch.setattr(runner.core, 'verify_model', lambda job: dict(metrics=dict(n=100)))
    runner.write(tmp_path/'run_plan.json', plan)
    for name in ('steps', 'logs', 'resource_admission'):
        (tmp_path/name).mkdir()
    models, cpu = runner.all_jobs(tmp_path, plan)
    reused = next(job for job in models if job['name'] in recovery.REUSED)
    directory = Path(reused['directory'])
    (directory/'source').mkdir(parents=True)
    (directory/'source/model.py').write_text('# original source snapshot\n')
    (directory/'predictions.jsonl').write_text('complete mock predictions\n')
    runner.write(directory/'metrics.json', dict(n=100))
    log = tmp_path/'logs'/f'{reused["name"]}.log'
    log.write_text('original successful log\n')
    env = runner.core.environment(reused['devices'])
    runner.write(tmp_path/'steps'/f'{reused["name"]}.json', dict(name=reused['name'], status='completed',
        exit_code=0, phase='full', command=reused['command'], lane=reused['lane'], pid=123, process_group=123,
        cuda_visible_devices=reused['devices'], spawn_command=runner.core.shell_command(reused['command']),
        environment={key: env[key] for key in runner.core.ENV_KEYS}, log=str(log), log_sha256=runner.digest(log),
        output_validation=dict(passed=True), started_at='original timestamp'))
    (tmp_path/'steps'/f'{reused["name"]}.exitcode').write_text('0\n')
    runner.write(tmp_path/'resource_admission'/f'{reused["name"]}.json', dict(status='admitted',
        devices='7', settings=plan['admission'], samples=[dict(free_mib={'7': 24000})]))
    retained = {path: runner.digest(path) for path in recovery.retained_paths(tmp_path, reused)}
    archive = tmp_path/'failed_attempts/resource_contention1'
    archive.mkdir(parents=True)
    runner.write(archive/'experiment.json', dict(status='failed', error='old OOM', schema='original'))
    runner.write(archive/'queue_execution.json', dict(status='failed', completed_model_steps=1))
    manifest = archive/'manifest.json'
    runner.write(manifest, dict(schema='lvb_proportional_failed_attempt_archive_v1', files={str(path.relative_to(archive)):
        dict(bytes=path.stat().st_size, sha256=runner.digest(path)) for path in archive.rglob('*') if path.is_file()}))
    extra = tmp_path/'recovery_launcher.sh'
    extra.write_text('# frozen recovery launcher\n')
    proof = dict(schema='lvb_proportional_recovery_v1', passed=True, reused_model_steps=sorted(recovery.REUSED),
        retained_files=retained, archive=dict(path=str(archive), manifest_path=str(manifest),
        manifest_sha256=runner.digest(manifest)), driver=dict(path=str(Path(recovery.__file__).resolve()),
        sha256=runner.digest(recovery.__file__)), additional_sources={str(extra): runner.digest(extra)},
        required_absent=[str(tmp_path/name) for name in ('queue_execution.json', 'experiment.json', 'full.exitcode')])
    runner.write(tmp_path/'recovery_preflight.json', proof)
    return tmp_path, plan, models, cpu, proof


def test_check_only_is_read_only_without_gpu_or_launch_and_allows_wrapper_log(ready, monkeypatch):
    base, _, _, _, _ = ready
    (base/'full.log').write_text('wrapper already opened\n')
    runner.write(base/'launch_record.json', dict(wrapper_pid=100))
    before = hashes(base)
    monkeypatch.setattr(recovery, 'resource_snapshot', lambda *a: pytest.fail('不能查询 GPU'))
    monkeypatch.setattr(runner.core, 'run_step', lambda *a: pytest.fail('不能启动进程'))
    result = recovery.execute(base, check_only=True)
    assert result['reused_model_steps'] == 1 and result['new_model_steps'] == 11 and result['cpu_steps'] == 7
    assert result['resource_query_executed'] is False and before == hashes(base)


def test_missing_preflight_is_read_only(tmp_path):
    with pytest.raises(ValueError, match='请先由 root 原样归档'):
        recovery.execute(tmp_path, check_only=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['retained', 'archive', 'extra_archive', 'missing_retained', 'driver', 'additional'])
def test_hash_and_full_coverage_guards_reject_changed_evidence(ready, change):
    base, _, _, _, proof = ready
    if change == 'retained':
        Path(next(iter(proof['retained_files']))).write_text('changed')
    elif change == 'archive':
        (Path(proof['archive']['path'])/'queue_execution.json').write_text('changed')
    elif change == 'extra_archive':
        (Path(proof['archive']['path'])/'unexpected').write_text('new')
    elif change == 'missing_retained':
        proof['retained_files'].pop(next(path for path in proof['retained_files'] if '/source/' in path))
    elif change == 'driver':
        proof['driver']['sha256'] = '0'*64
    else:
        Path(next(iter(proof['additional_sources']))).write_text('changed source')
    runner.write(base/'recovery_preflight.json', proof)
    with pytest.raises(ValueError):
        recovery.execute(base, check_only=True)


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('output_validation', dict(passed=False)),
    ('command', ['false']), ('spawn_command', ['false']), ('process_group', 999), ('pid', None)])
def test_reused_step_requires_original_valid_success(ready, field, value):
    base, _, models, _, proof = ready
    reused = next(job for job in models if job['name'] in recovery.REUSED)
    path = base/'steps'/f'{reused["name"]}.json'
    step = runner.read(path)
    step[field] = value
    runner.write(path, step)
    proof['retained_files'][str(path)] = runner.digest(path)
    runner.write(base/'recovery_preflight.json', proof)
    with pytest.raises(ValueError, match='复用步骤'):
        recovery.execute(base, check_only=True)


def test_reused_model_requires_all_100_questions(ready, monkeypatch):
    monkeypatch.setattr(runner.core, 'verify_model', lambda job: dict(metrics=dict(n=99)))
    with pytest.raises(ValueError, match='完整 100 题'):
        recovery.execute(ready[0], check_only=True)


@pytest.mark.parametrize('name', ['queue_execution.json', 'experiment.json', 'full.exitcode',
    'resource_admission/seed42_videojev_controller.json', 'seed42/videojev_controller/partial.json'])
def test_unarchived_or_new_attempt_never_overwritten(ready, name):
    path = ready[0]/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('old or new attempt\n')
    before = hashes(ready[0])
    with pytest.raises(ValueError):
        recovery.execute(ready[0], check_only=True)
    assert hashes(ready[0]) == before


def test_resource_query_limits_gpu_set_and_records_compute_pids(monkeypatch):
    calls = []
    outputs = iter(['0, GPU-zero, 24000\n6, GPU-six, 23999\n', 'GPU-six, 1234\n'])
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs))
    monkeypatch.setattr(recovery.subprocess, 'run', query)
    assert recovery.resource_snapshot('0,6') == dict(free_mib={'0': 24000, '6': 23999},
                                                    compute_pids={'0': [], '6': [1234]})
    assert all('--id=0,6' in command and kwargs['timeout'] == 30 for command, kwargs in calls)
    assert all(kwargs['env']['HF_HUB_OFFLINE'] == '1' for _, kwargs in calls)


def clock_gate(tmp_path, monkeypatch):
    (tmp_path/'recovery_stability').mkdir()
    clock = [0.]
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    stopped = threading.Event()
    monkeypatch.setattr(stopped, 'wait', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    return dict(name='new_model', devices='7'), stopped, clock


def test_stability_needs_full_60_seconds_and_resets_on_busy_pid(tmp_path, monkeypatch):
    job, stopped, clock = clock_gate(tmp_path, monkeypatch)
    def snapshot(_):
        return dict(free_mib={'7': 24000}, compute_pids={'7': [100] if clock[0] == 20 else []})
    monkeypatch.setattr(recovery, 'resource_snapshot', snapshot)
    recovery.stable_admission(tmp_path, job, stopped)
    saved = runner.read(tmp_path/'recovery_stability/new_model.json')
    assert clock[0] == 90 and saved['status'] == 'admitted'
    assert saved['samples'][-1]['stable_seconds'] == 60
    assert saved['samples'][2]['qualifying'] is False


def test_stability_busy_then_timeout_never_admits(tmp_path, monkeypatch):
    job, stopped, clock = clock_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'TIMEOUT_SECONDS', 25)
    monkeypatch.setattr(recovery, 'resource_snapshot', lambda _: dict(free_mib={'7': 100}, compute_pids={'7': []}))
    with pytest.raises(ValueError, match='六小时'):
        recovery.stable_admission(tmp_path, job, stopped)
    assert clock[0] == 25
    assert runner.read(tmp_path/'recovery_stability/new_model.json')['status'] == 'failed'


def test_stability_query_crossing_deadline_cannot_admit(tmp_path, monkeypatch):
    job, stopped, clock = clock_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'TIMEOUT_SECONDS', 65)
    def snapshot(_):
        if clock[0] == 60:
            clock[0] = 65
        return dict(free_mib={'7': 24000}, compute_pids={'7': []})
    monkeypatch.setattr(recovery, 'resource_snapshot', snapshot)
    with pytest.raises(ValueError, match='六小时'):
        recovery.stable_admission(tmp_path, job, stopped)
    assert runner.read(tmp_path/'recovery_stability/new_model.json')['status'] == 'failed'


def test_stability_cancelled_does_not_query_or_signal_external_process(tmp_path, monkeypatch):
    job, stopped, _ = clock_gate(tmp_path, monkeypatch)
    stopped.set()
    monkeypatch.setattr(recovery, 'resource_snapshot', lambda _: pytest.fail('取消后不能查询'))
    monkeypatch.setattr(recovery.os, 'killpg', lambda *a: pytest.fail('不能发送信号'))
    with pytest.raises(ValueError, match='取消稳定门禁'):
        recovery.stable_admission(tmp_path, job, stopped)
    assert runner.read(tmp_path/'recovery_stability/new_model.json')['status'] == 'cancelled'


def test_recovery_runs_only_11_new_models_and_7_cpu_preserving_success_bytes(ready, monkeypatch):
    base, _, _, _, proof = ready
    original = dict(proof['retained_files'])
    launched, stability = [], []
    monkeypatch.setattr(recovery, 'stable_admission', lambda master, job, *_: stability.append(job['name']))
    monkeypatch.setattr(runner, 'admission', lambda *a: None)
    def step(master, job, *_):
        assert job['name'] not in recovery.REUSED
        launched.append(job['name'])
        runner.write(master/'steps'/f'{job["name"]}.json', dict(status='completed', exit_code=0))
    monkeypatch.setattr(runner.core, 'run_step', step)
    monkeypatch.setattr(runner, 'validate_outputs', lambda *a: dict(passed=True, model_steps=12, cpu_steps=7,
        step_sha256={}, summary_source_hashes={}, report_sha256='mock'))
    result = recovery.execute(base)
    assert result['reused_model_steps'] == 1 and result['new_model_steps'] == 11 and result['cpu_steps'] == 7
    assert len(launched) == 18 and len(stability) == 11
    assert {path: runner.digest(path) for path in original} == original
    state = runner.read(base/'queue_execution.json')
    assert state['completed_model_steps'] == 12 and state['completed_cpu_steps'] == 7
    assert len(state['completed_steps']) == 19 and state['status'] == 'completed'
    assert runner.read(base/'experiment.json')['status'] == 'awaiting_finalization'
    assert runner.read(base/'validation.json')['passed'] is True
    assert not (base/'full.exitcode').exists()


def test_real_model_failure_stops_without_cpu_or_automatic_retry(ready, monkeypatch):
    base, _, _, _, proof = ready
    original = dict(proof['retained_files'])
    monkeypatch.setattr(recovery, 'stable_admission', lambda *a: None)
    monkeypatch.setattr(runner, 'admission', lambda *a: None)
    calls = []
    def fail(master, job, *_):
        calls.append(job['name'])
        raise ValueError('actual model exit1')
    monkeypatch.setattr(runner.core, 'run_step', fail)
    with pytest.raises(ValueError, match='actual model exit1'):
        recovery.execute(base)
    assert len(calls) <= 2 and not any(name.endswith('_summary') for name in calls)
    assert runner.read(base/'queue_execution.json')['status'] == 'failed'
    assert runner.read(base/'experiment.json')['status'] == 'failed'
    assert not (base/'full.exitcode').exists() and not (base/'validation.json').exists()
    assert {path: runner.digest(path) for path in original} == original
    with pytest.raises(ValueError, match='拒绝覆盖|旧尝试未归档'):
        recovery.execute(base)


def test_finalize_checks_recovery_sources_then_preserves_original_validator(ready, monkeypatch):
    base = ready[0]
    runner.write(base/'validation.json', dict(recovery_preflight_sha256=runner.digest(base/'recovery_preflight.json')))
    seen = []
    monkeypatch.setattr(runner, 'finalize', lambda args: seen.append(args) or dict(checked=True))
    before = hashes(base)
    assert recovery.finalize(base, check_only=True) == dict(checked=True)
    assert before == hashes(base) and len(seen) == 1
    assert seen[0].out == base and seen[0].devices == '0,1,2,3,4,6' and seen[0].videojev_devices == '7'
    assert seen[0].phase == 'finalize' and seen[0].check_only is True


@pytest.mark.parametrize('change', ['preflight', 'additional', 'archive'])
def test_finalize_rejects_recovery_provenance_drift_before_sealing(ready, monkeypatch, change):
    base, _, _, _, proof = ready
    runner.write(base/'validation.json', dict(recovery_preflight_sha256=runner.digest(base/'recovery_preflight.json')))
    if change == 'preflight':
        proof['passed'] = False
        runner.write(base/'recovery_preflight.json', proof)
    elif change == 'additional':
        Path(next(iter(proof['additional_sources']))).write_text('new launcher source')
    else:
        (Path(proof['archive']['path'])/'experiment.json').write_text('changed failed experiment')
    monkeypatch.setattr(runner, 'finalize', lambda args: pytest.fail('来源变化不能调用封存'))
    with pytest.raises(ValueError):
        recovery.finalize(base)
