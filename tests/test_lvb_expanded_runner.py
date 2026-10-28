import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import pytest

from scripts import run_longvideobench_expanded_groups as runner


def plan(tmp_path):
    cells = [dict(group=group, seed=seed, directory=str(tmp_path/f'group{group}/seed{seed}'),
        manifest=str(tmp_path/f'group{group}/seed{seed}/sample.jsonl')) for group in (2, 3) for seed in (42, 43, 44)]
    return dict(cells=cells, devices='0,1,2,3,4,6', videojev_devices=['5', '7'])


def job(tmp_path, name='example'):
    return dict(name=name, lane='qwen', devices='0,1,2,3,4,6',
                directory=str(tmp_path/'outputs'/name), command=['python', 'mock.py', name])


def master(tmp_path, monkeypatch):
    for name in ('steps', 'logs'):
        (tmp_path/name).mkdir()
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: None)
    return tmp_path


def test_six_cells_four_protocols_and_three_nonoverlapping_lanes(tmp_path):
    frozen = plan(tmp_path)
    lanes = runner.model_lanes(frozen['cells'], frozen)
    assert {name: len(jobs) for name, jobs in lanes.items()} == dict(qwen=12, videojev4=6, videojev2=6)
    jobs = [item for lane in lanes.values() for item in lane]
    assert len({item['name'] for item in jobs}) == len({item['directory'] for item in jobs}) == 24
    assert all('--resume' not in item['command'] for item in jobs)
    assert [item['role'] for item in lanes['qwen']] == ['qwen_baseline', 'qwen_controller']*6
    for name, visible in [('qwen', '0,1,2,3,4,6'), ('videojev4', '5'), ('videojev2', '7')]:
        assert all(item['devices'] == visible for item in lanes[name])
    for item in jobs:
        cmd = item['command']
        assert 'group1' not in item['manifest']
        assert cmd[cmd.index('--device')+1] == 'cuda:0'
        assert cmd[cmd.index('--context')+1] == '262144'
        assert cmd[cmd.index('--visual-token-budget')+1] == '24576'
        assert (item['fps'], item['max_frames']) == ((4, 4096) if item['role'].endswith('baseline') else (2, 2048))
        if item['method'] == 'qwen36':
            assert cmd[cmd.index('--gpu-count')+1] == '6'


def test_single_videojev_card_keeps_protocol_commands_and_serial_order(tmp_path):
    dual = plan(tmp_path)
    single = dict(dual, videojev_devices=['7'])
    lanes = runner.model_lanes(single['cells'], single)
    assert {name: len(jobs) for name, jobs in lanes.items()} == dict(qwen=12, videojev=12)
    assert [item['role'] for item in lanes['videojev']] == ['videojev_baseline', 'videojev_controller']*6
    assert all(item['devices'] == '7' for item in lanes['videojev'])
    assert not set(single['devices'].split(',')) & set(single['videojev_devices'])
    old = {item['name']: item for lane in runner.model_lanes(dual['cells'], dual).values() for item in lane}
    for lane in lanes.values():
        for item in lane:
            assert {k: v for k, v in item.items() if k not in ('lane', 'devices')} == {
                k: v for k, v in old[item['name']].items() if k not in ('lane', 'devices')}


def test_single_videojev_lane_does_not_overlap_itself_but_runs_with_qwen(tmp_path, monkeypatch):
    frozen = dict(plan(tmp_path), videojev_devices=['7'])
    lanes = runner.model_lanes(frozen['cells'], frozen)
    active, maximum, seen = {}, {}, []
    barrier, lock = threading.Barrier(2), threading.Lock()
    def step(master, job, phase, stopped):
        with lock:
            active[job['devices']] = active.get(job['devices'], 0)+1
            maximum[job['devices']] = max(maximum.get(job['devices'], 0), active[job['devices']])
            seen.append(job['name'])
        if job['name'].startswith('group2_seed42') and job['role'].endswith('baseline'):
            barrier.wait(timeout=5)
        time.sleep(.002)
        with lock:
            active[job['devices']] -= 1
    monkeypatch.setattr(runner, 'run_step', step)
    monkeypatch.setattr(runner, 'verify_model', lambda _: {})
    runner.run_lanes(tmp_path, lanes, 'pilot', threading.Event())
    assert len(seen) == 24 and maximum == {'0,1,2,3,4,6': 1, '7': 1}
    assert not any(active.values())


@pytest.mark.parametrize('value', ['7', '5,7'])
def test_videojev_device_parser_accepts_single_or_dual(value):
    assert runner.devices(value, (1, 2)) == value.split(',')


@pytest.mark.parametrize('value', ['', '7,7', '5,7,8'])
def test_videojev_device_parser_rejects_invalid_card_sets(value):
    with pytest.raises(ValueError):
        runner.devices(value, (1, 2))


@pytest.mark.parametrize('videojev', ['6', '5,6'])
def test_qwen_videojev_overlap_rejected_before_model_checks(tmp_path, monkeypatch, videojev):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, 'old_preflight', lambda: pytest.fail('设备冲突时不应检查模型'))
    args = SimpleNamespace(out=tmp_path, devices='0,1,2,3,4,6', videojev_devices=videojev)
    with pytest.raises(ValueError, match='不得共享'):
        runner.inspect_inputs(args)


@pytest.mark.parametrize('videojev', [['7'], ['5', '7']])
def test_pilot_check_only_binds_single_or_dual_devices_to_frozen_plan(tmp_path, monkeypatch, videojev):
    from types import SimpleNamespace
    frozen = dict(plan(tmp_path), videojev_devices=videojev)
    frozen['pilot'] = dict(group=0, seed=42, directory=str(tmp_path/'pilot'), manifest=str(tmp_path/'pilot/sample.jsonl'))
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'old_preflight', lambda: dict(qwen_files={}))
    monkeypatch.setattr(runner, 'packages', lambda: {})
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *_a, **_k: pytest.fail('check-only不可启动子进程'))
    runner.write(tmp_path/'preflight.json', dict(qwen_files={}))
    runner.write(tmp_path/'launch_environment.json', dict(python=runner.sys.executable, packages={}))
    args = SimpleNamespace(out=tmp_path, phase='pilot', devices=frozen['devices'],
                           videojev_devices=','.join(videojev), check_only=True)
    result = runner.execute(args)
    assert result['videojev_devices'] == videojev and result['model_steps'] == 4
    assert not (tmp_path/'pilot/queue_execution.json').exists()
    args.videojev_devices = '5,7' if len(videojev) == 1 else '7'
    with pytest.raises(ValueError, match='选卡与prepare冻结计划不同'):
        runner.execute(args)


@pytest.mark.parametrize('value,count', [('0,0', 2), ('0,a', 2), ('0,1,2', 2)])
def test_invalid_devices_rejected(value, count):
    with pytest.raises(ValueError):
        runner.devices(value, count)


def test_cpu_fusion_has_two_fps_sources_beta2_and_explicit_qwen36(tmp_path):
    fusion, summary = runner.cpu_jobs(plan(tmp_path)['cells'][0])
    assert fusion['devices'] == summary['devices'] == ''
    command = fusion['command']
    assert command[command.index('--qwen')+1].endswith('/qwen_controller')
    assert command[command.index('--videojev')+1].endswith('/videojev_controller')
    assert command[command.index('--beta')+1] == '2'
    assert command[command.index('--qwen-method')+1] == 'qwen36'
    assert summary['command'][summary['command'].index('--videojev-baseline')+1].endswith('/videojev_baseline')


def test_child_environment_and_shell_preserve_arguments(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '99')
    env = runner.environment('7')
    assert env['CUDA_VISIBLE_DEVICES'] == '7'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert env['FLA_INTRACARD_CP'] == '0'
    assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '2'
    assert all(env[key] == 'http://127.0.0.1:7890' for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'))
    command = ['python', '/space in path/script.py', '--value', 'literal;$(not-a-command)']
    spawned = runner.shell_command(command)
    assert spawned[:2] == ['bash', '-c'] and spawned[-len(command):] == command
    assert 'exec "$@"' in spawned[2] and 'source "$1"' in spawned[2]


def test_step_records_actual_pid_exit_environment_and_sha(tmp_path, monkeypatch):
    master(tmp_path, monkeypatch)
    seen = {}

    class Process:
        pid = 321
        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        seen.update(command=command, **kwargs)
        kwargs['stdout'].write(b'mock child output\n')
        return Process()

    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    item = runner.run_step(tmp_path, job(tmp_path), 'pilot', threading.Event())
    assert item['pid'] == 321 and item['exit_code'] == 0 and item['status'] == 'completed'
    assert item['log_sha256'] == runner.digest(tmp_path/'logs/example.log')
    assert (tmp_path/'steps/example.exitcode').read_text() == '0\n'
    assert seen['start_new_session'] is True
    assert seen['env']['CUDA_VISIBLE_DEVICES'] == item['cuda_visible_devices']
    assert item['environment']['FLA_INTRACARD_CP'] == '0'
    assert 'TOKEN' not in item['environment']
    with pytest.raises(ValueError, match='禁止自动重试'):
        runner.run_step(tmp_path, job(tmp_path), 'pilot', threading.Event())


def test_peer_failure_terminates_only_owned_children_and_waits_for_all(tmp_path, monkeypatch):
    master(tmp_path, monkeypatch)
    barrier, processes, killed = threading.Barrier(3), {}, []
    lock = threading.Lock()

    class Process:
        def __init__(self, name):
            with lock:
                self.pid = len(processes)+1000
                processes[self.pid] = self
            self.name, self.terminated, self.finished = name, False, False
            barrier.wait(timeout=5)

        def wait(self, timeout=None):
            if self.name == 'broken':
                self.finished = True
                return 17
            if self.terminated:
                self.finished = True
                return -15
            time.sleep(.01)
            raise subprocess.TimeoutExpired('mock', timeout)

    def popen(command, **kwargs):
        return Process(command[-1])

    def killpg(pid, sig):
        assert pid in processes and sig == signal.SIGTERM
        killed.append(pid)
        processes[pid].terminated = True

    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.os, 'killpg', killpg)
    monkeypatch.setattr(runner, 'verify_model', lambda _: {})
    lanes = {name: [job(tmp_path, name), job(tmp_path, name+'_must_not_run')] for name in ('broken', 'peer1', 'peer2')}
    with pytest.raises(ValueError, match='所有本队列活动child已收尾'):
        runner.run_lanes(tmp_path, lanes, 'full', threading.Event())
    assert len(processes) == 3 and len(killed) == 2 and all(p.finished for p in processes.values())
    for name in lanes:
        item = runner.read(tmp_path/'steps'/f'{name}.json')
        assert item['status'] == 'failed' and item['exit_code'] == (17 if name == 'broken' else -15)
        assert not (tmp_path/'steps'/f'{name}_must_not_run.json').exists()


def test_spawn_failure_does_not_invent_pid_or_exitcode(tmp_path, monkeypatch):
    master(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise OSError('mock spawn failure')
    monkeypatch.setattr(runner.subprocess, 'Popen', fail)
    stopped = threading.Event()
    with pytest.raises(OSError):
        runner.run_step(tmp_path, job(tmp_path), 'pilot', stopped)
    item = runner.read(tmp_path/'steps/example.json')
    assert stopped.is_set() and item['pid'] is None and item['exit_code'] is None
    assert not (tmp_path/'steps/example.exitcode').exists()


def test_metadata_error_after_spawn_still_terminates_and_waits(tmp_path, monkeypatch):
    master(tmp_path, monkeypatch)
    signals, waited = [], []
    class Process:
        pid = 654
        def wait(self, timeout=None):
            waited.append(timeout)
            return -15
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *_a, **_k: Process())
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    original = runner.write
    def write(path, value, **kwargs):
        if value.get('status') == 'running':
            raise OSError('mock metadata write error')
        original(path, value, **kwargs)
    monkeypatch.setattr(runner, 'write', write)
    with pytest.raises(OSError):
        runner.run_step(tmp_path, job(tmp_path), 'pilot', threading.Event())
    assert signals == [(654, signal.SIGTERM)] and waited == [10]
    item = runner.read(tmp_path/'steps/example.json')
    assert item['pid'] == 654 and item['exit_code'] == -15 and item['status'] == 'failed'


def test_invalid_model_output_stops_lane_without_faking_child_exit(tmp_path, monkeypatch):
    master(tmp_path, monkeypatch)
    def step(master, job, phase, stopped):
        runner.write(master/'steps'/f'{job["name"]}.json',dict(status='completed',exit_code=0))
    def verify(_):
        raise ValueError('invalid model output')
    monkeypatch.setattr(runner, 'run_step', step)
    monkeypatch.setattr(runner, 'verify_model', verify)
    stopped = threading.Event()
    with pytest.raises(ValueError):
        runner.run_lanes(tmp_path, {'one':[job(tmp_path),job(tmp_path,'not_started')]},'pilot',stopped)
    item = runner.read(tmp_path/'steps/example.json')
    assert stopped.is_set() and item['exit_code'] == 0 and item['output_validation']['passed'] is False
    assert not (tmp_path/'steps/not_started.json').exists()


def test_nonempty_output_is_not_implicitly_resumed(tmp_path):
    candidate = job(tmp_path)
    directory = Path(candidate['directory'])
    directory.mkdir(parents=True)
    (directory/'predictions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='输出非空'):
        runner.unused(tmp_path, candidate)


def frozen_fixture(tmp_path, monkeypatch):
    root, out = tmp_path/'repo', tmp_path/'master'
    root.mkdir();out.mkdir()
    monkeypatch.setattr(runner, 'ROOT', root)
    source = root/'source.py';source.write_text('original\n')
    (out/'source').mkdir();(out/'source/source.py').write_bytes(source.read_bytes())
    runner.write(out/'source_fingerprints.json', {'source.py': runner.digest(source)})
    pilot = out/'pilot.jsonl';pilot.write_text('{}\n')
    plan = dict(references={}, media={}, pilot=dict(manifest=str(pilot),manifest_sha256=runner.digest(pilot)))
    runner.write(out/'run_plan.json', plan)
    runner.write(out/'preflight.json', dict(run_plan_sha256=runner.digest(out/'run_plan.json'),
        source_fingerprints_sha256=runner.digest(out/'source_fingerprints.json')))
    return root, out


@pytest.mark.parametrize('tamper', ['source', 'snapshot', 'source_fingerprints', 'plan', 'pilot'])
def test_frozen_sources_plan_and_pilot_tampering_rejected(tmp_path, monkeypatch, tamper):
    root, out = frozen_fixture(tmp_path, monkeypatch)
    assert runner.validate_frozen(out)['references'] == {}
    paths = dict(source=root/'source.py', snapshot=out/'source/source.py',
        source_fingerprints=out/'source_fingerprints.json', plan=out/'run_plan.json', pilot=out/'pilot.jsonl')
    paths[tamper].write_text('{"changed":true}\n')
    with pytest.raises(ValueError):
        runner.validate_frozen(out)


def test_full_requires_pilot_before_any_process(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = SimpleNamespace(out=tmp_path, phase='full', devices='0,1,2,3,4,6', videojev_devices='5,7', check_only=False)
    frozen = plan(tmp_path)
    monkeypatch.setattr(runner, 'validate_frozen', lambda _: frozen)
    monkeypatch.setattr(runner, 'old_preflight', lambda: dict(qwen_files={}))
    runner.write(tmp_path/'preflight.json',dict(qwen_files={}))
    runner.write(tmp_path/'launch_environment.json',dict(python=runner.sys.executable,packages={}))
    monkeypatch.setattr(runner,'packages',lambda:{})
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *_a,**_k:pytest.fail('pilot未通过时不可启动任何进程'))
    with pytest.raises(FileNotFoundError):
        runner.execute(args)


def test_prepare_check_only_has_no_writes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    frozen = plan(tmp_path)
    frozen['pilot'] = dict(selected_ids=['short','long'])
    monkeypatch.setattr(runner, 'inspect_inputs', lambda _: (frozen, {}, '', {}))
    assert runner.prepare(SimpleNamespace(out=tmp_path/'absent',check_only=True))['pilot_ids'] == ['short','long']
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('tamper', ['log', 'model'])
def test_pilot_gate_rechecks_actual_logs_and_outputs(tmp_path, monkeypatch, tamper):
    frozen = plan(tmp_path)
    frozen['pilot'] = dict(group=0,seed=42,directory=str(tmp_path/'pilot'),manifest=str(tmp_path/'pilot/sample.jsonl'))
    for name in ('steps','logs','pilot'):
        (tmp_path/name).mkdir()
    jobs = [job for lane in runner.model_lanes([frozen['pilot']],frozen,pilot=True).values() for job in lane]
    expected = dict(metrics=dict(n=2),source=dict(sha256='original'))
    runner.write(tmp_path/'pilot/execution.json',dict(passed=True,models={job['role']:expected for job in jobs}))
    for candidate in jobs:
        log=tmp_path/'logs'/f'{candidate["name"]}.log';log.write_text('original')
        runner.write(tmp_path/'steps'/f'{candidate["name"]}.json',dict(status='completed',exit_code=0,
            command=candidate['command'],cuda_visible_devices=candidate['devices'],log=str(log),log_sha256=runner.digest(log)))
        (tmp_path/'steps'/f'{candidate["name"]}.exitcode').write_text('0\n')
    if tamper == 'log':
        (tmp_path/'logs'/f'{jobs[0]["name"]}.log').write_text('changed')
    monkeypatch.setattr(runner,'verify_model',lambda _:dict(metrics=dict(n=2),source=dict(sha256='changed')) if tamper=='model' else expected)
    with pytest.raises(ValueError):
        runner.require_pilot(tmp_path,frozen)


def test_dependencies_include_new_sampling_summary_and_random_selector():
    assert {'sample_longvideobench_additional_groups.py', 'summarize_longvideobench_expanded_groups.py',
            'simulate_longvideobench_random_choice.py'} <= set(runner.SCRIPTS)
