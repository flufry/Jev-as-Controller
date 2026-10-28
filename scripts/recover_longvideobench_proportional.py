"""原样保留一个成功步骤，按冻结协议恢复比例抽样实验的其余 11 个模型步骤。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_longvideobench_proportional as runner

BASE = runner.DEFAULT_OUT
REUSED = {'seed42_videojev_baseline'}
MIN_FREE_MIB = 23000
STABLE_SECONDS = 60
INTERVAL_SECONDS = 10
TIMEOUT_SECONDS = 6*3600


def files_under(directory):
    directory = Path(directory)
    runner.require(directory.is_dir() and not directory.is_symlink(), f'目录缺失或是符号链接：{directory}')
    result = set()
    for path in directory.rglob('*'):
        runner.require(not path.is_symlink(), f'目录含符号链接：{path}')
        if path.is_file():
            result.add(str(path))
    return result


def check_files(files):
    runner.require(isinstance(files, dict) and files, '缺少冻结文件 SHA 清单')
    for name, expected in files.items():
        path = Path(name)
        runner.require(path.is_absolute() and path.is_file() and not path.is_symlink(),
                       f'冻结文件缺失或不是普通绝对路径：{name}')
        runner.require(runner.digest(path) == expected, f'冻结文件 SHA 变化：{name}')


def check_archive(master, archive):
    directory, manifest = Path(archive['path']), Path(archive['manifest_path'])
    runner.require(directory.is_absolute() and directory.parent == master/'failed_attempts'
            and directory.name == 'resource_contention1' and manifest == directory/'manifest.json',
            '失败归档路径不符')
    runner.require(not manifest.is_symlink() and runner.digest(manifest) == archive['manifest_sha256'],
                   '失败归档清单 SHA 不符')
    actual = files_under(directory)-{str(manifest)}
    contents = runner.read(manifest)
    runner.require(contents['schema'] == 'lvb_proportional_failed_attempt_archive_v1', '失败归档 schema 不符')
    entries = contents['files']
    runner.require(isinstance(entries, dict) and entries, '失败归档清单为空')
    paths = set()
    for relative, expected in entries.items():
        path = directory/relative
        runner.require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
                and path.resolve().is_relative_to(directory.resolve()) and path != manifest,
                f'失败归档清单路径越界：{relative}')
        runner.require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected['bytes'] and runner.digest(path) == expected['sha256'],
                f'失败归档文件 SHA/大小不符：{relative}')
        paths.add(str(path))
    runner.require(paths == actual, '失败归档清单没有完整覆盖实际文件')


def retained_paths(master, job):
    return files_under(job['directory']) | {
        str(master/'steps'/f'{job["name"]}.json'),
        str(master/'steps'/f'{job["name"]}.exitcode'),
        str(master/'logs'/f'{job["name"]}.log'),
        str(master/'resource_admission'/f'{job["name"]}.json')}


def verify_reused(master, job, plan):
    path = master/'steps'/f'{job["name"]}.json'
    step, env = runner.read(path), runner.core.environment(job['devices'])
    log = master/'logs'/f'{job["name"]}.log'
    runner.require(step['name'] == job['name'] and step['status'] == 'completed'
        and step['exit_code'] == 0 and step['phase'] == 'full' and step['command'] == job['command']
        and step['spawn_command'] == runner.core.shell_command(job['command'])
        and step['cuda_visible_devices'] == job['devices'] and step['lane'] == job['lane']
        and step['environment'] == {key: env[key] for key in runner.core.ENV_KEYS}
        and type(step['pid']) is int and step['pid'] > 0 and step['process_group'] == step['pid']
        and step.get('output_validation', {}).get('passed') is True
        and step['log'] == str(log) and runner.digest(log) == step['log_sha256']
        and path.with_suffix('.exitcode').read_text().strip() == '0',
        f'复用步骤真实命令、环境、验证或退出记录不符：{job["name"]}')
    gate = runner.read(master/'resource_admission'/f'{job["name"]}.json')
    runner.require(gate['status'] == 'admitted' and gate['devices'] == job['devices']
        and gate['settings'] == plan['admission'] and gate['samples']
        and set(gate['samples'][-1]['free_mib']) == set(job['devices'].split(','))
        and min(gate['samples'][-1]['free_mib'].values()) >= MIN_FREE_MIB,
        '原成功步骤缺少匹配的显存准入证据')
    runner.require(runner.core.verify_model(job)['metrics']['n'] == 100, '原成功模型结果不是完整 100 题')


def check_guard(master, recovery, preflight_sha):
    runner.require(runner.digest(master/'recovery_preflight.json') == preflight_sha,
                   '恢复前置证据 SHA 变化')
    driver = recovery['driver']
    runner.require(Path(driver['path']) == Path(__file__).resolve(), '恢复执行器路径不符')
    check_files({driver['path']: driver['sha256']})
    check_files(recovery['additional_sources'])
    check_files(recovery['retained_files'])
    check_archive(master, recovery['archive'])
    plan = runner.validate_frozen(master)
    reused_job = next(job for job in runner.all_jobs(master, plan)[0] if job['name'] in REUSED)
    runner.require(set(recovery['retained_files']) == retained_paths(master, reused_job),
                   '保留 SHA 清单必须完整且仅覆盖原成功模型及其步骤、日志、退出和准入现场')
    return plan


def preflight(master):
    path = master/'recovery_preflight.json'
    runner.require(path.is_file(), '缺少 recovery_preflight.json；请先由 root 原样归档失败现场')
    recovery = runner.read(path)
    runner.require(recovery.get('schema') == 'lvb_proportional_recovery_v1'
        and recovery.get('passed') is True and recovery['reused_model_steps'] == sorted(REUSED),
        '恢复前置证据或唯一可复用步骤不符')
    frozen = runner.digest(path)
    plan = check_guard(master, recovery, frozen)
    runner.require(plan['devices'] == '0,1,2,3,4,6' and plan['videojev_devices'] == ['7'],
                   '恢复不得变更冻结的两 lane 设备')
    runner.check_environment(master, plan, SimpleNamespace(devices=plan['devices'], videojev_devices='7'))
    models, cpu = runner.all_jobs(master, plan)
    wanted = {job['name'] for job in models}
    runner.require(len(models) == 12 and len(cpu) == 7 and REUSED <= wanted,
                   '恢复计划必须是原 12 模型和 7 CPU 步骤')
    for job in models+cpu:
        if job['name'] in REUSED:
            verify_reused(master, job, plan)
        else:
            runner.core.unused(master, job)
            if job in models:
                runner.require(not (master/'resource_admission'/f'{job["name"]}.json').exists(),
                               f'失败或未知准入现场尚未归档：{job["name"]}')
    absent = recovery['required_absent']
    runner.require(isinstance(absent, list) and absent and len(set(absent)) == len(absent),
                   '缺少原样归档后应为空的路径清单')
    for name in absent:
        target = Path(name)
        runner.require(target.is_absolute() and target.is_relative_to(master) and not target.exists(),
                       f'旧尝试未归档或已有新现场，拒绝覆盖：{name}')
    for name in ('queue_execution.json', 'experiment.json', 'full.exitcode', 'validation.json',
                 'finalization.json', 'artifact_manifest.json', 'recovery_stability'):
        runner.require(not (master/name).exists(), f'恢复已有现场，拒绝覆盖：{name}')
    previous = runner.read(Path(recovery['archive']['path'])/'experiment.json')
    runner.require(previous['status'] == 'failed', '归档的原实验状态不是 failed')
    return recovery, frozen, plan, models, cpu, previous


def resource_snapshot(visible):
    """仅查询本任务选定 GPU 的空闲显存与 compute PID；查询失败直接保留现场。"""
    common = dict(check=True, capture_output=True, text=True, timeout=30,
                  env=runner.core.environment(''), cwd=ROOT)
    output = subprocess.run(['nvidia-smi', '--id='+visible,
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], **common).stdout
    free, uuids = {}, {}
    for line in output.splitlines():
        values = [value.strip() for value in line.split(',')]
        runner.require(len(values) == 3 and values[0].isdigit() and values[2].isdigit()
            and values[0] not in free and values[1] not in uuids, 'GPU 显存/UUID 查询格式无效')
        index, uuid, amount = values
        free[index], uuids[uuid] = int(amount), index
    runner.require(set(free) == set(visible.split(',')), 'GPU 稳定门禁查询未覆盖计划设备')
    output = subprocess.run(['nvidia-smi', '--id='+visible,
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], **common).stdout
    pids = {index: [] for index in free}
    for line in output.splitlines():
        values = [value.strip() for value in line.split(',')]
        runner.require(len(values) == 2 and values[0] in uuids and values[1].isdigit(),
                       'GPU compute PID 查询格式或设备无效')
        pids[uuids[values[0]]].append(int(values[1]))
    return dict(free_mib=free, compute_pids=pids)


def stable_admission(master, job, stopped, progress=None):
    path = master/'recovery_stability'/f'{job["name"]}.json'
    settings = dict(min_free_mib=MIN_FREE_MIB, stable_seconds=STABLE_SECONDS,
                    interval_seconds=INTERVAL_SECONDS, timeout_seconds=TIMEOUT_SECONDS,
                    no_compute_pids=True)
    state = dict(name=job['name'], devices=job['devices'], status='waiting',
                 settings=settings, started_at=runner.now(), samples=[])
    runner.write(path, state, new=True)
    started, quiet_since = time.monotonic(), None
    try:
        while True:
            runner.require(not stopped.is_set(), '队列已停止，取消稳定门禁')
            runner.require(time.monotonic()-started < TIMEOUT_SECONDS, '稳定门禁等待超过六小时')
            snapshot = resource_snapshot(job['devices'])
            current = time.monotonic()
            runner.require(current-started < TIMEOUT_SECONDS, '稳定门禁等待超过六小时')
            quiet = min(snapshot['free_mib'].values()) >= MIN_FREE_MIB and not any(snapshot['compute_pids'].values())
            quiet_since = (current if quiet_since is None else quiet_since) if quiet else None
            stable = 0 if quiet_since is None else current-quiet_since
            state['samples'].append(dict(checked_at=runner.now(), elapsed_seconds=current-started,
                                         stable_seconds=stable, qualifying=quiet, **snapshot))
            if quiet and stable >= STABLE_SECONDS:
                runner.require(time.monotonic()-started < TIMEOUT_SECONDS, '稳定门禁等待超过六小时')
                state.update(status='admitted', finished_at=runner.now())
                runner.write(path, state)
                print(f'[{runner.now()}] {job["name"]} 已连续 {stable:.0f}s 空闲，稳定门禁通过', flush=True)
                return
            runner.write(path, state)
            print(f'[{runner.now()}] {job["name"]} 等待稳定资源：空闲 {snapshot["free_mib"]} MiB，'
                  f'compute PID {snapshot["compute_pids"]}，已连续空闲 {stable:.0f}/{STABLE_SECONDS}s', flush=True)
            if progress:
                progress(job, 'waiting_resource', dict(gate='stability', stable_seconds=stable, **snapshot))
            remaining = TIMEOUT_SECONDS-(time.monotonic()-started)
            runner.require(remaining > 0, '稳定门禁等待超过六小时')
            stopped.wait(min(INTERVAL_SECONDS, remaining))
    except BaseException as error:
        state.update(status='cancelled' if stopped.is_set() else 'failed',
                     finished_at=runner.now(), error=repr(error))
        runner.write(path, state)
        raise


def run_lanes(master, plan, recovery, stopped, unchanged, completed, progress):
    def lane(jobs):
        for job in jobs:
            if stopped.is_set():
                return
            try:
                unchanged()
                if job['name'] in REUSED:
                    verify_reused(master, job, plan)
                    completed(job)
                    continue
                stable_admission(master, job, stopped, progress)
                unchanged()
                runner.admission(master, job, stopped, plan['admission'], progress)
                unchanged()
                progress(job, 'running_model')
                runner.core.run_step(master, job, 'full', stopped)
                path = master/'steps'/f'{job["name"]}.json'
                step = runner.read(path)
                try:
                    runner.require(runner.core.verify_model(job)['metrics']['n'] == 100, '新模型结果不是完整 100 题')
                except Exception as error:
                    step['output_validation'] = dict(passed=False, checked_at=runner.now(), error=repr(error))
                    runner.write(path, step)
                    raise
                step['output_validation'] = dict(passed=True, checked_at=runner.now())
                runner.write(path, step)
                completed(job)
            except BaseException:
                stopped.set()
                raise
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(lane, jobs): name for name, jobs in runner.model_lanes(plan).items()}
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                stopped.set()
                errors.append(dict(lane=futures[future], error=repr(error)))
    runner.require(not errors and not stopped.is_set(), f'恢复队列失败，所属 child 已收尾：{errors}')


def execute(master=BASE, *, check_only=False):
    master = Path(master).resolve()
    recovery, frozen, plan, models, cpu, previous = preflight(master)
    if check_only:
        return dict(checked=True, reused_model_steps=1, new_model_steps=11, cpu_steps=7,
            resource_query_executed=False, devices=plan['devices'], videojev_devices=plan['videojev_devices'],
            stability_seconds=STABLE_SECONDS, max_wait_seconds=TIMEOUT_SECONDS)
    def unchanged():
        check_guard(master, recovery, frozen)
    for folder in ('steps', 'logs', 'resource_admission'):
        (master/folder).mkdir(exist_ok=True)
    (master/'recovery_stability').mkdir()
    state = dict(schema='lvb_proportional_recovery_queue_v1', status='running', phase='full',
        attempt='resource_contention_recovery1', activity='starting', started_at=runner.now(),
        pid=os.getpid(), hostname=socket.gethostname(), command=sys.argv,
        run_plan_sha256=runner.digest(master/'run_plan.json'), recovery_driver=recovery['driver'],
        recovery_preflight=dict(path=str(master/'recovery_preflight.json'), sha256=frozen),
        reused_model_steps=sorted(REUSED), new_model_steps=[j['name'] for j in models if j['name'] not in REUSED],
        prior_failed_archive=recovery['archive'], completed_model_steps=0, completed_cpu_steps=0,
        completed_steps=[], lanes={},
        note='一个原成功模型步骤只读复用；其余 11 个从空输出重跑，另有 7 个 CPU 步骤，无自动重试。')
    experiment = dict(previous, status='running', attempt=state['attempt'],
        recovery_started_at=state['started_at'], queue_pid=os.getpid(), recovery_driver=recovery['driver'],
        recovery_preflight=state['recovery_preflight'], prior_failed_archive=recovery['archive'])
    for key in ('error', 'finished_at'):
        experiment.pop(key, None)
    runner.write(master/'queue_execution.json', state, new=True)
    runner.write(master/'experiment.json', experiment, new=True)
    stopped, lock = threading.Event(), threading.Lock()
    def progress(job, activity, details=None):
        with lock:
            state['lanes'][job['lane']] = dict(status=activity, name=job['name'], devices=job['devices'],
                                              updated_at=runner.now(), **(details or {}))
            active = {value['status'] for value in state['lanes'].values()}
            state['activity'] = next((name for name in ('running_model', 'running_cpu', 'waiting_resource', 'admitted')
                                      if name in active), 'between_steps')
            runner.write(master/'queue_execution.json', state)
    def completed(job):
        with lock:
            runner.require(job['name'] not in state['completed_steps'], '不能重复计数同一步骤')
            key = 'completed_cpu_steps' if job['lane'] == 'cpu' else 'completed_model_steps'
            state[key] += 1
            state['completed_steps'].append(job['name'])
            state['lanes'][job['lane']] = dict(status='completed', name=job['name'], updated_at=runner.now())
            runner.write(master/'queue_execution.json', state)
    handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_lanes(master, plan, recovery, stopped, unchanged, completed, progress)
        for job in cpu:
            unchanged()
            progress(job, 'running_cpu')
            runner.core.run_step(master, job, 'full', stopped)
            completed(job)
        unchanged()
        runner.require(not stopped.is_set(), '收到停止信号，不能报告恢复成功')
        checked = runner.validate_outputs(master, plan)
        runner.require(state['completed_model_steps'] == 12 and state['completed_cpu_steps'] == 7,
                       '模型或 CPU 完成计数不完整')
        runner.write(master/'validation.json', dict(checked, checked_at=runner.now(),
            run_plan_sha256=runner.digest(master/'run_plan.json'), recovery_preflight_sha256=frozen,
            reused_model_steps=1, new_model_steps=11,
            note='12 模型含 1 个原成功步骤和 11 个新成功步骤；7 CPU 全部真实 exit0，仍需外部 full 退出及监控终态。'), new=True)
        state.update(status='completed', activity='completed', finished_at=runner.now())
        runner.write(master/'queue_execution.json', state)
        experiment.update(status='awaiting_finalization', full_steps_finished_at=runner.now())
        runner.write(master/'experiment.json', experiment)
    except BaseException as error:
        stopped.set()
        state.update(status='failed', activity='failed', finished_at=runner.now(), error=repr(error))
        runner.write(master/'queue_execution.json', state)
        experiment.update(status='failed', finished_at=runner.now(), error=repr(error))
        runner.write(master/'experiment.json', experiment)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return dict(completed=True, phase='full', model_steps=12, reused_model_steps=1,
                new_model_steps=11, cpu_steps=7, status='awaiting_finalization')


def finalize(master=BASE, *, check_only=False):
    master = Path(master).resolve()
    frozen = runner.read(master/'validation.json')['recovery_preflight_sha256']
    runner.require(runner.digest(master/'recovery_preflight.json') == frozen, '恢复前置证据 SHA 与全量完成记录不同')
    recovery = runner.read(master/'recovery_preflight.json')
    check_guard(master, recovery, frozen)
    return runner.finalize(SimpleNamespace(out=master, devices='0,1,2,3,4,6', videojev_devices='7',
                                           phase='finalize', check_only=check_only))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=lambda value: Path(value).resolve(), default=BASE)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--finalize', action='store_true', help='核验新增恢复来源后调用原完整最终验收')
    args = parser.parse_args()
    operation = finalize if args.finalize else execute
    print(json.dumps(operation(args.out, check_only=args.check_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
