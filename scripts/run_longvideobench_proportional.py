"""按官方题量比例抽样的三 seed 评测：冻结输入、执行与最终只读验收。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_longvideobench_expanded_groups as core

SEEDS = (42, 43, 44)
PRIOR = ROOT/'runs/lvb100_expanded3groups_seed42_43_44'
DEFAULT_OUT = ROOT/'runs/lvb100_proportional_seed42_43_44'
SAMPLING = 'proportional_official_duration_groups'
NEW_SOURCES = ('run_longvideobench_proportional.py', 'sample_longvideobench_proportional.py',
    'eval_longvideobench_proportional_controller.py', 'summarize_longvideobench_proportional.py',
    'launch_longvideobench_proportional.sh')
require, read, digest, now = core.require, core.read, core.digest, core.now


def write(path, value, *, new=False):
    """状态原子替换，避免监控恰好读到半段 JSON；新产物拒绝覆盖。"""
    path = Path(path)
    if new:
        core.write(path, value, new=True)
        return
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_paths():
    return sorted(set(core.source_paths()+[ROOT/'scripts'/name for name in NEW_SOURCES]
                      +[ROOT/'docs/LVB_PROPORTIONAL.md']))


def sample_reader():
    from scripts.sample_longvideobench_proportional import read_sample
    return read_sample


def summary_validation(master):
    from scripts.summarize_longvideobench_proportional import validate_all
    checked = validate_all(master)
    require(checked.get('passed') is True, '比例抽样汇总深验未通过')
    return checked


def model_lanes(plan):
    lanes = {'qwen': [], 'videojev': []}
    for cell in plan['cells']:
        for role in core.ROLES:
            job = core.model_job(cell, role, plan['devices'], plan['videojev_devices'])
            job['name'] = f'seed{cell["seed"]}_{role}'
            lanes[job['lane']].append(job)
    return lanes


def cpu_jobs(cell):
    out, manifest, seed = Path(cell['directory']), cell['manifest'], cell['seed']
    common = dict(devices='', lane='cpu')
    return [dict(common, name=f'seed{seed}_controller', directory=str(out/'controller'),
        command=[sys.executable, str(ROOT/'scripts/eval_longvideobench_proportional_controller.py'),
            '--manifest', manifest, '--qwen', str(out/'qwen_controller'),
            '--videojev', str(out/'videojev_controller'), '--out', str(out/'controller'),
            '--seed', str(seed), '--beta', '2', '--qwen-method', 'qwen36']),
        dict(common, name=f'seed{seed}_summary', directory=str(out/'summary'),
        command=[sys.executable, str(ROOT/'scripts/summarize_longvideobench_proportional.py'),
            '--mode', 'single', '--seed', str(seed), '--manifest', manifest,
            '--qwen-baseline', str(out/'qwen_baseline'), '--qwen-controller', str(out/'qwen_controller'),
            '--videojev-baseline', str(out/'videojev_baseline'),
            '--videojev-controller', str(out/'videojev_controller'), '--controller', str(out/'controller'),
            '--out', str(out/'summary')])]


def final_summary_job(master):
    return dict(name='proportional_summary', lane='cpu', devices='', directory=str(master/'summary'),
        command=[sys.executable, str(ROOT/'scripts/summarize_longvideobench_proportional.py'),
            '--mode', 'all', '--base', str(master), '--out', str(master/'summary')])


def all_jobs(master, plan):
    models = [job for lane in model_lanes(plan).values() for job in lane]
    cpu = [job for cell in plan['cells'] for job in cpu_jobs(cell)]+[final_summary_job(master)]
    require(len(models) == 12 and len(cpu) == 7, '比例抽样需要 12 个模型步骤和 7 个 CPU 步骤')
    return models, cpu


def prior_pilot():
    plan = core.validate_frozen(PRIOR)
    core.require_pilot(PRIOR, plan)
    return plan['pilot']


def inspect_inputs(args):
    qwen, jev = core.devices(args.devices, 6), core.devices(args.videojev_devices, 1)
    require(not set(qwen) & set(jev), 'Qwen 与 VideoJev 不得共享 GPU')
    require(args.out != ROOT and not any(args.out == p or p in args.out.parents or args.out in p.parents
            for p in (PRIOR, core.OLD)), '输出不得覆盖已有实验或仓库根目录')
    for path in source_paths():
        require(path.is_file(), f'比例方案源码尚未就绪：{path}')
    evidence = core.old_preflight()
    pilot = prior_pilot()
    cells, references, media, seen = [], {}, {}, {}
    for seed in SEEDS:
        directory = args.out/f'seed{seed}'
        manifest = directory/'sample.jsonl'
        rows, sample = sample_reader()(manifest, seed=seed, n=100)
        cells.append(dict(group=1, seed=seed, directory=str(directory), manifest=str(manifest),
                          manifest_sha256=sample['sha256']))
        for path in (manifest, manifest.with_suffix('.meta.json')):
            references[str(path)] = digest(path)
        for key in ('source_manifest', 'source_reference'):
            references[sample[key]['path']] = sample[key]['sha256']
        for row in rows:
            require(row['id'] not in seen or seen[row['id']] == row, '跨 seed 同题内容不同')
            seen[row['id']] = row
            path = Path(row['video_path'])
            stat = path.stat()
            media[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    for relative in ('run_plan.json', 'preflight.json', 'source_fingerprints.json',
                     'pilot/execution.json', 'artifact_manifest.json'):
        references[str(PRIOR/relative)] = digest(PRIOR/relative)
    references[str(args.out/'sampling_audit.json')] = digest(args.out/'sampling_audit.json')
    plan = dict(schema='lvb_proportional_runner_v1', sampling=SAMPLING, unit='question',
        cells=cells, seeds=list(SEEDS), n_per_seed=100, devices=','.join(qwen), videojev_devices=jev,
        model_steps=12, cpu_steps=7, baseline=dict(fps=4, max_frames=4096),
        controller=dict(fps=2, max_frames=2048, beta=2), context=262144,
        visual_token_budget=24576, qwen_dtype='bfloat16', quantized=False,
        new_gpu_pilot_runs=0, pilot=pilot, reused_pilot_master=str(PRIOR),
        pilot_note='复用同一推理源码、环境、模型的四协议真实两题 pilot；本次重新评测全部抽样题。',
        admission=dict(min_free_mib=23000, interval_seconds=60, timeout_seconds=21600),
        references=references, media=media,
        unique_questions=len(seen), unique_videos=len(media),
        prior_predictions_reused=False)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}
    return plan, sources, evidence


def validate_frozen(master):
    plan = core.validate_frozen(master)
    require(plan['schema'] == 'lvb_proportional_runner_v1' and plan['sampling'] == SAMPLING
            and plan['seeds'] == list(SEEDS) and plan['n_per_seed'] == 100
            and plan['model_steps'] == 12 and plan['cpu_steps'] == 7
            and len(plan['cells']) == 3 and plan['new_gpu_pilot_runs'] == 0
            and plan['reused_pilot_master'] == str(PRIOR), '比例方案冻结计划错误')
    for seed, cell in zip(SEEDS, plan['cells']):
        directory = master/f'seed{seed}'
        require(cell['seed'] == seed and Path(cell['directory']) == directory
                and Path(cell['manifest']) == directory/'sample.jsonl', 'seed 或输出路径与冻结布局不符')
        _, sample = sample_reader()(cell['manifest'], seed=seed, n=100)
        require(sample['sha256'] == cell['manifest_sha256'], '比例抽样清单与运行计划不同')
    return plan


def check_environment(master, plan, args):
    launch = read(master/'launch_environment.json')
    require(Path(sys.executable).resolve() == Path(launch['python']).resolve(), '执行解释器与 prepare 不同')
    require(core.packages() == launch['packages'], '执行依赖版本与 prepare 不同')
    require(args.devices == plan['devices'] and args.videojev_devices.split(',') == plan['videojev_devices'],
            '执行选卡与 prepare 冻结计划不同')
    evidence = core.old_preflight()
    require(evidence['qwen_files'] == read(master/'preflight.json')['qwen_files'], '模型与 prepare 不同')
    require(prior_pilot() == plan['pilot'], '既有四协议真实 pilot 与 prepare 不同')


def prepare(args):
    if (args.out/'preflight.json').exists():
        require(args.check_only, 'prepare 产物已存在，拒绝覆盖')
        plan = validate_frozen(args.out)
        check_environment(args.out, plan, args)
        return dict(checked=True, phase='prepare', cells=3, model_steps=12, cpu_steps=7)
    reserved = ('source', 'source_fingerprints.json', 'run_plan.json', 'launch_environment.json',
                'preflight.json', 'experiment.json', 'steps', 'logs', 'queue_execution.json', 'validation.json',
                'finalization.json', 'artifact_manifest.json', 'resource_admission')
    require(not any((args.out/name).exists() for name in reserved), 'prepare 产物或执行现场已存在，拒绝覆盖')
    plan, sources, evidence = inspect_inputs(args)
    if args.check_only:
        return dict(checked=True, phase='prepare', cells=3, model_steps=12, cpu_steps=7,
                    new_gpu_pilot_runs=0, unique_questions=plan['unique_questions'])
    (args.out/'source').mkdir(parents=True)
    for relative in sources:
        if not relative.endswith('.so'):
            target = args.out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
    write(args.out/'source_fingerprints.json', sources, new=True)
    write(args.out/'run_plan.json', plan, new=True)
    write(args.out/'launch_environment.json', dict(created_at=now(), python=sys.executable,
        hostname=socket.gethostname(), pid=os.getpid(), command=sys.argv,
        environment={k: core.environment(plan['devices'])[k] for k in core.ENV_KEYS},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        packages=core.packages()), new=True)
    write(args.out/'preflight.json', dict(passed=True, prepared_at=now(),
        run_plan_sha256=digest(args.out/'run_plan.json'),
        source_fingerprints_sha256=digest(args.out/'source_fingerprints.json'),
        qwen_files=evidence['qwen_files'], new_gpu_pilot_runs=0,
        reused_pilot_master=str(PRIOR), limitations=[
            '权重复用已有完整 SHA 证据并核对 size/mtime；实际 evaluator 仍重算权重与媒体完整 SHA。',
            '未获得完整 QA 训练清单，无法确认训练与验证集隔离。',
            '显存准入门禁不预留 GPU，不能阻止外部作业稍后抢占。']), new=True)
    write(args.out/'experiment.json', dict(schema='lvb_proportional_experiment_v1',
        experiment_id=args.out.name, status='prepared', sampling=SAMPLING, seeds=list(SEEDS),
        n_per_seed=100, model_steps=12, cpu_steps=7, prepared_at=now(),
        run_plan_sha256=digest(args.out/'run_plan.json')), new=True)
    return dict(prepared=True, cells=3, model_steps=12, cpu_steps=7, new_gpu_pilot_runs=0)


def free_memory(visible):
    output = subprocess.check_output(['nvidia-smi', '--id='+visible,
        '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=30)
    values = {}
    for line in output.splitlines():
        index, free = [part.strip() for part in line.split(',')]
        require(index not in values and free.isdigit(), 'nvidia-smi 显存输出无效')
        values[index] = int(free)
    require(set(values) == set(visible.split(',')), '显存准入查询未覆盖计划 GPU')
    return values


def admission(master, job, stopped, settings, progress=None):
    """调度准入，只检查当前任务选定 GPU；不重试已启动的模型。"""
    path = master/'resource_admission'/f'{job["name"]}.json'
    require(not path.exists(), f'已有准入现场，禁止重试：{job["name"]}')
    started = time.monotonic()
    state = dict(name=job['name'], devices=job['devices'], settings=settings,
                 status='waiting', started_at=now(), samples=[])
    write(path, state, new=True)
    try:
        while True:
            require(not stopped.is_set(), '队列已停止，不再等待显存或启动任务')
            require(time.monotonic()-started < settings['timeout_seconds'], '显存准入等待超时，保留现场')
            values = free_memory(job['devices'])
            state['samples'].append(dict(checked_at=now(), free_mib=values))
            require(time.monotonic()-started < settings['timeout_seconds'], '显存准入等待超时，保留现场')
            if min(values.values()) >= settings['min_free_mib']:
                require(time.monotonic()-started < settings['timeout_seconds'], '显存准入等待超时，保留现场')
                state.update(status='admitted', finished_at=now())
                write(path, state)
                print(f'[{now()}] {job["name"]} 显存准入通过：{values} MiB', flush=True)
                if progress:
                    progress(job, 'admitted', dict(free_mib=values))
                return state
            print(f'[{now()}] {job["name"]} 等待显存：{values} MiB；'
                  f'每卡需 {settings["min_free_mib"]} MiB，尚未启动模型', flush=True)
            if progress:
                progress(job, 'waiting_resource', dict(free_mib=values,
                         min_free_mib=settings['min_free_mib']))
            elapsed = time.monotonic()-started
            require(elapsed < settings['timeout_seconds'], '显存准入等待超时，保留现场')
            write(path, state)
            stopped.wait(min(settings['interval_seconds'], settings['timeout_seconds']-elapsed))
    except BaseException as error:
        state.update(status='failed', finished_at=now(), error=repr(error))
        write(path, state)
        raise


def run_lanes(master, plan, stopped, completed, progress=None):
    def lane(jobs):
        for job in jobs:
            if stopped.is_set():
                return
            try:
                admission(master, job, stopped, plan['admission'], progress)
                if progress:
                    progress(job, 'running_model')
                core.run_step(master, job, 'full', stopped)
                path = master/'steps'/f'{job["name"]}.json'
                item = read(path)
                try:
                    core.verify_model(job)
                except Exception as error:
                    item['output_validation'] = dict(passed=False, checked_at=now(), error=repr(error))
                    write(path, item)
                    raise
                item['output_validation'] = dict(passed=True, checked_at=now())
                write(path, item)
                completed(job)
            except BaseException:
                stopped.set()
                raise
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(lane, jobs): name for name, jobs in model_lanes(plan).items()}
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                stopped.set()
                errors.append(dict(lane=futures[future], error=repr(error)))
    require(not errors and not stopped.is_set(), f'模型队列失败，所属 child 已收尾：{errors}')


def validate_outputs(master, plan):
    models, cpu = all_jobs(master, plan)
    names = {job['name'] for job in models+cpu}
    require({path.stem for path in (master/'steps').glob('*.json')} == names
            and {path.stem for path in (master/'steps').glob('*.exitcode')} == names
            and {path.stem for path in (master/'logs').glob('*.log')} == names,
            '实际步骤/退出码/日志集合与 19 个冻结任务不同')
    hashes = {}
    for job in models+cpu:
        path = master/'steps'/f'{job["name"]}.json'
        step = read(path)
        require(step['status'] == 'completed' and step['exit_code'] == 0
                and step['command'] == job['command'] and step['cuda_visible_devices'] == job['devices']
                and step['spawn_command'] == core.shell_command(job['command'])
                and type(step['pid']) is int and step['pid'] > 0 and step['process_group'] == step['pid']
                and step['lane'] == job['lane'] and step['phase'] == 'full'
                and Path(step['log']) == master/'logs'/f'{job["name"]}.log'
                and digest(step['log']) == step['log_sha256']
                and (master/'steps'/f'{job["name"]}.exitcode').read_text().strip() == '0',
                f'步骤退出、命令、环境或日志指纹不符：{job["name"]}')
        expected_env = core.environment(job['devices'])
        require(step['environment'] == {key: expected_env[key] for key in core.ENV_KEYS},
                f'步骤代理/离线/原生后端环境不符：{job["name"]}')
        if job in models:
            require(step.get('output_validation', {}).get('passed') is True,
                    f'模型输出验证未成功：{job["name"]}')
            gate = read(master/'resource_admission'/f'{job["name"]}.json')
            require(gate['status'] == 'admitted' and gate['devices'] == job['devices']
                    and gate['settings'] == plan['admission'] and gate['samples']
                    and set(gate['samples'][-1]['free_mib']) == set(job['devices'].split(','))
                    and min(gate['samples'][-1]['free_mib'].values()) >= plan['admission']['min_free_mib'],
                    f'实际显存准入证据不符：{job["name"]}')
            core.verify_model(job)
        hashes[job['name']] = digest(path)
    checked = summary_validation(master)
    return dict(passed=True, model_steps=len(models), cpu_steps=len(cpu),
                step_sha256=hashes, summary_source_hashes=checked['source_hashes'],
                report_sha256=digest(master/'summary/results.json'))


def execute(args):
    master = args.out
    plan = validate_frozen(master)
    check_environment(master, plan, args)
    models, cpu = all_jobs(master, plan)
    for job in models+cpu:
        core.unused(master, job)
    require(not (master/'queue_execution.json').exists() and not (master/'validation.json').exists(),
            'full 已有执行现场，拒绝自动重试或覆盖')
    experiment = read(master/'experiment.json')
    require(experiment['status'] == 'prepared', '实验不是待启动状态')
    if args.check_only:
        return dict(checked=True, phase='full', model_steps=12, cpu_steps=7,
                    devices=plan['devices'], videojev_devices=plan['videojev_devices'])
    for name in ('steps', 'logs', 'resource_admission'):
        (master/name).mkdir(exist_ok=True)
    state = dict(status='running', phase='full', started_at=now(), pid=os.getpid(),
        hostname=socket.gethostname(), command=sys.argv, run_plan_sha256=digest(master/'run_plan.json'),
        completed_model_steps=0, completed_cpu_steps=0, completed_steps=[],
        activity='starting', lanes={})
    write(master/'queue_execution.json', state, new=True)
    experiment.update(status='running', started_at=state['started_at'], queue_pid=os.getpid())
    write(master/'experiment.json', experiment)
    stopped, lock = threading.Event(), threading.Lock()
    def progress(job, activity, details=None):
        with lock:
            state['lanes'][job['lane']] = dict(status=activity, name=job['name'],
                devices=job['devices'], updated_at=now(), **(details or {}))
            active = {value['status'] for value in state['lanes'].values()}
            state['activity'] = next((name for name in ('running_model', 'running_cpu', 'waiting_resource', 'admitted')
                                      if name in active), 'between_steps')
            write(master/'queue_execution.json', state)
    def completed(job):
        with lock:
            key = 'completed_cpu_steps' if job['lane'] == 'cpu' else 'completed_model_steps'
            state[key] += 1
            state['completed_steps'].append(job['name'])
            state['lanes'][job['lane']] = dict(status='completed', name=job['name'], updated_at=now())
            write(master/'queue_execution.json', state)
    handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_lanes(master, plan, stopped, completed, progress)
        for job in cpu:
            progress(job, 'running_cpu')
            core.run_step(master, job, 'full', stopped)
            completed(job)
        require(not stopped.is_set(), '队列收到停止信号，不能报告成功')
        checked = validate_outputs(master, plan)
        write(master/'validation.json', dict(checked, checked_at=now(),
            run_plan_sha256=digest(master/'run_plan.json'),
            note='12 模型及 7 CPU 子步骤真实 exit0 且深验通过；仍需外部 full 实际退出和监控终态后封存。'), new=True)
        state.update(status='completed', activity='completed', finished_at=now())
        write(master/'queue_execution.json', state)
        experiment.update(status='awaiting_finalization', full_steps_finished_at=now())
        write(master/'experiment.json', experiment)
    except BaseException as error:
        stopped.set()
        state.update(status='failed', activity='failed', finished_at=now(), error=repr(error))
        write(master/'queue_execution.json', state)
        experiment.update(status='failed', finished_at=now(), error=repr(error))
        write(master/'experiment.json', experiment)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return dict(completed=True, phase='full', model_steps=12, cpu_steps=7,
                status='awaiting_finalization')


def artifact_files(master):
    files = [path for path in master.rglob('*') if path.is_file() and path != master/'artifact_manifest.json']
    require(not any(path.is_symlink() for path in files), '实验封存不接受文件符号链接')
    return {str(path.relative_to(master)): dict(bytes=path.stat().st_size, sha256=digest(path))
            for path in sorted(files)}


def verify_seal(master):
    seal = read(master/'artifact_manifest.json')
    require(seal['schema'] == 'lvb_proportional_artifacts_v1' and seal['files'] == artifact_files(master),
            '封存清单内容、SHA 或文件覆盖范围不符')
    return len(seal['files'])


def finalize(args):
    master = args.out
    plan = validate_frozen(master)
    check_environment(master, plan, args)
    require((master/'full.exitcode').read_text().strip() == '0', '外部 full 实际退出码不是 0')
    monitor = read(master/'monitor/status.json')
    require(monitor.get('monitor_status') == 'completed', '指定监控尚未完成最后一次写入，不能封存')
    state = read(master/'queue_execution.json')
    require(state['status'] == 'completed' and state['completed_model_steps'] == 12
            and state['completed_cpu_steps'] == 7, '队列不是完整成功终态')
    checked = validate_outputs(master, plan)
    saved = read(master/'validation.json')
    require(saved['passed'] is True and all(saved.get(key) == value for key, value in checked.items())
            and saved['run_plan_sha256'] == digest(master/'run_plan.json'), '实际深验与 full 完成证据不同')
    experiment = read(master/'experiment.json')
    seal_exists = (master/'artifact_manifest.json').exists()
    if seal_exists:
        require(args.check_only and experiment['status'] == 'completed', '已有封存，拒绝改写')
        count = verify_seal(master)
        return dict(checked=True, phase='finalize', sealed=True, files=count, **checked)
    require(experiment['status'] == 'awaiting_finalization', '实验未进入最终验收阶段')
    require(not (master/'finalization.json').exists(), '存在未完成的封存现场，拒绝覆盖')
    if args.check_only:
        return dict(checked=True, phase='finalize', sealed=False, **checked)
    experiment.update(status='completed', finished_at=now())
    write(master/'experiment.json', experiment)
    write(master/'finalization.json', dict(passed=True, finished_at=now(),
        full_exitcode_sha256=digest(master/'full.exitcode'),
        monitor_status_sha256=digest(master/'monitor/status.json'),
        validation_sha256=digest(master/'validation.json'),
        run_plan_sha256=digest(master/'run_plan.json'), model_steps=12, cpu_steps=7,
        note='封存后不能继续写入本目录；finalize 日志与外部真实退出码保存在目录外。'), new=True)
    write(master/'artifact_manifest.json', dict(schema='lvb_proportional_artifacts_v1',
        created_at=now(), files=artifact_files(master), excludes=['artifact_manifest.json'],
        external_inputs='权重和媒体保留在原目录，其内容指纹由各 evaluator 的 run.json 记录。'), new=True)
    return dict(completed=True, phase='finalize', files=verify_seal(master), **checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'full', 'finalize'), required=True)
    parser.add_argument('--out', type=lambda value: Path(value).resolve(), default=DEFAULT_OUT)
    parser.add_argument('--devices', default='0,1,2,3,4,6')
    parser.add_argument('--videojev-devices', default='7')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    operation = dict(prepare=prepare, full=execute, finalize=finalize)[args.phase]
    print(json.dumps(operation(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
