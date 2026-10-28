"""新增 group2/3 × seed42/43/44：冻结、四协议 pilot 和独立 GPU 执行队列。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
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
OLD = ROOT/'runs/lvb100_qwen36_35b_repeat3_seed42_43_44'
sys.path.insert(0, str(ROOT))
SEEDS = (42, 43, 44)
ROLES = ('qwen_baseline', 'qwen_controller', 'videojev_baseline', 'videojev_controller')
SCRIPTS = (
    'run_longvideobench_expanded_groups.py', 'sample_longvideobench_additional_groups.py',
    'sample_longvideobench.py', 'summarize_longvideobench_expanded_groups.py',
    'eval_longvideobench_qwen36.py', 'eval_longvideobench_qwen.py',
    'eval_longvideobench_videojev.py', 'eval_longvideobench_controller.py',
    'summarize_longvideobench_qwen36.py', 'summarize_longvideobench_qwen36_repeats.py',
    'summarize_longvideobench_repeats.py', 'compare_longvideobench_sampling.py',
    'simulate_longvideobench_random_choice.py',
    'env.sh', 'env_qwen36.sh',
)
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM',
    'FLA_INTRACARD_CP', 'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, *, new=False):
    with Path(path).open('x' if new else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def devices(value, count):
    values = value.split(',')
    counts = (count,) if isinstance(count, int) else count
    require(len(values) in counts and len(set(values)) == len(values) and all(x.isdigit() for x in values),
            f'需要 {"或".join(map(str, counts))} 个不同的 GPU 编号')
    return values


def environment(visible):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=visible, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false', FLA_INTRACARD_CP='0', PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://127.0.0.1:7890'
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1'
    return env


def shell_command(command):
    # Bash source 保证每个 child 使用独立 overlay；所有命令参数通过 argv 原样传递。
    return ['bash', '-c', 'set -e\nsource "$1"\nshift\nexec "$@"',
            'lvb-expanded', str(ROOT/'scripts/env_qwen36.sh'), *command]


def model_job(cell, role, qwen_devices, jev_devices, *, pilot=False):
    qwen, baseline = role.startswith('qwen'), role.endswith('baseline')
    single_jev = len(jev_devices) == 1
    fps, frames = (4, 4096) if baseline else (2, 2048)
    checkpoint = ROOT/'models'/('Qwen3.6-35B-A3B' if qwen else 'VideoJev-4B')
    script = 'eval_longvideobench_qwen36.py' if qwen else 'eval_longvideobench_videojev.py'
    command = [sys.executable, '-u', str(ROOT/'scripts'/script), '--manifest', cell['manifest'],
        '--out', str(Path(cell['directory'])/role), '--checkpoint', str(checkpoint),
        '--seed', str(cell['seed']), '--device', 'cuda:0', '--fps', str(fps), '--max-frames', str(frames),
        '--context', '262144', '--visual-token-budget', '24576', '--memory-fraction', '.9']
    if qwen:
        command += ['--gpu-count', '6', '--max-memory-gib', '18.5']
    prefix = 'pilot' if pilot else f'group{cell["group"]}_seed{cell["seed"]}'
    return dict(name=f'{prefix}_{role}', lane='qwen' if qwen else 'videojev' if single_jev else 'videojev4' if baseline else 'videojev2',
        role=role, method='qwen36' if qwen else 'videojev', manifest=cell['manifest'], seed=cell['seed'],
        directory=str(Path(cell['directory'])/role), fps=fps, max_frames=frames, command=command,
        devices=qwen_devices if qwen else jev_devices[0 if single_jev or baseline else 1])


def model_lanes(cells, plan, *, pilot=False):
    keys = ('qwen', 'videojev') if len(plan['videojev_devices']) == 1 else ('qwen', 'videojev4', 'videojev2')
    lanes = {key: [] for key in keys}
    for cell in cells:
        for role in ROLES:
            job = model_job(cell, role, plan['devices'], plan['videojev_devices'], pilot=pilot)
            lanes[job['lane']].append(job)
    return lanes


def cpu_jobs(cell):
    out, manifest = Path(cell['directory']), cell['manifest']
    prefix = f'group{cell["group"]}_seed{cell["seed"]}'
    common = dict(devices='', lane='cpu')
    return [dict(common, name=prefix+'_controller', directory=str(out/'controller'), command=[sys.executable,
        str(ROOT/'scripts/eval_longvideobench_controller.py'), '--manifest', manifest,
        '--qwen', str(out/'qwen_controller'), '--videojev', str(out/'videojev_controller'),
        '--out', str(out/'controller'), '--beta', '2', '--qwen-method', 'qwen36']),
        dict(common, name=prefix+'_summary', directory=str(out/'summary'), command=[sys.executable,
        str(ROOT/'scripts/summarize_longvideobench_qwen36.py'), '--seed', str(cell['seed']), '--manifest', manifest,
        '--qwen-baseline', str(out/'qwen_baseline'), '--qwen-controller', str(out/'qwen_controller'),
        '--videojev-baseline', str(out/'videojev_baseline'), '--videojev-controller', str(out/'videojev_controller'),
        '--controller', str(out/'controller'), '--out', str(out/'summary')])]


def final_summary_job(master):
    return dict(name='expanded_groups_summary', lane='cpu', devices='', directory=str(master/'summary'),
        command=[sys.executable, str(ROOT/'scripts/summarize_longvideobench_expanded_groups.py'), '--all', '--base', str(master)])


def source_paths():
    paths = [ROOT/'scripts'/name for name in SCRIPTS]
    paths += sorted((ROOT/'timejev').glob('*.py'))
    paths += sorted((ROOT/'models/VideoJev-4B').glob('*.py'))
    paths += [ROOT/'.deps-qwen36/grouped_gemm/backend.py', ROOT/'.deps-qwen36/grouped_gemm/__init__.py',
        ROOT/'.deps-qwen36/grouped_gemm_backend.cpython-312-x86_64-linux-gnu.so',
        ROOT/'.deps-qwen36/src/grouped_gemm/csrc/grouped_gemm.cu', ROOT/'.deps-qwen36/src/grouped_gemm/csrc/ops.cu',
        ROOT/'reports/setup-qwen36-runtime/sm89-cutlass.patch']
    return sorted(set(paths))


def packages():
    found = {}
    for name in ('torch', 'torchvision', 'transformers', 'triton', 'flash-attn',
                 'causal-conv1d', 'fla-core', 'numpy', 'av', 'accelerate'):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def old_preflight():
    for relative, expected in read(OLD/'source_fingerprints.json').items():
        require(digest(ROOT/relative) == expected, f'旧实验冻结源码变化：{relative}')
    for entries in read(OLD/'extra_source_fingerprints.json').values():
        for relative, expected in entries.items():
            require(digest(ROOT/relative) == expected, f'旧实验额外冻结内容变化：{relative}')
    result = load_script('lvb_expanded_old_preflight', OLD/'full_preflight.py').preflight(check_only=True)
    original = read(OLD/'full_preflight.json')
    require(original['passed'] is True and result['qwen_files'] == original['qwen_files'],
            '模型 size/mtime 与旧实验已核验内容不同')
    identity = read(ROOT/'runs/lvb100_qwen36_35b_controller_seed42/qwen_baseline/run.json')['identity']
    require(packages() == dict(identity['packages'], accelerate=identity['settings']['backend']['accelerate_version']),
            '当前依赖版本与旧实验实际环境不同')
    return result


def inspect_inputs(args):
    qwen = devices(args.devices, 6)
    jev = devices(args.videojev_devices, (1, 2))
    require(not set(qwen) & set(jev), 'Qwen 和 VideoJev 两条 lane 不得共享 GPU')
    for path in source_paths():
        require(path.is_file(), f'准备所需源码/后端尚未就绪：{path}')
    evidence = old_preflight()
    from scripts.summarize_longvideobench_qwen36 import read_sample
    cells, rows_by_id, references, media = [], {}, {}, {}
    for group in (2, 3):
        for seed in SEEDS:
            directory = args.out/f'group{group}/seed{seed}'
            manifest = directory/'sample.jsonl'
            rows, sample = read_sample(manifest, seed)
            cells.append(dict(group=group, seed=seed, directory=str(directory), manifest=str(manifest),
                              manifest_sha256=sample['sha256']))
            for path in (manifest, manifest.with_suffix('.meta.json')):
                references[str(path)] = digest(path)
            for key in ('source_manifest', 'source_reference'):
                references[sample[key]['path']] = sample[key]['sha256']
            for row in rows:
                if row['id'] in rows_by_id:
                    require(rows_by_id[row['id']] == row, '跨seed相同问题的内容不同')
                rows_by_id[row['id']] = row
                path = Path(row['video_path'])
                stat = path.stat()
                media[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    ordered = sorted(rows_by_id.values(), key=lambda row: (row['duration'], row['id']))
    pilot_rows = [ordered[0], ordered[-1]]
    require(pilot_rows[0]['id'] != pilot_rows[1]['id'], 'pilot 需要不同的最短、最长样本')
    pilot_text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in pilot_rows)
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}
    plan = dict(schema='lvb_expanded_runner_v1', cells=cells, devices=','.join(qwen), videojev_devices=jev,
        context=262144, visual_token_budget=24576, beta=2, qwen_dtype='bfloat16', quantized=False,
        model_steps=24, new_groups=[2, 3], seeds=list(SEEDS), references=references, media=media,
        pilot=dict(group=0, seed=42, directory=str(args.out/'pilot'), manifest=str(args.out/'pilot/sample.jsonl'),
            selected_ids=[r['id'] for r in pilot_rows], selected_durations=[r['duration'] for r in pilot_rows],
            manifest_sha256=hashlib.sha256(pilot_text.encode()).hexdigest()))
    return plan, source_hashes, pilot_text, evidence


def validate_frozen(master):
    require(digest(master/'source_fingerprints.json') == read(master/'preflight.json')['source_fingerprints_sha256'],
            '源码指纹清单变化')
    for relative, expected in read(master/'source_fingerprints.json').items():
        require(digest(ROOT/relative) == expected, f'新增实验冻结源码变化：{relative}')
        if not relative.endswith('.so'):
            require(digest(master/'source'/relative) == expected, f'新增实验源码快照变化：{relative}')
    plan = read(master/'run_plan.json')
    require(digest(master/'run_plan.json') == read(master/'preflight.json')['run_plan_sha256'], '运行计划变化')
    for path, expected in plan['references'].items():
        require(digest(path) == expected, f'样本或源标注变化：{path}')
    for path, expected in plan['media'].items():
        stat = Path(path).stat()
        require(dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) == expected, f'媒体 size/mtime 变化：{path}')
    require(digest(plan['pilot']['manifest']) == plan['pilot']['manifest_sha256'], 'pilot 样本变化')
    return plan


def prepare(args):
    if args.check_only and (args.out/'preflight.json').exists():
        plan = validate_frozen(args.out)
        old_preflight()
        return dict(checked=True, phase='prepare', cells=len(plan['cells']), model_steps=24)
    plan, sources, pilot_text, evidence = inspect_inputs(args)
    if args.check_only:
        return dict(checked=True, phase='prepare', cells=len(plan['cells']), model_steps=24,
                    pilot_ids=plan['pilot']['selected_ids'])
    require(not any((args.out/name).exists() for name in ('source', 'source_fingerprints.json',
        'run_plan.json', 'launch_environment.json', 'preflight.json', 'pilot')), 'prepare 产物已存在，拒绝覆盖')
    (args.out/'source').mkdir(parents=True)
    for relative in sources:
        if not relative.endswith('.so'):
            target = args.out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
    write(args.out/'source_fingerprints.json', sources, new=True)
    write(args.out/'run_plan.json', plan, new=True)
    (args.out/'pilot').mkdir()
    (args.out/'pilot/sample.jsonl').write_text(pilot_text)
    write(args.out/'launch_environment.json', dict(created_at=now(), python=sys.executable, hostname=socket.gethostname(),
        pid=os.getpid(), command=sys.argv, environment={k: environment(plan['devices'])[k] for k in ENV_KEYS},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        packages=packages()), new=True)
    write(args.out/'preflight.json', dict(passed=True, prepared_at=now(), run_plan_sha256=digest(args.out/'run_plan.json'),
        source_fingerprints_sha256=digest(args.out/'source_fingerprints.json'),
        prior_preflight_path=str(OLD/'full_preflight.json'), prior_preflight_sha256=digest(OLD/'full_preflight.json'),
        qwen_files=evidence['qwen_files'], limitations=[
            '权重复用既有完整SHA并核对size/mtime；每个实际evaluator仍重新计算checkpoint和媒体完整SHA。',
            '准备期只记录新增媒体size/mtime；不代表已重新解码所有视频或验证训练/验证隔离。']), new=True)
    return dict(prepared=True, cells=6, model_steps=24)


def unused(master, job):
    for folder, suffix in (('steps', '.json'), ('steps', '.exitcode'), ('logs', '.log')):
        require(not (master/folder/f'{job["name"]}{suffix}').exists(), f'步骤已有现场，禁止自动重试：{job["name"]}')
    if job.get('directory'):
        path = Path(job['directory'])
        require(not path.exists() or path.is_dir() and not any(path.iterdir()), f'模型或汇总输出非空：{path}')


def run_step(master, job, phase, stopped):
    unused(master, job)
    validate_frozen(master)
    env = environment(job['devices'])
    path, log = master/'steps'/f'{job["name"]}.json', master/'logs'/f'{job["name"]}.log'
    item = dict(name=job['name'], phase=phase, command=job['command'], spawn_command=shell_command(job['command']),
        lane=job['lane'], environment={k: env[k] for k in ENV_KEYS}, cuda_visible_devices=job['devices'],
        started_at=now(), hostname=socket.gethostname(), parent_pid=os.getpid(), log=str(log),
        status='starting', pid=None, signals=[])
    write(path, item, new=True)
    child, code, cancelled_at, killed = None, None, None, False
    try:
        require(not stopped.is_set(), '同阶段已有失败，不再启动子进程')
        with log.open('xb') as stream:
            child = subprocess.Popen(item['spawn_command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            item.update(pid=child.pid, process_group=child.pid, status='running')
            write(path, item)
            while True:
                try:
                    code = child.wait(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    if stopped.is_set() and cancelled_at is None:
                        cancelled_at = time.monotonic()
                        item['signals'].append(dict(signal='SIGTERM', requested_at=now(), reason='peer_failure_or_queue_signal'))
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    elif cancelled_at is not None and not killed and time.monotonic()-cancelled_at > 10:
                        killed = True
                        item['signals'].append(dict(signal='SIGKILL', requested_at=now(), reason='termination_timeout'))
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            stream.flush()
            os.fsync(stream.fileno())
        if code != 0:
            stopped.set()
        item.update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=now(), log_sha256=digest(log))
        write(path, item)
        (master/'steps'/f'{job["name"]}.exitcode').write_text(str(code)+'\n')
        require(code == 0, f'{job["name"]} 实际退出码 {code}')
    except BaseException as error:
        stopped.set()
        if child is not None and code is None:
            item['signals'].append(dict(signal='SIGTERM', requested_at=now(), reason='step_exception'))
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                item['signals'].append(dict(signal='SIGKILL', requested_at=now(), reason='termination_timeout'))
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
        item.update(status='failed', exit_code=code, finished_at=now(), error=repr(error))
        if log.exists():
            item['log_sha256'] = digest(log)
        write(path, item)
        if code is not None:
            (master/'steps'/f'{job["name"]}.exitcode').write_text(str(code)+'\n')
        raise
    return item


def run_lanes(master, lanes, phase, stopped):
    def lane(jobs):
        for job in jobs:
            if stopped.is_set():
                return
            try:
                run_step(master, job, phase, stopped)
                try:
                    verify_model(job)
                except Exception as error:
                    path = master/'steps'/f'{job["name"]}.json'
                    item = read(path)
                    item['output_validation'] = dict(passed=False, error=repr(error), checked_at=now())
                    write(path, item)
                    raise ValueError(f'{job["name"]} 输出核验失败：{error}') from error
            except BaseException:
                stopped.set()
                raise

    errors = []
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {pool.submit(lane, jobs): name for name, jobs in lanes.items()}
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                stopped.set()
                errors.append(dict(lane=futures[future], error=repr(error)))
    require(not errors and not stopped.is_set(), f'并行阶段失败，所有本队列活动child已收尾：{errors}')


def verify_model(job):
    from scripts.eval_longvideobench_controller import read_source
    from scripts.summarize_longvideobench_qwen36 import validate_audits
    from timejev.lvb_experiment import load_sample
    rows = load_sample(job['manifest'])
    predictions, source, metrics = read_source(job['directory'], rows, digest(job['manifest']), job['method'])
    identity = source['identity']
    metadata = read(Path(job['directory'])/'run.json')
    require(metadata['cuda_visible_devices'] == job['devices'] and metadata['device'] == 'cuda:0'
            and metadata['memory_fraction'] == .9, '模型实际设备或显存比例与计划不同')
    require(identity['seed'] == job['seed'] and all(identity['protocol'][key] == expected for key, expected in
        dict(fps=job['fps'], max_frames=job['max_frames'], context=262144, visual_token_budget=24576).items()),
        '模型实际seed或输入协议与计划不同')
    validate_audits(rows, predictions, identity)
    if job['method'] == 'qwen36':
        loading = read(Path(job['directory'])/'loading.json')
        require(loading['patched_expert_layers'] == 40 and loading['quantized'] is False
                and loading['cpu_or_disk_offload'] is False and set(loading['parameter_dtypes']) == {'torch.bfloat16'}
                and len(loading['parameter_bytes_by_gpu']) == 6
                and all(not loading['loading_info'].get(key) for key in
                    ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), 'Qwen 六卡BF16加载审计不符')
    return dict(source=source, metrics=metrics, preprocessing={key: value['preprocessing'] for key, value in predictions.items()})


def require_pilot(master, plan):
    saved = read(master/'pilot/execution.json')
    require(saved.get('passed') is True, '四种模型协议pilot尚未完整成功')
    jobs = [job for jobs in model_lanes([plan['pilot']], plan, pilot=True).values() for job in jobs]
    for job in jobs:
        step = read(master/'steps'/f'{job["name"]}.json')
        require(step['status'] == 'completed' and step['exit_code'] == 0 and step['command'] == job['command']
                and step['cuda_visible_devices'] == job['devices']
                and digest(step['log']) == step['log_sha256']
                and (master/'steps'/f'{job["name"]}.exitcode').read_text().strip() == '0', 'pilot步骤真实退出/日志不符')
        fresh = verify_model(job)
        require(fresh == saved['models'][job['role']], 'pilot产物与完成核验记录不一致')


def execute(args):
    plan = validate_frozen(args.out)
    require(Path(sys.executable).resolve() == Path(read(args.out/'launch_environment.json')['python']).resolve(),
            '执行解释器与prepare记录不同')
    require(packages() == read(args.out/'launch_environment.json')['packages'], '执行依赖版本与prepare不同')
    require(args.devices == plan['devices'] and args.videojev_devices.split(',') == plan['videojev_devices'],
            '执行选卡与prepare冻结计划不同')
    evidence = old_preflight()
    require(evidence['qwen_files'] == read(args.out/'preflight.json')['qwen_files'], '模型与prepare时证据不同')
    pilot = args.phase == 'pilot'
    if not pilot:
        require_pilot(args.out, plan)
    lanes = model_lanes([plan['pilot']] if pilot else plan['cells'], plan, pilot=pilot)
    jobs = [job for lane in lanes.values() for job in lane]
    for job in jobs:
        unused(args.out, job)
    if not pilot:
        for cell in plan['cells']:
            for job in cpu_jobs(cell):
                unused(args.out, job)
        unused(args.out, final_summary_job(args.out))
        for group in (1, 2, 3):
            directory = args.out/f'group{group}_summary'
            require(not directory.exists() or directory.is_dir() and not any(directory.iterdir()),
                    f'分组汇总输出已非空：{directory}')
    queue_path = args.out/('pilot/queue_execution.json' if pilot else 'queue_execution.json')
    require(not queue_path.exists(), '该阶段已执行，禁止自动重试或覆盖')
    if args.check_only:
        return dict(checked=True, phase=args.phase, model_steps=len(jobs), devices=plan['devices'],
                    videojev_devices=plan['videojev_devices'])
    for name in ('steps', 'logs'):
        (args.out/name).mkdir(exist_ok=True)
    state = dict(status='running', phase=args.phase, started_at=now(), pid=os.getpid(), hostname=socket.gethostname(),
        command=sys.argv, run_plan_sha256=digest(args.out/'run_plan.json'), completed_model_steps=0)
    write(queue_path, state, new=True)
    stopped = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_lanes(args.out, lanes, args.phase, stopped)
        checked = {job['name']: verify_model(job) for job in jobs}
        state['completed_model_steps'] = len(jobs)
        if pilot:
            by_role = {job['role']: checked[job['name']] for job in jobs}
            require(all(value['metrics']['n'] == 2 for value in by_role.values()), 'pilot不是完整两题')
            for suffix in ('baseline', 'controller'):
                require(by_role[f'qwen_{suffix}']['preprocessing'] == by_role[f'videojev_{suffix}']['preprocessing'],
                        f'pilot两模型{suffix}几何不一致')
            write(args.out/'pilot/execution.json', dict(passed=True, finished_at=now(), models=by_role,
                note='最短/最长两题，仅验证四种协议实现，不用于调参或正式指标'), new=True)
        else:
            for cell in plan['cells']:
                for job in cpu_jobs(cell):
                    run_step(args.out, job, 'full', stopped)
            run_step(args.out, final_summary_job(args.out), 'full', stopped)
        require(not stopped.is_set(), '队列收到停止信号，保留现场而不报告成功')
        state.update(status='completed', finished_at=now(), note='子步骤实际退出及汇总通过，仍需根最终独立验收')
        write(queue_path, state)
    except BaseException as error:
        stopped.set()
        state['model_steps'] = {job['name']: read(args.out/'steps'/f'{job["name"]}.json') for job in jobs
                                if (args.out/'steps'/f'{job["name"]}.json').exists()}
        state['completed_model_steps'] = sum(item.get('exit_code') == 0
            and item.get('output_validation', {}).get('passed', True) for item in state['model_steps'].values())
        state.update(status='failed', finished_at=now(), error=repr(error))
        write(queue_path, state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return dict(completed=True, phase=args.phase, model_steps=len(jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'full'), required=True)
    parser.add_argument('--out', type=lambda value: Path(value).resolve(), default=ROOT/'runs/lvb100_expanded3groups_seed42_43_44')
    parser.add_argument('--devices', default='0,1,2,3,4,6')
    parser.add_argument('--videojev-devices', default='5,7', help='一张卡串行两协议，或两张卡分别执行4/2 FPS')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args) if args.phase == 'prepare' else execute(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
