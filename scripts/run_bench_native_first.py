"""按已冻结队列优先执行native三seed；不改评测参数、源码指纹或其他阶段。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def now():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(files):
    for path,value in files.items():
        if digest(path)!=value:raise ValueError(f'文件指纹不一致：{path}')


def save(path,state):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def reject_active_processes(base):
    # 当前队列没有跨入口锁；Linux下拒绝同base尚存活的runner/模型子进程。
    proc=Path('/proc')
    if not proc.is_dir():return
    entries=('run_bench_full.py','resume_bench_fast.py','eval_bench_full_source.py','run_bench_native_first.py')
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:args=[s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s]
        except (OSError,PermissionError):continue
        if not any(Path(a).name in entries for a in args):continue
        for flag in ('--base','--out'):
            if flag in args and args.index(flag)+1<len(args):
                target=Path(args[args.index(flag)+1])
                if not target.is_absolute():
                    try:target=(path/'cwd').resolve()/target
                    except OSError:continue
                target=target.resolve()
                if target==base or target.is_relative_to(base):
                    raise RuntimeError(f'同一实验仍有活动进程PID={path.name}；先停止旧队列并等待子进程退出')


def selected_stages(state,base):
    stages=[]
    for seed in (42,43,44):
        matches=[s for s in state['identity']['stages'] if s['name']==f'native_seed{seed}']
        if len(matches)!=1:raise ValueError('原队列必须含唯一native seed42/43/44')
        stage=matches[0];command=stage['command']
        def value(flag):
            if command.count(flag)!=1:raise ValueError(f'原命令参数缺失/重复：{flag}')
            return command[command.index(flag)+1]
        if (stage['kind']!='native' or Path(command[1]).resolve()!=ROOT/'scripts/eval_bench_full_source.py'
                or value('--kind')!='native' or int(value('--seed'))!=seed
                or float(value('--fps'))!=4. or int(value('--max-frames'))!=4096
                or Path(value('--out')).resolve()!=base/f'native/seed{seed}'
                or Path(stage['directory']).resolve()!=base/f'native/seed{seed}'):
            raise ValueError('原native阶段参数/路径与预期不一致')
        stages.append(stage)
    return stages


def run(args):
    base=Path(args.base).resolve();state_path=base/'experiment.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    stages=selected_stages(state,base)
    trust_source=getattr(args,'trust_input_hashes_from',None)
    trust_source=Path(trust_source).resolve() if trust_source else None
    trust_digest=None
    if trust_source is not None:
        from scripts.resume_bench_fast import CachedInputHashes,validate_cross_stage_source
        raw=trust_source.read_bytes();trust_digest=hashlib.sha256(raw).hexdigest()
        metadata=json.loads(raw.decode('utf-8'))
        cache=CachedInputHashes(metadata,trust_digest);cache.preflight()
        for stage in stages:validate_cross_stage_source(metadata,stage['command'][2:],cache)
    hashes=state['identity']['code_sha256']
    files={}
    for name,value in hashes.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('非法源码路径')
        files[str(ROOT/rel)]=value
    verify(files)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,stages=stages,trust_input_hashes_from=str(trust_source) if trust_source else None,note='仅优先native；原identity不变'),ensure_ascii=False,indent=2))
        return
    reject_active_processes(base)
    from timejev.bench_full import read_full,read_source
    manifest=state['identity']['full_dataset']['path']
    rows,full=read_full(manifest)
    if full!=state['identity']['full_dataset']:raise ValueError('全量题集身份不同')
    logs=base/'logs';logs.mkdir(exist_ok=True)
    try:
        for stage in stages:
            verify(files)
            step=state['steps'].get(stage['name'],{})
            if step.get('status')=='completed':
                verify(step['files']);continue
            command=list(stage['command']);directory=Path(stage['directory'])
            if directory.exists() and (directory/'run.json').exists():command.append('--resume')
            elif directory.exists() and any(directory.iterdir()):
                raise ValueError(f'目录非空且缺少run.json，保留现场：{directory}')
            model_command=list(command)
            if trust_source is not None:
                verify({str(trust_source):trust_digest})
                helper=ROOT/'scripts/resume_bench_fast.py'
                command=[command[0],str(helper),'--child-from',str(trust_source),trust_digest,*command[1:]]
            attempts=step.get('attempts',[])
            number=len(attempts)+1
            log=logs/f'{stage["name"]}.attempt{number}.log'
            exit_path=logs/f'{stage["name"]}.attempt{number}.exitcode'
            attempt=dict(command=command,log=str(log),started_at=now(),launcher=str(Path(__file__).resolve()),launcher_sha256=digest(__file__))
            if trust_source is not None:
                attempt.update(model_command=model_command,trust_input_hashes_from=str(trust_source),
                    input_hash_source_sha256=trust_digest,hash_helper_sha256=digest(ROOT/'scripts/resume_bench_fast.py'),
                    input_hash_policy='trust_prior_video_and_weight_hashes_stat_checks_only')
            attempts.append(attempt)
            step=dict(stage,status='running',attempts=attempts)
            state['steps'][stage['name']]=step
            state.update(status='running',active_stage=stage['name'])
            save(state_path,state)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=stage['cuda_visible_devices'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            print(json.dumps(dict(event='start_native_first',stage=stage['name'],log=str(log)),ensure_ascii=False),flush=True)
            started=time.perf_counter()
            interrupted=False
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                attempt['pid']=process.pid;save(state_path,state)
                try:code=process.wait()
                except KeyboardInterrupt:
                    process.terminate();code=process.wait();interrupted=True
            exit_path.write_text(str(code)+'\n',encoding='utf-8')
            attempt.update(exit_code=code,finished_at=now(),wall_seconds=time.perf_counter()-started)
            if interrupted or code:
                step['status']='failed'
                if interrupted:raise KeyboardInterrupt
                raise RuntimeError(f'{stage["name"]}退出码{code}，未自动重试')
            read_source(directory,rows,full,stage['name'],expected_seed=stage['seed'])
            step.update(status='completed',files={str(p):digest(p) for p in directory.rglob('*') if p.is_file()})
            step['files'].update({str(log):digest(log),str(exit_path):digest(exit_path)})
            save(state_path,state)
        # 尚有VideoJev/汇总未完成，不能将整个实验标为completed。
        state.update(status='awaiting_remaining_stages',active_stage=None,native_priority_finished_at=now())
        save(state_path,state)
        print('三个native阶段已完成；之后用原run_bench_full.py及原参数 --resume 继续VideoJev和汇总。',flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=now());save(state_path,state)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--trust-input-hashes-from',metavar='RUN_JSON',
        help='显式信任该Qwen run.json的原视频/权重SHA；三seed均跳过大文件重哈希（保留stat/配置/源码校验）')
    return run(parser.parse_args(argv))


if __name__=='__main__':main()
