"""执行单个benchmark的全量三seed；可选Qwen/VideoJev双通道并行，不抢占或自动重试GPU。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from timejev.bench_input_cache import prepare_cache
from timejev.bench_full import BENCHMARKS, SEEDS, code_files, json_bytes, read_full, read_json, read_source, sha256, snapshot, verify_files, candidate_directories


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(json_bytes(value));temp.replace(path)


def make_plan(args):
    base=Path(args.base).resolve();manifest=Path(args.manifest or base/'questions.jsonl').resolve()
    devices=[x.strip() for x in args.qwen_gpus.split(',')]
    if len(devices) not in (4,5,6) or len(set(devices))!=len(devices) or any(not x.isdigit() for x in devices):
        raise ValueError('Qwen需要4/5/6个互异非负GPU编号')
    if args.videojev_gpu<0:raise ValueError('VideoJev GPU编号必须非负')
    if getattr(args,'parallel_models',False) and args.videojev_gpu in {int(x) for x in devices}:
        raise ValueError('并行模式要求VideoJev GPU与所有Qwen GPU互不重叠')
    if args.max_memory_gib is not None and args.max_memory_gib_per_gpu is not None:
        raise ValueError('两种显存预算参数不能同时指定')
    # 保留虚拟环境的解释器入口：resolve()会跟随uv/venv符号链接，
    # 直接启动底层Python后无法发现虚拟环境中的torch等依赖。
    qwen_python=os.path.abspath(os.path.expanduser(args.qwen_python))
    videojev_python=os.path.abspath(os.path.expanduser(args.videojev_python))
    common=['--manifest',str(manifest),'--context',str(args.context),'--visual-token-budget',str(args.visual_token_budget),
            '--memory-fraction',str(args.memory_fraction)]
    qwen_args=['--gpu-count',str(len(devices))]
    for flag,value in [('--max-memory-gib',args.max_memory_gib),('--max-memory-gib-per-gpu',args.max_memory_gib_per_gpu)]:
        if value is not None:qwen_args += [flag,str(value)]
    reuse=getattr(args,'reuse_sources_from',None)
    source_dirs=candidate_directories(reuse) if reuse else {role:base/'sources'/role for role in
        ('qwen_controller','videojev_controller','videojev_baseline')}
    stages=[]
    for role,kind,fps,cap in [('qwen_controller','qwen_logits',2,2048),
            ('videojev_controller','videojev',2,2048),('videojev_baseline','videojev',4,4096)]:
        qwen=kind=='qwen_logits';directory=source_dirs[role]
        command=[qwen_python if qwen else videojev_python,str(ROOT/'scripts/eval_bench_full_source.py'),
            '--kind',kind,*common,'--out',str(directory),
            '--checkpoint',str(Path(args.qwen_checkpoint if qwen else args.videojev_checkpoint).resolve()),
            '--seed','42','--fps',str(fps),'--max-frames',str(cap),*(qwen_args if qwen else [])]
        stages.append(dict(name=role,kind=kind,seed=42,source_role=role,directory=str(directory),command=command,
            reuse=bool(reuse),cuda_visible_devices=','.join(devices) if qwen else str(args.videojev_gpu)))
    for seed in SEEDS:
        directory=base/f'native/seed{seed}'
        command=[qwen_python,str(ROOT/'scripts/eval_bench_full_source.py'),'--kind','native',*common,
            '--out',str(directory),'--checkpoint',str(Path(args.qwen_checkpoint).resolve()),
            '--seed',str(seed),'--fps','4','--max-frames','4096','--max-new-tokens',str(args.max_new_tokens),*qwen_args]
        stages.append(dict(name=f'native_seed{seed}',kind='native',seed=seed,source_role=f'native_seed{seed}',
            directory=str(directory),command=command,reuse=False,cuda_visible_devices=','.join(devices)))
        directory=base/f'per_seed/seed{seed}'
        command=[qwen_python,str(ROOT/'scripts/summarize_bench_full.py'),'--manifest',str(manifest),
            '--base',str(base),'--out',str(directory),'--seed',str(seed)]
        if reuse:command+=['--source-base',str(Path(reuse).resolve())]
        stages.append(dict(name=f'summary_seed{seed}',kind='cpu',seed=seed,directory=str(directory),
            cuda_visible_devices='',command=command))
    command=[qwen_python,str(ROOT/'scripts/summarize_bench_full.py'),'--manifest',str(manifest),
        '--base',str(base),'--out',str(base/'summary')]
    if reuse:command+=['--source-base',str(Path(reuse).resolve())]
    stages.append(dict(name='summary',kind='cpu',directory=str(base/'summary'),cuda_visible_devices='',command=command))
    # 串行默认先跑完所有Qwen，再跑VideoJev；CPU汇总必须等待两类来源就绪。
    priority={'qwen_logits':0,'native':1,'videojev':2,'cpu':3}
    return sorted(stages,key=lambda stage:priority[stage['kind']])


def verify_summary(manifest, base, directory, seed=None, source_base=None):
    from scripts.summarize_bench_full import main
    argv=['--manifest', str(manifest), '--base', str(base), '--out', str(directory), '--verify']
    if seed is not None:argv+=['--seed',str(seed)]
    if source_base is not None:argv+=['--source-base',str(source_base)]
    return main(argv)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench',choices=BENCHMARKS,required=True)
    p.add_argument('--base',required=True)
    p.add_argument('--manifest',help='默认BASE/questions.jsonl')
    p.add_argument('--qwen-gpus',default='0,1,2,3,4,5')
    p.add_argument('--videojev-gpu',type=int,default=0)
    p.add_argument('--qwen-python',default=sys.executable,help='解释器完整路径；可与VideoJev环境分开')
    p.add_argument('--videojev-python',default=sys.executable)
    p.add_argument('--qwen-checkpoint',default='models/Qwen3.6-35B-A3B')
    p.add_argument('--videojev-checkpoint',default='models/VideoJev-4B')
    p.add_argument('--max-memory-gib',type=float)
    p.add_argument('--max-memory-gib-per-gpu')
    p.add_argument('--context',type=int,default=262144)
    p.add_argument('--visual-token-budget',type=int,default=24576)
    p.add_argument('--max-new-tokens',type=int,default=32768)
    p.add_argument('--memory-fraction',type=float,default=.9)
    p.add_argument('--parallel-models',action='store_true',
        help='Qwen与VideoJev使用不同GPU并行；各通道内部顺序执行，全部GPU阶段完成后统一汇总')
    p.add_argument('--input-hash-mode',choices=('once','always'),default='once',
        help='默认once：首次内容哈希，后续复用大文件SHA并检查大小/mtime；always：各模型阶段完整重新哈希')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--reuse-sources-from',help='只读复用该旧实验已完整完成的三份seed42候选来源；不修改旧目录，不复用部分预测')
    p.add_argument('--resume',action='store_true')
    return p


def validate_reused_sources(stages, rows, full):
    sources={};predictions={}
    for stage in stages:
        if not stage.get('reuse'):continue
        command=stage['command']
        predictions[stage['name']],source,_=read_source(stage['directory'],rows,full,stage['source_role'],expected_seed=42)
        for flag,key in [('--context','context'),('--visual-token-budget','visual_token_budget')]:
            if source['identity']['protocol'].get(key)!=int(command[command.index(flag)+1]):
                raise ValueError(f'复用来源的{key}与本次参数不一致')
        checkpoint=Path(command[command.index('--checkpoint')+1]).resolve()
        if Path(source['identity']['checkpoint']['path']).resolve()!=checkpoint:
            raise ValueError('复用来源checkpoint路径与本次参数不一致')
        sources[stage['name']]=source
    if sources:
        q=sources['qwen_controller']['identity'];j=sources['videojev_controller']['identity'];b=sources['videojev_baseline']['identity']
        if q['protocol']!=j['protocol'] or b['protocol']!=dict(q['protocol'],fps=4.,max_frames=4096):
            raise ValueError('三份复用来源协议不匹配')
        if q['media']!=j['media'] or q['media']!=b['media'] or j['checkpoint']!=b['checkpoint']:
            raise ValueError('三份复用来源媒体或VideoJev checkpoint指纹不一致')
        for row in rows:
            qa=predictions['qwen_controller'][row['id']]['preprocessing']
            ja=predictions['videojev_controller'][row['id']]['preprocessing']
            ba=predictions['videojev_baseline'][row['id']]['preprocessing']
            if qa!=ja or qa['sampled_duration']!=ba['sampled_duration']:
                raise ValueError('三份复用来源逐题预处理不匹配')
    return sources


def cached_model_command(command, state):
    """实际子进程使用缓存包装；原模型命令及sys.argv保持不变。"""
    cached=state.get('input_hash_cache')
    if not cached:return command
    verify_files({cached['path']:cached['sha256']})
    return [command[0],str(ROOT/'scripts/run_bench_cached_source.py'),
        '--cache',cached['path'],'--cache-sha256',cached['sha256'],'--',*command[1:]]


def run_parallel_models(stages, rows, full, hashes, state, state_path, logs, reused_sources):
    """单调度进程写状态；最多一个Qwen子进程和一个VideoJev子进程。"""
    active={}
    models=[s for s in stages if s['kind']!='cpu']
    frozen={str(ROOT/name):value for name,value in hashes.items()}

    def persist():
        names=[item['stage']['name'] for item in active.values()]
        state.update(active_stages=names,active_stage=names[0] if len(names)==1 else None)
        save(state_path,state)

    def record_exit(item,code,status):
        item['stream'].close()
        item['exit_path'].write_text(str(code)+'\n',encoding='utf-8')
        item['record'].update(exit_code=code,finished_at=now(),wall_seconds=time.perf_counter()-item['started'])
        state['steps'][item['stage']['name']]['status']=status

    try:
        for stage in models:
            step=state['steps'].get(stage['name'],{})
            if step.get('status') in ('completed','reused'):
                verify_files(step['files'])
            elif stage.get('reuse'):
                source=reused_sources[stage['name']];verify_files(source['files'])
                state['steps'][stage['name']]=dict(stage,status='reused',verified_at=now(),files=source['files'])
        state.update(status='running');persist()
        while True:
            # 先回收退出进程；发现失败时禁止启动任何后续阶段。
            for lane,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                del active[lane]
                record_exit(item,code,'failed' if code else 'verifying')
                persist()
                stage=item['stage'];step=state['steps'][stage['name']]
                if code:raise RuntimeError(f'{stage["name"]}退出码{code}；停止本队列另一通道，保留可恢复产物')
                try:
                    verify_files(frozen)
                    directory=Path(stage['directory'])
                    read_source(directory,rows,full,stage['source_role'],expected_seed=stage['seed'])
                    files={str(p):sha256(p) for p in directory.rglob('*') if p.is_file()}
                    files.update({str(item['log']):sha256(item['log']),str(item['exit_path']):sha256(item['exit_path'])})
                    step.update(status='completed',files=files)
                except BaseException:
                    step['status']='failed';raise
                print(json.dumps(dict(event='completed',stage=stage['name']),ensure_ascii=False),flush=True)
                persist()
            pending=[s for s in models if state['steps'].get(s['name'],{}).get('status') not in ('completed','reused')]
            if not pending and not active:break
            for lane in ('qwen','videojev'):
                if lane in active:continue
                stage=next((s for s in pending if ('videojev' if s['kind']=='videojev' else 'qwen')==lane),None)
                if stage is None:continue
                verify_files(frozen)
                directory=Path(stage['directory']);command=list(stage['command'])
                if directory.exists():
                    if (directory/'run.json').is_file():command.append('--resume')
                    elif any(directory.iterdir()):
                        raise ValueError(f'输出非空但缺少run.json，需保留现场检查：{directory}')
                    elif stage['kind']=='native':
                        # NativeExperiment拒绝已有空目录；仅移除本阶段空目录。
                        directory.rmdir()
                previous=state['steps'].get(stage['name'],{})
                attempts=list(previous.get('attempts',[]));attempt=len(attempts)+1
                log=logs/f'{stage["name"]}.attempt{attempt}.log'
                exit_path=logs/f'{stage["name"]}.attempt{attempt}.exitcode'
                model_command=list(command)
                command=cached_model_command(command,state)
                record=dict(command=command,model_command=model_command,log=str(log),started_at=now(),lane=lane)
                attempts.append(record)
                state['steps'][stage['name']]=dict(stage,status='starting',attempts=attempts)
                persist()
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=stage['cuda_visible_devices'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                stream=log.open('x',encoding='utf-8')
                started=time.perf_counter()
                try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                except BaseException as error:
                    stream.close();record.update(launch_error=repr(error),finished_at=now())
                    state['steps'][stage['name']]['status']='failed';raise
                record['pid']=process.pid
                state['steps'][stage['name']]['status']='running'
                active[lane]=dict(stage=stage,process=process,stream=stream,log=log,exit_path=exit_path,record=record,started=started)
                persist()
                print(json.dumps(dict(event='start',stage=stage['name'],lane=lane,log=str(log),
                    cuda_visible_devices=stage['cuda_visible_devices']),ensure_ascii=False),flush=True)
            if active:time.sleep(.5)
        persist()
    except BaseException:
        # 只终止本调度器启动且仍在运行的进程；不触碰服务器其他任务。
        for lane,item in list(active.items()):
            process=item['process'];code=process.poll()
            if code is None:
                process.terminate()
                try:code=process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill();code=process.wait()
            record_exit(item,code,'interrupted')
            del active[lane]
        persist()
        raise


def run(args):
    base=Path(args.base).resolve();manifest=Path(args.manifest or base/'questions.jsonl').resolve()
    rows,full=read_full(manifest)
    if full['benchmark']!=args.bench:raise ValueError('benchmark与全量清单不一致')
    stages=make_plan(args)
    from scripts.eval_bench_full_source import parser as source_parser, validate
    for stage in stages:
        if stage['kind']!='cpu':validate(source_parser().parse_args(stage['command'][2:]))
    reused_sources=validate_reused_sources(stages,rows,full)
    hashes=code_files()
    identity=dict(input_hash_mode=args.input_hash_mode,reused_sources=reused_sources,execution_protocol='shared_candidate_sources_three_sampling_seeds_v1',full_dataset=full,inference_seeds=list(SEEDS),selection_seed=None,stages=stages,code_sha256=hashes)
    identity['scheduling']=('parallel_qwen_videojev_then_cpu_v1' if getattr(args,'parallel_models',False)
        else 'qwen_then_videojev_then_cpu_v1')
    if args.dry_run:
        result=dict(dry_run=True,gpu_started=False,output_created=False,identity=identity)
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    state_path=base/'experiment.json'
    if state_path.exists():
        state=read_json(state_path)
        if not args.resume or state['identity']!=identity:raise ValueError('队列已存在；恢复必须显式--resume且题集/源码/设置完全一致')
    else:
        if args.resume:raise ValueError('没有可恢复的experiment.json')
        if any(Path(s['directory']).exists() for s in stages if not s.get('reuse')):raise FileExistsError('阶段输出已存在，拒绝覆盖或冒用')
        base.mkdir(parents=True,exist_ok=True)
        snapshot(base,hashes)
        state=dict(schema='video_bench_full_queue_v4',identity=identity,status='running',started_at=now(),steps={},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
        save(state_path,state)
    logs=base/'logs';logs.mkdir(exist_ok=True)
    try:
        if args.input_hash_mode=='once':
            state.update(status='checking_input_hash_cache',active_stage=None,active_stages=[])
            save(state_path,state)
            previous=state.get('input_hash_cache')
            cache=prepare_cache(base/'input_hash_cache.json',rows,
                [args.qwen_checkpoint,args.videojev_checkpoint],manifest,
                expected_sha256=previous['sha256'] if previous else None)
            state.update(input_hash_cache=cache,status='running');save(state_path,state)
        if getattr(args,'parallel_models',False):
            run_parallel_models(stages,rows,full,hashes,state,state_path,logs,reused_sources)
        for stage in stages:
            verify_files({str(ROOT/name):digest for name,digest in hashes.items()})
            step=state['steps'].get(stage['name'],{})
            if step.get('status') in ('completed','reused'):
                verify_files(step['files'])
                continue
            command=list(stage['command']);directory=Path(stage['directory'])
            if stage.get('reuse'):
                source=reused_sources[stage['name']]
                verify_files(source['files'])
                step=dict(stage,status='reused',verified_at=now(),files=source['files'])
                state['steps'][stage['name']]=step;save(state_path,state)
                print(json.dumps(dict(event='reused',stage=stage['name'],directory=str(directory)),ensure_ascii=False),flush=True)
                continue
            if directory.exists():
                if stage['kind']=='cpu':
                    if not (directory/'artifact_manifest.json').is_file():
                        raise ValueError('汇总目录不完整，请保留现场并移走该summary目录后--resume；模型来源不必重跑')
                    command.append('--verify')
                else:command.append('--resume')
            attempt=len(step.get('attempts',[]))+1
            log=logs/f'{stage["name"]}.attempt{attempt}.log'
            exit_path=logs/f'{stage["name"]}.attempt{attempt}.exitcode'
            attempts=step.get('attempts',[])
            model_command=list(command)
            if stage['kind']!='cpu':command=cached_model_command(command,state)
            record=dict(command=command,model_command=model_command,log=str(log),started_at=now())
            attempts.append(record)
            step=dict(stage,status='running',attempts=attempts);state['steps'][stage['name']]=step
            state.update(status='running',active_stage=stage['name'],active_stages=[stage['name']]);save(state_path,state)
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=stage['cuda_visible_devices'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            started=time.perf_counter()
            print(json.dumps(dict(event='start',stage=stage['name'],log=str(log)),ensure_ascii=False),flush=True)
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                record['pid']=process.pid;save(state_path,state)
                try:code=process.wait()
                except KeyboardInterrupt:
                    process.terminate();process.wait();raise
            exit_path.write_text(str(code)+'\n',encoding='utf-8')
            record.update(exit_code=code,finished_at=now(),wall_seconds=time.perf_counter()-started)
            if code:
                step['status']='failed';raise RuntimeError(f'{stage["name"]}退出码{code}，已保留日志；不自动重试')
            if stage['kind']!='cpu':
                read_source(directory,rows,full,stage['source_role'],expected_seed=stage['seed'])
            else:
                verify_summary(manifest,base,directory,stage.get('seed'),getattr(args,'reuse_sources_from',None))
            step.update(status='completed',files={str(p):sha256(p) for p in directory.rglob('*') if p.is_file()})
            step['files'].update({str(log):sha256(log),str(exit_path):sha256(exit_path)})
            save(state_path,state)
        state.update(status='completed',finished_at=now(),active_stage=None,active_stages=[]);save(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=now());save(state_path,state)
        raise
    print(json.dumps(dict(status='completed',summary=str(base/'summary/results.md')),ensure_ascii=False))
    return state


def main(argv=None):return run(parser().parse_args(argv))


if __name__=='__main__':main()
