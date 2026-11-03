"""顺序执行单个benchmark的全量三seed；不依赖历史实验，不自动抢占或重试GPU。"""
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
from timejev.bench_full import BENCHMARKS, SEEDS, code_files, json_bytes, read_full, read_json, read_source, sha256, snapshot, verify_files


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
    stages=[]
    for role,kind,fps,cap in [('qwen_controller','qwen_logits',2,2048),('videojev_controller','videojev',2,2048),('videojev_baseline','videojev',4,4096)]:
        qwen=kind=='qwen_logits';directory=base/'sources'/role
        command=[qwen_python if qwen else videojev_python,str(ROOT/'scripts/eval_bench_full_source.py'),
            '--kind',kind,*common,'--out',str(directory),'--checkpoint',str(Path(args.qwen_checkpoint if qwen else args.videojev_checkpoint).resolve()),
            '--seed','42','--fps',str(fps),'--max-frames',str(cap),*(qwen_args if qwen else [])]
        stages.append(dict(name=role,kind=kind,directory=str(directory),command=command,
            cuda_visible_devices=','.join(devices) if qwen else str(args.videojev_gpu)))
    for seed in SEEDS:
        directory=base/f'native/seed{seed}'
        stages.append(dict(name=f'native_seed{seed}',kind='native',directory=str(directory),cuda_visible_devices=','.join(devices),
            command=[qwen_python,str(ROOT/'scripts/eval_bench_full_source.py'),'--kind','native',*common,
                '--out',str(directory),'--checkpoint',str(Path(args.qwen_checkpoint).resolve()),'--seed',str(seed),
                '--fps','4','--max-frames','4096','--max-new-tokens',str(args.max_new_tokens),*qwen_args]))
    stages.append(dict(name='summary',kind='cpu',directory=str(base/'summary'),cuda_visible_devices='',
        command=[qwen_python,str(ROOT/'scripts/summarize_bench_full.py'),'--manifest',str(manifest),'--base',str(base),'--out',str(base/'summary')]))
    return stages


def verify_summary(manifest, base, directory):
    from scripts.summarize_bench_full import main
    return main(['--manifest', str(manifest), '--base', str(base), '--out', str(directory), '--verify'])


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
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p


def run(args):
    base=Path(args.base).resolve();manifest=Path(args.manifest or base/'questions.jsonl').resolve()
    rows,full=read_full(manifest)
    if full['benchmark']!=args.bench:raise ValueError('benchmark与全量清单不一致')
    stages=make_plan(args)
    from scripts.eval_bench_full_source import parser as source_parser, validate
    for stage in stages:
        if stage['kind']!='cpu':validate(source_parser().parse_args(stage['command'][2:]))
    hashes=code_files()
    identity=dict(full_dataset=full,inference_seeds=list(SEEDS),selection_seed=None,stages=stages,code_sha256=hashes)
    if args.dry_run:
        result=dict(dry_run=True,gpu_started=False,output_created=False,identity=identity)
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    state_path=base/'experiment.json'
    if state_path.exists():
        state=read_json(state_path)
        if not args.resume or state['identity']!=identity:raise ValueError('队列已存在；恢复必须显式--resume且题集/源码/设置完全一致')
    else:
        if args.resume:raise ValueError('没有可恢复的experiment.json')
        if any(Path(s['directory']).exists() for s in stages):raise FileExistsError('阶段输出已存在，拒绝覆盖或冒用')
        base.mkdir(parents=True,exist_ok=True)
        snapshot(base,hashes)
        state=dict(schema='video_bench_full_queue_v1',identity=identity,status='running',started_at=now(),steps={},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
        save(state_path,state)
    logs=base/'logs';logs.mkdir(exist_ok=True)
    try:
        for stage in stages:
            verify_files({str(ROOT/name):digest for name,digest in hashes.items()})
            step=state['steps'].get(stage['name'],{})
            if step.get('status')=='completed':
                verify_files(step['files'])
                continue
            command=list(stage['command']);directory=Path(stage['directory'])
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
            record=dict(command=command,log=str(log),started_at=now())
            attempts.append(record)
            step=dict(stage,status='running',attempts=attempts);state['steps'][stage['name']]=step
            state.update(status='running',active_stage=stage['name']);save(state_path,state)
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
                read_source(directory,rows,full,stage['name'])
            else:
                verify_summary(manifest,base,directory)
            step.update(status='completed',files={str(p):sha256(p) for p in directory.rglob('*') if p.is_file()})
            step['files'].update({str(log):sha256(log),str(exit_path):sha256(exit_path)})
            save(state_path,state)
        state.update(status='completed',finished_at=now(),active_stage=None);save(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=now());save(state_path,state)
        raise
    print(json.dumps(dict(status='completed',summary=str(base/'summary/results.md')),ensure_ascii=False))
    return state


def main(argv=None):return run(parser().parse_args(argv))


if __name__=='__main__':main()
