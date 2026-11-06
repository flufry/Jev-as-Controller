"""双GPU通道调度：真实子命令不变、单写状态、失败回收、恢复不重复计算。"""
import json
from pathlib import Path
import pytest
from scripts import run_bench_full as runner


def test_parallel_rejects_overlapping_gpus_and_keeps_model_commands(tmp_path):
    common=['--bench','mlvu','--base',str(tmp_path),'--qwen-gpus','0,1,2,3','--videojev-gpu','7']
    serial=runner.make_plan(runner.parser().parse_args(common))
    parallel=runner.make_plan(runner.parser().parse_args(common+['--parallel-models']))
    assert serial==parallel
    with pytest.raises(ValueError,match='不重叠'):
        runner.make_plan(runner.parser().parse_args(common+['--videojev-gpu','3','--parallel-models']))
    # 串行仍允许两模型先后使用同一张卡。
    runner.make_plan(runner.parser().parse_args(common+['--videojev-gpu','3']))


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    args=runner.parser().parse_args(['--bench','mlvu','--base',str(tmp_path),'--qwen-gpus','0,1,2,3','--videojev-gpu','7','--parallel-models'])
    stages=runner.make_plan(args);state=dict(identity={'unchanged':True},steps={},status='running')
    logs=tmp_path/'logs';logs.mkdir()
    events=[];processes=[];fail={'name':None};checked=[]
    class Process:
        def __init__(self,command,**kw):
            self.command=command;self.output=Path(command[command.index('--out')+1]);self.output.mkdir(parents=True,exist_ok=True)
            self.name=next(s['name'] for s in stages if s['directory']==str(self.output))
            self.lane='videojev' if command[command.index('--kind')+1]=='videojev' else 'qwen'
            assert not any(p.lane==self.lane and p.code is None for p in processes)
            self.pid=100+len(processes);self.code=None;self.polls=0
            self.env=kw['env'];processes.append(self);events.append(('start',self.name))
            (self.output/'run.json').write_text('{}')
            assert self.env['CUDA_VISIBLE_DEVICES']==('7' if self.lane=='videojev' else '0,1,2,3')
        def poll(self):
            if self.code is not None:return self.code
            self.polls+=1
            threshold=5 if fail['name']=='qwen_controller' and self.lane=='videojev' else 3
            if self.polls<threshold:return None
            self.code=1 if self.name==fail['name'] else 0
            (self.output/'metrics.json').write_text('{}')
            events.append(('exit',self.name));return self.code
        def terminate(self):events.append(('terminate',self.name));self.code=-15
        def wait(self,timeout=None):return self.code
        def kill(self):self.code=-9
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(runner,'read_source',lambda directory,*a,**kw:checked.append(str(directory)))
    def execute():runner.run_parallel_models(stages,[],{}, {},state,tmp_path/'experiment.json',logs,{})
    return execute,state,events,processes,fail,stages,checked,tmp_path


def test_two_lanes_overlap_keep_order_and_no_cpu_launch(scheduler):
    execute,state,events,processes,_,stages,checked,base=scheduler
    execute()
    assert events[:2]==[('start','qwen_controller'),('start','videojev_controller')]
    assert [p.name for p in processes if p.lane=='qwen']==['qwen_controller','native_seed42','native_seed43','native_seed44']
    assert [p.name for p in processes if p.lane=='videojev']==['videojev_controller','videojev_baseline']
    assert len(processes)==6 and len(checked)==6
    assert all(s['status']=='completed' for s in state['steps'].values())
    assert state['active_stages']==[] and state['active_stage'] is None
    assert state['identity']=={'unchanged':True}
    for step in state['steps'].values():
        runner.verify_files(step['files'])
        assert step['attempts'][0]['exit_code']==0
    execute();assert len(processes)==6
    (base/'sources/qwen_controller/metrics.json').write_text('tampered')
    with pytest.raises(ValueError,match='SHA'):execute()


def test_failure_cancels_peer_and_resume_only_unfinished(scheduler):
    execute,state,events,processes,fail,stages,checked,_=scheduler
    fail['name']='qwen_controller'
    with pytest.raises(RuntimeError,match='退出码1'):execute()
    assert len(processes)==2
    assert state['steps']['qwen_controller']['status']=='failed'
    assert state['steps']['videojev_controller']['status']=='interrupted'
    assert ('terminate','videojev_controller') in events
    assert state['active_stages']==[]
    fail['name']=None
    execute()
    assert len(processes)==8
    assert '--resume' in processes[2].command and '--resume' in processes[3].command
    assert len(state['steps']['qwen_controller']['attempts'])==2


def test_interrupt_terminates_both_lanes(scheduler,monkeypatch):
    execute,state,events,processes,*_=scheduler
    def stop(_):raise KeyboardInterrupt
    monkeypatch.setattr(runner.time,'sleep',stop)
    with pytest.raises(KeyboardInterrupt):execute()
    assert len(processes)==2 and all(p.code==-15 for p in processes)
    assert all(s['status']=='interrupted' for s in state['steps'].values())
    assert state['active_stages']==[]


def test_reused_sources_do_not_start_gpu(scheduler):
    execute,state,events,processes,fail,stages,checked,base=scheduler
    for s in stages:
        if s['kind'] in ('qwen_logits','videojev'):state['steps'][s['name']]=dict(status='reused',files={})
    execute()
    assert [p.name for p in processes]==['native_seed42','native_seed43','native_seed44']


def test_result_validation_failure_stops_pending_stages(scheduler,monkeypatch):
    execute,state,events,processes,*_=scheduler
    def invalid(*a,**kw):raise ValueError('invalid predictions')
    monkeypatch.setattr(runner,'read_source',invalid)
    with pytest.raises(ValueError,match='invalid predictions'):execute()
    assert len(processes)==2
    assert state['steps']['qwen_controller']['status']=='failed'
    assert all(p.code is not None for p in processes)
    assert state['active_stages']==[]


def test_launch_failure_reaps_already_started_peer(scheduler,monkeypatch):
    execute,state,events,processes,*_=scheduler
    original=runner.subprocess.Popen
    def launch(command,**kwargs):
        if command[command.index('--kind')+1]=='videojev':raise OSError('launch failed')
        return original(command,**kwargs)
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    with pytest.raises(OSError,match='launch failed'):execute()
    assert len(processes)==1 and processes[0].code==-15
    assert state['steps']['qwen_controller']['status']=='interrupted'
    assert state['steps']['videojev_controller']['status']=='failed'
    assert 'exit_code' not in state['steps']['videojev_controller']['attempts'][0]


def test_runner_parallel_barrier_before_cpu_summaries(tmp_path,monkeypatch):
    full={'benchmark':'mlvu'}
    monkeypatch.setattr(runner,'read_full',lambda _:([],full))
    monkeypatch.setattr(runner,'code_files',lambda:{})
    monkeypatch.setattr(runner,'snapshot',lambda *a:None)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'fixture')
    barrier=[];cpu=[]
    def parallel(stages,rows,full,hashes,state,path,logs,reused):
        for s in stages:
            if s['kind']!='cpu':state['steps'][s['name']]=dict(status='completed',files={})
        barrier.append(True)
    monkeypatch.setattr(runner,'run_parallel_models',parallel)
    class Process:
        pid=123
        def __init__(self,command,**kwargs):
            assert barrier==[True]
            assert '--kind' not in command
            cpu.append(command)
            out=Path(command[command.index('--out')+1]);out.mkdir(parents=True)
            (out/'results.json').write_text('{}')
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner,'verify_summary',lambda *a:None)
    args=runner.parser().parse_args(['--bench','mlvu','--base',str(tmp_path),'--parallel-models','--videojev-gpu','7','--input-hash-mode','always'])
    state=runner.run(args)
    assert state['status']=='completed' and len(cpu)==4
    assert state['identity']['scheduling']=='parallel_qwen_videojev_then_cpu_v1'
    args.resume=True
    runner.run(args)
    assert len(cpu)==4
    args.parallel_models=False
    with pytest.raises(ValueError,match='恢复'):runner.run(args)


def test_parallel_models_share_one_read_only_cache(scheduler):
    execute,state,events,processes,_,_,_,base=scheduler
    cache=base/'input_hash_cache.json';cache.write_text('{}')
    state['input_hash_cache']={'path':str(cache),'sha256':runner.sha256(cache)}
    execute()
    assert len(processes)==6
    for process in processes:
        assert Path(process.command[1]).name=='run_bench_cached_source.py'
        assert process.command[process.command.index('--cache')+1]==str(cache)
        attempt=state['steps'][process.name]['attempts'][0]
        assert Path(attempt['model_command'][1]).name=='eval_bench_full_source.py'
    assert cache.read_text()=='{}'
