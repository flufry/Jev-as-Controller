"""一次输入哈希缓存：真实读取计数、跨模型复用、恢复及审计不改变评测身份。"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from timejev import bench_input_cache as cachemod
from scripts import run_bench_full as runner, run_bench_cached_source as child


@pytest.fixture
def inputs(tmp_path):
    video=tmp_path/'v.mp4';video.write_bytes(b'video')
    rows=[dict(id='q',query='Q?',options=['A. a','B. b'],answer_index=0,task='choice',
        video_key='v',video_path=str(video),duration=2.,duration_group=15)]
    manifest=tmp_path/'questions.jsonl';manifest.write_text(json.dumps(rows[0])+'\n',encoding='utf-8')
    models=[]
    for name in ('qwen','jev'):
        root=tmp_path/name;root.mkdir()
        (root/'config.json').write_text('{}');(root/'model.safetensors').write_bytes(name.encode())
        models.append(root)
    path=tmp_path/'input_hash_cache.json'
    return rows,manifest,models,path


def test_hashes_each_unique_input_once_and_resume_reads_no_large_files(inputs,monkeypatch):
    rows,manifest,models,path=inputs
    calls=[];original=cachemod.sha256
    def tracked(p):calls.append(Path(p));return original(p)
    monkeypatch.setattr(cachemod,'sha256',tracked)
    # 同视频多个问题也只读一次。
    report=cachemod.prepare_cache(path,rows+rows,models,manifest)
    large=[Path(rows[0]['video_path']),*[p/'model.safetensors' for p in models]]
    assert all(calls.count(p)==1 for p in large)
    before=path.read_bytes();calls.clear()
    second=cachemod.prepare_cache(path,rows,models,manifest,report['sha256'])
    assert second==report and path.read_bytes()==before
    assert not any(p in large for p in calls)
    for model in (models[0],models[0],models[0],models[0],models[1],models[1]):
        cache=cachemod.InputHashCache(path,report['sha256'],manifest,model)
        for p in (large[0],model/'model.safetensors'):
            assert cache.hash(p,lambda _:pytest.fail('large file rehash'))==original(p)
        config=model/'config.json'
        assert cache.hash(config,tracked)==original(config)
    assert not any(p in large for p in calls)


def test_incomplete_initial_hashing_can_resume_without_rereading_completed(inputs,monkeypatch):
    rows,manifest,models,path=inputs
    original=cachemod.sha256;video=Path(rows[0]['video_path'])
    def interrupted(p):
        if Path(p).name=='model.safetensors':raise KeyboardInterrupt
        return original(p)
    monkeypatch.setattr(cachemod,'sha256',interrupted)
    with pytest.raises(KeyboardInterrupt):cachemod.prepare_cache(path,rows,models,manifest)
    assert str(video) in json.loads(path.read_text())['hashes']
    def resume(p):
        assert Path(p)!=video
        return original(p)
    monkeypatch.setattr(cachemod,'sha256',resume)
    report=cachemod.prepare_cache(path,rows,models,manifest)
    assert report['files']==5


@pytest.mark.parametrize('mutation',['video_size','weight_size','weight_mtime','inventory','cache','config_content'])
def test_changed_inputs_or_cache_rejected(inputs,mutation):
    rows,manifest,models,path=inputs
    report=cachemod.prepare_cache(path,rows,models,manifest)
    video=Path(rows[0]['video_path']);weight=models[0]/'model.safetensors'
    if mutation=='video_size':video.write_bytes(b'other data')
    if mutation=='weight_size':weight.write_bytes(b'much longer')
    if mutation=='weight_mtime':os.utime(weight,ns=(weight.stat().st_atime_ns,weight.stat().st_mtime_ns+10000000))
    if mutation=='inventory':(models[0]/'extra.json').write_text('{}')
    if mutation=='cache':path.write_bytes(path.read_bytes()+b' ')
    if mutation=='config_content':
        config=models[0]/'config.json';stat=config.stat();config.write_text('[]')
        os.utime(config,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError):cachemod.InputHashCache(path,report['sha256'],manifest,models[0])


def test_modified_during_hash_is_not_cached(inputs,monkeypatch):
    rows,manifest,models,path=inputs
    original=cachemod.sha256;video=Path(rows[0]['video_path'])
    def changing(p):
        result=original(p)
        if Path(p)==video:video.write_bytes(b'changed during read')
        return result
    monkeypatch.setattr(cachemod,'sha256',changing)
    with pytest.raises(ValueError,match='变化'):cachemod.prepare_cache(path,rows,models,manifest)
    assert str(video) not in json.loads(path.read_text())['hashes']


def test_cached_child_preserves_entry_argv_and_writes_audit(inputs,tmp_path,monkeypatch):
    from timejev import lvb_experiment as lvb
    rows,manifest,models,path=inputs;report=cachemod.prepare_cache(path,rows,models,manifest)
    out=tmp_path/'result';args=[str(child.ROOT/'scripts/eval_bench_full_source.py'),'--manifest',str(manifest),'--checkpoint',str(models[0]),'--out',str(out)]
    previous=sys.argv
    def run(entry,run_name):
        assert sys.argv==args and run_name=='__main__'
        assert lvb.sha256(rows[0]['video_path'])==cachemod.sha256(rows[0]['video_path'])
        out.mkdir()
    monkeypatch.setattr(child.runpy,'run_path',run)
    child.main(['--cache',str(path),'--cache-sha256',report['sha256'],'--',*args])
    assert sys.argv is previous
    audit=json.loads((out/'input_hash_cache_use.jsonl').read_text(encoding='utf-8'))
    assert audit['reused_large_files']==1
    assert audit['cache_sha256']==report['sha256']


def test_real_experiment_identity_same_with_or_without_cache(inputs,tmp_path,monkeypatch):
    from timejev import lvb_experiment as lvb,lvb_native_generation as native
    rows,manifest,models,path=inputs;report=cachemod.prepare_cache(path,rows,models,manifest)
    args=SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'regular'),checkpoint=str(models[0]),
        context=262144,visual_token_budget=24576,fps=2.,max_frames=2048,memory_fraction=.9,seed=42,device='cpu',resume=False,limit=None)
    monkeypatch.setattr(sys,'argv',[str(child.ROOT/'scripts/eval_bench_full_source.py')])
    regular=lvb.Experiment(args,'qwen36',{})
    args.out=str(tmp_path/'cached')
    cache=cachemod.InputHashCache(path,report['sha256'],manifest,models[0])
    with cachemod.install(cache):cached=lvb.Experiment(args,'qwen36',{})
    assert regular.metadata['identity']==cached.metadata['identity']
    args.out=str(tmp_path/'native_regular')
    normal=native.NativeExperiment(args,{})
    args.out=str(tmp_path/'native_cached')
    with cachemod.install(cache):fast=native.NativeExperiment(args,{})
    assert normal.metadata['identity']==fast.metadata['identity']


def test_runner_default_order_and_cache_six_stages_resume(inputs,tmp_path,monkeypatch):
    rows,manifest,models,_=inputs;base=tmp_path/'run';events=[]
    monkeypatch.setattr(runner,'read_full',lambda _: (rows,{'benchmark':'videomme'}))
    monkeypatch.setattr(runner,'code_files',lambda:{})
    monkeypatch.setattr(runner,'snapshot',lambda *a:None)
    monkeypatch.setattr(runner,'read_source',lambda *a,**k:None)
    monkeypatch.setattr(runner,'verify_summary',lambda *a:None)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'fixture')
    class Process:
        pid=15
        def __init__(self,command,**kwargs):
            events.append(command)
            out=Path(command[command.index('--out')+1]);out.mkdir(parents=True)
            (out/'run.json').write_text('{}')
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    args=runner.parser().parse_args(['--bench','videomme','--manifest',str(manifest),'--base',str(base),
        '--qwen-checkpoint',str(models[0]),'--videojev-checkpoint',str(models[1])])
    state=runner.run(args)
    assert args.input_hash_mode=='once'
    assert [s['name'] for s in state['identity']['stages']]==['qwen_controller','native_seed42','native_seed43','native_seed44',
        'videojev_controller','videojev_baseline','summary_seed42','summary_seed43','summary_seed44','summary']
    assert all(Path(command[1]).name=='run_bench_cached_source.py' for command in events[:6])
    assert all('--cache' not in command for command in events[6:])
    hashes={command[command.index('--cache-sha256')+1] for command in events[:6]};assert len(hashes)==1
    before=len(events);args.resume=True
    runner.run(args)
    assert len(events)==before
    args.input_hash_mode='always'
    with pytest.raises(ValueError,match='恢复'):runner.run(args)


def test_cache_known_limit_same_size_preserved_mtime_content_change(inputs):
    rows,manifest,models,path=inputs
    report=cachemod.prepare_cache(path,rows,models,manifest)
    weight=models[0]/'model.safetensors';stat=weight.stat();before=cachemod.sha256(weight)
    weight.write_bytes(b'x'*stat.st_size)
    os.utime(weight,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    cache=cachemod.InputHashCache(path,report['sha256'],manifest,models[0])
    assert cachemod.sha256(weight)!=before
    assert cache.hash(weight,cachemod.sha256)==before
    assert '同大小且保留mtime' in cachemod.POLICY


def test_child_only_accepts_original_model_entry(inputs):
    rows,manifest,models,path=inputs
    report=cachemod.prepare_cache(path,rows,models,manifest)
    with pytest.raises(ValueError,match='只允许'):
        child.main(['--cache',str(path),'--cache-sha256',report['sha256'],'--','unrelated.py'])
