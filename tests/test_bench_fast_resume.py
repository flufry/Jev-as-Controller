"""显式快速恢复：只复用大文件旧哈希，原实验身份校验和小文件校验保留。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import resume_bench_fast as fast


@pytest.fixture
def inputs(tmp_path):
    video=tmp_path/'video.mp4';video.write_bytes(b'original video')
    checkpoint=tmp_path/'model';checkpoint.mkdir()
    config=checkpoint/'config.json';config.write_text('{}')
    weight=checkpoint/'model.safetensors';weight.write_bytes(b'original weights')
    digest=lambda p:dict(bytes=p.stat().st_size,sha256=fast.file_hash(p))
    metadata=dict(identity=dict(media={'v':dict(path=str(video),mtime_ns=video.stat().st_mtime_ns,**digest(video))},
        checkpoint=dict(path=str(checkpoint),files={p.name:digest(p) for p in (config,weight)})))
    return metadata,video,weight,config


def test_reuse_skips_large_file_reads_but_hashes_config(inputs):
    metadata,video,weight,config=inputs
    cache=fast.CachedInputHashes(metadata,'fixture');cache.preflight()
    calls=[]
    def original(path):
        calls.append(Path(path));return fast.file_hash(path)
    for path in (video,weight,config):
        assert cache.reuse_or_hash(path,original)==fast.file_hash(path)
    assert calls==[config]
    assert cache.counts=={'video':1,'weight':1}
    assert cache.audit()['weight_mtime_unavailable']==1


@pytest.mark.parametrize('change',['video_size','video_mtime','weight_size','missing'])
def test_changed_stats_are_rejected(inputs,change):
    metadata,video,weight,_=inputs
    cache=fast.CachedInputHashes(metadata,'fixture')
    if change=='video_size':video.write_bytes(b'changed long video')
    if change=='video_mtime':os.utime(video,ns=(video.stat().st_atime_ns,video.stat().st_mtime_ns+10_000_000))
    if change=='weight_size':weight.write_bytes(b'changed')
    if change=='missing':video.unlink()
    with pytest.raises(ValueError):cache.preflight()


def test_documented_limit_same_size_weight_is_not_content_verified(inputs):
    metadata,_,weight,_=inputs
    cache=fast.CachedInputHashes(metadata,'fixture')
    before=fast.file_hash(weight)
    weight.write_bytes(b'x'*weight.stat().st_size)
    cache.preflight()
    assert fast.file_hash(weight)!=before
    assert cache.reuse_or_hash(weight,lambda _:pytest.fail('权重不应重新读取'))==before
    assert cache.audit()['weight_mtime_unavailable']==1


def test_if_weight_mtime_was_recorded_it_is_checked(inputs):
    metadata,_,weight,_=inputs
    metadata['identity']['checkpoint']['files'][weight.name]['mtime_ns']=weight.stat().st_mtime_ns
    cache=fast.CachedInputHashes(metadata,'fixture')
    os.utime(weight,ns=(weight.stat().st_atime_ns,weight.stat().st_mtime_ns+10_000_000))
    with pytest.raises(ValueError,match='修改时间'):cache.preflight()


def test_no_cache_entry_falls_back_to_real_hash(inputs,tmp_path):
    metadata,*_=inputs
    cache=fast.CachedInputHashes(metadata,'fixture')
    other=tmp_path/'unrecorded.mp4';other.write_bytes(b'new')
    assert cache.reuse_or_hash(other,fast.file_hash)==fast.file_hash(other)
    assert other not in cache.used


def test_child_requires_existing_resume_and_exact_entry(tmp_path):
    with pytest.raises(ValueError,match='全量模型入口'):fast.run_child(['other.py'])
    with pytest.raises(ValueError,match='--resume'):fast.run_child([str(fast.SOURCE_ENTRY)])
    with pytest.raises(ValueError,match='run.json'):
        fast.run_child([str(fast.SOURCE_ENTRY),'--resume','--out',str(tmp_path)])
    with pytest.raises(ValueError,match='--resume'):fast.main(['--bench','videomme'])


def test_native_and_regular_sha_aliases_restored(inputs):
    from timejev import lvb_experiment as lvb,lvb_native_generation as native
    metadata,video,weight,config=inputs
    cache=fast.CachedInputHashes(metadata,'fixture')
    old=(lvb.sha256,native.sha256)
    with fast.reuse_hashes(cache):
        assert lvb.sha256(video)==fast.file_hash(video)
        assert native.sha256(video)==fast.file_hash(video)
        assert lvb.checkpoint_identity(weight.parent)==metadata['identity']['checkpoint']
    assert (lvb.sha256,native.sha256)==old


def test_child_keeps_original_argv_identity_and_records_audit(inputs,tmp_path,monkeypatch):
    metadata,video,weight,_=inputs
    out=tmp_path/'run';out.mkdir()
    raw=json.dumps(metadata).encode();(out/'run.json').write_bytes(raw)
    entry_args=['--resume','--out',str(out),'--checkpoint',str(weight.parent)]
    original_argv=sys.argv
    def run(path,run_name):
        assert path==str(fast.SOURCE_ENTRY) and run_name=='__main__'
        assert sys.argv==[str(fast.SOURCE_ENTRY),*entry_args]
        from timejev import lvb_experiment as lvb
        assert lvb.sha256(video)==fast.file_hash(video)
    monkeypatch.setattr(fast.runpy,'run_path',run)
    fast.run_child([str(fast.SOURCE_ENTRY),*entry_args])
    assert sys.argv is original_argv
    assert (out/'run.json').read_bytes()==raw
    audit=json.loads((out/'input_hash_reuse.jsonl').read_text(encoding='utf-8'))
    assert audit['used_files']==1
    assert audit['source_run_sha256']==hashlib.sha256(raw).hexdigest()


def test_parent_only_wraps_resumed_sources_with_metadata(tmp_path,monkeypatch):
    from scripts import run_bench_full as runner
    out=tmp_path/'out';out.mkdir();(out/'run.json').write_text('{}')
    venv_python='/opt/jev/bin/python'
    resumed=[venv_python,str(fast.SOURCE_ENTRY),'--out',str(out),'--resume']
    fresh=[venv_python,str(fast.SOURCE_ENTRY),'--out',str(out)]
    summary=[venv_python,str(fast.ROOT/'scripts/summarize_bench_full.py')]
    seen=[]
    monkeypatch.setattr(subprocess,'Popen',lambda command,*a,**kw:seen.append(command))
    def main(argv):
        assert sys.argv==[str(fast.RUNNER),'--resume']
        for command in (resumed,fresh,summary):runner.subprocess.Popen(command)
    monkeypatch.setattr(runner,'main',main)
    fast.main(['--resume'])
    assert seen[0]==[venv_python,str(fast.HELPER),'--child',*resumed[1:]]
    assert seen[1]==fresh and seen[2]==summary


def test_real_experiment_resume_keeps_identity_and_never_rehashes_large_inputs(tmp_path,monkeypatch):
    from timejev import lvb_experiment as lvb
    video=tmp_path/'video.mp4';video.write_bytes(b'video')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    weight=model/'model.safetensors';weight.write_bytes(b'weights')
    row=dict(id='q',task='choice',query='Q?',options=['A. one','B. two'],answer_index=0,
        video_key='v',video_path=str(video),duration=2.,duration_group=15)
    manifest=tmp_path/'questions.jsonl';manifest.write_text(json.dumps(row)+'\n')
    args=SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'out'),checkpoint=str(model),fps=2.,max_frames=2048,
        context=262144,visual_token_budget=24576,memory_fraction=.9,seed=42,device='cpu',resume=False)
    monkeypatch.setattr(sys,'argv',[str(fast.SOURCE_ENTRY)])
    run=lvb.Experiment(args,'qwen36',dict(test=True))
    run.append(row,[.9,.1])
    raw=(Path(args.out)/'run.json').read_bytes();cache=fast.CachedInputHashes(json.loads(raw),'fixture')
    original=lvb.sha256
    def spy(path):
        assert Path(path) not in (video,weight),'不应重新读取大文件'
        return original(path)
    monkeypatch.setattr(lvb,'sha256',spy)
    args.resume=True
    with fast.reuse_hashes(cache):
        resumed=lvb.Experiment(args,'qwen36',dict(test=True))
    assert resumed.done=={'q'} and resumed.metadata['identity']==run.metadata['identity']
    # 改了小配置文件仍会使身份校验失败，不允许用输入缓存掩盖。
    (model/'config.json').write_text('{"changed":1}')
    with fast.reuse_hashes(cache),pytest.raises(ValueError,match='指纹'):
        lvb.Experiment(args,'qwen36',dict(test=True))

@pytest.fixture
def cross_source(inputs,tmp_path):
    metadata,video,weight,config=inputs
    manifest=tmp_path/'questions.jsonl'
    manifest.write_text(json.dumps(dict(video_key='v',video_path=str(video)))+'\n',encoding='utf-8')
    metadata['identity'].update(method='qwen36',manifest_sha256=fast.file_hash(manifest))
    source=tmp_path/'qwen_run.json';source.write_text(json.dumps(metadata),encoding='utf-8')
    out=tmp_path/'new_native'
    args=['--kind','native','--manifest',str(manifest),'--checkpoint',str(weight.parent),'--out',str(out),'--seed','43']
    return metadata,source,args,out


def test_cross_stage_new_native_skips_weights_and_video(cross_source,monkeypatch):
    metadata,source,args,out=cross_source
    from timejev import lvb_experiment as lvb
    original_hash=lvb.sha256
    large={Path(metadata['identity']['media']['v']['path']),Path(metadata['identity']['checkpoint']['path'])/'model.safetensors'}
    def fail_large(path):
        assert Path(path) not in large,'大文件不应重哈希'
        return original_hash(path)
    monkeypatch.setattr(lvb,'sha256',fail_large)
    def run(entry,run_name):
        assert '--resume' not in sys.argv
        for path in large:
            expected=(metadata['identity']['media']['v']['sha256'] if path.suffix=='.mp4'
                else metadata['identity']['checkpoint']['files'][path.name]['sha256'])
            assert lvb.sha256(path)==expected
        out.mkdir();(out/'run.json').write_text('{}')
    monkeypatch.setattr(fast.runpy,'run_path',run)
    fast.run_child([str(fast.SOURCE_ENTRY),*args],source,fast.file_hash(source))
    audit=json.loads((out/'input_hash_reuse.jsonl').read_text(encoding='utf-8'))
    assert audit['cross_stage_reuse'] and audit['source_run_path']==str(source)
    assert audit['used_files']==2


@pytest.mark.parametrize('change',['manifest','model_config','inventory','media_path','model_type','source_digest'])
def test_cross_stage_rejects_incompatible_source(cross_source,change,monkeypatch):
    metadata,source,args,out=cross_source
    old_digest=fast.file_hash(source)
    if change=='manifest':Path(fast.argument(args,'--manifest')).write_text('{}\n')
    if change=='model_config':(Path(fast.argument(args,'--checkpoint'))/'config.json').write_text('{"x":1}')
    if change=='inventory':(Path(fast.argument(args,'--checkpoint'))/'extra.json').write_text('{}')
    if change=='media_path':
        other=source.parent/'different.mp4';other.write_bytes(b'original video')
        metadata['identity']['media']['v']['path']=str(other)
        metadata['identity']['media']['v']['mtime_ns']=other.stat().st_mtime_ns
    if change=='model_type':metadata['identity']['method']='videojev'
    if change in ('media_path','model_type'):source.write_text(json.dumps(metadata))
    expected=old_digest if change=='source_digest' else fast.file_hash(source)
    if change=='source_digest':source.write_text(json.dumps(metadata)+' ')
    monkeypatch.setattr(fast.runpy,'run_path',lambda *a,**k:pytest.fail('不应进入模型加载'))
    with pytest.raises(ValueError):fast.run_child([str(fast.SOURCE_ENTRY),*args],source,expected)
    assert not out.exists()
