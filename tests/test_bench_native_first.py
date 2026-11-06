import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import run_bench_native_first as entry
from scripts.run_bench_full import make_plan,parser


@pytest.fixture
def setup(tmp_path,monkeypatch):
    from timejev import bench_full
    args=parser().parse_args(['--bench','videomme','--base',str(tmp_path),'--videojev-gpu','7'])
    plan=make_plan(args)
    full={'path':str(tmp_path/'questions.jsonl')}
    state=dict(identity=dict(stages=plan,full_dataset=full,code_sha256={}),steps={'qwen_controller':{'status':'completed','files':{}}},status='failed')
    entry.save(tmp_path/'experiment.json',state)
    monkeypatch.setattr(entry,'reject_active_processes',lambda base:None)
    monkeypatch.setattr(bench_full,'read_full',lambda path:([],full))
    monkeypatch.setattr(bench_full,'read_source',lambda *a,**kw:None)
    calls=[]
    class Process:
        pid=123
        def __init__(self,command,**kwargs):
            calls.append(command)
            path=Path(command[command.index('--out')+1]);path.mkdir(parents=True,exist_ok=True)
            (path/'run.json').write_text('{}')
        def wait(self):return 0
    monkeypatch.setattr(entry.subprocess,'Popen',Process)
    return tmp_path,state,calls


def test_only_native_runs_then_original_runner_can_skip(setup):
    base,state,calls=setup;identity=deepcopy(state['identity'])
    entry.run(SimpleNamespace(base=str(base),dry_run=False))
    saved=json.loads((base/'experiment.json').read_text())
    assert saved['identity']==identity
    assert saved['status']=='awaiting_remaining_stages'
    assert saved['steps']['qwen_controller']==state['steps']['qwen_controller']
    assert len(calls)==3
    for seed,command in zip((42,43,44),calls):
        assert command[command.index('--seed')+1]==str(seed)
        assert command[command.index('--kind')+1]=='native'
        step=saved['steps'][f'native_seed{seed}']
        assert step['status']=='completed' and step['attempts'][0]['exit_code']==0
        entry.verify(step['files'])
    entry.run(SimpleNamespace(base=str(base),dry_run=False))
    assert len(calls)==3


def test_partial_native_uses_resume(setup):
    base,state,calls=setup
    path=base/'native/seed42';path.mkdir(parents=True);(path/'run.json').write_text('{}')
    entry.run(SimpleNamespace(base=str(base),dry_run=False))
    assert '--resume' in calls[0] and '--resume' not in calls[1]


def test_dry_run_never_writes_or_launches(setup):
    base,state,calls=setup;before=(base/'experiment.json').read_bytes()
    entry.run(SimpleNamespace(base=str(base),dry_run=True))
    assert not calls and (base/'experiment.json').read_bytes()==before


def test_changed_source_stops_before_launch(setup):
    base,state,calls=setup
    state['identity']['code_sha256']={'scripts/run_bench_full.py':'0'*64}
    entry.save(base/'experiment.json',state)
    with pytest.raises(ValueError,match='指纹'):entry.run(SimpleNamespace(base=str(base),dry_run=False))
    assert not calls


def test_failure_preserves_exit_and_stops(setup,monkeypatch):
    base,state,calls=setup
    class Failed:
        pid=12
        def __init__(self,*a,**kw):pass
        def wait(self):return 1
    monkeypatch.setattr(entry.subprocess,'Popen',Failed)
    with pytest.raises(RuntimeError,match='退出码1'):entry.run(SimpleNamespace(base=str(base),dry_run=False))
    saved=json.loads((base/'experiment.json').read_text())
    assert saved['status']=='failed'
    assert saved['steps']['native_seed42']['attempts'][0]['exit_code']==1
    assert 'native_seed43' not in saved['steps']


def test_native_first_forwards_same_hash_source_to_three_seeds(setup,monkeypatch):
    from scripts import resume_bench_fast as fast
    base,state,calls=setup
    source=base/'cached_run.json'
    source.write_text(json.dumps(dict(identity=dict(media={},checkpoint={'path':str(base),'files':{}}))))
    monkeypatch.setattr(fast,'validate_cross_stage_source',lambda *a:None)
    original_identity=deepcopy(state['identity'])
    entry.run(SimpleNamespace(base=str(base),dry_run=False,trust_input_hashes_from=str(source)))
    assert len(calls)==3
    for seed,command in zip((42,43,44),calls):
        assert command[2:5]==['--child-from',str(source),entry.digest(source)]
        assert command[command.index('--seed')+1]==str(seed)
    saved=json.loads((base/'experiment.json').read_text())
    assert saved['identity']==original_identity
    for seed in (42,43,44):
        attempt=saved['steps'][f'native_seed{seed}']['attempts'][0]
        assert attempt['trust_input_hashes_from']==str(source)
        assert attempt['model_command'][1]==str(entry.ROOT/'scripts/eval_bench_full_source.py')
