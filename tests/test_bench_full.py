"""全量三seed协议：官方库存、CDF采样、M-Avg、4FPS基线OR与安全恢复。"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from timejev import bench_full as full
from scripts import run_bench_full as runner
from scripts.bench_io import json_bytes, jsonl_bytes, sha256


def rows_fixture(tmp_path):
    video=tmp_path/'video.mp4';video.write_bytes(b'test media')
    # 七任务不等题数，确保M-Avg不能偷换为微平均accuracy。
    from timejev.mlvu_benchmark import TASK_COUNTS
    rows=[]
    for index,task in enumerate(TASK_COUNTS):
        for i in range(2 if index==0 else 1):
            rows.append(dict(id=f'mlvu/{task}/{i}',question_id=f'{task}/{i}',video_key='mlvu/v',
                video_path=str(video),media_source=dict(kind='video',path=str(video)),duration=2.,
                task='choice',benchmark='mlvu',stratum=task,duration_group=task,query=f'{task} question {i}',
                options=['A. one','B. two','C. three'],answer_index=0))
    return rows


def test_cdf_sampling_matches_latest_and_does_not_use_gold(tmp_path):
    rows=rows_fixture(tmp_path)
    dist={r['id']:dict(r,probabilities=[.2,.3,.5],answer_index=2) for r in rows}
    for method,namespace in full.RNG_METHODS.items():
        for seed in full.SEEDS:
            rng=random.Random(f'{seed}:{namespace}')
            expected=[]
            for row in rows:
                u=rng.random();expected.append(0 if u<.2 else 1 if u<.5 else 2)
            result=full.sample_predictions(rows,dist,seed,method)
            assert [r['answer_index'] for r in result.values()]==expected
            changed=[dict(r,answer_index=2) for r in rows]
            assert full.sample_predictions(changed,dist,seed,method)==result
    # 相同分布下两版controller逐题选词完全一致；差异只来自融合分布。
    a=full.sample_predictions(rows,dist,42,'controller_probability')
    b=full.sample_predictions(rows,dist,42,'controller_loglinear')
    assert [r['answer_index'] for r in a.values()]==[r['answer_index'] for r in b.values()]
    assert set(full.sample_index([0.,1.,0.],random.Random(s)) for s in range(50))=={1}


@pytest.mark.parametrize('probabilities',[[0.,0.],[float('nan'),1.],[True,0.],[-.1,1.1],[.3,.3]])
def test_bad_sampling_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):full.sample_index(probabilities,random.Random(42))


def test_mlvu_macro_counts_invalid_and_does_not_require_argmax(tmp_path):
    rows=rows_fixture(tmp_path)
    by_id={r['id']:dict(r,answer_index=1,probabilities=[.9,.05,.05]) for r in rows}
    for r in rows[:2]:by_id[r['id']]['answer_index']=0
    by_id[rows[-1]['id']].update(answer_index=None,valid=False,truncated=True)
    metrics=full.score_answers(rows,by_id,'mlvu')
    assert metrics['accuracy']==2/8 and metrics['m_avg']==1/7
    assert metrics['invalid']==1 and metrics['n']==8


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    rows=rows_fixture(tmp_path)
    source=tmp_path/'source.jsonl';source.write_bytes(jsonl_bytes(rows))
    reference=tmp_path/'official.json';reference.write_text('[]',encoding='utf-8')
    def official(benchmark,path):
        assert benchmark=='mlvu'
        return {r['id']:dict(r) for r in rows},{str(reference):sha256(reference)}
    monkeypatch.setitem(full.COUNTS,'mlvu',len(rows))
    monkeypatch.setattr(full,'_official_rows',official)
    manifest=tmp_path/'suite/questions.jsonl'
    full.prepare('mlvu',source,reference,manifest)
    actual,metadata=full.read_full(manifest)
    return manifest,actual,metadata


def test_full_manifest_exact_population_sources_and_order(prepared,tmp_path):
    manifest,rows,metadata=prepared
    assert metadata['selection_seed'] is None and metadata['n']==8
    assert [r['id'] for r in rows]==metadata['selected_ids']
    assert b'\r\n' not in manifest.read_bytes()
    with pytest.raises(FileExistsError):full.prepare('mlvu',metadata['source_manifest'],metadata['reference'],manifest)
    originals=full.read_jsonl(metadata['source_manifest'])
    Path(metadata['source_manifest']).write_bytes(jsonl_bytes(originals[:-1]))
    with pytest.raises(ValueError,match='SHA'):full.read_full(manifest)


def test_same_count_unknown_id_cannot_impersonate_full(prepared,tmp_path):
    _,rows,metadata=prepared
    rows[-1]=dict(rows[-1],id='mlvu/forged')
    Path(metadata['source_manifest']).write_bytes(jsonl_bytes(rows))
    with pytest.raises(ValueError,match='完整官方题集'):
        full.prepare('mlvu',metadata['source_manifest'],metadata['reference'],tmp_path/'fake.jsonl')


def test_true_official_inventory_rejects_tiny_reference(tmp_path):
    reference=tmp_path/'bad.json';reference.write_text('[]')
    for benchmark in full.BENCHMARKS:
        with pytest.raises(ValueError):full._official_rows(benchmark,reference)


def audit(fps):
    frames=int(fps*2)
    return dict(fps=fps,frames=frames,effective_fps=fps,processed_hw=[32,32],grid_thw=[frames//2,2,2],
        tokens=frames//2,sampled_duration=2.,source=dict(kind='video',path='fixture'),sampling_mode='fixed_fps')


@pytest.fixture
def suite(prepared):
    manifest,rows,metadata=prepared;base=manifest.parent
    hashes=full.code_files()
    checkpoint=lambda name:dict(path=f'/models/{name}',files={'config.json':{'sha256':'fixture'}})
    from scripts.videojev_candidate_logits import softmax
    from timejev.lvb_native_generation import extract_answer, generation_parameters
    for role,seed in [(role,42) for role in ['qwen_controller','videojev_controller','videojev_baseline']]+[(f'native_seed{s}',s) for s in full.SEEDS]:
        native=role.startswith('native');fps=4. if native or role=='videojev_baseline' else 2.
        directory=base/(f'native/seed{role.removeprefix("native_seed")}' if native else f'sources/{role}')
        directory.mkdir(parents=True)
        protocol=dict(name='video_bench_full_rgb_v1',fps=fps,max_frames=4096 if fps==4 else 2048,
            context=262144,visual_token_budget=24576,use_audio=False,use_subtitles=False)
        method='qwen36_native_instruct' if native else 'qwen36' if role=='qwen_controller' else 'videojev'
        identity=dict(method=method,manifest_sha256=metadata['manifest_sha256'],protocol=protocol,media={'v':'same'},
            checkpoint=checkpoint('qwen' if native or role=='qwen_controller' else 'videojev'),code_sha256=hashes,
            settings=dict(thinking=False,decoding=generation_parameters() if native else dict(do_sample=False,max_new_tokens=1)))
        if native:identity['settings'].update(presence_penalty=1.5,scoring='native_generation_then_option_parsing')
        if native:identity.update(full_dataset=metadata,inference_seed=int(role.removeprefix('native_seed')))
        else:identity.update(seed=seed);identity['settings']['full_dataset']=metadata
        preds={}
        for i,row in enumerate(rows):
            pred=dict(row,preprocessing=audit(fps))
            if native:
                # 每seed不同正确集合；末题始终无效，用于验证OR分母。
                seed=identity['inference_seed'];response='A' if i==(seed-42) else 'B'
                if i==len(rows)-1:response='unclear'
                parsed=extract_answer(response,row['options'])
                choice=parsed['answer_index']
                pred.update(response=response,raw_response=response,generated_token_ids=[32],generated_tokens=1,
                    truncated=False,parsed=parsed,answer_index=choice,valid=parsed['valid'],
                    invalid_reason=None if parsed['valid'] else parsed['reason'],answer=row['options'][choice] if choice is not None else None)
            else:
                # 4FPS J与2FPS J故意不同，防止OR误用融合源。
                values=[0.,float(seed-42),0.] if role=='videojev_baseline' else [1.,0.,.9] if role=='qwen_controller' else [-2.,8.,0.]
                probabilities=softmax(values);choice=max(range(3),key=values.__getitem__)
                pred.update(answer_index=choice,answer=row['options'][choice],probabilities=probabilities)
                if role=='qwen_controller':pred.update(candidate_logits=values,candidate_token_ids=[32,33,34])
                else:pred['logits']=values
            preds[row['id']]=pred
        (directory/'run.json').write_bytes(json_bytes(dict(identity=identity,status='completed')))
        (directory/'predictions.jsonl').write_bytes(jsonl_bytes(preds.values()))
        (directory/'metrics.json').write_bytes(json_bytes(dict(full.score_answers(rows,preds,'mlvu'),method=method,protocol=protocol)))
        full.snapshot(directory,hashes)
    return manifest,rows,metadata


def test_summary_recomputes_sampling_or_and_variance(suite):
    manifest,rows,_=suite;base=manifest.parent;out=base/'summary'
    result=full.summarize(manifest,base,out)
    assert set(result['methods'])==set(full.NAMES)
    assert result['selection_seed'] is None and result['inference_seeds']==[42,43,44]
    assert result['schema']=='video_bench_full_three_seed_summary_v4'
    assert result['methods']['qwen_logits']['decoding']=='categorical_sampling_temperature_1'
    assert result['rng_methods']['qwen_logits']=='qwen36'
    assert result['argmax_diagnostics']['qwen_logits']['correct']==len(rows)
    assert result['candidate_gpu_passes']==3 and result['native_gpu_passes']==3
    assert result['source_seed']==42
    assert result['candidate_source_scores_reused'] is True
    assert result['or_scope']=='same_seed_sampled_2fps_qwen_and_sampled_4fps_videojev'
    assert len(result['sources'])==6
    assert result['primary_metric']=='m_avg'
    for seed in full.SEEDS:
        sampled={r['id']:r for r in full.read_jsonl(out/f'qwen_logits.seed{seed}.predictions.jsonl')}
        jev={r['id']:r for r in full.read_jsonl(out/f'videojev.seed{seed}.predictions.jsonl')}
        oracle=full.read_jsonl(out/f'baseline_or.seed{seed}.predictions.jsonl')
        for p in oracle:
            assert (p['answer_index'] is not None)==(sampled[p['id']]['answer_index']==0 or jev[p['id']]['answer_index']==0)
        sampled_qwen=full.read_jsonl(out/f'qwen_logits.seed{seed}.predictions.jsonl')
        source_qwen={r['id']:r for r in full.read_jsonl(base/'sources/qwen_controller/predictions.jsonl')}
        rng=random.Random(f'{seed}:qwen36')
        expected=[full.sample_index(source_qwen[r['id']]['probabilities'],rng) for r in rows]
        assert [r['answer_index'] for r in sampled_qwen]==expected
        assert any(index!=0 for index in expected)
        trial=result['methods']['baseline_or']['trials'][seed-42]
        assert sum(trial['overlap'].values())==len(rows)
    for method in full.RNG_METHODS:
        s=result['methods'][method]
        mean=sum(s['values'])/3
        assert s['sample_variance_percentage_points_squared']==pytest.approx(sum((x-mean)**2 for x in s['values'])/2*10000)
    from scripts.summarize_bench_full import main
    assert main(['--manifest',str(manifest),'--base',str(base),'--out',str(out),'--verify'])==result
    with pytest.raises(FileExistsError):full.summarize(manifest,base,out)


@pytest.mark.parametrize('mutation',['missing','answer','protocol','seed','metric','source','checkpoint','audit','generation','code'])
def test_summary_rejects_incomplete_or_mismatched_sources(suite,mutation):
    manifest,rows,_=suite;base=manifest.parent
    directory=base/'sources/videojev_baseline'
    path=directory/'predictions.jsonl';values=full.read_jsonl(path)
    if mutation in ('missing','answer','audit'):
        if mutation=='missing':values.pop()
        if mutation=='answer':values[0]['answer_index']=2
        if mutation=='audit':values[0]['preprocessing']['fps']=2.
        path.write_bytes(jsonl_bytes(values))
    elif mutation=='metric':
        p=directory/'metrics.json';data=full.read_json(p);data['correct']+=1;p.write_bytes(json_bytes(data))
    elif mutation=='code':
        (directory/'source/timejev/bench_full.py').write_text('changed')
    elif mutation=='generation':
        p=base/'native/seed42/predictions.jsonl';values=full.read_jsonl(p);values[0]['response']='C';p.write_bytes(jsonl_bytes(values))
    else:
        p=directory/'run.json';data=full.read_json(p);i=data['identity']
        if mutation=='protocol':i['protocol']['fps']=2.
        if mutation=='seed':i['seed']=43
        if mutation=='source':i['settings']['full_dataset']['manifest_sha256']='bad'
        if mutation=='checkpoint':i['checkpoint']['path']='/other'
        p.write_bytes(json_bytes(data))
    with pytest.raises((ValueError,KeyError)):full.summarize(manifest,base)


def test_plan_all_methods_and_no_sampling_seed_used_for_selection(prepared):
    manifest,_,_=prepared
    args=runner.parser().parse_args(['--bench','mlvu','--base',str(manifest.parent),'--qwen-gpus','0,1,2,3',
        '--max-memory-gib-per-gpu','30,45,45,45','--dry-run'])
    stages=runner.make_plan(args)
    assert len(stages)==10
    assert [s['name'] for s in stages]==(['qwen_controller']+[f'native_seed{s}' for s in full.SEEDS]+
        ['videojev_controller','videojev_baseline']+[f'summary_seed{s}' for s in full.SEEDS]+['summary'])
    for stage in stages[:-1]:
        assert stage['command'][stage['command'].index('--seed')+1]==str(stage['seed'])
        assert '--sample-seed' not in stage['command']
        if stage['kind']=='native':assert stage['command'][stage['command'].index('--fps')+1]=='4'
        if stage['kind']=='qwen_logits':assert stage['command'][stage['command'].index('--max-frames')+1]=='2048'
    assert len({s['directory'] for s in stages})==10
    assert sum(s['kind']!='cpu' for s in stages)==6
    assert all(s['seed']==42 for s in stages if s['kind'] in ('qwen_logits','videojev'))
    assert args.input_hash_mode=='once'
    runner.run(args)
    assert not (manifest.parent/'experiment.json').exists()
    assert not (manifest.parent/'sources').exists()


def test_native_full_metadata_and_score_without_model(prepared,monkeypatch,tmp_path):
    from timejev import lvb_native_generation as native
    manifest,rows,metadata=prepared
    args=SimpleNamespace(manifest=str(manifest),checkpoint='fake',out=str(tmp_path/'native'),limit=None,resume=False,seed=43)
    monkeypatch.setattr(native,'checkpoint_identity',lambda _:dict(fake=True))
    protocol=dict(name='video_bench_full_rgb_v1',fps=4.,max_frames=4096)
    run=native.NativeExperiment(args,dict(test=True),rows=rows,code_paths=[Path(native.__file__)],
        benchmark='mlvu',full_dataset=metadata,input_protocol=protocol)
    for row in rows:run.append_response(row,'A',[32])
    result=run.finish()
    assert result['m_avg']==1 and result['split']=='dev_mcq'
    assert run.metadata['identity']['selection_seed'] is None
    assert run.metadata['identity']['inference_seed']==43
    assert 'sample_seed' not in run.metadata['identity']


def test_full_source_adapter_restores_globals_and_scores_raw(prepared,tmp_path,monkeypatch):
    from timejev import lvb_experiment as lvb
    manifest,rows,metadata=prepared
    names=('load_sample','protocol','score','Experiment','write_json')
    before={k:getattr(lvb,k) for k in names}
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'config.json').write_text('{}')
    monkeypatch.setattr('sys.argv',[str(full.ROOT/'scripts/eval_bench_full_source.py')])
    args=SimpleNamespace(manifest=str(manifest),checkpoint=str(checkpoint),out=str(tmp_path/'source_run'),
        fps=2.,max_frames=2048,context=262144,visual_token_budget=24576,memory_fraction=.9,seed=42,device='cpu',resume=False)
    with full.install_sources(rows,metadata):
        run=lvb.Experiment(args,'videojev',dict(test=True))
        for row in rows:run.append(row,logits=[4.,2.,1.],preprocessing=audit(2.))
        result=run.finish()
        assert result['m_avg']==1 and result['selection_seed'] is None
        meta=full.read_json(Path(args.out)/'run.json')
        assert meta['split']=='dev_mcq' and meta['identity']['protocol']['name']=='video_bench_full_rgb_v1'
    assert all(getattr(lvb,k) is v for k,v in before.items())

@pytest.mark.parametrize('benchmark',['mlvu','longvideobench'])
def test_full_native_entry_generates_all_questions_and_records_seed(prepared,tmp_path,monkeypatch,benchmark):
    import torch
    import transformers
    import timejev.rgb_media
    from timejev import lvb_experiment as lvb
    from scripts import eval_longvideobench_qwen36 as qwen
    from scripts import eval_bench_full_source as entry
    manifest,rows,metadata=prepared
    rows=deepcopy(rows);metadata=deepcopy(metadata)
    if benchmark=='longvideobench':
        for row in rows:row.update(benchmark=benchmark,stratum='15',duration_group=15)
        metadata.update(benchmark=benchmark,split='validation')
    monkeypatch.setattr(entry,'read_full',lambda _: (rows,metadata))
    checkpoint=tmp_path/'model';checkpoint.mkdir()
    for name in ('config.json','generation_config.json','tokenizer.json','preprocessor_config.json'):(checkpoint/name).write_text('{}')
    (checkpoint/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}))
    (checkpoint/'model.safetensors').write_bytes(b'fixture')
    config=SimpleNamespace(model_type='qwen3_5_moe',architectures=['Qwen3_5MoeForConditionalGeneration'],tie_word_embeddings=False,
        text_config=SimpleNamespace(num_hidden_layers=40,num_experts=256,num_experts_per_tok=8,tie_word_embeddings=False))
    class Tokenizer:
        def decode(self,tokens,skip_special_tokens):
            assert tokens==[32,248046]
            return 'A' if skip_special_tokens else 'A<|im_end|>'
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**kw:config)
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *a,**kw:SimpleNamespace(tokenizer=Tokenizer()))
    calls=[]
    class Model:
        generation_config=SimpleNamespace(eos_token_id=[248046])
        def generate(self,**kw):
            assert kw['do_sample'] is True and kw['temperature']==.7 and kw['max_new_tokens']==32768
            calls.append(kw)
            return SimpleNamespace(sequences=torch.tensor([[9,10,32,248046]]))
    monkeypatch.setattr(qwen,'load_model',lambda *a:(Model(),{}))
    monkeypatch.setattr(qwen,'load_backend',lambda:(object(),{'test':True}))
    monkeypatch.setattr(qwen,'synchronize',lambda *a:None)
    monkeypatch.setattr(lvb,'prepare_record',lambda *a,**kw:audit(kw['fps']))
    monkeypatch.setattr(lvb,'record_audit',lambda record:record)
    monkeypatch.setattr(timejev.rgb_media,'preprocess',lambda *a:[torch.zeros((1,2))])
    from scripts import eval_longvideobench_qwen as helpers
    monkeypatch.setattr(helpers,'text_inputs',lambda *a:{'input_ids':torch.tensor([[9,10]])})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:6)
    for name in ('set_per_process_memory_fraction','reset_peak_memory_stats','manual_seed_all'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda *a:0)
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:self if a and str(a[0]).startswith('cuda') else original_to(self,*a,**kw))
    out=tmp_path/'native_actual'
    result=entry.main(['--kind','native','--manifest',str(manifest),'--out',str(out),'--checkpoint',str(checkpoint),
        '--fps','4','--max-frames','4096','--seed','44'])
    assert len(calls)==len(rows) and result['correct']==len(rows)
    run=full.read_json(out/'run.json')
    assert run['status']=='completed' and run['identity']['inference_seed']==44
    assert run['identity']['full_dataset']==metadata and run['identity']['protocol']['name']=='video_bench_full_rgb_v1'
    assert run['identity']['selection_seed'] is None and 'sample_seed' not in run['identity']
    values=full.read_jsonl(out/'predictions.jsonl')
    from timejev.lvb_native_generation import question_seed
    assert [v['question_seed'] for v in values]==[question_seed(44,r['id']) for r in rows]
    assert all(v['benchmark']==benchmark for v in values)


def test_runner_records_failure_resumes_and_checks_completed_hashes(prepared,monkeypatch):
    manifest,rows,metadata=prepared;base=manifest.parent
    args=runner.parser().parse_args(['--bench','mlvu','--base',str(base),'--qwen-gpus','0,1,2,3','--input-hash-mode','always'])
    stages=runner.make_plan(args)
    calls=[];fail={'once':True}
    class Process:
        pid=12345
        def __init__(self,command,**kwargs):
            calls.append(command)
            out=Path(command[command.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            self.out=out
        def wait(self):
            if self.out.name=='videojev_controller' and fail['once']:
                fail['once']=False
                (self.out/'run.json').write_text('{}')
                return 7
            (self.out/'metrics.json').write_text('{}')
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'fixture')
    monkeypatch.setattr(runner,'read_source',lambda *a,**kw:None)
    monkeypatch.setattr(runner,'verify_summary',lambda *a:None)
    with pytest.raises(RuntimeError,match='退出码7'):runner.run(args)
    state=full.read_json(base/'experiment.json')
    assert state['status']=='failed'
    assert state['steps']['qwen_controller']['status']=='completed'
    assert (base/'logs/videojev_controller.attempt1.exitcode').read_text().strip()=='7'
    args.resume=True
    runner.run(args)
    assert len(calls)==11  # 10 stages plus the failed attempt; completed source not rerun.
    assert '--resume' in calls[5]
    assert full.read_json(base/'experiment.json')['status']=='completed'
    (base/'sources/qwen_controller/metrics.json').write_text('tampered')
    with pytest.raises(ValueError,match='SHA'):runner.run(args)


@pytest.mark.parametrize('explicit', [False, True])
def test_runner_preserves_virtualenv_interpreter_symlink(tmp_path, monkeypatch, explicit):
    # 模拟uv/venv入口指向独立Python安装，Windows也无需创建特权符号链接。
    qwen = tmp_path/'qwen-env/bin/python'
    videojev = tmp_path/'jev-env/bin/python'
    base_python = tmp_path/'uv/cpython/bin/python'
    original_resolve = Path.resolve
    def follow_link(path, *args, **kwargs):
        if path in (qwen, videojev):
            return base_python
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', follow_link)
    monkeypatch.setattr(runner.sys, 'executable', str(qwen))
    argv = ['--bench', 'videomme', '--base', str(tmp_path/'run')]
    if explicit:
        argv += ['--qwen-python', str(qwen), '--videojev-python', str(videojev)]
    stages = runner.make_plan(runner.parser().parse_args(argv))
    for stage in stages:
        expected = videojev if explicit and stage['kind']=='videojev' else qwen
        assert stage['command'][0] == str(expected)
        assert stage['command'][0] != str(base_python)


def test_six_row_table_layout_and_mlvu_label():
    methods={name:dict(mean=.6+i*.01,values=[.6+i*.01]*3,
        sample_variance_percentage_points_squared=0.,trials=[]) for i,name in enumerate(full.NAMES)}
    report=dict(benchmark='longvideobench',inference_seeds=[42,43,44],n=1337,primary_metric='accuracy',methods=methods,
        note='fixture',rng='fixture',argmax_diagnostics={})
    text=full.markdown(report)
    table=text.split('\n\n')[2].splitlines()
    assert len(table)==8  # 标题行、分隔行、六个方法。
    assert all(len(line.split('|'))==8 for line in table)
    assert table[0]=='| 方法 | seed42 | seed43 | seed44 | 平均准确率 | 样本方差 |'
    assert '（单步生成）' in table[3]
    assert '**64.00%**' in table[6]  # 可部署方法中最好，OR不参与加粗比较。
    assert '**' not in table[7]
    assert '标准差' not in table[0]
    report.update(benchmark='mlvu',primary_metric='m_avg')
    assert '平均 M-Avg' in full.markdown(report)


def test_per_seed_summary_uses_shared_scores_and_own_native(suite):
    manifest,rows,_=suite;base=manifest.parent
    result=full.summarize(manifest,base,seeds=(43,))
    assert result['inference_seeds']==[43] and len(result['sources'])==4
    assert result['candidate_gpu_passes']==3 and result['native_gpu_passes']==1
    assert set(result['sources'])=={'qwen_controller','videojev_controller','videojev_baseline','native_seed43'}
    assert all(s['sample_variance_percentage_points_squared'] is None for s in result['methods'].values())
    assert '—（单seed）' in full.markdown(result)
    all_seeds=full.summarize(manifest,base)
    for method in result['methods']:
        assert result['methods'][method]['trials'][0]==all_seeds['methods'][method]['trials'][1]
    # native43的种子错误只阻止该seed，不影响单独核验seed42。
    p=base/'native/seed43/run.json'
    run=full.read_json(p);run['identity']['inference_seed']=42;p.write_bytes(json_bytes(run))
    with pytest.raises(ValueError,match='seed'):full.summarize(manifest,base,seeds=(43,))
    assert full.summarize(manifest,base,seeds=(42,))['inference_seeds']==[42]


def test_each_seed_samples_same_videojev_distribution_with_own_rng(suite):
    manifest,rows,_=suite;base=manifest.parent;out=base/'summary'
    full.summarize(manifest,base,out)
    seen=[]
    for seed in full.SEEDS:
        source=full.read_jsonl(base/'sources/videojev_baseline/predictions.jsonl')
        probs={p['id']:p['probabilities'] for p in source}
        actual=full.read_jsonl(out/f'videojev.seed{seed}.predictions.jsonl')
        rng=random.Random(f'{seed}:videojev')
        assert [p['answer_index'] for p in actual]==[full.sample_index(probs[r['id']],rng) for r in rows]
        assert all(p['probabilities']==probs[p['id']] for p in actual)
        seen.append(actual[0]['probabilities'])
    assert seen[0]==seen[1]==seen[2]


@pytest.mark.parametrize('kind',['qwen_logits','videojev'])
@pytest.mark.parametrize('seed',[42,43,44])
def test_candidate_entry_requires_shared_seed42(kind,seed):
    from scripts.eval_bench_full_source import parser,validate
    args=parser().parse_args(['--kind',kind,'--manifest','manifest','--out','out','--fps','2','--max-frames','2048','--seed',str(seed)])
    if seed==42:validate(args)
    else:
        with pytest.raises(ValueError,match='seed42'):validate(args)


def test_or_uses_sampled_qwen_not_native(suite):
    manifest,rows,_=suite;base=manifest.parent
    before=full.summarize(manifest,base)
    from timejev.lvb_native_generation import extract_answer
    for seed in full.SEEDS:
        d=base/f'native/seed{seed}'
        preds=full.read_jsonl(d/'predictions.jsonl')
        for pred in preds:
            parsed=extract_answer('A',pred['options'])
            pred.update(response='A',raw_response='A',parsed=parsed,answer_index=0,valid=True,invalid_reason=None,answer=pred['options'][0])
        (d/'predictions.jsonl').write_bytes(jsonl_bytes(preds))
        saved=full.read_json(d/'metrics.json');saved.update(full.score_answers(rows,{p['id']:p for p in preds},'mlvu'))
        (d/'metrics.json').write_bytes(json_bytes(saved))
    after=full.summarize(manifest,base)
    assert before['methods']['baseline_or']==after['methods']['baseline_or']
    assert before['methods']['qwen_native']!=after['methods']['qwen_native']


def test_shared_source_layout_rejects_ambiguity(tmp_path):
    (tmp_path/'sources/qwen_controller').mkdir(parents=True)
    (tmp_path/'sources/seed42/qwen_controller').mkdir(parents=True)
    with pytest.raises(ValueError,match='两种布局'):full.candidate_directories(tmp_path)


def test_reuse_completed_sources_keeps_originals_and_skips_gpu(suite,tmp_path,monkeypatch):
    manifest,rows,metadata=suite;old=manifest.parent;new=tmp_path/'new_queue'
    args=runner.parser().parse_args(['--bench','mlvu','--manifest',str(manifest),'--base',str(new),
        '--reuse-sources-from',str(old),'--qwen-checkpoint','/models/qwen','--videojev-checkpoint','/models/videojev','--input-hash-mode','always'])
    stages=runner.make_plan(args)
    assert sum(s.get('reuse',False) for s in stages)==3
    assert all(Path(s['directory']).is_relative_to(old) for s in stages if s.get('reuse'))
    originals={str(p):p.read_bytes() for p in (old/'sources').rglob('*') if p.is_file()}
    calls=[]
    class StopBeforeNative:
        def __init__(self,command,**kwargs):
            calls.append(command)
            raise RuntimeError('stop before GPU')
    monkeypatch.setattr(runner.subprocess,'Popen',StopBeforeNative)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'fixture')
    with pytest.raises(RuntimeError,match='stop before GPU'):runner.run(args)
    state=full.read_json(new/'experiment.json')
    assert state['steps']['qwen_controller']['status']=='reused'
    assert 'videojev_controller' not in state['steps']  # Qwen优先，尚未轮到Jev
    assert len(calls)==1 and calls[0][calls[0].index('--kind')+1]=='native'
    assert all(Path(p).read_bytes()==data for p,data in originals.items())
    assert not (new/'sources').exists()


def test_partial_source_reuse_fails_before_gpu(suite,tmp_path,monkeypatch):
    manifest,rows,metadata=suite;old=manifest.parent
    p=old/'sources/videojev_controller/predictions.jsonl'
    p.write_bytes(jsonl_bytes(full.read_jsonl(p)[:-1]))
    args=runner.parser().parse_args(['--bench','mlvu','--manifest',str(manifest),'--base',str(tmp_path/'new'),
        '--reuse-sources-from',str(old),'--qwen-checkpoint','/models/qwen','--videojev-checkpoint','/models/videojev','--input-hash-mode','always'])
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('不应启动GPU'))
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'fixture')
    with pytest.raises(ValueError,match='结果缺题'):runner.run(args)

def test_summary_reads_old_seed42_layout_without_relabeling(suite,tmp_path):
    import shutil
    manifest,rows,_=suite;old=manifest.parent;copied=tmp_path/'old_per_seed'
    target=copied/'sources/seed42'
    shutil.copytree(old/'sources',target)
    originals={str(p):p.read_bytes() for p in copied.rglob('*') if p.is_file()}
    expected=full.summarize(manifest,old)
    actual=full.summarize(manifest,old,source_base=copied)
    assert actual['methods']==expected['methods']
    assert all('seed42' in actual['sources'][role]['directory'] for role in
        ('qwen_controller','videojev_controller','videojev_baseline'))
    assert all(Path(p).read_bytes()==content for p,content in originals.items())


def test_reuse_dry_run_validates_sources_without_outputs(suite,tmp_path):
    manifest,_,_=suite;old=manifest.parent;out=tmp_path/'dry_queue'
    args=runner.parser().parse_args(['--bench','mlvu','--manifest',str(manifest),'--base',str(out),
        '--reuse-sources-from',str(old),'--qwen-checkpoint','/models/qwen','--videojev-checkpoint','/models/videojev','--dry-run'])
    report=runner.run(args)
    assert set(report['identity']['reused_sources'])=={'qwen_controller','videojev_controller','videojev_baseline'}
    assert not out.exists()
    path=old/'sources/qwen_controller/predictions.jsonl'
    path.write_bytes(jsonl_bytes(full.read_jsonl(path)[:-1]))
    with pytest.raises(ValueError,match='结果缺题'):runner.run(args)
    assert not out.exists()
