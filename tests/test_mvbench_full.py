"""MVBench全量覆盖、时段素材约束和三seed汇总，全部使用本地合成素材。"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.bench_io import json_bytes,jsonl_bytes
from scripts.build_bench_manifest import adapt,mvbench_reference
from scripts.run_bench_full import make_plan,parser
from timejev import bench_full as full
from timejev.mvbench_benchmark import TASK_SPECS,DEFAULT_TASKS,materialized_video_path


@pytest.fixture
def population(tmp_path):
    annotations=tmp_path/'json';annotations.mkdir();videos=tmp_path/'videos';videos.mkdir()
    for task,spec in TASK_SPECS.items():
        row=dict(video='clip' if spec[2]=='frames' else 'clip.mp4',question='What happens?',
            candidates=['one','two','three'],answer='two')
        if spec[3]:row.update(start=1.,end=3.)
        reference = [dict(row) for _ in range(200)]
        if task == 'counterfactual_inference':
            reference[81] = dict(row, candidates=['two'])
        (annotations/f'{task}.json').write_bytes(json_bytes(reference))
        p=videos/materialized_video_path(task,row);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    reference,_=mvbench_reference(annotations)
    rows=adapt('mvbench',reference,videos)
    for row in rows:row['duration']=2.
    return annotations,rows


@pytest.fixture
def prepared(population,tmp_path):
    annotation,all_rows=population
    rows=[r for r in all_rows if r['stratum'] in DEFAULT_TASKS]
    source=tmp_path/'public19.jsonl';source.write_bytes(jsonl_bytes(rows))
    manifest=tmp_path/'suite/questions.jsonl'
    metadata=full.prepare('mvbench',source,annotation,manifest)
    return manifest,metadata,source,annotation


def test_public19_and_full20_explicit_coverage(population,prepared,tmp_path):
    manifest,meta,_,annotation=prepared
    rows,verified=full.read_full(manifest)
    assert len(rows)==3800 and meta['coverage']['official_full_benchmark'] is False
    assert meta['coverage']['excluded_tasks']==['fine_grained_pose']
    assert len(meta['verified_inputs'])==20  # 19 annotation files + source manifest
    source=tmp_path/'full20.jsonl';source.write_bytes(jsonl_bytes(population[1]))
    out=tmp_path/'suite20/questions.jsonl'
    meta20=full.prepare('mvbench',source,annotation,out,mvbench_scope='full20')
    assert len(full.read_full(out)[0])==4000 and meta20['coverage']['official_full_benchmark'] is True
    with pytest.raises(ValueError,match='完整官方题集'):
        full.prepare('mvbench',source,annotation,tmp_path/'wrong.jsonl')


@pytest.mark.parametrize('change',['drop','duplicate','label','option','boundary','transform','path'])
def test_reject_tampered_population_and_clip_metadata(population,tmp_path,change):
    annotation,all_rows=population
    rows=[dict(r) for r in all_rows if r['stratum'] in DEFAULT_TASKS]
    if change=='drop':rows.pop()
    if change=='duplicate':rows[-1]=rows[0]
    if change=='label':rows[0]['answer_index']=0
    if change=='option':rows[0]['options']=list(reversed(rows[0]['options']))
    bound=next(r for r in rows if r['task_type']=='action_sequence')
    if change=='boundary':bound['clip_start']=0
    if change=='transform':bound['media_transform']='none'
    if change=='path':
        path=tmp_path/'full_video.mp4';path.write_bytes(b'fixture')
        bound.update(video_path=str(path),media_source=dict(kind='video',path=str(path)))
    source=tmp_path/'bad.jsonl';source.write_bytes(jsonl_bytes(rows))
    with pytest.raises(ValueError):full.prepare('mvbench',source,annotation,tmp_path/'out/questions.jsonl')


def test_missing_task_or_changed_reference_rejected(prepared,tmp_path):
    manifest,meta,source,annotation=prepared
    task=annotation/'action_count.json';saved=task.read_bytes()
    task.write_bytes(json_bytes(json.loads(saved)[:-1]))
    with pytest.raises(ValueError,match='SHA'):full.read_full(manifest)
    with pytest.raises(ValueError):full.prepare('mvbench',source,annotation,tmp_path/'new.jsonl')


def test_scores_all_tasks_and_invalid_answers(prepared):
    manifest,_,_,_=prepared;rows,meta=full.read_full(manifest)
    predictions={r['id']:dict(r) for r in rows}
    predictions[rows[0]['id']].update(answer_index=None,valid=False,truncated=True)
    scored=full.score_answers(rows,predictions,'mvbench')
    assert scored['correct']==3799 and scored['n']==3800 and scored['invalid']==1
    assert scored['accuracy']==pytest.approx(scored['task_average_accuracy'])
    assert len(scored['by_task'])==19 and scored['coverage']==meta['coverage']
    with pytest.raises(ValueError):full.score_answers(rows[:-1],{r['id']:predictions[r['id']] for r in rows[:-1]},'mvbench')


def test_mvbench_dry_plan_and_native_metadata(prepared,tmp_path,monkeypatch):
    manifest,_,_,_=prepared;rows,metadata=full.read_full(manifest)
    args=parser().parse_args(['--bench','mvbench','--base',str(manifest.parent)])
    stages=make_plan(args)
    assert args.input_hash_mode=='once'
    assert [s['name'] for s in stages[:6]]==['qwen_controller','native_seed42','native_seed43','native_seed44','videojev_controller','videojev_baseline']
    from timejev import lvb_native_generation as native
    monkeypatch.setattr(native,'checkpoint_identity',lambda _:dict(fake=True))
    args=SimpleNamespace(manifest=str(manifest),checkpoint='fake',out=str(tmp_path/'native'),limit=None,resume=False,seed=43)
    run=native.NativeExperiment(args,{},rows=rows,code_paths=[Path(native.__file__)],benchmark='mvbench',full_dataset=metadata,
        input_protocol=dict(name='video_bench_full_rgb_v1',fps=4.,max_frames=4096))
    assert run.metadata['identity']['full_dataset']['coverage']==metadata['coverage']


def test_six_method_full_summary_mvbench(prepared,tmp_path,monkeypatch):
    manifest,_,_,_=prepared;rows,metadata=full.read_full(manifest)
    from scripts.eval_longvideobench_controller import softmax
    def source(directory,targets,info,role,expected_seed):
        native=role.startswith('native');baseline=role=='videojev_baseline';is_qwen=role=='qwen_controller' or native
        fps=4. if native or baseline else 2.
        protocol=dict(name='video_bench_full_rgb_v1',fps=fps,max_frames=4096 if fps==4 else 2048,context=262144,visual_token_budget=24576)
        values={}
        for row in rows:
            logits = [0.,1.,.5][:len(row['options'])]
            p=dict(row,preprocessing={'fps':fps},probabilities=softmax(logits),
                   answer_index=max(range(len(logits)), key=logits.__getitem__))
            if native:p.update(valid=True,truncated=False)
            elif is_qwen:p['candidate_logits']=logits
            else:p['logits']=logits
            values[row['id']]=p
        identity=dict(protocol=protocol,media={'fixture':True},checkpoint={'model':'qwen' if is_qwen else 'jev'},settings={})
        return values,dict(identity=identity,files={}),{}
    monkeypatch.setattr(full,'read_source',source)
    result=full.summarize(manifest,manifest.parent)
    assert result['n']==3800 and result['coverage']['scope']=='public19'
    assert len(result['methods'])==6
    assert result['or_scope']=='same_seed_sampled_2fps_qwen_and_sampled_4fps_videojev'
    for method in result['methods'].values():
        assert len(method['values'])==3 and all(len(t['by_task'])==19 for t in method['trials'])
    text=full.markdown(result)
    assert '# MVBench 19个公开任务（3800题）' in text
    assert 'VideoChat2' in text
