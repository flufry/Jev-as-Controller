import json
from pathlib import Path
import shutil

import pytest

from scripts.eval_longvideobench_controller import evaluate, softmax
from scripts.sample_longvideobench import sample
from scripts.summarize_longvideobench_qwen36 import summarize
from timejev.lvb_experiment import load_sample, read_predictions, score, sha256, write_json
from timejev.user_data import write_jsonl
from timejev.video import sampling_grid


def make_source(out, manifest, checkpoint, method, fps, hits, seed=42):
    out.mkdir(parents=True)
    rows, media, predictions = load_sample(manifest), {}, []
    cap = 4096 if fps == 4 else 2048
    protocol = dict(fps=fps, max_frames=cap, context=262144, visual_token_budget=24576,
                    use_subtitles=False, use_audio=False)
    for index, row in enumerate(rows):
        path = Path(row['video_path'])
        content = dict(path=str(path), bytes=path.stat().st_size,
                       mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        media[row['video_key']] = content
        times, effective, mode = sampling_grid(row['duration'], fps, cap)
        groups = (len(times)+1)//2
        audit = dict(frames=len(times), fps=fps, effective_fps=effective, sampled_duration=row['duration'],
            sampling_mode=mode, grid_thw=[groups, 2, 2], tokens=groups, processed_hw=[32, 32],
            source=dict(kind='video', path=str(path), identity={k: content[k] for k in ('path', 'bytes', 'mtime_ns')}))
        hit = index < hits
        logits = [1., 0.] if hit else [0., 1.]
        values = softmax(logits) if method == 'qwen36' else [float(hit), float(not hit)]
        prediction = dict(id=row['id'], task='choice', query=row['query'], options=row['options'],
            answer_index=int(not hit), probabilities=values, preprocessing=audit)
        if method == 'qwen36':
            prediction.update(candidate_logits=logits, input_tokens=groups+32,
                timing=dict(video_preprocessing_seconds=2., question_inference_seconds=.5))
        else:
            prediction['timing'] = dict(video_group_total_seconds=3., question_count=25)
        predictions.append(prediction)
    config = checkpoint/'config.json'
    identity = dict(method=method, manifest_sha256=sha256(manifest), seed=seed, protocol=protocol,
        media=media, checkpoint=dict(path=str(checkpoint), files={
            'config.json': dict(bytes=config.stat().st_size, sha256=sha256(config)),
            'weights.safetensors': dict(bytes=1, sha256='a'*64)}),
        settings=dict(scoring=method), packages={'torch': 'test'},
        code_sha256={'mock.py': ('b' if method == 'qwen36' or fps == 2 else 'c')*64})
    write_json(out/'run.json', dict(identity=identity))
    write_jsonl(predictions, out/'predictions.jsonl')
    metrics = score(rows, read_predictions(out/'predictions.jsonl', rows))
    metrics.update(method=method, protocol=protocol)
    write_json(out/'metrics.json', metrics)
    write_json(out/'runtime.json', dict(wall_seconds=10., peak_cuda_allocated_bytes=1024))


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path/'source'
    (source/'videos').mkdir(parents=True)
    rows, official = [], []
    for group in (15, 60, 600, 3600):
        (source/'videos'/f'{group}.mp4').write_bytes(b'mock local video')
        for i in range(30):
            target = dict(id=f'{group}_{i}', video_id=str(group), video_path=f'{group}.mp4',
                duration=float(group), duration_group=group, question=f'question {group} {i}',
                candidates=['right', 'wrong'], correct_choice=0)
            official.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+str(group), video_path='videos/'+target['video_path'],
                duration=float(group), task='choice', query=target['question'], options=['A. right', 'B. wrong'], answer_index=0))
    write_jsonl(rows, source/'questions.jsonl')
    write_json(source/'official.json', official)
    manifest = tmp_path/'sample.jsonl'
    sample(source/'questions.jsonl', source/'official.json', manifest, seed=42)
    checkpoints = {}
    for method, model_type in (('qwen36', 'qwen3_5_moe'), ('videojev', 'videojev')):
        checkpoint = tmp_path/'models'/method
        checkpoint.mkdir(parents=True)
        write_json(checkpoint/'config.json', dict(model_type=model_type))
        checkpoints[method] = checkpoint
    paths = dict(manifest=manifest, out=tmp_path/'summary', controller=tmp_path/'controller')
    for role, method, fps, hits in (('qwen_baseline', 'qwen36', 4., 40), ('qwen_controller', 'qwen36', 2., 50),
            ('videojev_baseline', 'videojev', 4., 75), ('videojev_controller', 'videojev', 2., 72)):
        paths[role] = tmp_path/role
        make_source(paths[role], manifest, checkpoints[method], method, fps, hits)
    evaluate(manifest, paths['qwen_controller'], paths['videojev_controller'], paths['controller'], qwen_method='qwen36')
    return paths


def test_single_seed_scores_source_gain_and_resources(experiment):
    result = summarize(**experiment)
    assert result['seed'] == 42 and result['repetitions'] == 1
    assert result['methods']['qwen36']['accuracy'] == .4
    assert result['methods']['videojev']['accuracy'] == .75
    assert result['methods']['jev_as_controller']['accuracy'] == .72
    assert result['controller_source_metrics']['qwen36']['accuracy'] == .5
    assert result['controller_source_metrics']['videojev']['accuracy'] == .72
    assert result['controller2_minus_qwen2']['accuracy_difference'] == pytest.approx(.22)
    assert result['controller_minus_baseline']['qwen36']['accuracy_difference'] == pytest.approx(.32)
    assert result['controller_minus_baseline']['videojev']['accuracy_difference'] == pytest.approx(-.03)
    assert result['training_overlap_verified'] is False
    assert set(result['methods']['qwen36']['by_duration_group']) == {'15', '60', '600', '3600'}
    assert all('variance' not in key for key in result)
    audits = result['resource_audits']
    assert audits['qwen_baseline']['videos']['longvideobench/15']['frames'] == 60
    assert audits['qwen_baseline']['videos']['longvideobench/3600']['frames'] == 4096
    assert audits['qwen_controller']['videos']['longvideobench/3600']['frames'] == 2048
    assert audits['qwen_baseline']['timing_seconds']['video_preprocessing_seconds']['total'] == 8.
    assert audits['qwen_controller']['timing_seconds']['question_inference_seconds']['total'] == 50.
    assert len(result['provenance']['runs']) == 5
    assert json.loads((experiment['out']/'results.json').read_text()) == result
    markdown = (experiment['out']/'results.md').read_text()
    assert '同协议融合增益' in markdown and '不能全部归因于融合' in markdown
    assert '仅运行 seed42 一次' in markdown
    with pytest.raises(FileExistsError):
        summarize(**experiment)


@pytest.mark.parametrize('kind', ['fps', 'context', 'method', 'model', 'settings', 'packages', 'code',
    'manifest', 'partial', 'metrics', 'logits', 'preprocessing', 'media', 'source_sha', 'metadata'])
def test_incompatible_inputs_rejected(experiment, kind):
    directory = experiment['qwen_baseline']
    if kind in ('fps', 'context', 'method', 'model', 'settings', 'packages', 'code', 'manifest', 'media'):
        path = directory/'run.json'
        run = json.loads(path.read_text())
        identity = run['identity']
        if kind in ('fps', 'context'):
            identity['protocol'][kind] = 2 if kind == 'fps' else 131072
            metrics = json.loads((directory/'metrics.json').read_text())
            metrics['protocol'] = identity['protocol']
            write_json(directory/'metrics.json', metrics)
        elif kind == 'method':
            identity['method'] = 'qwen35'
        elif kind == 'manifest':
            identity['manifest_sha256'] = 'wrong'
        elif kind == 'media':
            identity['media'][next(iter(identity['media']))]['sha256'] = 'd'*64
        else:
            key = dict(model='checkpoint', code='code_sha256').get(kind, kind)
            identity[key]['changed'] = True
        write_json(path, run)
    elif kind == 'metrics':
        path = directory/'metrics.json'
        metrics = json.loads(path.read_text())
        metrics['accuracy'] = .99
        write_json(path, metrics)
    elif kind in ('source_sha', 'metadata'):
        path = experiment['manifest'].with_suffix('.meta.json')
        metadata = json.loads(path.read_text())
        if kind == 'source_sha':
            metadata['source_reference']['sha256'] = 'wrong'
        else:
            metadata['seed'] = 43
        write_json(path, metadata)
    else:
        path = directory/'predictions.jsonl'
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == 'partial':
            predictions.pop()
        elif kind == 'logits':
            predictions[0]['candidate_logits'].reverse()
        else:
            predictions[0]['preprocessing']['source']['identity']['bytes'] += 1
        write_jsonl(predictions, path)
    with pytest.raises(ValueError):
        summarize(**experiment)
    assert not (experiment['out']/'results.json').exists()


@pytest.mark.parametrize('kind', ['four_fps_sources', 'source_hash', 'formula', 'components', 'beta', 'comparison', 'old_qwen35'])
def test_wrong_controller_sources_or_math_rejected(experiment, kind):
    directory = experiment['controller']
    if kind == 'four_fps_sources':
        shutil.rmtree(directory)
        evaluate(experiment['manifest'], experiment['qwen_baseline'], experiment['videojev_baseline'],
                 directory, qwen_method='qwen36')
    elif kind == 'source_hash':
        path = directory/'run.json'
        run = json.loads(path.read_text())
        run['identity']['sources']['qwen']['predictions_sha256'] = 'wrong'
        write_json(path, run)
    elif kind == 'old_qwen35':
        path = directory/'run.json'
        run = json.loads(path.read_text())
        run['identity']['sources']['qwen']['identity']['method'] = 'qwen35'
        write_json(path, run)
    elif kind == 'comparison':
        path = directory/'comparison.json'
        comparison = json.loads(path.read_text())
        comparison['qwen35'] = comparison.pop('qwen36')
        write_json(path, comparison)
    else:
        path = directory/'predictions.jsonl'
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == 'beta':
            predictions[0]['beta'] = 0
        elif kind == 'formula':
            winner = predictions[0]['answer_index']
            predictions[0]['probabilities'] = [.6 if i == winner else .4 for i in range(2)]
        else:
            predictions[0]['qwen_probabilities'].reverse()
        write_jsonl(predictions, path)
    with pytest.raises(ValueError):
        summarize(**experiment)


def test_four_b_checkpoint_cannot_be_relabelled_qwen36(experiment):
    for role in ('qwen_baseline', 'qwen_controller'):
        path = experiment[role]/'run.json'
        run = json.loads(path.read_text())
        checkpoint = Path(run['identity']['checkpoint']['path'])
        write_json(checkpoint/'config.json', dict(model_type='qwen3_5'))
        file = run['identity']['checkpoint']['files']['config.json']
        file.update(bytes=(checkpoint/'config.json').stat().st_size, sha256=sha256(checkpoint/'config.json'))
        write_json(path, run)
    # 来源也重新生成，使失败来自模型架构检查而非陈旧 Controller 哈希。
    shutil.rmtree(experiment['controller'])
    evaluate(experiment['manifest'], experiment['qwen_controller'], experiment['videojev_controller'],
             experiment['controller'], qwen_method='qwen36')
    with pytest.raises(ValueError, match='不能冒充原 4B 模型'):
        summarize(**experiment)


@pytest.mark.parametrize('seed', [43, 44])
def test_other_original_seeds_are_supported(experiment, tmp_path, seed):
    meta = json.loads(experiment['manifest'].with_suffix('.meta.json').read_text())
    root = tmp_path/f'seed{seed}'
    manifest = root/'sample.jsonl'
    sample(meta['source_manifest']['path'], meta['source_reference']['path'], manifest, seed=seed)
    args = dict(manifest=manifest, controller=root/'controller', out=root/'summary', seed=seed)
    for role in ('qwen_baseline', 'qwen_controller', 'videojev_baseline', 'videojev_controller'):
        identity = json.loads((experiment[role]/'run.json').read_text())['identity']
        args[role] = root/role
        make_source(args[role], manifest, Path(identity['checkpoint']['path']), identity['method'],
                    identity['protocol']['fps'], 50, seed=seed)
    evaluate(manifest, args['qwen_controller'], args['videojev_controller'], args['controller'], qwen_method='qwen36')
    result = summarize(**args)
    assert result['seed'] == seed
    assert result['provenance']['sample']['seed'] == seed
    assert f'仅运行 seed{seed} 一次' in (args['out']/'results.md').read_text()


def test_revalidation_does_not_write_outputs(experiment):
    result = summarize(**dict(experiment, out=None))
    assert result['n'] == 100
    assert not experiment['out'].exists()
