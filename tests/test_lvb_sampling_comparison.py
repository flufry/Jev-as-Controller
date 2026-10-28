import json
from pathlib import Path

import pytest

from scripts.compare_longvideobench_sampling import compare
from scripts.eval_longvideobench_controller import evaluate, softmax
from scripts.sample_longvideobench import sample
from timejev.lvb_experiment import load_sample, read_predictions, score, sha256, write_json
from timejev.user_data import write_jsonl
from timejev.video import sampling_grid


def make_baseline(directory, manifest, seed, method, fps, hits):
    directory.mkdir(parents=True)
    rows = load_sample(manifest)
    cap = 2048 if fps == 2 else 4096
    protocol = dict(fps=fps, max_frames=cap, context=262144, visual_token_budget=24576,
                    use_audio=False, use_subtitles=False)
    media = {}
    predictions = []
    for index, row in enumerate(rows):
        path = Path(row['video_path'])
        content = dict(path=str(path), bytes=path.stat().st_size,
                       mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        media[row['video_key']] = content
        times, effective, mode = sampling_grid(row['duration'], fps, cap)
        groups = (len(times)+1)//2
        audit = dict(frames=len(times), fps=fps, effective_fps=effective,
            grid_thw=[groups, 2, 2], processed_hw=[32, 32], tokens=groups,
            sampled_duration=row['duration'], sampling_mode=mode,
            source=dict(kind='video', path=str(path), identity={k: content[k] for k in ('path', 'bytes', 'mtime_ns')}))
        correct = index < hits
        logits = [1., 0.] if correct else [0., 1.]
        probabilities = softmax(logits) if method == 'qwen35' else [float(correct), float(not correct)]
        prediction = dict(id=row['id'], task='choice', query=row['query'], options=row['options'],
            probabilities=probabilities, answer_index=int(not correct), preprocessing=audit)
        if method == 'qwen35':
            prediction.update(candidate_logits=logits, input_tokens=groups+32,
                timing=dict(video_preprocessing_seconds=2., question_inference_seconds=.5))
        else:
            prediction['timing'] = dict(video_group_total_seconds=3., question_count=25)
        predictions.append(prediction)
    identity = dict(method=method, manifest_sha256=sha256(manifest), seed=seed, protocol=protocol,
        media=media, checkpoint=dict(path='models/'+method, files={'weights':dict(bytes=1, sha256='a'*64)}),
        settings=dict(scoring=method), packages={'torch': 'test'}, code_sha256={'mock.py': ('b' if fps == 2 else 'c')*64})
    write_json(directory/'run.json', dict(identity=identity))
    write_jsonl(predictions, directory/'predictions.jsonl')
    metrics = score(rows, read_predictions(directory/'predictions.jsonl', rows))
    metrics.update(method=method, protocol=protocol)
    write_json(directory/'metrics.json', metrics)
    write_json(directory/'runtime.json', dict(wall_seconds=10., peak_cuda_allocated_bytes=1024))


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path/'source'
    (source/'videos').mkdir(parents=True)
    rows, reference = [], []
    for group in (15, 60, 600, 3600):
        (source/'videos'/f'{group}.mp4').write_bytes(b'local mock video')
        for index in range(30):
            target = dict(id=f'{group}_{index}', video_id=str(group), video_path=f'{group}.mp4',
                duration=float(group), duration_group=group, question=f'question {group} {index}',
                candidates=['right', 'wrong'], correct_choice=0)
            reference.append(target)
            rows.append(dict(id='longvideobench/'+target['id'], question_id=target['id'],
                video_key='longvideobench/'+str(group), video_path='videos/'+target['video_path'],
                duration=float(group), task='choice', query=target['question'],
                options=['A. right', 'B. wrong'], answer_index=0))
    write_jsonl(rows, source/'questions.jsonl')
    write_json(source/'official.json', reference)
    baseline_root = tmp_path/'new'
    for index, seed in enumerate((42, 43, 44)):
        original = tmp_path/'old'/f'seed{seed}'
        manifest = original/'sample.jsonl'
        sample(source/'questions.jsonl', source/'official.json', manifest, seed=seed)
        for method, folder in (('qwen35', 'qwen'), ('videojev', 'videojev')):
            make_baseline(original/folder, manifest, seed, method, 2.,
                          (20 if method == 'qwen35' else 30)+20*index)
            make_baseline(baseline_root/f'seed{seed}'/folder, manifest, seed, method, 1.,
                          (40 if method == 'qwen35' else 35)+10*index)
        evaluate(manifest, original/'qwen', original/'videojev', original/'controller')
    return baseline_root, str(tmp_path/'old'/'seed{seed}')


def invoke(experiment, tmp_path):
    baseline_root, pattern = experiment
    return compare(baseline_root, pattern, tmp_path/'summary')


def four_fps_baselines(experiment, tmp_path):
    _, pattern = experiment
    baseline_root = tmp_path/'new4'
    for index, seed in enumerate((42, 43, 44)):
        manifest = Path(pattern.format(seed=seed))/'sample.jsonl'
        for method, folder in (('qwen35', 'qwen'), ('videojev', 'videojev')):
            make_baseline(baseline_root/f'seed{seed}'/folder, manifest, seed, method, 4.,
                          (40 if method == 'qwen35' else 35)+10*index)
    return baseline_root, pattern


def test_statistics_fixed_controller_and_resource_deduplication(experiment, tmp_path):
    baseline_root, pattern = experiment
    old_controller_hashes = [sha256(Path(pattern.format(seed=s))/'controller/predictions.jsonl') for s in (42, 43, 44)]
    result = invoke(experiment, tmp_path)
    assert result['methods']['qwen35']['overall']['values'] == [.4, .5, .6]
    assert result['methods']['qwen35']['overall']['mean'] == pytest.approx(.5)
    assert result['methods']['qwen35']['overall']['sample_variance'] == pytest.approx(.01)
    assert result['methods']['jev_as_controller']['overall']['values'] == [.3, .5, .7]
    assert result['controller_minus_baseline']['qwen35']['values'] == pytest.approx([-.1, 0., .1])
    assert result['controller_minus_baseline']['qwen35']['better_each_seed'] == [False, False, True]
    assert result['controller_minus_baseline']['qwen35']['better_mean'] is False
    assert result['controller_minus_baseline']['videojev']['mean'] == pytest.approx(.05)
    assert result['controller_minus_baseline']['videojev']['better_mean'] is True
    assert result['training_overlap_verified'] is False
    assert result['baseline_sampling'] == dict(fps=1., max_frames=4096)
    assert result['reference_fingerprints']['qwen35']['code_sha256'] != result['new_fingerprints']['qwen35']['code_sha256']
    resources = result['resource_audits'][0]['methods']
    assert resources['qwen35']['frames_per_video']['n'] == 4
    assert resources['qwen35']['input_tokens_per_question']['n'] == 100
    assert resources['qwen35']['timing_seconds']['video_preprocessing_seconds']['total'] == 8.
    assert resources['qwen35']['timing_seconds']['question_inference_seconds']['total'] == 50.
    assert resources['videojev']['timing_seconds']['video_group_total_seconds']['total'] == 12.
    assert resources['qwen35']['runtime']['wall_seconds'] == 10.
    assert resources['qwen35']['videos']['longvideobench/3600']['frames'] == 3600
    assert resources['jev_as_controller']['videos']['longvideobench/3600']['frames'] == 2048
    assert resources['jev_as_controller']['runtime'] is None
    assert 'source_baselines' in resources['jev_as_controller']
    assert old_controller_hashes == [sha256(Path(pattern.format(seed=s))/'controller/predictions.jsonl') for s in (42, 43, 44)]
    assert json.loads((tmp_path/'summary/results.json').read_text()) == result
    markdown = (tmp_path/'summary/results.md').read_text()
    assert '未使用新基线重新融合' in markdown and '不代表严格算力对齐' in markdown
    assert '100.0000' in markdown and '百分点²' in markdown
    assert '帧数均值/最大' in markdown and '峰值已分配显存（GiB）' in markdown
    assert '同视频多题之间恢复运行' in result['resource_timing_limitation']
    assert result['resource_timing_limitation'] in markdown
    with pytest.raises(FileExistsError):
        invoke(experiment, tmp_path)


@pytest.mark.parametrize('kind', ['manifest', 'seed', 'fps', 'context', 'settings', 'checkpoint',
    'packages', 'code', 'partial', 'metrics', 'preprocessing', 'old_cache', 'media', 'metadata'])
def test_invalid_or_partial_inputs_rejected(experiment, tmp_path, kind):
    baseline_root, pattern = experiment
    directory = baseline_root/'seed43/qwen'
    if kind in ('manifest', 'seed', 'fps', 'context', 'settings', 'checkpoint', 'packages', 'code', 'media'):
        path = directory/'run.json'
        metadata = json.loads(path.read_text())
        identity = metadata['identity']
        if kind == 'manifest':
            identity['manifest_sha256'] = 'wrong'
        elif kind == 'seed':
            identity['seed'] = 42
        elif kind in ('fps', 'context'):
            identity['protocol'][kind] = 4 if kind == 'fps' else 131072
            metrics = json.loads((directory/'metrics.json').read_text())
            metrics['protocol'] = identity['protocol']
            write_json(directory/'metrics.json', metrics)
        elif kind == 'media':
            identity['media'][next(iter(identity['media']))]['sha256'] = 'd'*64
        else:
            key = 'code_sha256' if kind == 'code' else kind
            identity[key]['changed'] = True
        write_json(path, metadata)
    elif kind == 'metrics':
        path = directory/'metrics.json'
        metrics = json.loads(path.read_text())
        metrics['accuracy'] = .99
        write_json(path, metrics)
    elif kind == 'metadata':
        path = Path(pattern.format(seed=43))/'sample.meta.json'
        metadata = json.loads(path.read_text())
        metadata['seed'] = 44
        write_json(path, metadata)
    else:
        path = directory/'predictions.jsonl'
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == 'partial':
            predictions.pop()
        elif kind == 'old_cache':
            old_path = Path(pattern.format(seed=43))/'qwen/predictions.jsonl'
            predictions = [json.loads(line) for line in old_path.read_text().splitlines()]
        else:
            predictions[0]['preprocessing']['source']['identity']['bytes'] += 1
        write_jsonl(predictions, path)
        if kind == 'old_cache':
            rows = load_sample(Path(pattern.format(seed=43))/'sample.jsonl')
            metrics = score(rows, read_predictions(path, rows))
            metadata = json.loads((directory/'run.json').read_text())
            metrics.update(method='qwen35', protocol=metadata['identity']['protocol'])
            write_json(directory/'metrics.json', metrics)
    with pytest.raises(ValueError):
        invoke(experiment, tmp_path)
    assert not (tmp_path/'summary/results.json').exists()


@pytest.mark.parametrize('location,value', [('prediction', float('nan')), ('prediction', -1.),
    ('runtime', float('inf')), ('runtime', -.1)])
def test_nonfinite_or_negative_timings_rejected(experiment, tmp_path, location, value):
    baseline_root, _ = experiment
    directory = baseline_root/'seed43/qwen'
    if location == 'prediction':
        path = directory/'predictions.jsonl'
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        predictions[0]['timing']['question_inference_seconds'] = value
        path.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    else:
        path = directory/'runtime.json'
        runtime = json.loads(path.read_text())
        runtime['wall_seconds'] = value
        path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match='计时必须是有限非负'):
        invoke(experiment, tmp_path)


@pytest.mark.parametrize('method', ['qwen', 'videojev'])
def test_coherently_shortened_duration_rejected(experiment, tmp_path, method):
    baseline_root, _ = experiment
    directory = baseline_root/'seed43'/method
    metadata_path = directory/'run.json'
    original_media = json.loads(metadata_path.read_text())['identity']['media']
    path = directory/'predictions.jsonl'
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    source_path = predictions[0]['preprocessing']['source']['path']
    for prediction in predictions:
        audit = prediction['preprocessing']
        if audit['source']['path'] != source_path:
            continue
        audit['sampled_duration'] /= 2
        times, effective, mode = sampling_grid(audit['sampled_duration'], 1., 4096)
        audit.update(frames=len(times), effective_fps=effective, sampling_mode=mode)
        audit['grid_thw'][0] = (len(times)+1)//2
        audit['tokens'] = audit['grid_thw'][0]*(audit['grid_thw'][1]//2)*(audit['grid_thw'][2]//2)
        if method == 'qwen':
            prediction['input_tokens'] = audit['tokens']+32
    write_jsonl(predictions, path)
    assert json.loads(metadata_path.read_text())['identity']['media'] == original_media
    with pytest.raises(ValueError, match='新旧同方法的完整视频采样时长不一致'):
        invoke(experiment, tmp_path)


def test_four_fps_full_comparison_keeps_old_controller(experiment, tmp_path):
    baseline_root, pattern = four_fps_baselines(experiment, tmp_path)
    result = compare(baseline_root, pattern, tmp_path/'summary4', fps=4., max_frames=4096)
    assert result['baseline_sampling'] == dict(fps=4., max_frames=4096)
    assert result['controller_sampling'] == dict(fps=2., max_frames=2048, beta=2)
    assert result['methods']['qwen35']['overall']['values'] == [.4, .5, .6]
    assert result['methods']['jev_as_controller']['overall']['values'] == [.3, .5, .7]
    videos = result['resource_audits'][0]['methods']['qwen35']['videos']
    assert videos['longvideobench/15']['frames'] == 60
    assert videos['longvideobench/600']['frames'] == 2400
    assert videos['longvideobench/3600']['frames'] == 4096
    assert result['new_fingerprints']['qwen35']['protocol']['fps'] == 4.
    assert '两条新基线为 4 FPS、最多 4096 帧' in (tmp_path/'summary4/results.md').read_text()


@pytest.mark.parametrize('retag_metadata', [False, True])
def test_one_fps_cache_cannot_impersonate_four(experiment, tmp_path, retag_metadata):
    baseline_root, pattern = experiment
    if retag_metadata:
        for seed in (42, 43, 44):
            for folder in ('qwen', 'videojev'):
                directory = baseline_root/f'seed{seed}'/folder
                run = json.loads((directory/'run.json').read_text())
                run['identity']['protocol']['fps'] = 4.
                write_json(directory/'run.json', run)
                metrics = json.loads((directory/'metrics.json').read_text())
                metrics['protocol']['fps'] = 4.
                write_json(directory/'metrics.json', metrics)
    with pytest.raises(ValueError):
        compare(baseline_root, pattern, tmp_path/'summary4', fps=4., max_frames=4096)


@pytest.mark.parametrize('group,wrong_frames', [('15', 15), ('3600', 4097)])
def test_four_fps_actual_short_grid_and_long_cap_are_enforced(experiment, tmp_path, group, wrong_frames):
    baseline_root, pattern = four_fps_baselines(experiment, tmp_path)
    path = baseline_root/'seed43/qwen/predictions.jsonl'
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    for prediction in predictions:
        audit = prediction['preprocessing']
        if Path(audit['source']['path']).stem != group:
            continue
        audit['frames'] = wrong_frames
        audit['grid_thw'][0] = (wrong_frames+1)//2
        audit['tokens'] = audit['grid_thw'][0]*(audit['grid_thw'][1]//2)*(audit['grid_thw'][2]//2)
        prediction['input_tokens'] = audit['tokens']+32
    write_jsonl(predictions, path)
    with pytest.raises(ValueError, match='实际帧或视觉 token 记录与 4 FPS / 4096 帧协议不符'):
        compare(baseline_root, pattern, tmp_path/'summary4', fps=4., max_frames=4096)


@pytest.mark.parametrize('changes', [dict(fps=0), dict(fps=float('nan')), dict(fps=True),
    dict(max_frames=0), dict(max_frames=4096.)])
def test_invalid_comparison_sampling_fails_before_reading_runs(tmp_path, changes):
    with pytest.raises(ValueError):
        compare(tmp_path/'absent', 'absent{seed}', tmp_path/'summary', **changes)
