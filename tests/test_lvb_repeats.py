import json
import math
import shutil

import pytest

from scripts.eval_longvideobench_controller import evaluate, softmax
from scripts.sample_longvideobench import sample
from scripts.summarize_longvideobench_repeats import accuracy_statistics, read_repeat, summarize
from timejev.lvb_experiment import load_sample, read_predictions, score, sha256, write_json
from timejev.user_data import write_jsonl


@pytest.fixture
def repeats(tmp_path):
    source = tmp_path/'source'
    (source/'videos').mkdir(parents=True)
    rows, reference = [], []
    for group in (15, 60, 600, 3600):
        (source/'videos'/f'{group}.mp4').touch()
        for i in range(100):
            question = dict(id=f'{group}_{i}', video_id=str(group), video_path=f'{group}.mp4',
                duration=float(group), duration_group=group, question=f'question {group} {i}',
                candidates=['right', 'wrong'], correct_choice=0)
            reference.append(question)
            rows.append(dict(id='longvideobench/'+question['id'], question_id=question['id'],
                video_key='longvideobench/'+str(group), video_path=f'videos/{group}.mp4',
                duration=float(group), task='choice', query=question['question'],
                options=['A. right', 'B. wrong'], answer_index=0))
    manifest, official = source/'questions.jsonl', source/'official.json'
    write_jsonl(rows, manifest)
    write_json(official, reference)
    runs = []
    protocol = dict(fps=2., max_frames=2048, context=262144, visual_token_budget=24576,
                    use_subtitles=False, use_audio=False)
    for index, seed in enumerate((42, 43, 44)):
        run = tmp_path/f'seed{seed}'
        sampled = run/'sample.jsonl'
        sample(manifest, official, sampled, seed=seed)
        selected = load_sample(sampled)
        media = {}
        for row in selected:
            path = source/'videos'/f'{int(row["duration"])}.mp4'
            media[row['video_key']] = dict(path=str(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        for method, folder in (('qwen35', 'qwen'), ('videojev', 'videojev')):
            out = run/folder
            out.mkdir()
            predictions = []
            for number, row in enumerate(selected):
                correct = number < (20+index*20 if method == 'qwen35' else 30+index*20)
                values = [1., 0.] if correct else [0., 1.]
                probabilities = softmax(values) if method == 'qwen35' else [float(correct), float(not correct)]
                prediction = dict(id=row['id'], task='choice', query=row['query'], options=row['options'],
                    probabilities=probabilities, answer_index=int(not correct),
                    preprocessing=dict(frames=30, grid_thw=[15, 4, 4], source=dict(kind='video',
                        path=media[row['video_key']]['path'], identity={
                            key: media[row['video_key']][key] for key in ('path', 'bytes', 'mtime_ns')})))
                if method == 'qwen35':
                    prediction['candidate_logits'] = values
                predictions.append(prediction)
            write_jsonl(predictions, out/'predictions.jsonl')
            identity = dict(method=method, manifest_sha256=sha256(sampled), seed=seed,
                protocol=protocol, settings=dict(scoring=method), packages={'torch': 'test'},
                checkpoint=dict(path='models/'+method, files={'weights':dict(bytes=1, sha256='a'*64)}),
                code_sha256={'mock.py': 'b'*64}, media=media)
            write_json(out/'run.json', dict(identity=identity))
            metrics = score(selected, read_predictions(out/'predictions.jsonl', selected))
            metrics.update(method=method, protocol=protocol)
            write_json(out/'metrics.json', metrics)
        evaluate(sampled, run/'qwen', run/'videojev', run/'controller')
        runs.append(run)
    return runs


def test_known_statistics_and_ddof():
    stats = accuracy_statistics([.2, .4, .6])
    assert stats['mean'] == pytest.approx(.4)
    assert stats['sample_variance'] == pytest.approx(.04)
    assert stats['population_variance'] == pytest.approx(.08/3)
    assert stats['sample_std'] == pytest.approx(.2)
    assert stats['population_std'] == pytest.approx(math.sqrt(.08/3))
    with pytest.raises(ValueError):
        accuracy_statistics([.5])


def test_complete_repeats_and_markdown_units(repeats, tmp_path):
    result = summarize(repeats, tmp_path/'summary')
    assert result['methods']['qwen35']['overall']['values'] == [.2, .4, .6]
    assert result['methods']['qwen35']['overall']['mean'] == pytest.approx(.4)
    assert result['methods']['qwen35']['overall']['sample_variance'] == pytest.approx(.04)
    assert result['methods']['qwen35']['by_duration_group']['15']['values'] == [.8, 1., 1.]
    assert result['methods']['qwen35']['by_duration_group']['15']['sample_variance'] == pytest.approx(.04/3)
    assert result['methods']['videojev']['overall']['values'] == [.3, .5, .7]
    assert result['methods']['jev_as_controller']['overall']['values'] == [.3, .5, .7]
    assert result['variance_unit'] == 'accuracy_fraction_squared'
    assert result['sample_variance_ddof'] == 1
    assert result['population_variance_ddof'] == 0
    assert result['training_overlap_verified'] is False
    assert '无法按视频或问题确认' in result['limitation']
    assert result['unique_questions'] < 300
    expected_overlap = len(read_repeat(repeats[0])['question_ids'] & read_repeat(repeats[1])['question_ids'])
    assert result['overlaps'][0]['shared_questions'] == expected_overlap
    saved = json.loads((tmp_path/'summary/results.json').read_text())
    assert saved == result
    markdown = (tmp_path/'summary/results.md').read_text()
    assert '40.00%' in markdown and '400.0000' in markdown and '20.0000' in markdown
    assert 'ddof=1' in markdown and '不是三种方法之间的方差' in markdown
    assert '不能称已验证无训练污染' in markdown
    with pytest.raises(FileExistsError):
        summarize(repeats, tmp_path/'summary')


@pytest.mark.parametrize('kind', ['same_seed', 'same_sample', 'config', 'checkpoint', 'packages',
    'code', 'coverage', 'metrics', 'controller_source', 'controller_formula', 'sample_sha', 'group_count'])
def test_incompatible_or_forged_repeats_rejected(repeats, tmp_path, kind):
    run = repeats[1]
    if kind == 'same_sample':
        shutil.rmtree(run)
        shutil.copytree(repeats[0], run)
        shutil.rmtree(run/'controller')
        evaluate(run/'sample.jsonl', run/'qwen', run/'videojev', run/'controller')
    elif kind in ('same_seed', 'sample_sha', 'group_count'):
        path = run/'sample.meta.json'
        metadata = json.loads(path.read_text())
        if kind == 'same_seed':
            metadata['seed'] = 42
        elif kind == 'sample_sha':
            metadata['sample_sha256'] = 'wrong'
        else:
            metadata['selected_counts']['15'] = 24
        write_json(path, metadata)
        if kind == 'same_seed':
            for folder in ('qwen', 'videojev'):
                source_path = run/folder/'run.json'
                source = json.loads(source_path.read_text())
                source['identity']['seed'] = 42
                write_json(source_path, source)
            shutil.rmtree(run/'controller')
            evaluate(run/'sample.jsonl', run/'qwen', run/'videojev', run/'controller')
    elif kind in ('config', 'checkpoint', 'packages', 'code'):
        path = run/'qwen/run.json'
        metadata = json.loads(path.read_text())
        key = dict(config='settings', checkpoint='checkpoint', packages='packages', code='code_sha256')[kind]
        metadata['identity'][key]['changed'] = True
        write_json(path, metadata)
        # 重新运行控制器，保证拒绝原因来自跨次配置校验，而非旧来源哈希。
        shutil.rmtree(run/'controller')
        evaluate(run/'sample.jsonl', run/'qwen', run/'videojev', run/'controller')
    elif kind == 'metrics':
        path = run/'qwen/metrics.json'
        metadata = json.loads(path.read_text())
        metadata['accuracy'] = .99
        write_json(path, metadata)
    elif kind == 'controller_source':
        path = run/'controller/run.json'
        metadata = json.loads(path.read_text())
        metadata['identity']['sources']['qwen']['predictions_sha256'] = 'wrong'
        write_json(path, metadata)
    else:
        path = run/('qwen' if kind == 'coverage' else 'controller')/'predictions.jsonl'
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == 'coverage':
            predictions.pop()
        else:
            predictions[0]['qwen_probabilities'].reverse()
        write_jsonl(predictions, path)
    with pytest.raises(ValueError):
        summarize(repeats, tmp_path/'summary')
    assert not (tmp_path/'summary/results.json').exists()


def test_requires_three_runs(repeats, tmp_path):
    with pytest.raises(ValueError):
        summarize(repeats[:2], tmp_path/'summary')


@pytest.mark.parametrize('kind', ['within_run', 'missing_video', 'preprocessing_identity',
    'cross_seed_content', 'source_manifest', 'source_reference'])
def test_media_and_actual_annotation_hashes_checked(repeats, tmp_path, kind):
    run = repeats[1]
    if kind in ('source_manifest', 'source_reference'):
        path = run/'sample.meta.json'
        metadata = json.loads(path.read_text())
        metadata[kind]['sha256'] = 'c'*64
        write_json(path, metadata)
    else:
        for folder in ('qwen', 'videojev'):
            path = run/folder/'run.json'
            metadata = json.loads(path.read_text())
            media = metadata['identity']['media']
            key = next(iter(media))
            if kind == 'missing_video':
                media.pop(key)
            elif kind == 'cross_seed_content' or kind == 'within_run' and folder == 'qwen':
                media[key]['sha256'] = 'c'*64
            write_json(path, metadata)
            if kind == 'preprocessing_identity':
                path = run/folder/'predictions.jsonl'
                predictions = [json.loads(line) for line in path.read_text().splitlines()]
                predictions[0]['preprocessing']['source']['identity']['bytes'] += 1
                write_jsonl(predictions, path)
        shutil.rmtree(run/'controller')
        evaluate(run/'sample.jsonl', run/'qwen', run/'videojev', run/'controller')
    message = ('跨次重叠视频' if kind == 'cross_seed_content' else
               '原始标注文件' if kind.startswith('source_') else
               '逐题预处理源身份' if kind == 'preprocessing_identity' else '两基线媒体指纹')
    with pytest.raises(ValueError, match=message):
        summarize(repeats, tmp_path/'summary')
