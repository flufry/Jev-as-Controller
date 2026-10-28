import argparse
import json
from types import SimpleNamespace

import pytest

from timejev.lvb_experiment import (
    Experiment, load_sample, prepare_record, protocol, read_predictions, score, validate_prediction,
)


def row(identity='q1', group=15, answer=1):
    return dict(id=identity, video_key=identity, video_path='video.mp4', duration=10.,
                task='choice', query='Question?', options=['A. One', 'B. Two'],
                answer_index=answer, duration_group=group)


def prediction(target, probabilities=(.2, .8)):
    return dict(id=target['id'], query=target['query'], options=target['options'],
                probabilities=list(probabilities), answer_index=probabilities.index(max(probabilities)))


def test_subset_scores_include_every_duration_group():
    rows = [row(str(i), group) for i, group in enumerate((15, 60, 600, 3600))]
    predictions = {r['id']: prediction(r) for r in rows}
    predictions['0'] = prediction(rows[0], (.9, .1))
    metrics = score(rows, predictions)
    assert metrics['n'] == 4 and metrics['correct'] == 3 and metrics['accuracy'] == .75
    assert metrics['by_duration_group']['15']['accuracy'] == 0.
    assert metrics['by_duration_group']['3600']['n'] == 1
    with pytest.raises(ValueError, match='完整覆盖'):
        score(rows, {})


@pytest.mark.parametrize('change', [
    dict(probabilities=[.2, .2]), dict(probabilities=[float('nan'), .8]),
    dict(probabilities=[-.2, 1.2]), dict(probabilities=[.2]),
    dict(answer_index=0), dict(answer_index=True), dict(options=['B. Two', 'A. One']),
])
def test_invalid_prediction_rejected(change):
    target = row()
    value = prediction(target)
    value.update(change)
    with pytest.raises(ValueError):
        validate_prediction(target, value)


def test_prediction_file_duplicate_and_partial_rejected(tmp_path):
    target, path = row(), tmp_path/'predictions.jsonl'
    with pytest.raises(ValueError, match='未覆盖'):
        read_predictions(path, [target])
    assert read_predictions(path, [target], complete=False) == {}
    value = json.dumps(prediction(target))+'\n'
    path.write_text(value*2)
    with pytest.raises(ValueError, match='重复'):
        read_predictions(path, [target])


def test_sample_uses_relative_media_path(tmp_path):
    (tmp_path/'video.mp4').touch()
    path = tmp_path/'sample.jsonl'
    target = row()
    path.write_text(json.dumps(target)+'\n')
    assert load_sample(path)[0]['video_path'] == str(tmp_path/'video.mp4')
    target.pop('duration_group')
    path.write_text(json.dumps(target)+'\n')
    with pytest.raises(ValueError, match='duration_group'):
        load_sample(path)


def test_geometry_caps_tokens_without_removing_frames(monkeypatch):
    import timejev.rgb_data
    import timejev.rgb_media
    base = dict(frames=2048, original_hw=[720, 1280], effective_fps=1., fps=2.,
                sampled_duration=2048., source={'kind': 'video'})
    monkeypatch.setattr(timejev.rgb_media, 'metadata', lambda *_: base.copy())
    monkeypatch.setattr(timejev.rgb_data, 'geometries', lambda *_args, **_kwargs: [(320, 640), (96, 192), (32, 64)])
    record = prepare_record(row(), 24576)
    assert record['frames'] == 2048
    assert record['processed_hw'] == [96, 192]
    assert record['tokens'] == 18432
    with pytest.raises(ValueError, match='完整视频'):
        prepare_record(row(), 100)


def test_protocol_rejects_impossible_budget():
    args = argparse.Namespace(context=262144, visual_token_budget=24576, memory_fraction=.9)
    assert protocol(args)['fps'] == 2.
    args.visual_token_budget = args.context
    with pytest.raises(ValueError):
        protocol(args)


def test_videojev_labels_never_reach_predict_features(monkeypatch):
    import torch
    import timejev.rgb_media
    from scripts.eval_longvideobench_videojev import predict_group
    target = row()
    seen = {}
    monkeypatch.setattr(timejev.rgb_media, 'preprocess', lambda *_: [torch.zeros(8, 3)])
    def fake_predict(_processor, features, record, questions, **kwargs):
        seen['questions'] = questions
        assert features.shape == (2, 3)
        return [dict(id=target['id'], probabilities=[.2, .8])]
    model = SimpleNamespace(encode_video=lambda *_: torch.zeros(2, 3), predict_features=fake_predict)
    result = predict_group(model, object(), [target], dict(grid_thw=[2, 2, 2]), 262144)
    assert result[0]['id'] == target['id']
    assert set(seen['questions'][0]) == {'id', 'task', 'query', 'options'}


def test_resume_requires_identical_sample_checkpoint_and_settings(tmp_path):
    target = row()
    (tmp_path/'video.mp4').touch()
    manifest = tmp_path/'sample.jsonl'
    manifest.write_text(json.dumps(target)+'\n')
    checkpoint = tmp_path/'model'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    (checkpoint/'tiny.safetensors').write_bytes(b'test')
    args = argparse.Namespace(manifest=str(manifest), out=str(tmp_path/'run'),
        checkpoint=str(checkpoint), device='cuda:0', context=262144,
        visual_token_budget=24576, seed=42, memory_fraction=.9, resume=False)
    run = Experiment(args, 'test', {'scoring': 'test'})
    run.append(run.rows[0], [.2, .8])
    assert run.finish()['accuracy'] == 1.
    with pytest.raises(ValueError, match='输出已存在'):
        Experiment(args, 'test', {'scoring': 'test'})
    args.resume = True
    assert Experiment(args, 'test', {'scoring': 'test'}).done == {'q1'}
    with pytest.raises(ValueError, match='指纹不一致'):
        Experiment(args, 'test', {'scoring': 'changed'})
    (tmp_path/'video.mp4').write_bytes(b'changed media')
    with pytest.raises(ValueError, match='指纹不一致'):
        Experiment(args, 'test', {'scoring': 'test'})
    (tmp_path/'video.mp4').write_bytes(b'')
    # 媒体 mtime 已变化；使用独立目录再验证权重本身变化也会拒绝恢复。
    args.out = str(tmp_path/'run2')
    args.resume = False
    Experiment(args, 'test', {'scoring': 'test'})
    args.resume = True
    (checkpoint/'tiny.safetensors').write_bytes(b'new!')
    with pytest.raises(ValueError, match='指纹不一致'):
        Experiment(args, 'test', {'scoring': 'test'})
