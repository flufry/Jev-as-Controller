from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from timejev.lvb_experiment import Experiment, evaluation_parser, prepare_record, protocol
from timejev.rgb_media import metadata, preprocess, sampled_chunks
from timejev.video import sampling_grid


def args(**changes):
    values = dict(context=262144, visual_token_budget=24576, memory_fraction=.9)
    return Namespace(**(values | changes))


def media(tmp_path, monkeypatch, duration):
    import timejev.rgb_data
    import timejev.rgb_media
    path = tmp_path/'video.mp4'
    path.touch()
    monkeypatch.setattr(timejev.rgb_media, 'probe_video_geometry', lambda _: (4, 2, duration))
    monkeypatch.setattr(timejev.rgb_data, 'geometries', lambda *_args, **_kwargs: [(32, 32)])
    return dict(id='q', video_key='v', video_path=str(path), duration=duration)


@pytest.mark.parametrize('fps', [1., 4.])
def test_parser_and_old_namespaces_keep_defaults(fps):
    parser = evaluation_parser('test', 'models/test')
    parsed = parser.parse_args(['--manifest', 'input.jsonl', '--out', 'output'])
    assert parsed.fps == 2. and parsed.max_frames == 2048
    assert protocol(args()) == protocol(args(fps=2., max_frames=2048))
    parsed = parser.parse_args(['--manifest', 'input.jsonl', '--out', 'output',
                               '--fps', str(fps), '--max-frames', '4096'])
    assert protocol(parsed)['fps'] == fps and protocol(parsed)['max_frames'] == 4096


@pytest.mark.parametrize('change', [
    dict(fps=0), dict(fps=-1), dict(fps=float('nan')), dict(fps=float('inf')),
    dict(fps=True), dict(fps='1'), dict(max_frames=0), dict(max_frames=-1),
    dict(max_frames=4096.), dict(max_frames=True),
])
def test_invalid_sampling_rejected_before_opening_media(change):
    settings = dict(fps=1., max_frames=4096) | change
    with pytest.raises(ValueError):
        protocol(args(**settings))
    with pytest.raises(ValueError):
        prepare_record({}, 24576, **settings)


def test_default_metadata_is_unchanged(tmp_path, monkeypatch):
    row = media(tmp_path, monkeypatch, 3601.)
    expected = metadata(dict(row, media_source=dict(kind='video', path=row['video_path'])), 'longvideobench')
    actual = prepare_record(row, 24576)
    assert {key: actual[key] for key in expected} == expected
    assert actual['frames'] == 2048
    assert actual['fps'] == 2.
    assert actual['effective_fps'] == pytest.approx(2048/3601)
    assert actual['score_bounds'][-1][-1] == 3601.


def test_one_fps_keeps_more_than_2048_frames_and_odd_tail(tmp_path, monkeypatch):
    row = media(tmp_path, monkeypatch, 3601.)
    record = prepare_record(row, 24576, fps=1., max_frames=4096)
    assert record['frames'] == 3601 and record['effective_fps'] == 1.
    assert record['sampling_mode'] == 'fixed_fps'
    assert record['grid_thw'] == [1801, 2, 2]
    assert record['score_times'][-1] == 3600.
    assert record['score_bounds'][-1][-1] == 3601.
    assert record['source']['kind'] == 'video'


@pytest.mark.parametrize('duration,frames,last_center', [(7.97, 32, 7.625), (8.01, 33, 8.)])
def test_four_fps_short_video_preserves_fractional_and_odd_tail(
        tmp_path, monkeypatch, duration, frames, last_center):
    row = media(tmp_path, monkeypatch, duration)
    record = prepare_record(row, 24576, fps=4., max_frames=4096)
    assert record['fps'] == record['effective_fps'] == 4.
    assert record['frames'] == frames
    assert record['grid_thw'] == [(frames+1)//2, 2, 2]
    assert record['sampling_mode'] == 'fixed_fps'
    assert record['sampled_duration'] == duration
    assert record['score_times'][-1] == last_center
    assert record['score_bounds'][-1][-1] == duration


@pytest.mark.parametrize('duration,fps', [(4095, 1.), (5001, 1.), (8.25, 4.), (1500.25, 4.)])
def test_capped_and_odd_videos_stream_all_planned_frames(tmp_path, monkeypatch, duration, fps):
    import av

    row = media(tmp_path, monkeypatch, float(duration))
    record = prepare_record(row, 24576, fps=fps, max_frames=4096)
    expected, effective, mode = sampling_grid(duration, fps, 4096)
    source_frames = int(np.ceil(duration*fps))
    assert record['frames'] == len(expected)
    assert record['effective_fps'] == effective and record['sampling_mode'] == mode
    if source_frames > 4096:
        assert len(expected) == 4096 and effective == pytest.approx(4096/duration)
        assert duration-expected[-1] == pytest.approx(1/effective)
    else:
        assert len(expected) == source_frames and effective == fps
        assert source_frames % 2 == 1 and record['score_times'][-1] == expected[-1]
    assert record['sampled_duration'] == duration
    assert record['score_bounds'][-1][-1] == duration

    class Frame:
        time_base = 1/fps

        def __init__(self, index):
            self.pts, self.duration = index, 1

        def to_ndarray(self, format):
            assert format == 'rgb24'
            return np.full((2, 4, 3), self.pts % 251, dtype=np.uint8)

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=fps)])
            self.decoded = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def decode(self, stream):
            for index in range(source_frames):
                self.decoded += 1
                yield Frame(index)

    opened = []

    def open_video(_):
        opened.append(Container())
        return opened[-1]

    monkeypatch.setattr(av, 'open', open_video)
    chunks = sampled_chunks(record, chunk_frames=32)
    first = next(chunks)
    assert len(first) == 32 and opened[-1].decoded < source_frames
    decoded = torch.cat([first, *chunks])
    assert len(decoded) == len(expected)
    assert decoded[-1, 0, 0, 0] == int(np.floor(expected[-1]*fps+.5)) % 251

    class VideoProcessor:
        resample = 'mock'

        def resize(self, frames, size, resample):
            assert len(frames) <= 32
            assert size.height == size.width == 32
            return frames

        def __call__(self, videos, **kwargs):
            assert kwargs == dict(do_resize=False, do_sample_frames=False, return_tensors='pt')
            frames = videos[0]
            groups = (len(frames)+1)//2
            return dict(video_grid_thw=torch.tensor([[groups, 2, 2]]),
                pixel_values_videos=torch.full((groups*4, 3), float(frames[-1, 0, 0, 0])))

    patches = preprocess(SimpleNamespace(video_processor=VideoProcessor()), record)
    assert sum(len(chunk) for chunk in patches) == record['grid_thw'][0]*4
    assert patches[-1][-1, 0] == decoded[-1, 0, 0, 0]
    assert all(chunk.dtype == torch.bfloat16 for chunk in patches)


def test_runtime_wall_span_is_explicit(tmp_path):
    (tmp_path/'video.mp4').touch()
    row = dict(id='q', video_key='v', video_path='video.mp4', duration=1., task='choice',
               query='Question?', options=['A. one', 'B. two'], answer_index=0, duration_group=15)
    manifest = tmp_path/'sample.jsonl'
    manifest.write_text(json.dumps(row)+'\n')
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    settings = args(manifest=str(manifest), out=str(tmp_path/'run'), checkpoint=str(checkpoint),
                    seed=42, resume=False, device='cuda:0')
    run = Experiment(settings, 'test', {})
    run.append(run.rows[0], [.8, .2])
    run.finish()
    runtime = json.loads((tmp_path/'run/runtime.json').read_text())
    assert runtime['wall_seconds'] >= 0
    assert runtime['wall_seconds_is_active_gpu_time'] is False
    assert '首次指纹' in runtime['wall_seconds_scope'] and '中断间隔' in runtime['wall_seconds_scope']
