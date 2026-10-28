"""LongVideoBench 分层子集实验的输入、断点保存和统一计分。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from .user_data import load_manifest

GROUPS = (15, 60, 600, 3600)
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def evaluation_parser(description, checkpoint):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default=checkpoint)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--visual-token-budget', type=int, default=24576)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--max-frames', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    parser.add_argument('--resume', action='store_true')
    return parser


def sampling_parameters(fps, max_frames):
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps 必须是有限正数')
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError('max_frames 必须是正整数')
    return float(fps), max_frames


def protocol(args):
    fps, max_frames = sampling_parameters(getattr(args, 'fps', 2.), getattr(args, 'max_frames', 2048))
    if (args.context < 1024 or not 0 < args.visual_token_budget <= args.context*4//5
            or not 0 < args.memory_fraction <= 1):
        raise ValueError('上下文、视觉 token 预算或显存比例无效')
    return dict(name='lvb_equal_duration_rgb_subset_v1', fps=fps, max_frames=max_frames,
                context=args.context, visual_token_budget=args.visual_token_budget,
                geometry='largest_native_grid_under_budget_max_area_409600',
                frame_sampling='uniform_full_video', input_source='raw_video',
                use_subtitles=False, use_audio=False)


def load_sample(path):
    rows = load_manifest(path)
    if not rows or any(r['task'] != 'choice' or r.get('duration_group') not in GROUPS for r in rows):
        raise ValueError('采样清单需要非空选择题及官方 duration_group')
    if any(r.get('media_source', {}).get('kind', 'video') != 'video' for r in rows):
        raise ValueError('本次子集对照实验使用原始视频，不读取帧缓存')
    return rows


def checkpoint_identity(directory):
    directory = Path(directory).resolve()
    if not (directory/'config.json').is_file():
        raise ValueError(f'需要完整本地 checkpoint：{directory}')
    files = sorted(p for p in directory.iterdir()
                   if p.suffix in ('.json', '.jinja', '.py', '.safetensors'))
    return dict(path=str(directory), files={p.name: dict(bytes=p.stat().st_size,
                sha256=sha256(p)) for p in files})


def validate_prediction(row, prediction):
    values = prediction.get('probabilities', [])
    if (prediction.get('id') != row['id'] or prediction.get('query') != row['query']
            or prediction.get('options') != row['options'] or len(values) != len(row['options'])
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in values)
            or not math.isclose(sum(values), 1., abs_tol=1e-5)
            or type(prediction.get('answer_index')) is not int
            or prediction['answer_index'] != max(range(len(values)), key=values.__getitem__)):
        raise ValueError(f'预测内容或候选概率无效：{row["id"]}')


def read_predictions(path, rows, complete=True):
    path = Path(path)
    predictions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    by_id, targets = {}, {r['id']: r for r in rows}
    for prediction in predictions:
        identity = prediction['id']
        if identity in by_id or identity not in targets:
            raise ValueError(f'重复或未知预测 id：{identity}')
        validate_prediction(targets[identity], prediction)
        by_id[identity] = prediction
    if complete and set(by_id) != set(targets):
        raise ValueError(f'预测未覆盖全部题目：{len(by_id)}/{len(rows)}')
    return by_id


def score(rows, by_id):
    if set(by_id) != {r['id'] for r in rows}:
        raise ValueError('计分必须完整覆盖采样清单')
    totals, hits = Counter(), Counter()
    for row in rows:
        validate_prediction(row, by_id[row['id']])
        group = row['duration_group']
        totals[group] += 1
        hits[group] += by_id[row['id']]['answer_index'] == row['answer_index']
    correct = sum(hits.values())
    return dict(n=len(rows), videos=len({r['video_key'] for r in rows}), correct=correct,
                accuracy=correct/len(rows), split='validation_subset',
                by_duration_group={str(g): dict(n=totals[g], correct=hits[g],
                    accuracy=hits[g]/totals[g]) for g in GROUPS if totals[g]})


class Experiment:
    def __init__(self, args, method, settings):
        self.rows = load_sample(args.manifest)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        code = [Path(__file__), ROOT/'timejev/rgb_media.py', ROOT/'timejev/rgb_data.py',
                ROOT/'timejev/video.py', ROOT/'timejev/temporal.py', Path(sys.argv[0]).resolve()]
        packages = {}
        for name in ('torch', 'torchvision', 'transformers', 'triton', 'flash-attn',
                     'causal-conv1d', 'fla-core', 'numpy', 'av'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        media = {}
        for row in self.rows:
            if row['video_key'] not in media:
                path = Path(row['video_path'])
                media[row['video_key']] = dict(path=str(path), bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        identity = dict(method=method, manifest_sha256=sha256(args.manifest),
                        checkpoint=checkpoint_identity(args.checkpoint), protocol=protocol(args),
                        settings=settings, seed=args.seed, media=media, packages=packages,
                        code_sha256={os.path.relpath(p, ROOT): sha256(p) for p in code})
        self.meta_path = self.out/'run.json'
        self.predictions_path = self.out/'predictions.jsonl'
        if self.meta_path.exists():
            previous = json.loads(self.meta_path.read_text())
            if not args.resume or previous['identity'] != identity:
                raise ValueError('输出已存在或恢复指纹不一致；请选择新目录或匹配的 --resume')
            self.metadata = previous
        else:
            if any(self.out.iterdir()):
                raise ValueError(f'拒绝覆盖非空输出目录：{self.out}')
            self.metadata = dict(identity=identity, manifest=str(Path(args.manifest).resolve()),
                command=sys.argv, started_at=datetime.now(timezone.utc).isoformat(),
                hostname=socket.gethostname(), pid=os.getpid(), device=args.device,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                memory_fraction=args.memory_fraction,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                python=sys.version,
                packages=packages)
            write_json(self.meta_path, self.metadata)
            for path in code:
                relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path(path.name)
                target = self.out/'source'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            (self.out/'source'/'git.diff').write_bytes(
                subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
        self.done = set(read_predictions(self.predictions_path, self.rows, complete=False))

    def append(self, row, probabilities, **details):
        if row['id'] in self.done:
            raise ValueError(f'重复写入预测：{row["id"]}')
        values = list(probabilities)
        index = max(range(len(values)), key=values.__getitem__)
        prediction = dict(details, id=row['id'], task='choice', query=row['query'],
            options=row['options'], video_key=row['video_key'], duration_group=row['duration_group'],
            probabilities=values, answer_index=index, answer=row['options'][index])
        validate_prediction(row, prediction)
        with self.predictions_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.done.add(row['id'])
        print(json.dumps(dict(done=len(self.done), total=len(self.rows), id=row['id']),
                         ensure_ascii=False), flush=True)

    def finish(self):
        report = score(self.rows, read_predictions(self.predictions_path, self.rows))
        finished_at = datetime.now(timezone.utc)
        report.update(method=self.metadata['identity']['method'],
                      protocol=self.metadata['identity']['protocol'],
                      finished_at=finished_at.isoformat())
        write_json(self.out/'metrics.json', report)
        runtime = dict(
            wall_seconds=(finished_at-datetime.fromisoformat(self.metadata['started_at'])).total_seconds(),
            wall_seconds_scope='从 started_at 到完成，包含模型加载及运行，不含首次指纹计算；恢复时包含中断间隔',
            wall_seconds_is_active_gpu_time=False,
        )
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            runtime.update(
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.metadata['device']),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.metadata['device']))
        write_json(self.out/'runtime.json', runtime)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report


def prepare_record(row, visual_token_budget, *, fps=2., max_frames=2048):
    from .rgb_data import geometries
    from .rgb_media import fingerprint, metadata, probe_video_geometry, timeline
    from .video import sampling_grid
    fps, max_frames = sampling_parameters(fps, max_frames)
    if fps == 2. and max_frames == 2048:
        record = metadata(dict(row, media_source=dict(kind='video', path=row['video_path'])), 'longvideobench')
    else:
        # 自定义采样仅从原视频探测；不改变训练和已有帧缓存的固定协议。
        path = Path(row['video_path'])
        width, height, media_duration = probe_video_geometry(path)
        duration = min(float(row['duration']), media_duration)
        times, effective, mode = sampling_grid(duration, fps, max_frames)
        record = dict(video_key=row['video_key'],
            source=dict(kind='video', path=str(path), identity=fingerprint(path)),
            frames=len(times), fps=fps, effective_fps=effective, original_hw=[height, width],
            sampled_duration=duration, media_duration=duration, sampling_mode=mode)
    groups = (record['frames']+1)//2
    for h, w in geometries(*record['original_hw'], minimum=1024):
        tokens = groups*(h//32)*(w//32)
        if tokens <= visual_token_budget:
            return timeline(dict(record, processed_hw=[h, w],
                                 grid_thw=[groups, h//16, w//16], tokens=tokens))
    raise ValueError(f'完整视频不能放入视觉 token 预算：{row["video_key"]}')


def record_audit(record):
    return {key: record[key] for key in ('frames', 'fps', 'effective_fps', 'grid_thw',
            'processed_hw', 'tokens', 'sampled_duration', 'source', 'sampling_mode')}
