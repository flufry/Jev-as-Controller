"""Download benchmark inputs and arrange video or frame-derived media."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_io import sha256
from scripts.build_bench_manifest import mlvu_reference, validate_lvbench
from timejev.mvbench_benchmark import (TASK_ARCHIVES, TASK_SPECS, materialized_video_path,
    selected_tasks, source_relative_path, validate_reference as validate_mvbench, video_relative_path)
from timejev.videomme_benchmark import validate_reference as validate_videomme

REPOS = {"videomme": "lmms-eval/Video-MME", "mlvu": "MLVU/MVLU", "lvbench": "zai-org/LVBench",
         "mvbench": "OpenGVLab/MVBench"}
SCREENING_BENCHES = ("videomme", "mlvu", "lvbench")


def write_json(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs; use a new data directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def copy_annotation(source, target):
    target = Path(target)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"Existing annotation differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def snapshot(bench, base, annotations_only, revision, workers, mvbench_tasks=None):
    from huggingface_hub import HfApi, snapshot_download
    repo = REPOS[bench]
    record = base / "download_revision.json"
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if saved["repo"] != repo or (revision and revision != saved["revision"]):
            raise ValueError(f"Revision/repository differs from {record}; use a new data directory")
        revision = saved["revision"]
    else:
        revision = HfApi().dataset_info(repo, revision=revision).sha
        write_json(record, {"repo": repo, "revision": revision})
    patterns = {
        "videomme": ["videomme/test-*.parquet"],
        "mlvu": ["MLVU/json/[1-7]_*.json"],
        "lvbench": ["video_info.meta.jsonl"],
        "mvbench": [f"json/{task}.json" for task in selected_tasks(mvbench_tasks)],
    }[bench]
    if not annotations_only:
        if bench == "videomme":
            patterns += ["videos_chunked_*.zip"]
        elif bench == "mlvu":
            patterns += ["MLVU/video/[1-7]_*/**"]
        elif bench == "mvbench":
            patterns += [f"video/{archive}" for archive in dict.fromkeys(
                TASK_ARCHIVES[task] for task in selected_tasks(mvbench_tasks) if TASK_ARCHIVES[task])]
            patterns += ["video/data0613.zip"]
    print(f"Downloading {repo}@{revision} -> {base / 'download'}", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                      local_dir=base / "download", allow_patterns=patterns, max_workers=workers)
    return base / "download"


def unpack_videomme(archives, videos, expected):
    videos.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path in archives:
        print(f"Extracting MP4 files: {path.name}", flush=True)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                # Select by official video ID; never extract arbitrary ZIP paths.
                name = PurePosixPath(info.filename).name
                key = Path(name).stem
                if info.is_dir() or not name.endswith(".mp4") or key not in expected:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate video in ZIP files: {key}")
                seen.add(key)
                target = videos / (key + ".mp4")
                if target.exists():
                    if target.stat().st_size != info.file_size:
                        raise ValueError(f"Existing video has a different size: {target}")
                    continue
                partial = target.with_suffix(".mp4.part")
                with archive.open(info) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                partial.replace(target)
    missing = sorted(key for key in expected if not (videos / (key + ".mp4")).is_file())
    if missing:
        raise ValueError(f"Missing {len(missing)} Video-MME videos; first IDs: {missing[:10]}")


def videomme(base, download, annotations_only):
    import pyarrow.parquet as pq
    shards = sorted((download / "videomme").glob("test-*.parquet"))
    if not shards:
        raise ValueError("Video-MME annotation Parquet files were not found")
    rows = [row for shard in shards for row in pq.read_table(shard).to_pylist()]
    validate_videomme(rows)
    write_json(base / "official.json", rows)
    if not annotations_only:
        archives = sorted(download.glob("videos_chunked_*.zip"))
        if not archives:
            raise ValueError("Video-MME video ZIP archives were not found")
        unpack_videomme(archives, base / "videos", {row["videoID"] for row in rows})


def mlvu_video_map(files, video_root):
    mapping, originals = {}, {}
    for file in files:
        rows = json.loads(file.read_text(encoding="utf-8"))
        for row in rows:
            key = row["video"]
            relative = PurePosixPath(key)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid MLVU video field: {key}")
            # Exact paths from the official task folder, not a recursive basename search.
            candidates = [video_root / file.stem / key, video_root / key]
            candidates = [path.resolve() for path in candidates if path.is_file()]
            if not candidates:
                raise ValueError(f"Missing MLVU video: task={file.stem}, video={key}")
            if len(set(candidates)) > 1 and sha256(candidates[0]) != sha256(candidates[1]):
                raise ValueError(f"Ambiguous MLVU root/task path for {key}")
            source = candidates[0]
            if not source.is_relative_to(video_root.resolve()):
                raise ValueError(f"Video outside MLVU root: {source}")
            if key in originals and originals[key] != source and sha256(originals[key]) != sha256(source):
                raise ValueError(f"Ambiguous MLVU video field with different task files: {key}")
            if key not in mapping:
                originals[key] = source
                mapping[key] = source.relative_to(video_root.resolve()).as_posix()
    return mapping


def mlvu(base, download, annotations_only):
    _, files = mlvu_reference(download / "MLVU/json")
    for source in files:
        copy_annotation(source, base / "json" / source.name)
    if not annotations_only:
        mapping = mlvu_video_map(files, download / "MLVU/video")
        write_json(base / "video-map.json", mapping)


def _unpack_mvbench_archive(archive_path, videos, expected, *, frame_dirs=False, member_root=None):
    """Extract annotated media, retaining nested names and skipping prior files."""
    videos.mkdir(parents=True, exist_ok=True)
    expected = {video_relative_path(value, frames=frame_dirs).parts: value for value in expected}
    by_depth = {}
    for parts in expected:
        by_depth.setdefault(len(parts), set()).add(parts)
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.is_absolute() or '..' in member.parts or chr(92) in info.filename:
                continue
            if member_root and member.parts[:len(member_root)] != member_root:
                continue
            if frame_dirs:
                if member.suffix.lower() != '.jpg':
                    continue
                parent = member.parts[:-1]
                matches = [parent[-depth:] for depth, values in by_depth.items()
                           if len(parent) >= depth and parent[-depth:] in values]
            else:
                matches = [member.parts[-depth:] for depth, values in by_depth.items()
                           if len(member.parts) >= depth and member.parts[-depth:] in values]
            if not matches:
                continue
            if len(matches) != 1 or (not frame_dirs and matches[0] in seen):
                raise ValueError(f'Ambiguous MVBench ZIP member: {info.filename}')
            parts = matches[0]
            seen.add(parts)
            target = videos.joinpath(*parts, member.name) if frame_dirs else videos.joinpath(*parts)
            if target.exists():
                if target.stat().st_size != info.file_size:
                    raise ValueError(f'Existing MVBench video has a different size: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            with archive.open(info) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            partial.replace(target)
    missing = [value for parts, value in expected.items() if
               not (any(videos.joinpath(*parts).glob('*.jpg')) if frame_dirs
                    else videos.joinpath(*parts).is_file())]
    return missing


def unpack_mvbench(archive_path, videos, expected, *, frame_dirs=False, patch_archive=None):
    """Resolve official update media before enforcing complete task coverage."""
    expected = set(expected)
    missing = _unpack_mvbench_archive(archive_path, videos, expected, frame_dirs=frame_dirs)
    if missing and patch_archive is not None and patch_archive.is_file():
        print(f'Supplementing {len(missing)} media for {archive_path.name} from {patch_archive.name}', flush=True)
        # The update stores data0613/<original archive root>/<media subdir>/...
        # Scope by source dataset so same-named videos cannot cross datasets.
        missing = _unpack_mvbench_archive(patch_archive, videos, set(missing), frame_dirs=frame_dirs,
                                         member_root=('data0613', archive_path.stem))
    if missing:
        hint = ' Ensure video/data0613.zip is downloaded and supplied.' if patch_archive is not None else ''
        raise ValueError(f'Missing {len(missing)} MVBench media from {archive_path.name}; '
                         f'first: {sorted(missing)[:10]}.{hint}')


def materialize_mvbench_clip(source, target, start, end, *, frames=False):
    """Make the official time-bound interval a normal video for all evaluators."""
    if target.is_file() and target.stat().st_size:
        return
    if not shutil.which('ffmpeg'):
        raise RuntimeError('MVBench bounded/frame tasks require ffmpeg')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + '.part.mp4')
    if frames:
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-framerate', '3', '-start_number', '1', '-i', str(source / '%05d.jpg'),
                   '-ss', str(start)]
    else:
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-ss', str(start), '-i', str(source)]
    command += ['-t', str(end - start), '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '18', '-pix_fmt', 'yuv420p', str(partial)]
    subprocess.run(command, check=True)
    if not partial.is_file() or not partial.stat().st_size:
        raise ValueError(f'MVBench produced an empty clip: {target}')
    partial.replace(target)


def mvbench(base, download, annotations_only, tasks):
    groups = {}
    clips = {}
    if not annotations_only and any(TASK_SPECS[task][3] for task in selected_tasks(tasks)):
        if not shutil.which('ffmpeg'):
            raise RuntimeError('MVBench bounded/frame tasks require ffmpeg before extraction')
    for task in selected_tasks(tasks):
        source = download / 'json' / f'{task}.json'
        rows = json.loads(source.read_text(encoding='utf-8'))
        validate_mvbench(task, rows)
        copy_annotation(source, base / 'json' / source.name)
        if annotations_only:
            continue
        archive, owner, media_type, bounded = TASK_SPECS[task]
        if archive:
            group = groups.setdefault(archive, (owner, media_type == 'frames', set()))
            group[2].update(video_relative_path(row['video'], frames=media_type == 'frames').as_posix()
                            for row in rows)
        else:
            missing = [str(base / 'videos' / source_relative_path(task, row)) for row in rows
                       if not (base / 'videos' / source_relative_path(task, row)).is_file()]
            if missing:
                raise FileNotFoundError(f'Licensed NTU RGB+D videos must be staged first: {missing[:3]}')
        if bounded:
            for row in rows:
                target = base / 'videos' / materialized_video_path(task, row)
                clips[target] = (base / 'videos' / source_relative_path(task, row),
                                 row['start'], row['end'], media_type == 'frames')
    for archive_name, (owner, frames, expected) in groups.items():
        archive = download / 'video' / archive_name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print(f'Extracting {len(expected)} annotated media from {archive.name}', flush=True)
        unpack_mvbench(archive, base / 'videos' / owner, expected, frame_dirs=frames,
                      patch_archive=download / 'video' / 'data0613.zip')
    if clips:
        print(f'Materializing {len(clips)} unique time-bound MVBench clips', flush=True)
        for index, (target, (source, start, end, frames)) in enumerate(clips.items(), 1):
            materialize_mvbench_clip(source, target, start, end, frames=frames)
            if index % 50 == 0 or index == len(clips):
                print(f'MVBench clips: {index}/{len(clips)}', flush=True)


def lvbench(base, download, annotations_only, js_runtime, cookies=None, prepare_local=False):
    annotation = download / "video_info.meta.jsonl"
    rows = [json.loads(line) for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_lvbench(rows)
    copy_annotation(annotation, base / "video_info.meta.jsonl")
    if annotations_only:
        return
    keys = [row["key"] for row in rows]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", key) for key in keys):
        raise ValueError("Invalid YouTube video ID in LVBench metadata")
    videos = base / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    urls = base / "youtube-urls.txt"
    urls.write_text("".join(f"https://www.youtube.com/watch?v={key}\n" for key in keys), encoding="utf-8")
    status = 0
    if not prepare_local:
        runtime = js_runtime.split(":", 1)
        if not shutil.which(runtime[-1]):
            raise RuntimeError(f"JavaScript runtime not found: {js_runtime}; see 05 download instructions")
        command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--ignore-errors", "--no-playlist",
                   "--continue", "--no-overwrites", "--retries", "5", "--fragment-retries", "5",
                   "--js-runtimes", js_runtime, "--format", "bestvideo[ext=mp4]/best[ext=mp4]",
                   "--output", str(videos / "%(id)s.%(ext)s"),
                   "--download-archive", str(base / "youtube-completed.txt"), "--batch-file", str(urls)]
        if cookies:
            command += ["--cookies", str(cookies)]
        status = subprocess.run(command, check=False).returncode
    missing = [key for key in keys if not (videos / (key + ".mp4")).is_file()
               or (videos / (key + ".mp4")).stat().st_size == 0]
    # This is a live status file, updated after each attempt; missing videos are
    # reported, never removed from the official benchmark inventory.
    (base / "missing-videos.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    if missing or status:
        raise RuntimeError(f"LVBench download incomplete: {len(missing)} missing videos, yt-dlp exit={status}; see {base / 'missing-videos.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=[*REPOS, "all"], required=True,
                        help="all retains the original three benchmarks; select mvbench explicitly")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--prepare-local", action="store_true", help="Arrange already uploaded files; no network downloads")
    parser.add_argument("--revision", help="Exact HF revision for one benchmark; recorded/reused automatically")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--js-runtime", default="deno", help="yt-dlp runtime, e.g. deno or node")
    parser.add_argument("--cookies", type=Path, help="Optional yt-dlp cookies file for LVBench")
    parser.add_argument("--mvbench-tasks", nargs='+', choices=sorted(TASK_ARCHIVES),
                        help="MVBench only; default: all 19 publicly downloadable tasks")
    args = parser.parse_args()
    if args.bench == "all" and args.revision:
        parser.error("--revision applies to one benchmark; call the three separately")
    if args.mvbench_tasks and args.bench != 'mvbench':
        parser.error('--mvbench-tasks requires --bench mvbench')
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not args.prepare_local and os.environ.get("HF_HUB_OFFLINE", "0").upper() in ("1", "TRUE", "YES", "ON"):
        parser.error("unset HF_HUB_OFFLINE before downloading")
    for bench in SCREENING_BENCHES if args.bench == "all" else [args.bench]:
        base = args.data_root.resolve() / bench
        if args.prepare_local:
            record = json.loads((base / "download_revision.json").read_text(encoding="utf-8"))
            if record["repo"] != REPOS[bench] or (args.revision and args.revision != record["revision"]):
                raise ValueError("Local revision/repository mismatch")
            download = base / "download"
        else:
            download = snapshot(bench, base, args.annotations_only, args.revision, args.workers,
                                args.mvbench_tasks)
        if bench == "lvbench":
            lvbench(base, download, args.annotations_only, args.js_runtime, args.cookies, args.prepare_local)
        elif bench == 'mvbench':
            mvbench(base, download, args.annotations_only, args.mvbench_tasks)
        else:
            {"videomme": videomme, "mlvu": mlvu}[bench](base, download, args.annotations_only)
        print(f"{bench}: {'annotations' if args.annotations_only else 'download and file inventory'} ready at {base}", flush=True)


if __name__ == "__main__":
    main()
