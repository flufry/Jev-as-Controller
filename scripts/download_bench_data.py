"""Download the three screening benchmarks and arrange their official inputs."""
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
from timejev.videomme_benchmark import validate_reference as validate_videomme

REPOS = {"videomme": "lmms-eval/Video-MME", "mlvu": "MLVU/MVLU", "lvbench": "zai-org/LVBench"}


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


def snapshot(bench, base, annotations_only, revision, workers):
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
    }[bench]
    if not annotations_only:
        if bench == "videomme":
            patterns += ["videos_chunked_*.zip"]
        elif bench == "mlvu":
            patterns += ["MLVU/video/[1-7]_*/**"]
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


def lvbench(base, download, annotations_only, js_runtime, cookies=None):
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
    parser.add_argument("--bench", choices=[*REPOS, "all"], required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--revision", help="Exact HF revision for one benchmark; recorded/reused automatically")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--js-runtime", default="deno", help="yt-dlp runtime, e.g. deno or node")
    parser.add_argument("--cookies", type=Path, help="Optional yt-dlp cookies file for LVBench")
    args = parser.parse_args()
    if args.bench == "all" and args.revision:
        parser.error("--revision applies to one benchmark; call the three separately")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if os.environ.get("HF_HUB_OFFLINE", "0").upper() in ("1", "TRUE", "YES", "ON"):
        parser.error("unset HF_HUB_OFFLINE before downloading")
    for bench in REPOS if args.bench == "all" else [args.bench]:
        base = args.data_root.resolve() / bench
        download = snapshot(bench, base, args.annotations_only, args.revision, args.workers)
        if bench == "lvbench":
            lvbench(base, download, args.annotations_only, args.js_runtime, args.cookies)
        else:
            {"videomme": videomme, "mlvu": mlvu}[bench](base, download, args.annotations_only)
        print(f"{bench}: {'annotations' if args.annotations_only else 'download and file inventory'} ready at {base}", flush=True)


if __name__ == "__main__":
    main()
