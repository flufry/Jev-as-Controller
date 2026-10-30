"""Download one asset at a time, SCP to Linux, verify SHA256, optionally remove locally."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_io import sha256
from scripts.download_bench_data import REPOS, validate_lvbench

MODELS = {
    "qwen": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    "videojev": ("ZhuYuhan/VideoJev-4B", "1c1a253bf90dfa66176124d25a48ffb791d517d5"),
}
PATTERNS = {
    "videomme": ["videomme/test-*.parquet", "videos_chunked_*.zip"],
    "mlvu": ["MLVU/json/[1-7]_*.json", "MLVU/video/[1-7]_*/**"],
    "lvbench": ["video_info.meta.jsonl"],
}


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or any(c in name for c in "\r\n\t"):
        raise ValueError(f"Unsafe relative asset path: {name!r}")
    return path


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


class Uploader:
    def __init__(self, server, remote_root, stage, key=None, port=22, remove=False):
        if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.-]+", server) or server.startswith("-"):
            raise ValueError("Use an SSH host or user@host (without spaces)")
        root = PurePosixPath(remote_root)
        if not root.is_absolute() or root == PurePosixPath("/") or ".." in root.parts:
            raise ValueError("--remote-root must be an absolute Linux data directory, not /")
        self.server, self.root = server, root
        self.stage, self.remove = Path(stage).resolve(), remove
        self.stage.mkdir(parents=True, exist_ok=True)
        self.ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-p", str(port)]
        # Legacy SCP mode has explicit remote-shell quoting, including spaces
        # and apostrophes in official filenames. Requires a modern OpenSSH client.
        self.scp = ["scp", "-O", "-o", "BatchMode=yes", "-P", str(port)]
        if key:
            self.ssh += ["-i", str(Path(key).expanduser())]
            self.scp += ["-i", str(Path(key).expanduser())]

    def remote(self, command):
        return subprocess.run(self.ssh + [self.server, command], capture_output=True, text=True)

    def receipt(self, relative):
        digest = hashlib.sha256(relative.encode()).hexdigest()
        return self.stage / "receipts" / (digest + ".json")

    def verified(self, remote, size, digest):
        # A mismatching file is a normal cache miss. SSH errors must abort.
        q = shlex.quote(str(remote))
        command = (f"if [ -f {q} ] && [ \"$(stat -c %s -- {q})\" = {shlex.quote(str(size))} ]; then "
                   f"sha256sum -- {q}; else printf 'MISSING\\n'; fi")
        result = self.remote(command)
        if result.returncode:
            raise RuntimeError(f"SSH verification failed: {result.stderr.strip()}")
        return result.stdout.split(maxsplit=1)[0:1] == [digest]

    def already_uploaded(self, relative, revision):
        safe_relative(relative)
        path = self.receipt(relative)
        if not path.exists():
            return False
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["revision"] != revision or saved["server"] != self.server or saved["root"] != str(self.root):
            return False
        if self.verified(self.root / relative, saved["size"], saved["sha256"]):
            print(f"Verified on server; skipping download: {relative}", flush=True)
            return True
        return False

    def publish(self, local, relative, revision, keep=False):
        safe_relative(relative)
        local = Path(local).resolve()
        if not local.is_relative_to(self.stage):
            raise ValueError("Local asset must be inside --staging-dir")
        size, digest = local.stat().st_size, sha256(local)
        target = self.root / relative
        if not self.verified(target, size, digest):
            partial = str(target) + ".jev-upload.part"
            mkdir = self.remote("mkdir -p -- " + shlex.quote(str(target.parent)))
            if mkdir.returncode:
                raise RuntimeError(f"Cannot create remote directory: {mkdir.stderr.strip()}")
            print(f"Uploading {relative} ({size / 1024**3:.2f} GiB)", flush=True)
            subprocess.run(self.scp + [str(local), self.server + ":" + shlex.quote(partial)], check=True)
            if not self.verified(partial, size, digest):
                raise RuntimeError(f"Remote SHA256 mismatch; local file retained: {relative}")
            result = self.remote(f"mv -fT -- {shlex.quote(partial)} {shlex.quote(str(target))}")
            if result.returncode:
                raise RuntimeError(f"Cannot publish remote file: {result.stderr.strip()}")
        save_json(self.receipt(relative), dict(server=self.server, root=str(self.root), revision=revision,
                                             size=size, sha256=digest, path=relative))
        print(f"SHA256 verified: {relative}", flush=True)
        if self.remove and not keep:
            local.unlink()


def download_hf(target, upload):
    from huggingface_hub import HfApi, hf_hub_download
    if target in MODELS:
        repo, revision = MODELS[target]
        kind, prefix = "model", "models/" + repo.split("/")[1]
    else:
        repo, kind, prefix = REPOS[target], "dataset", "data/benchmarks/" + target
        pin = upload.stage / "pins" / (target + ".json")
        if not pin.exists():
            save_json(pin, dict(repo=repo, revision=HfApi().repo_info(repo, repo_type=kind).sha))
        saved = json.loads(pin.read_text(encoding="utf-8"))
        if saved["repo"] != repo:
            raise ValueError(f"Repository mismatch: {pin}")
        revision = saved["revision"]
    files = HfApi().list_repo_files(repo, repo_type=kind, revision=revision)
    if kind == "dataset":
        files = [name for name in files if any(fnmatch.fnmatchcase(name, p) for p in PATTERNS[target])]
    if not files:
        raise ValueError(f"No matching files in {repo}@{revision}")
    raw = upload.stage / "payload" / target
    metadata = None
    for name in sorted(files):
        safe_relative(name)
        relative = prefix + ("/download/" if kind == "dataset" else "/") + name
        # LVBench metadata is also needed locally to enumerate video IDs.
        keep_metadata = target == "lvbench"
        if not keep_metadata and upload.already_uploaded(relative, revision):
            continue
        local = Path(hf_hub_download(repo, name, repo_type=kind, revision=revision, local_dir=raw))
        if keep_metadata:
            metadata = local
        upload.publish(local, relative, revision, keep=keep_metadata)
    record = upload.stage / "generated" / target / "download_revision.json"
    save_json(record, dict(repo=repo, revision=revision))
    upload.publish(record, prefix + ("/.jev-download.json" if kind == "model" else "/download_revision.json"), revision)
    return metadata, revision, prefix


def download_lvbench(upload, annotation, revision, prefix, runtime, cookies):
    rows = [json.loads(s) for s in annotation.read_text(encoding="utf-8").splitlines() if s.strip()]
    validate_lvbench(rows)
    missing = []
    for row in rows:
        key = row["key"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", key):
            raise ValueError(f"Invalid video ID: {key}")
        relative = prefix + "/videos/" + key + ".mp4"
        if upload.already_uploaded(relative, revision):
            continue
        local = upload.stage / "payload/lvbench/videos" / (key + ".mp4")
        local.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--continue",
                   "--no-overwrites", "--retries", "5", "--fragment-retries", "5", "--js-runtimes", runtime,
                   "--format", "bestvideo[ext=mp4]/best[ext=mp4]", "--output", str(local),
                   "https://www.youtube.com/watch?v=" + key]
        if cookies:
            command += ["--cookies", str(cookies)]
        status = subprocess.run(command).returncode
        if status or not local.is_file() or not local.stat().st_size:
            missing.append(key)
            # Keep one incomplete video's fragments, rather than accumulating
            # partial downloads for many failures on a small local disk.
            break
        upload.publish(local, relative, revision)
    report = upload.stage / "generated/lvbench/missing-videos.json"
    save_json(report, missing)
    upload.publish(report, prefix + "/missing-videos.json", revision)
    if missing:
        raise RuntimeError(f"LVBench stopped at {missing[0]}; later videos not attempted; rerun to retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[*MODELS, *REPOS], required=True)
    parser.add_argument("--server", required=True, help="SSH user@host; key/agent authentication required")
    parser.add_argument("--remote-root", default="/public/hyj/jev")
    parser.add_argument("--staging-dir", type=Path, default=ROOT / "data/local-stage")
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--remove-after-upload", action="store_true")
    parser.add_argument("--js-runtime", default="deno")
    parser.add_argument("--cookies", type=Path)
    args = parser.parse_args()
    for tool in ("ssh", "scp"):
        if not shutil.which(tool):
            parser.error(f"{tool} is required")
    if args.target == "lvbench" and not shutil.which(args.js_runtime.split(":", 1)[-1]):
        parser.error("LVBench needs Deno (or Node >=22 with --js-runtime node)")
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    upload = Uploader(args.server, args.remote_root, args.staging_dir, args.ssh_key,
                      args.ssh_port, args.remove_after_upload)
    check = upload.remote("command -v sha256sum >/dev/null && command -v stat >/dev/null")
    if check.returncode:
        raise RuntimeError("SSH key/agent login or server tools unavailable: " + check.stderr.strip())
    annotation, revision, prefix = download_hf(args.target, upload)
    if args.target == "lvbench":
        download_lvbench(upload, annotation, revision, prefix, args.js_runtime, args.cookies)
    print("Download, SCP and SHA256 verification complete. Benchmarks still need --prepare-local on server.", flush=True)


if __name__ == "__main__":
    main()
