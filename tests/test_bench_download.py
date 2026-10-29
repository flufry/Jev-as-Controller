"""Synthetic archive/path/resume checks; no real benchmark downloads."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts import download_bench_data as download


class BenchmarkDownloadTests(unittest.TestCase):
    def test_zip_paths_are_flattened_to_official_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "chunk.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../../outside/official.mp4", b"official video")
                output.writestr("unrelated.mp4", b"unrelated")
            download.unpack_videomme([archive], base / "videos", {"official"})
            self.assertEqual(b"official video", (base / "videos/official.mp4").read_bytes())
            self.assertFalse((base / "outside").exists())
            self.assertFalse((base / "videos/unrelated.mp4").exists())
            # Repeating the same download/extraction keeps existing files.
            download.unpack_videomme([archive], base / "videos", {"official"})

    def test_missing_zip_video_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing 1"):
                download.unpack_videomme([], Path(directory) / "videos", {"missing"})

    def test_existing_annotation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            download.write_json(path, [1, 2])
            download.write_json(path, [1, 2])
            with self.assertRaisesRegex(ValueError, "differs"):
                download.write_json(path, [2, 1])
            self.assertEqual([1, 2], json.loads(path.read_text()))

    def test_mlvu_maps_exact_task_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = base / "1_plotQA.json"
            task.write_text(json.dumps([{"video": "clip.mp4"}]))
            video = base / "video/1_plotQA/clip.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            self.assertEqual({"clip.mp4": "1_plotQA/clip.mp4"}, download.mlvu_video_map([task], base / "video"))

    def test_mlvu_conflicting_task_videos_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            files = []
            for index, task_name in enumerate(["1_plotQA", "2_needle"]):
                task = base / (task_name + ".json")
                task.write_text(json.dumps([{"video": "same.mp4"}]))
                files.append(task)
                video = base / "video" / task_name / "same.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(bytes([index]))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                download.mlvu_video_map(files, base / "video")

    def test_mlvu_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = base / "1_plotQA.json"
            task.write_text(json.dumps([{"video": "../outside.mp4"}]))
            with self.assertRaisesRegex(ValueError, "Invalid MLVU"):
                download.mlvu_video_map([task], base / "video")

    def test_snapshot_resume_reuses_recorded_commit(self):
        module = ModuleType("huggingface_hub")
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        module.HfApi = Mock(return_value=api)
        module.snapshot_download = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"huggingface_hub": module}):
            base = Path(directory)
            download.snapshot("videomme", base, True, None, 2)
            api.dataset_info.return_value = SimpleNamespace(sha="b" * 40)
            download.snapshot("videomme", base, False, None, 2)
            api.dataset_info.assert_called_once()
            self.assertEqual("a" * 40, module.snapshot_download.call_args.kwargs["revision"])
            self.assertIn("videos_chunked_*.zip", module.snapshot_download.call_args.kwargs["allow_patterns"])

    def test_lvbench_failure_records_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "download"
            raw.mkdir()
            (raw / "video_info.meta.jsonl").write_text(json.dumps({"key": "abcdefghijk", "qa": []}) + "\n")
            with patch.object(download, "validate_lvbench"), patch.object(download.shutil, "which", return_value="deno"), \
                 patch.object(download.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                with self.assertRaisesRegex(RuntimeError, "1 missing videos"):
                    download.lvbench(base, raw, False, "deno")
            self.assertEqual(["abcdefghijk"], json.loads((base / "missing-videos.json").read_text()))


if __name__ == "__main__":
    unittest.main()
