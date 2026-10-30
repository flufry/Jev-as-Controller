"""Offline upload/cleanup checks; no SSH, SCP, model or video downloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.stage_local_assets import Uploader, safe_relative
from scripts import download_bench_data as bench


class LocalStageTests(unittest.TestCase):
    def upload_fixture(self, directory):
        root = Path(directory)
        local = root / "payload/model.bin"
        local.parent.mkdir()
        local.write_bytes(b"model fixture")
        return Uploader("hyj@example.test", "/public/hyj/jev", root, remove=True), local

    def test_cleanup_only_after_remote_verification_and_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.verified = Mock(side_effect=[False, True])
            upload.remote = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
            with patch("scripts.stage_local_assets.subprocess.run") as scp:
                upload.publish(local, "models/test/model.bin", "fixed")
            self.assertFalse(local.exists())
            saved = json.loads(upload.receipt("models/test/model.bin").read_text())
            self.assertEqual(13, saved["size"])
            self.assertEqual("fixed", saved["revision"])
            self.assertIn("-O", scp.call_args.args[0])
            self.assertIn("mv -fT", upload.remote.call_args.args[0])

    def test_scp_failure_keeps_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.verified = Mock(return_value=False)
            upload.remote = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
            with patch("scripts.stage_local_assets.subprocess.run", side_effect=subprocess.CalledProcessError(1, "scp")):
                with self.assertRaises(subprocess.CalledProcessError):
                    upload.publish(local, "models/test/model.bin", "fixed")
            self.assertTrue(local.exists())
            self.assertFalse(upload.receipt("models/test/model.bin").exists())

    def test_hash_mismatch_keeps_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.verified = Mock(return_value=False)
            upload.remote = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
            with patch("scripts.stage_local_assets.subprocess.run"):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    upload.publish(local, "models/test/model.bin", "fixed")
            self.assertTrue(local.exists())
            self.assertFalse(upload.receipt("models/test/model.bin").exists())

    def test_publish_failure_keeps_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.verified = Mock(side_effect=[False, True])
            upload.remote = Mock(side_effect=[subprocess.CompletedProcess([], 0, "", ""),
                                             subprocess.CompletedProcess([], 1, "", "permission denied")])
            with patch("scripts.stage_local_assets.subprocess.run"):
                with self.assertRaisesRegex(RuntimeError, "publish"):
                    upload.publish(local, "models/test/model.bin", "fixed")
            self.assertTrue(local.exists())

    def test_receipt_alone_does_not_skip_missing_remote_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.verified = Mock(return_value=True)
            upload.publish(local, "models/test/model.bin", "fixed")
            upload.verified = Mock(return_value=False)
            self.assertFalse(upload.already_uploaded("models/test/model.bin", "fixed"))
            upload.verified.assert_called_once()

    def test_ssh_failure_is_not_treated_as_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload, _ = self.upload_fixture(directory)
            upload.remote = Mock(return_value=subprocess.CompletedProcess([], 255, "", "authentication failed"))
            with self.assertRaisesRegex(RuntimeError, "SSH verification failed"):
                upload.verified("/public/file", 1, "a" * 64)

    def test_paths_cannot_escape_stage_or_remote_root(self):
        for name in ("../escape", "/absolute", "path/../../escape", "windows\\escape", "line\nbreak"):
            with self.assertRaises(ValueError):
                safe_relative(name)
        with tempfile.TemporaryDirectory() as directory:
            upload, local = self.upload_fixture(directory)
            upload.stage = local.parent / "different-stage"
            with self.assertRaisesRegex(ValueError, "inside"):
                upload.publish(local, "models/test/model.bin", "fixed")
            self.assertTrue(local.exists())

    def test_lvbench_local_preparation_never_calls_downloader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "download"
            raw.mkdir()
            (raw / "video_info.meta.jsonl").write_text(json.dumps({"key": "abcdefghijk", "qa": []}) + "\n")
            (root / "videos").mkdir()
            (root / "videos/abcdefghijk.mp4").write_bytes(b"video fixture")
            with patch.object(bench, "validate_lvbench"), patch.object(bench.shutil, "which", return_value=None), \
                    patch.object(bench.subprocess, "run") as run:
                bench.lvbench(root, raw, False, "deno", prepare_local=True)
            run.assert_not_called()
            self.assertEqual([], json.loads((root / "missing-videos.json").read_text()))

    def test_prepare_local_main_works_in_offline_mode_without_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "videomme"
            base.mkdir()
            (base / "download_revision.json").write_text(json.dumps({
                "repo": bench.REPOS["videomme"], "revision": "a" * 40}))
            with patch.object(sys, "argv", ["prepare", "--bench", "videomme", "--prepare-local", "--data-root", str(root)]), \
                    patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}), \
                    patch.object(bench, "snapshot") as snapshot, patch.object(bench, "videomme") as arrange:
                bench.main()
            snapshot.assert_not_called()
            arrange.assert_called_once_with(base, base / "download", False)

    def test_prepare_local_missing_video_fails_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "download"
            raw.mkdir()
            (raw / "video_info.meta.jsonl").write_text(json.dumps({"key": "abcdefghijk", "qa": []}) + "\n")
            with patch.object(bench, "validate_lvbench"), patch.object(bench.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "1 missing"):
                    bench.lvbench(root, raw, False, "deno", prepare_local=True)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
