"""Offline wheel preparation checks; no downloads, CUDA, or real model packages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_bench_wheels as prepare


def wheel(directory, name, version, tag="py3-none-any", dependencies=(), metadata_version=None):
    path = directory / f"{name}-{version}-{tag}.whl"
    info = f"{name}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {metadata_version or version}\n"
    metadata += "".join(f"Requires-Dist: {dependency}\n" for dependency in dependencies)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info + "/METADATA", metadata + "\n")
        archive.writestr(info + "/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: " + tag + "\n")
        archive.writestr(info + "/RECORD", "")
    return path


class WheelhouseTests(unittest.TestCase):
    def test_lock_preserves_post_and_cuda_versions(self):
        report = {"install": [{"metadata": {"name": "flash-attn", "version": "2.8.3.post2"}},
                              {"metadata": {"name": "torch", "version": "2.8.0+cu128"}}]}
        lock = prepare.constraints_from_report(report)
        self.assertIn("flash-attn==2.8.3.post2\n", lock)
        self.assertIn("torch==2.8.0+cu128\n", lock)

    def test_empty_resolution_cannot_report_ready(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            prepare.constraints_from_report({"install": []})

    def test_conflicting_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            prepare.constraints_from_report({"install": [
                {"metadata": {"name": "torch", "version": "2.8.0"}},
                {"metadata": {"name": "torch", "version": "2.9.0"}}]})

    def test_existing_extension_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            existing = wheel(destination, "flash_attn", "2.8.3.post2+cu12torch2.8cxx11abiTRUE",
                             "cp312-cp312-linux_x86_64")
            with patch.object(prepare, "WHEELS", destination), patch.object(prepare, "run") as run:
                actual = prepare.fetch_extension("flash_attn", "2.8.3.post2", "flash-attention", None, {})
            self.assertEqual(existing, actual)
            run.assert_not_called()

    def test_corrupt_extension_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl").write_bytes(b"truncated")
            with patch.object(prepare, "WHEELS", destination), self.assertRaises(zipfile.BadZipFile):
                prepare.fetch_extension("causal_conv1d", "1.7.0", "causal-conv1d", None, {})

    def test_failed_download_does_not_publish_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)

            def interrupted(command, env):
                Path(command[command.index("--output") + 1]).write_bytes(b"partial")
                raise subprocess.CalledProcessError(28, command)

            with patch.object(prepare, "WHEELS", destination), patch.object(prepare, "run", interrupted):
                with self.assertRaises(subprocess.CalledProcessError):
                    prepare.fetch_extension("causal_conv1d", "1.7.0", "causal-conv1d", None, {})
            self.assertEqual([], list(destination.glob("*.whl")))
            self.assertEqual(1, len(list(destination.glob("*.part"))))

    def test_real_offline_resolver_checks_closure_and_target_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            root = wheel(destination, "jev_fixture_root", "1.0", dependencies=["jev-fixture-dep>=2,<4"])
            # Official extension filenames can carry an ABI local version while
            # their METADATA contains only the public post-release version.
            extension = wheel(destination, "jev_fixture_native", "1.0.post2+cu12torch2.8cxx11abiTRUE",
                              "cp312-cp312-linux_x86_64", metadata_version="1.0.post2")
            dependency = wheel(destination, "jev_fixture_dep", "2.0", "cp312-cp312-manylinux_2_28_x86_64")
            # These newer releases are unusable in the target Docker image.
            wheel(destination, "jev_fixture_dep", "3.0", "cp312-cp312-win_amd64")
            wheel(destination, "jev_fixture_dep", "3.1", "cp312-cp312-manylinux_2_38_x86_64")
            report = destination / "report.json"
            env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
            env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
            command = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                       "--no-index", "--find-links", str(destination), "--report", str(report)]
            command += prepare.target_args() + [str(root), str(extension)]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            selected = {item["metadata"]["name"]: item["metadata"]["version"]
                        for item in json.loads(report.read_text())["install"]}
            self.assertEqual("2.0", selected["jev_fixture_dep"])
            self.assertEqual("1.0.post2", selected["jev_fixture_native"])
            # Reusing local wheel paths already inside --dest must also work.
            download = [sys.executable, "-m", "pip", "download", "--no-index", "--find-links",
                        str(destination), "--dest", str(destination)] + prepare.target_args() + [str(root), str(extension)]
            result = subprocess.run(download, env=env, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            dependency.unlink()
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("jev-fixture-dep", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
