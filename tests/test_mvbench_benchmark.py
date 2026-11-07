"""Offline checks for the selected official MVBench video tasks."""
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts import download_bench_data as download
from scripts.bench_io import load_rows
from scripts.bench_compat import benchmark_scope, manifest_scope
from scripts.build_bench_manifest import build
from scripts.sample_bench_subset import sample
from timejev.mvbench_benchmark import (DEFAULT_TASKS, TASK_SPECS, materialized_video_path,
    selected_tasks, validate_reference, video_relative_path)


def questions(video='left/clip.mp4'):
    return [dict(video=video, question=f'Question {index}?',
                 candidates=['left', 'forward', 'right', 'stop'], answer='forward')
            for index in range(200)]


class MVBenchTests(unittest.TestCase):
    def test_task_and_annotation_validation(self):
        self.assertEqual(19, len(selected_tasks()))
        self.assertNotIn('fine_grained_pose', DEFAULT_TASKS)
        self.assertEqual(20, len(TASK_SPECS))
        self.assertEqual(('left', 'clip.mp4'), video_relative_path('left/clip.mp4').parts)
        for path in ('../escape.mp4', '/absolute.mp4', 'left\\clip.mp4'):
            with self.assertRaises(ValueError):
                video_relative_path(path)
        with self.assertRaises(ValueError):
            selected_tasks(['action_antonym', 'action_antonym'])
        validate_reference('egocentric_navigation', questions())
        singleton = questions()
        singleton[81]['candidates'] = ['forward']
        validate_reference('counterfactual_inference', singleton)
        singleton[81]['candidates'] = []
        with self.assertRaisesRegex(ValueError, 'answer/options'):
            validate_reference('counterfactual_inference', singleton)
        invalid = questions()
        invalid[0]['answer'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'answer/options'):
            validate_reference('egocentric_navigation', invalid)
        bounded = questions('clip.mp4')
        for row in bounded:
            row.update(start=2.0, end=5.0)
        validate_reference('action_sequence', bounded)
        self.assertEqual('clips', materialized_video_path('action_sequence', bounded[0]).parts[0])
        bounded[0]['end'] = 1.0
        with self.assertRaisesRegex(ValueError, 'time bound'):
            validate_reference('action_sequence', bounded)

    def test_nested_archive_to_manifest_and_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'mvbench'
            raw_json = base / 'download/json/egocentric_navigation.json'
            raw_json.parent.mkdir(parents=True)
            reference = questions()
            reference[81]['candidates'] = ['forward']
            raw_json.write_text(json.dumps(reference), encoding='utf-8')
            archive = base / 'download/video/vlnqa.zip'
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('vlnqa/left/clip.mp4', b'video bytes')
                output.writestr('vlnqa/other.mp4', b'ignore')
                output.writestr('../escape.mp4', b'ignore')
            download.mvbench(base, base / 'download', False, ['egocentric_navigation'])
            self.assertEqual(b'video bytes', (base / 'videos/egocentric_navigation/left/clip.mp4').read_bytes())
            self.assertFalse((base / 'videos/egocentric_navigation/other.mp4').exists())
            metadata = build('mvbench', base, probe=lambda _: (640, 360, 5.0))
            rows = load_rows(base / 'questions.jsonl')
            self.assertEqual(200, metadata['n'])
            self.assertEqual(['egocentric_navigation'], metadata['coverage']['tasks'])
            self.assertFalse(metadata['coverage']['official_full_benchmark'])
            self.assertEqual(200, len(rows))
            self.assertEqual(1, len({row['video_key'] for row in rows}))
            self.assertEqual('B. forward', rows[0]['options'][1])
            self.assertEqual(1, rows[0]['answer_index'])
            self.assertEqual(['A. forward'], rows[81]['options'])
            self.assertEqual(0, rows[81]['answer_index'])
            from scripts.analyze_controller_changes import read_manifest
            self.assertEqual(200, len(read_manifest(base / 'questions.jsonl')))
            subset = sample('mvbench', base / 'sample.jsonl', manifest=base / 'questions.jsonl',
                            seed=42, n=20)
            self.assertEqual(20, subset['n'])
            self.assertEqual('proportional_mvbench_task', subset['sampling'])
            self.assertEqual(20, len(load_rows(base / 'sample.jsonl')))

    def test_snapshot_downloads_only_selected_archives(self):
        module = ModuleType('huggingface_hub')
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha='f' * 40)
        module.HfApi = Mock(return_value=api)
        module.snapshot_download = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'huggingface_hub': module}):
            download.snapshot('mvbench', Path(directory), False, None, 2,
                              ['egocentric_navigation'])
            patterns = module.snapshot_download.call_args.kwargs['allow_patterns']
            self.assertEqual(['json/egocentric_navigation.json', 'video/vlnqa.zip'], patterns)
            download.snapshot('mvbench', Path(directory), False, None, 2)
            patterns = module.snapshot_download.call_args.kwargs['allow_patterns']
            self.assertEqual(19, len([p for p in patterns if p.endswith('.json')]))
            self.assertEqual(10, len([p for p in patterns if p.endswith('.zip')]))

    def test_shared_archive_and_bounded_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for task in ('action_sequence', 'action_prediction'):
                source = base / 'download/json' / (task + '.json')
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(json.dumps([dict(row, start=2, end=5)
                                              for row in questions('clip.mp4')]), encoding='utf-8')
            archive = base / 'download/video/star.zip'
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('star/Charades_v1_480/clip.mp4', b'video bytes')
            def fake_clip(source, target, start, end, *, frames=False):
                self.assertTrue(source.is_file())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'clip bytes')
            with patch.object(download.shutil, 'which', return_value='ffmpeg'), \
                    patch.object(download, 'materialize_mvbench_clip', side_effect=fake_clip) as convert:
                download.mvbench(base, base / 'download', False,
                                 ['action_sequence', 'action_prediction'])
            self.assertEqual(1, convert.call_count)
            metadata = build('mvbench', base, probe=lambda _: (640, 360, 3.0))
            rows = load_rows(base / 'questions.jsonl')
            self.assertEqual(400, metadata['n'])
            self.assertEqual(1, metadata['videos'])
            self.assertEqual(1, len({row['video_key'] for row in rows}))
            self.assertEqual('action_sequence/clip.mp4', rows[0]['source_video'])
            self.assertEqual('time_bound_to_mp4', rows[0]['media_transform'])

    def test_frame_directory_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / 'tvqa.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('tvqa/frames_fps3_hq/episode/00001.jpg', b'one')
                output.writestr('tvqa/frames_fps3_hq/episode/00002.jpg', b'two')
                output.writestr('tvqa/frames_fps3_hq/other/00001.jpg', b'other')
            download.unpack_mvbench(archive, base / 'videos', {'episode'}, frame_dirs=True)
            self.assertEqual(2, len(list((base / 'videos/episode').glob('*.jpg'))))
            self.assertFalse((base / 'videos/other').exists())

    def test_clip_commands_keep_bounds_and_frame_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / 'input.mp4'
            raw.write_bytes(b'raw')
            frames = base / 'frames'
            frames.mkdir()
            (frames / '00001.jpg').write_bytes(b'frame')
            commands = []
            def fake_run(command, check):
                self.assertTrue(check)
                commands.append(command)
                Path(command[-1]).write_bytes(b'converted')
            with patch.object(download.shutil, 'which', return_value='ffmpeg'), \
                    patch.object(download.subprocess, 'run', side_effect=fake_run):
                download.materialize_mvbench_clip(raw, base / 'raw.mp4', 1.25, 3.75)
                download.materialize_mvbench_clip(frames, base / 'frames.mp4', 1.25, 3.75,
                                                  frames=True)
            self.assertEqual('2.5', commands[0][commands[0].index('-t') + 1])
            self.assertEqual('1.25', commands[0][commands[0].index('-ss') + 1])
            self.assertEqual('3', commands[1][commands[1].index('-framerate') + 1])
            self.assertIn('%05d.jpg', commands[1][commands[1].index('-i') + 1])
            self.assertEqual(b'converted', (base / 'raw.mp4').read_bytes())

    def test_all_public_tasks_form_one_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archives = {}
            for task in DEFAULT_TASKS:
                archive, _, media_type, bounded = TASK_SPECS[task]
                video = 'episode' if media_type == 'frames' else 'clip.mp4'
                rows = questions(video)
                if bounded:
                    rows = [dict(row, start=0, end=3) for row in rows]
                source = base / 'download/json' / (task + '.json')
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(json.dumps(rows), encoding='utf-8')
                archives[archive] = media_type
            for archive, media_type in archives.items():
                target = base / 'download/video' / archive
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target, 'w') as output:
                    member = ('tvqa/frames_fps3_hq/episode/00001.jpg' if media_type == 'frames'
                              else archive.removesuffix('.zip') + '/clip.mp4')
                    output.writestr(member, b'media bytes')
            def fake_clip(source, target, start, end, *, frames=False):
                self.assertTrue(source.is_dir() if frames else source.is_file())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'clip bytes')
            with patch.object(download.shutil, 'which', return_value='ffmpeg'), \
                    patch.object(download, 'materialize_mvbench_clip', side_effect=fake_clip):
                download.mvbench(base, base / 'download', False, None)
            metadata = build('mvbench', base, probe=lambda _: (640, 360, 3.0))
            self.assertEqual(3800, metadata['n'])
            self.assertEqual(19, len(metadata['coverage']['tasks']))
            self.assertFalse(metadata['coverage']['official_full_benchmark'])
            self.assertEqual('fine_grained_pose', metadata['coverage']['unavailable_public_task'])
            rows = load_rows(base / 'questions.jsonl')
            self.assertEqual(3800, len(rows))
            self.assertEqual('mvbench_public19', benchmark_scope(rows))
            self.assertEqual('mvbench_public19', manifest_scope(base / 'questions.jsonl'))

    def test_missing_video_fails_without_writing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            annotation = base / 'json/egocentric_navigation.json'
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps(questions()), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing video'):
                build('mvbench', base, probe=lambda _: (640, 360, 5.0))
            self.assertFalse((base / 'questions.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
