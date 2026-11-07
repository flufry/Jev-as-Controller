"""Official MVBench task inventory and deterministic media paths.

The NTU RGB+D archive was removed from the public release. Its task is
supported when the licensed videos are supplied locally, but is not downloaded.
"""
import hashlib
import json
import math
from pathlib import PurePosixPath


# task: (public ZIP, shared extraction directory, source type, temporal bound)
# The roots and bound flags follow the official MVBench evaluation notebook.
TASK_SPECS = {
    'action_sequence': ('star.zip', 'action_sequence', 'video', True),
    'action_prediction': ('star.zip', 'action_sequence', 'video', True),
    'action_antonym': ('ssv2_video.zip', 'action_antonym', 'video', False),
    'fine_grained_action': ('Moments_in_Time_Raw.zip', 'fine_grained_action', 'video', False),
    'unexpected_action': ('FunQA_test.zip', 'unexpected_action', 'video', False),
    'object_existence': ('clevrer.zip', 'object_existence', 'video', False),
    'object_interaction': ('star.zip', 'action_sequence', 'video', True),
    'object_shuffle': ('perception.zip', 'object_shuffle', 'video', False),
    'moving_direction': ('clevrer.zip', 'object_existence', 'video', False),
    'action_localization': ('sta.zip', 'action_localization', 'video', True),
    'scene_transition': ('scene_qa.zip', 'scene_transition', 'video', False),
    'action_count': ('perception.zip', 'object_shuffle', 'video', False),
    'moving_count': ('clevrer.zip', 'object_existence', 'video', False),
    'moving_attribute': ('clevrer.zip', 'object_existence', 'video', False),
    'state_change': ('perception.zip', 'object_shuffle', 'video', False),
    'fine_grained_pose': (None, 'fine_grained_pose', 'video', False),
    'character_order': ('perception.zip', 'object_shuffle', 'video', False),
    'egocentric_navigation': ('vlnqa.zip', 'egocentric_navigation', 'video', False),
    'episodic_reasoning': ('tvqa.zip', 'episodic_reasoning', 'frames', True),
    'counterfactual_inference': ('clevrer.zip', 'object_existence', 'video', False),
}
TASK_ARCHIVES = {task: spec[0] for task, spec in TASK_SPECS.items()}
DEFAULT_TASKS = tuple(task for task, spec in TASK_SPECS.items() if spec[0])
OFFICIAL_QUESTIONS_PER_TASK = 200


def selected_tasks(value=None):
    tasks = tuple(DEFAULT_TASKS if value is None else value)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASK_SPECS for task in tasks):
        raise ValueError(f'MVBench tasks must be nonempty, unique and from {sorted(TASK_SPECS)}')
    return tasks


def video_relative_path(value, *, frames=False):
    if not isinstance(value, str) or not value or chr(92) in value:
        raise ValueError(f'Invalid MVBench video path: {value!r}')
    path = PurePosixPath(value)
    if (path.is_absolute() or any(part in ('.', '..') for part in value.split('/'))
            or (not frames and not path.suffix)):
        raise ValueError(f'Invalid MVBench video path: {value!r}')
    return path


def source_relative_path(task, row):
    _, owner, media_type, _ = TASK_SPECS[task]
    return PurePosixPath(owner) / video_relative_path(row['video'], frames=media_type == 'frames')


def materialized_video_path(task, row):
    source = source_relative_path(task, row)
    if not TASK_SPECS[task][3]:
        return source
    identity = [source.as_posix(), float(row['start']), float(row['end'])]
    digest = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:20]
    return PurePosixPath('clips') / source.parts[0] / (digest + '.mp4')


def validate_reference(task, rows, *, expected_count=OFFICIAL_QUESTIONS_PER_TASK):
    selected_tasks((task,))
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f'MVBench {task} requires {expected_count} official questions')
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Invalid MVBench {task} question {index}')
        _, _, media_type, bounded = TASK_SPECS[task]
        video_relative_path(row.get('video'), frames=media_type == 'frames')
        options = row.get('candidates')
        if (not isinstance(row.get('question'), str) or not row['question'].strip()
                # The official counterfactual task includes single-candidate rows.
                # Preserve those rows and labels rather than inventing distractors.
                or not isinstance(options, list) or not 1 <= len(options) <= 8
                or any(not isinstance(option, str) or not option.strip() for option in options)
                or len(set(options)) != len(options) or options.count(row.get('answer')) != 1):
            raise ValueError(f'Invalid MVBench {task} answer/options at question {index}')
        if bounded:
            start, end = row.get('start'), row.get('end')
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start):
                raise ValueError(f'Invalid MVBench {task} time bound at question {index}')
        elif 'start' in row or 'end' in row:
            raise ValueError(f'MVBench {task} unexpectedly requires a temporal crop')
