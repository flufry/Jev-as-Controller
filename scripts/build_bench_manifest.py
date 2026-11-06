"""Build video manifests for supported QA benchmarks and prepared MVBench clips."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_io import BENCHES, json_bytes, jsonl_bytes, load_rows, portable_rows, sha256, write_new
from timejev.mlvu_benchmark import TASK_COUNTS, formatted_options, validate_reference as validate_mlvu
from timejev.mvbench_benchmark import (TASK_SPECS, materialized_video_path,
    selected_tasks as mvbench_tasks, source_relative_path, validate_reference as validate_mvbench)
from timejev.videomme_benchmark import validate_reference as validate_videomme
from timejev.qa_manifest import parse_choice

LV_CATEGORIES = {'key information retrieval', 'event understanding', 'summarization',
                 'entity recognition', 'reasoning', 'temporal grounding'}


def read_annotation(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, list):
        raise ValueError('Annotation must be a JSON list, JSONL or Parquet')
    return value


def validate_lvbench(reference):
    keys, uids = set(), set()
    for video in reference:
        if video['key'] in keys:
            raise ValueError('Duplicate LVBench video key')
        keys.add(video['key'])
        for qa in video['qa']:
            uid = str(qa['uid'])
            if uid in uids:
                raise ValueError('Duplicate LVBench UID')
            uids.add(uid)
            parse_choice(qa['question'], qa['answer'])
            types = qa['question_type']
            if (not isinstance(types, list) or not types or len(set(types)) != len(types)
                    or not set(types) <= LV_CATEGORIES):
                raise ValueError('Expected official LVBench category list')
    if len(uids) != 1549:
        raise ValueError('LVBench requires 1549 unique official UIDs')


def mlvu_reference(path):
    path = Path(path)
    # Original MLVU release: seven numbered JSON files; ignore the two generative tasks.
    files = sorted(p for p in path.glob('*.json') if p.stem.split('_', 1)[-1] in TASK_COUNTS) if path.is_dir() else [path]
    if not files:
        raise ValueError('No seven-task MLVU annotation files found')
    rows = []
    for file in files:
        for index, source in enumerate(read_annotation(file)):
            row = dict(source)
            task = row.get('task_type', file.stem.split('_', 1)[-1])
            if task not in TASK_COUNTS:
                raise ValueError(f'Not an MLVU-Dev MCQ task: {task}')
            row.update(task_type=task, official_id=str(row.get('official_id', f'{task}/{index}')))
            rows.append(row)
    validate_mlvu(rows)
    return rows, files


def mvbench_reference(path):
    path = Path(path)
    files = sorted(path.glob('*.json')) if path.is_dir() else [path]
    if not files:
        raise ValueError('No supported MVBench annotation files found')
    mvbench_tasks(tuple(file.stem for file in files))
    rows = []
    for file in files:
        source = read_annotation(file)
        validate_mvbench(file.stem, source)
        rows.extend(dict(item, task_type=file.stem, official_id=f'{file.stem}/{index:03d}')
                    for index, item in enumerate(source))
    return rows, files


def adapt(bench, reference, video_root, video_map=None, *, missing_videos=None):
    """Only question content determines strata; labels never affect selection."""
    video_root, video_map = Path(video_root).resolve(), video_map or {}

    def path_for(key, filename):
        path = (video_root / video_map.get(key, filename)).resolve()
        if not path.is_relative_to(video_root):
            raise ValueError(f'Video path outside video root: {path}')
        if missing_videos is not None and (not path.is_file() or path.stat().st_size == 0):
            return None
        if not path.is_relative_to(video_root) or not path.is_file():
            raise ValueError(f'Missing video or path outside video root: {path}')
        return str(path)

    rows = []
    if bench == 'videomme':
        validate_videomme(reference)
        for r in reference:
            qid, video = str(r['question_id']), r['videoID']
            rows.append(dict(id=f'videomme/{qid}', question_id=qid, video_key=f'videomme/{video}',
                video_path=path_for(video, video + '.mp4'), query=r['question'], options=r['options'],
                answer_index=ord(r['answer']) - 65, stratum=r['duration'], official_duration=r['duration'],
                **{k: r[k] for k in ('domain', 'sub_category', 'task_type')}))
    elif bench == 'mlvu':
        validate_mlvu(reference)
        for r in reference:
            rows.append(dict(id='mlvu/' + r['official_id'], question_id=r['official_id'],
                video_key='mlvu/' + Path(r['video']).stem, video_path=path_for(r['video'], r['video']),
                query=r['question'], options=formatted_options(r), answer_index=r['candidates'].index(r['answer']),
                stratum=r['task_type'], task_type=r['task_type'], official_duration=r['duration']))
    elif bench == 'mvbench':
        for r in reference:
            task = r['task_type']
            video = materialized_video_path(task, r).as_posix()
            row = dict(id='mvbench/' + r['official_id'], question_id=r['official_id'],
                video_key='mvbench/' + video, video_path=path_for(video, video), query=r['question'],
                options=[f'{chr(65+i)}. {value}' for i, value in enumerate(r['candidates'])],
                answer_index=r['candidates'].index(r['answer']), stratum=task, task_type=task,
                source_video=source_relative_path(task, r).as_posix())
            if TASK_SPECS[task][3]:
                row.update(clip_start=r['start'], clip_end=r['end'],
                           media_transform='frames_3fps_to_mp4' if TASK_SPECS[task][2] == 'frames'
                           else 'time_bound_to_mp4')
            rows.append(row)
    else:
        validate_lvbench(reference)
        for video in reference:
            path = path_for(video['key'], video['key'] + '.mp4')
            if path is None:
                missing_videos.append(dict(video_key='lvbench/' + video['key'], path=str(
                    (video_root / video_map.get(video['key'], video['key'] + '.mp4')).resolve()),
                    reason='missing_or_empty', question_ids=['lvbench/' + str(q['uid']) for q in video['qa']]))
                continue
            for qa in video['qa']:
                query, options, answer = parse_choice(qa['question'], qa['answer'])
                uid = str(qa['uid'])
                rows.append(dict(id='lvbench/' + uid, uid=uid, question_id=uid,
                    video_key='lvbench/' + video['key'], video_path=path,
                    query=query, options=options, answer_index=answer, question_type=qa['question_type'],
                    stratum=json.dumps(sorted(qa['question_type']), ensure_ascii=False, separators=(',', ':'))))
    for row in rows:
        row.update(task='choice', benchmark=bench, media_source=dict(kind='video', path=row['video_path']))
    return rows


def build(bench, base, annotation=None, videos=None, out=None, video_map=None, probe=None,
          allow_missing_videos=False):
    if allow_missing_videos and bench != 'lvbench':
        raise ValueError('--allow-missing-videos is only supported for LVBench')
    base = Path(base).resolve()
    default_annotation = base / ('official.jsonl' if bench == 'lvbench' else 'json' if bench == 'mvbench' else 'official.json')
    annotation = Path(annotation or default_annotation).resolve()
    out = Path(out or base / 'questions.jsonl').resolve()
    meta_path = out.with_suffix('.meta.json')
    if out.exists() or meta_path.exists():
        raise FileExistsError('Choose new manifest and metadata paths')
    if bench == 'mlvu':
        reference, files = mlvu_reference(annotation)
    elif bench == 'mvbench':
        reference, files = mvbench_reference(annotation)
    else:
        reference, files = read_annotation(annotation), [annotation]
    mapping = json.loads(Path(video_map).read_text(encoding='utf-8')) if video_map else None
    if video_map:
        files.append(Path(video_map).resolve())
    before = {str(p): sha256(p) for p in files}
    missing = [] if allow_missing_videos else None
    rows = adapt(bench, reference, videos or base / 'videos', mapping, missing_videos=missing)
    if not rows:
        raise ValueError('No available questions; no manifest written')
    if probe is None:
        from timejev.rgb_media import probe_video_geometry
        probe = probe_video_geometry
    cache = {}
    for row in rows:
        path = row['video_path']
        if path not in cache:
            width, height, seconds = probe(path)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f'Cannot establish complete video duration: {path}')
            cache[path] = dict(width=width, height=height, duration=seconds)
        row['duration'] = cache[path]['duration']
    if before != {str(p): sha256(p) for p in files}:
        raise ValueError('Annotation changed while constructing manifest')
    data = jsonl_bytes(portable_rows(rows, out))
    import hashlib
    meta = dict(schema='bench_manifest_v1', benchmark=bench, n=len(rows), videos=len(cache),
        sources=before, manifest_sha256=hashlib.sha256(data).hexdigest(),
        strata=dict(Counter(r['stratum'] for r in rows)), duration_source='decoded_video_probe',
        training_overlap_verified=False, builder_sha256=sha256(__file__))
    if bench == 'mvbench':
        tasks = sorted({r['task_type'] for r in rows})
        full = len(tasks) == len(TASK_SPECS)
        meta['coverage'] = dict(scope='official_full' if full else 'selected_official_tasks',
            official_full_benchmark=full, tasks=tasks, expected_questions_per_task=200,
            available_questions=len(rows), official_full_questions=4000,
            unavailable_public_task=None if 'fine_grained_pose' in tasks else 'fine_grained_pose')
    if allow_missing_videos:
        meta['coverage'] = dict(scope='available_videos_subset', official_full_benchmark=False,
            expected_questions=1549, expected_videos=len(reference),
            available_questions=len(rows), available_videos=len(cache),
            excluded_questions=sum(len(v['question_ids']) for v in missing),
            excluded_videos=missing, question_coverage=len(rows) / 1549,
            exclusion_policy='exclude_all_questions_of_missing_or_empty_video')
    write_new(out, data)
    load_rows(out, verify_meta=False)
    write_new(meta_path, json_bytes(meta))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', choices=BENCHES, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--annotation', help='JSON/JSONL/Parquet; MLVU and MVBench also accept their json/ directories')
    parser.add_argument('--videos', help='Root of decoded video files')
    parser.add_argument('--video-map', help='JSON: official video ID/MLVU video field -> path relative to --videos')
    parser.add_argument('--out')
    parser.add_argument('--allow-missing-videos', action='store_true',
                        help='LVBench only: explicitly build an available-video subset and record exclusions in metadata')
    print(json.dumps(build(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
