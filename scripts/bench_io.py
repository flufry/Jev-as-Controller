"""CPU-only manifest validation and deterministic JSON for benchmark screening."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

BENCHES = ('videomme', 'mlvu', 'lvbench')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def jsonl_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows).encode('utf-8')


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def stratum(row):
    value = row.get('stratum', row.get('duration_group'))
    if value is None or not str(value).strip():
        raise ValueError(f'Missing stratum: {row.get("id")}')
    return str(value)


def load_rows(path, *, verify_meta=True):
    """Same path/group invariants as user_data, without importing GPU libraries."""
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Expected nonempty multiple-choice manifest')
    ids, videos, owners = set(), {}, {}
    for row in rows:
        for key in ('id', 'video_key', 'video_path', 'query'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Expected nonempty {key}')
        options = row.get('options')
        if (row.get('task') != 'choice' or not isinstance(options, list) or not 2 <= len(options) <= 8
                or any(not isinstance(s, str) or not s.strip() or '<|' in s for s in [row['query'], *options])
                or type(row.get('answer_index')) is not int or not 0 <= row['answer_index'] < len(options)):
            raise ValueError(f'Invalid choice question: {row["id"]}')
        duration = row.get('duration')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'Invalid duration seconds: {row["id"]}')
        row['video_path'] = str((path.parent / row['video_path']).resolve())
        source = row.get('media_source', dict(kind='video', path=row['video_path']))
        source = dict(source, path=str((path.parent / source['path']).resolve()))
        if source.get('kind') != 'video' or source['path'] != row['video_path']:
            raise ValueError('Screening requires explicit raw video matching video_path')
        row['media_source'] = source
        if not Path(row['video_path']).is_file():
            raise FileNotFoundError(row['video_path'])
        if row['id'] in ids:
            raise ValueError(f'Duplicate ID: {row["id"]}')
        ids.add(row['id'])
        group = (row['video_path'], duration, source)
        if videos.setdefault(row['video_key'], group) != group:
            raise ValueError(f'Inconsistent video group: {row["video_key"]}')
        if owners.setdefault(row['video_path'], row['video_key']) != row['video_key']:
            raise ValueError('One video path must have one video_key')
        stratum(row)
    if len({r.get('benchmark') for r in rows}) != 1:
        raise ValueError('Do not mix benchmarks in one manifest')
    meta_path = path.with_suffix('.meta.json')
    if verify_meta and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('sample_sha256', meta.get('manifest_sha256')) != sha256(path):
            raise ValueError('Manifest SHA differs from metadata')
        if meta.get('schema') == 'bench_subset_v1':
            counts = dict(Counter(stratum(r) for r in rows))
            if (meta['selected_ids'] != [r['id'] for r in rows] or meta['n'] != len(rows)
                    or {k: v for k, v in meta['quotas'].items() if v} != counts):
                raise ValueError('Sample IDs/counts differ from metadata')
    return rows


def portable_rows(rows, output):
    parent = Path(output).resolve().parent
    for row in rows:
        yield dict(row, video_path=Path(os.path.relpath(row['video_path'], parent)).as_posix(),
                   media_source=dict(kind='video', path=Path(os.path.relpath(row['media_source']['path'], parent)).as_posix()))
