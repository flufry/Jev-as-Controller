"""实验级输入指纹缓存：首次计算内容SHA，后续仅大文件使用stat匹配的旧SHA。"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

SCHEMA = 'bench_input_hash_cache_v1'
SUFFIXES = ('.json', '.jinja', '.py', '.safetensors')
POLICY = ('首次计算全部输入SHA256；后续视频/权重只核对路径、大小和mtime_ns，复用旧SHA；'
          '配置、源码和预测继续内容校验。同大小且保留mtime的内容修改无法保证检出。')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def stat_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'输入文件不存在：{path}')
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def inventory(rows, checkpoints):
    files = {}
    for row in rows:
        path = str(Path(row['video_path']).resolve())
        files[path] = dict(kind='video', **stat_file(path))
    models = {}
    for directory in sorted({str(Path(p).resolve()) for p in checkpoints}):
        root = Path(directory)
        if not (root/'config.json').is_file():
            raise ValueError(f'checkpoint缺少config.json：{root}')
        members = sorted(p for p in root.iterdir() if p.suffix in SUFFIXES and p.is_file())
        models[directory] = [p.name for p in members]
        for p in members:
            name = str(p.resolve())
            value = dict(kind='weight' if p.suffix=='.safetensors' else 'config', **stat_file(p))
            if name in files and files[name] != value:
                raise ValueError('同一输入路径对应多种类型')
            files[name] = value
    return files, models


def prepare_cache(path, rows, checkpoints, manifest, expected_sha256=None):
    """单主进程写缓存；每完成一个文件持久化，中断后只续算尚未完成者。"""
    path = Path(path).resolve()
    files, models = inventory(rows, checkpoints)
    identity = dict(manifest_sha256=sha256(manifest), files=files, checkpoints=models)
    if expected_sha256 is not None and (not path.is_file() or sha256(path)!=expected_sha256):
        raise ValueError('输入哈希缓存缺失或SHA变化，拒绝静默重建')
    if path.exists():
        cache = json.loads(path.read_text(encoding='utf-8'))
        if cache.get('schema')!=SCHEMA or cache.get('identity')!=identity:
            raise ValueError('输入路径、文件库存、大小或修改时间已变化；请检查输入并使用新实验目录')
        if not set(cache.get('hashes', {})) <= set(files):
            raise ValueError('输入缓存包含未知文件')
        for value in cache['hashes'].values():
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('输入缓存SHA格式无效')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = dict(schema=SCHEMA, identity=identity, hashes={}, policy=POLICY,
                     created_at=datetime.now(timezone.utc).isoformat())
        write_atomic(path, cache)
    total = len(files)
    print(json.dumps(dict(event='input_hash_cache', cached=len(cache['hashes']), total=total, path=str(path)), ensure_ascii=False), flush=True)
    for name, record in files.items():
        if name in cache['hashes']:
            continue
        before = stat_file(name)
        print(json.dumps(dict(event='hash_input_start', path=name, completed=len(cache['hashes']), total=total), ensure_ascii=False), flush=True)
        value = sha256(name)
        if before!=stat_file(name) or before!={k:record[k] for k in ('bytes','mtime_ns')}:
            raise ValueError(f'输入在计算SHA时变化：{name}')
        cache['hashes'][name] = value
        write_atomic(path, cache)
    # 防止文件在计算其他文件时被修改。
    if inventory(rows, checkpoints)!=(files, models):
        raise ValueError('输入在建立缓存期间变化')
    result = dict(path=str(path), sha256=sha256(path), files=total, policy=POLICY)
    print(json.dumps(dict(event='input_hash_cache_ready', **result), ensure_ascii=False), flush=True)
    return result


class InputHashCache:
    def __init__(self, path, expected_sha256, manifest, checkpoint):
        self.path = Path(path).resolve()
        if sha256(self.path)!=expected_sha256:
            raise ValueError('输入哈希缓存SHA不一致')
        self.expected_sha256 = expected_sha256
        self.cache = json.loads(self.path.read_text(encoding='utf-8'))
        identity = self.cache['identity']
        if self.cache.get('schema')!=SCHEMA or identity['manifest_sha256']!=sha256(manifest):
            raise ValueError('缓存与当前题集不一致')
        if set(self.cache['hashes'])!=set(identity['files']):
            raise ValueError('输入哈希缓存尚未完整完成')
        directory = Path(checkpoint).resolve()
        names = sorted(p.name for p in directory.iterdir() if p.suffix in SUFFIXES and p.is_file())
        if names!=identity['checkpoints'].get(str(directory)):
            raise ValueError('模型文件库存或路径与缓存不一致')
        raw = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        videos = {str((Path(manifest).resolve().parent/r['video_path']).resolve()) for r in raw}
        if videos!={p for p,r in identity['files'].items() if r['kind']=='video'}:
            raise ValueError('视频路径集合与缓存不一致')
        self.selected = videos | {str((directory/name).resolve()) for name in names}
        self.used = set()
        for name in self.selected:
            self.check(name)
            if identity['files'][name]['kind']=='config' and sha256(name)!=self.cache['hashes'][name]:
                raise ValueError(f'模型配置内容与缓存不一致：{name}')

    def check(self, name):
        record = self.cache['identity']['files'][name]
        if stat_file(name)!={k:record[k] for k in ('bytes','mtime_ns')}:
            raise ValueError(f'输入大小或修改时间已变化：{name}')

    def hash(self, path, original):
        name = str(Path(path).resolve())
        if name not in self.selected or self.cache['identity']['files'][name]['kind']=='config':
            return original(path)
        self.check(name)
        self.used.add(name)
        return self.cache['hashes'][name]

    def audit(self):
        return dict(schema='bench_cached_input_use_v1', timestamp=datetime.now(timezone.utc).isoformat(),
            cache_path=str(self.path), cache_sha256=self.expected_sha256, policy=POLICY,
            reused_large_files=len(self.used), large_input_content_rehashed=False)


@contextmanager
def install(cache):
    from timejev import lvb_experiment as lvb, lvb_native_generation as native
    old_lvb, old_native = lvb.sha256, native.sha256
    with patch.object(lvb, 'sha256', lambda p:cache.hash(p, old_lvb)), \
         patch.object(native, 'sha256', lambda p:cache.hash(p, old_native)):
        yield
