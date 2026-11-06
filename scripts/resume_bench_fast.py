"""显式复用旧run.json中的输入大文件哈希；保留原队列源码/配置/预测校验。"""
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
import runpy
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).resolve()
RUNNER = ROOT/'scripts/run_bench_full.py'
SOURCE_ENTRY = ROOT/'scripts/eval_bench_full_source.py'
POLICY = ('显式信任先前输入SHA256：视频只检查路径/大小/mtime_ns，safetensors只检查路径/大小及旧记录中可用的mtime_ns；'
          '不重新验证这些文件内容。配置、源码、清单和预测仍走原校验。')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def argument(argv, flag):
    positions = [i for i,value in enumerate(argv) if value==flag]
    if len(positions)!=1 or positions[0]+1>=len(argv):
        raise ValueError(f'需要唯一参数及值：{flag}')
    return argv[positions[0]+1]


class CachedInputHashes:
    """仅缓存显式记录的原视频/权重；绝不伪造新文件的hash。"""
    def __init__(self, metadata, source_run_sha256):
        identity = metadata['identity']
        self.entries = {}
        self.counts = {'video':0, 'weight':0}
        self.used = set()
        self.source_run_sha256 = source_run_sha256
        for entry in identity['media'].values():
            self._add(entry['path'], entry, 'video', require_mtime=True)
        checkpoint = identity['checkpoint']
        directory = Path(checkpoint['path']).resolve()
        for name,entry in checkpoint['files'].items():
            relative = Path(name)
            if relative.is_absolute() or len(relative.parts)!=1:
                raise ValueError('checkpoint文件名不是单层相对路径')
            if relative.suffix=='.safetensors':
                self._add(directory/relative, entry, 'weight')

    def _add(self, path, entry, kind, require_mtime=False):
        path = Path(path).resolve()
        digest = entry.get('sha256')
        size,mtime = entry.get('bytes'),entry.get('mtime_ns')
        if (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
                or type(size) is not int or size<0 or require_mtime and type(mtime) is not int
                or mtime is not None and type(mtime) is not int):
            raise ValueError(f'先前输入指纹不完整或无效：{path}')
        value = dict(sha256=digest,bytes=size,mtime_ns=mtime,kind=kind)
        if path in self.entries and self.entries[path]!=value:
            raise ValueError(f'同一路径的历史指纹冲突：{path}')
        if path not in self.entries:
            self.entries[path]=value
            self.counts[kind]+=1

    def check_stat(self, path, entry):
        if not path.is_file():
            raise ValueError(f'缓存输入已不存在：{path}')
        stat = path.stat()
        if stat.st_size!=entry['bytes'] or entry['mtime_ns'] is not None and stat.st_mtime_ns!=entry['mtime_ns']:
            raise ValueError(f'输入大小或修改时间已变，拒绝复用旧hash：{path}；请使用原runner完整校验')

    def preflight(self):
        for path,entry in self.entries.items():
            self.check_stat(path,entry)

    def reuse_or_hash(self, path, original):
        absolute = Path(path).resolve()
        if absolute not in self.entries:
            return original(path)
        entry = self.entries[absolute]
        self.check_stat(absolute,entry)
        self.used.add(absolute)
        return entry['sha256']

    def audit(self):
        return dict(schema='explicit_input_hash_reuse_v1',timestamp=datetime.now(timezone.utc).isoformat(),
            helper_path=str(HELPER),helper_sha256=file_hash(HELPER),source_run_sha256=self.source_run_sha256,
            policy=POLICY,cached_files=self.counts,weight_mtime_unavailable=sum(
                e['kind']=='weight' and e['mtime_ns'] is None for e in self.entries.values()),
            used_files=len(self.used),large_input_content_rehashed=False)


@contextmanager
def reuse_hashes(cache):
    # 原生模块也直接导入sha256；checkpoint_identity内部则读取lvb全局sha256。
    lvb = importlib.import_module('timejev.lvb_experiment')
    native = importlib.import_module('timejev.lvb_native_generation')
    lvb_hash,native_hash = lvb.sha256,native.sha256
    with patch.object(lvb,'sha256',lambda p:cache.reuse_or_hash(p,lvb_hash)), \
         patch.object(native,'sha256',lambda p:cache.reuse_or_hash(p,native_hash)):
        yield


def validate_cross_stage_source(metadata, child_args, cache):
    """不同fps/seed可共享原始文件hash；题集、原视频、checkpoint必须相同。"""
    identity=metadata['identity']
    if identity.get('method') not in ('qwen36','qwen36_native_instruct'):
        raise ValueError('native必须复用Qwen来源的输入指纹，不能复用VideoJev权重')
    manifest=Path(argument(child_args,'--manifest')).resolve()
    if file_hash(manifest)!=identity.get('manifest_sha256'):
        raise ValueError('hash来源题集与当前manifest不同')
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    media=identity['media']
    if {row['video_key'] for row in rows}!=set(media):
        raise ValueError('hash来源没有覆盖相同视频集合')
    for row in rows:
        path=(manifest.parent/row['video_path']).resolve()
        if path!=Path(media[row['video_key']]['path']).resolve():
            raise ValueError('hash来源视频路径与当前manifest不同')
    checkpoint=identity['checkpoint']
    directory=Path(argument(child_args,'--checkpoint')).resolve()
    if directory!=Path(checkpoint['path']).resolve():
        raise ValueError('checkpoint路径与hash来源不同')
    files={p.name:p for p in directory.iterdir() if p.suffix in ('.json','.jinja','.py','.safetensors')}
    if set(files)!=set(checkpoint['files']):
        raise ValueError('checkpoint文件库存与hash来源不同')
    for name,path in files.items():
        previous=checkpoint['files'][name]
        if path.stat().st_size!=previous['bytes']:
            raise ValueError(f'checkpoint文件大小已变化：{name}')
        # 配置/tokenizer/处理器等小文件仍核对内容，只有权重跳过内容读取。
        if path.suffix!='.safetensors' and file_hash(path)!=previous['sha256']:
            raise ValueError(f'checkpoint配置内容已变化：{name}')


def run_child(argv, hash_source=None, expected_source_sha256=None):
    if not argv or Path(argv[0]).resolve()!=SOURCE_ENTRY:
        raise ValueError('快速恢复仅支持本仓库全量模型入口')
    entry,child_args = argv[0],argv[1:]
    if hash_source is None and '--resume' not in child_args:
        raise ValueError('没有--resume，不允许复用输入hash')
    if hash_source is not None and argument(child_args,'--kind')!='native':
        raise ValueError('跨阶段复用输入hash仅支持native')
    out = Path(argument(child_args,'--out')).resolve()
    run_path = Path(hash_source).resolve() if hash_source is not None else out/'run.json'
    if not run_path.is_file():
        raise ValueError('缺少已有run.json，无法复用输入hash；使用原runner完成首次初始化')
    raw = run_path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_source_sha256 is not None and source_sha256!=expected_source_sha256:
        raise ValueError('信任的hash来源run.json已发生变化')
    metadata = json.loads(raw.decode('utf-8'))
    if Path(argument(child_args,'--checkpoint')).resolve()!=Path(metadata['identity']['checkpoint']['path']).resolve():
        raise ValueError('checkpoint路径与历史运行不同，拒绝快速恢复')
    cache = CachedInputHashes(metadata,source_sha256)
    cache.preflight()
    if hash_source is not None:
        validate_cross_stage_source(metadata, child_args, cache)

    print(json.dumps(dict(event='input_hash_reuse',**cache.audit()),ensure_ascii=False),flush=True)
    old_argv = sys.argv
    try:
        # 保持原入口的sys.argv[0]和参数，原有代码/设置指纹仍独立核验。
        sys.argv = [entry,*child_args]
        with reuse_hashes(cache):
            runpy.run_path(entry,run_name='__main__')
    finally:
        sys.argv = old_argv
        # 不改写run.json中的身份字段；单独记录本次有意降低的输入校验强度。
        if out.is_dir():
            audit=dict(cache.audit(),source_run_path=str(run_path),cross_stage_reuse=hash_source is not None)
            with (out/'input_hash_reuse.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(audit,ensure_ascii=False,allow_nan=False)+'\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='--child':
        return run_child(argv[1:])
    if argv and argv[0]=='--child-from':
        if len(argv)<4:raise ValueError('需要来源run.json、SHA256和原入口命令')
        return run_child(argv[3:],hash_source=argv[1],expected_source_sha256=argv[2])
    if '--resume' not in argv:
        raise ValueError('此入口仅用于显式--resume恢复；首次运行请用run_bench_full.py')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    runner = importlib.import_module('scripts.run_bench_full')
    original_popen = subprocess.Popen

    def launch(command,*args,**kwargs):
        if (isinstance(command,(list,tuple)) and len(command)>1
                and Path(command[1]).resolve()==SOURCE_ENTRY and '--resume' in command):
            out = Path(argument(command,'--out'))
            if (out/'run.json').is_file():
                command = [command[0],str(HELPER),'--child',*command[1:]]
            else:
                print(json.dumps(dict(event='input_hash_reuse_unavailable',directory=str(out),
                    reason='尚未保存run.json，继续使用原初始化校验'),ensure_ascii=False),flush=True)
        return original_popen(command,*args,**kwargs)

    print(json.dumps(dict(event='fast_resume_requested',policy=POLICY,
        note='仅已有run.json的恢复子进程复用大文件hash；新阶段仍全量校验；不是跳过全部校验'),ensure_ascii=False),flush=True)
    old_argv = sys.argv
    try:
        sys.argv = [str(RUNNER),*argv]
        with patch.object(runner.subprocess,'Popen',launch):
            return runner.main(argv)
    finally:
        sys.argv = old_argv


if __name__=='__main__':
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    main()
