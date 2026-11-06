"""由全量runner调用，保留原模型入口argv，只复用实验级大文件SHA。"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from timejev.bench_input_cache import InputHashCache, install


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command or Path(command[0]).resolve()!=ROOT/'scripts/eval_bench_full_source.py':
        raise ValueError('只允许调用当前仓库全量模型入口')
    def value(flag):
        if command.count(flag)!=1:raise ValueError(f'需要唯一{flag}')
        return command[command.index(flag)+1]
    cache=InputHashCache(args.cache,args.cache_sha256,value('--manifest'),value('--checkpoint'))
    out=Path(value('--out'))
    print(json.dumps(dict(event='cached_inputs_verified',**cache.audit()),ensure_ascii=False),flush=True)
    previous=sys.argv
    try:
        sys.argv=list(command)
        with install(cache):runpy.run_path(command[0],run_name='__main__')
    finally:
        sys.argv=previous
        if out.is_dir():
            with (out/'input_hash_cache_use.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(cache.audit(),ensure_ascii=False)+'\n')


if __name__=='__main__':main()
