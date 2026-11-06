"""汇总/只读重验视频benchmark的全量三seed结果及单步Qwen/VideoJev采样OR。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from timejev.bench_full import SEEDS, read_json, sha256, summarize, verify_files


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--base',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--source-base',help='已有三份候选来源的实验目录；默认与--base相同，可读旧per-seed目录中的seed42')
    parser.add_argument('--seed',type=int,choices=SEEDS,help='只汇总该seed答案：共用候选来源及该seed native；省略时汇总全部三个seed')
    parser.add_argument('--verify',action='store_true',help='不写文件，重算全部来源并核对已保存汇总')
    args=parser.parse_args(argv)
    result=summarize(args.manifest,args.base,None if args.verify else args.out,
        seeds=(args.seed,) if args.seed is not None else SEEDS,source_base=args.source_base)
    if args.verify:
        out=Path(args.out)
        artifact=read_json(out/'artifact_manifest.json')['files']
        if set(artifact)!={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p!=out/'artifact_manifest.json'}:
            raise ValueError('汇总文件库存不完整')
        for name in artifact:
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('非法产物路径')
        verify_files({str(out/name):digest for name,digest in artifact.items()})
        if result!=read_json(out/'results.json'):
            raise ValueError('保存的汇总与完整重算不一致')
    print(json.dumps(dict(benchmark=result['benchmark'],n=result['n'],verified=args.verify,
        methods={k:{'mean':v['mean'],'variance_pp2':v['sample_variance_percentage_points_squared']} for k,v in result['methods'].items()}),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    main()
