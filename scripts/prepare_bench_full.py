"""准备视频 benchmark 的完整固定题集，不进行抽样。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timejev.bench_full import BENCHMARKS, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', choices=BENCHMARKS, required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--reference', required=True, help='官方标注；MLVU也接受七任务json目录')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mvbench-scope',choices=('public19','full20'),
        help='MVBench默认public19=3800题；有完整20任务素材时选full20=4000题')
    parser.add_argument('--allow-incomplete-population', action='store_true',
        help='仅LVBench：使用经缺失视频审计的全部可用题目，并记录覆盖率；不是官方1549题全量')
    args = parser.parse_args(argv)
    result = prepare(args.bench, args.source_manifest, args.reference, args.out,
        mvbench_scope=args.mvbench_scope, allow_incomplete_population=args.allow_incomplete_population)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
