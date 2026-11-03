"""全量模型来源入口：复用现有Qwen/VideoJev前向，替换全量清单和计分。"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timejev.bench_full import BENCHMARKS, code_files, install_sources, read_full, snapshot


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('qwen_logits','videojev','native'), required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--max-frames', type=int, required=True)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--visual-token-budget', type=int, default=24576)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.9)
    p.add_argument('--gpu-count', type=int, choices=(4,5,6), default=6)
    p.add_argument('--max-memory-gib', type=float)
    p.add_argument('--max-memory-gib-per-gpu')
    p.add_argument('--max-new-tokens', type=int, default=32768)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(args):
    if args.kind=='native':
        if args.seed not in (42,43,44) or (args.fps,args.max_frames)!=(4.,4096):
            raise ValueError('全量native固定4FPS/4096，seed为42/43/44')
    elif args.seed!=42 or (args.fps,args.max_frames) not in ((2.,2048),(4.,4096)):
        raise ValueError('确定性来源仅执行seed42，帧率/上限必须匹配')
    if args.kind=='qwen_logits' and args.fps!=2.:
        raise ValueError('融合用Qwen来源固定2FPS/2048')
    if args.context<1024 or not 0<args.visual_token_budget<=args.context*4//5 or not 0<args.memory_fraction<=1:
        raise ValueError('上下文、视觉预算或显存比例无效')
    if args.max_new_tokens<1 or args.max_new_tokens>=args.context:
        raise ValueError('输出上限必须为正且小于上下文')
    if args.max_memory_gib is not None and (not math.isfinite(args.max_memory_gib) or args.max_memory_gib<=0):
        raise ValueError('每卡显存预算无效')
    if args.max_memory_gib is not None and args.max_memory_gib_per_gpu is not None:
        raise ValueError('两种显存预算参数不能同时指定')
    if args.max_memory_gib_per_gpu is not None:
        budgets=[float(x) for x in args.max_memory_gib_per_gpu.split(',')]
        if len(budgets)!=args.gpu_count or any(not math.isfinite(x) or x<=0 for x in budgets):
            raise ValueError('逐卡预算数量或值无效')


def main(argv=None):
    args=parser().parse_args(argv)
    validate(args)
    rows,full=read_full(args.manifest)
    checkpoint=args.checkpoint or ('models/VideoJev-4B' if args.kind=='videojev' else 'models/Qwen3.6-35B-A3B')
    if args.dry_run:
        result=dict(dry_run=True,gpu_started=False,output_created=False,dataset=full,kind=args.kind,
            checkpoint=str(Path(checkpoint).resolve()),settings=vars(args))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return result
    entry={'qwen_logits':'eval_longvideobench_qwen36.py','videojev':'eval_longvideobench_videojev.py',
           'native':'eval_bench_qwen36_native.py'}[args.kind]
    command=['--manifest',args.manifest,'--out',args.out,'--checkpoint',checkpoint,'--fps',str(args.fps),
        '--max-frames',str(args.max_frames),'--context',str(args.context),'--visual-token-budget',str(args.visual_token_budget),
        '--seed',str(args.seed),'--device',args.device,'--memory-fraction',str(args.memory_fraction)]
    if args.kind!='videojev':
        command+=['--gpu-count',str(args.gpu_count)]
        for flag,value in [('--max-memory-gib',args.max_memory_gib),('--max-memory-gib-per-gpu',args.max_memory_gib_per_gpu)]:
            if value is not None:command += [flag,str(value)]
    if args.kind=='native':command += ['--max-new-tokens',str(args.max_new_tokens)]
    if args.resume:command.append('--resume')
    with install_sources(rows,full):
        spec=importlib.util.spec_from_file_location('_full_'+args.kind,ROOT/'scripts'/entry)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if args.kind=='native':
            from timejev.lvb_native_generation import NativeExperiment
            hashes=code_files()
            class FullNativeExperiment(NativeExperiment):
                def __init__(self,parsed,settings,**kwargs):
                    kwargs.update(full_dataset=full,code_paths=[ROOT/name for name in hashes],
                        input_protocol=dict(kwargs['input_protocol'],name='video_bench_full_rgb_v1'))
                    super().__init__(parsed,settings,**kwargs)
                    snapshot(self.out,hashes)
            module.BENCHES=BENCHMARKS
            module.load_rows=lambda path:rows
            module.sample_seed=lambda path:None
            module.NativeExperiment=FullNativeExperiment
        if args.kind=='videojev':
            return module.main(command,save_logits=True,logits_only=True)
        return module.main(command)


if __name__=='__main__':
    main()
