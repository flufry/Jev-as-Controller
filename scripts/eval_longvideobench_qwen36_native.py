"""Qwen3.6-35B-A3B完整Instruct采样生成，解析最终选项后评估LongVideoBench。"""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.eval_longvideobench_qwen import ANSWER_INSTRUCTION, text_inputs
from scripts.eval_longvideobench_qwen36 import device_plan, load_backend, load_model, synchronize
from scripts.sample_longvideobench_proportional import read_sample
from timejev.lvb_experiment import evaluation_parser, prepare_record, protocol, record_audit, write_json
from timejev.lvb_native_generation import (
    NativeExperiment, PresencePenalty, PROTOCOL_NOTE, SEED_RULE, generation_parameters, question_seed,
)
from timejev.rgb_media import preprocess


def main(argv=None):
    from transformers import AutoConfig, AutoProcessor, LogitsProcessorList
    parser = evaluation_parser(__doc__, 'models/Qwen3.6-35B-A3B')
    parser.set_defaults(fps=4., max_frames=4096)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--gpu-count', type=int, choices=(4, 5, 6), default=6)
    parser.add_argument('--max-memory-gib', type=float, default=18.5)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--limit', type=int, help='仅用于独立smoke目录；记录截取的题目ID')
    parser.add_argument('--dry-run', action='store_true', help='仅CPU校验输入/本地配置，不加载或hash权重，不创建输出')
    args = parser.parse_args(argv)
    if (args.device not in ('cuda', 'cuda:0') or not math.isfinite(args.max_memory_gib)
            or args.max_memory_gib <= 0 or args.limit is not None and args.limit < 1):
        raise ValueError('输入设备、每卡权重预算或limit无效')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    rows, sample = read_sample(args.manifest, seed=args.sample_seed, n=100)
    if args.limit is not None and args.limit > len(rows):
        raise ValueError('limit不能超过固定题集大小')
    parameters = generation_parameters(args.max_new_tokens)
    input_protocol = protocol(args)
    checkpoint = Path(args.checkpoint)
    for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json',
                 'model.safetensors.index.json'):
        if not (checkpoint/name).is_file():
            raise ValueError(f'缺少本地模型/处理器文件：{name}')
    weights = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(weights.values()):
        if not (checkpoint/name).is_file():
            raise ValueError(f'缺少权重分片：{name}')
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    plan = device_plan(config, args.gpu_count)
    settings = dict(prompt=ANSWER_INSTRUCTION, thinking=False, scoring='native_generation_then_option_parsing',
        decoding=parameters, presence_penalty=1.5, presence_penalty_scope='generated_tokens_only_unique',
        presence_penalty_order='before_temperature_top_k_top_p', seed_rule=SEED_RULE,
        sample=sample, gpu_count=args.gpu_count, max_memory_gib=args.max_memory_gib, device_map=plan,
        dtype='bfloat16', quantized=False, cache='dynamic_without_offload',
        invalid_answers='count_as_incorrect', parser='deterministic_no_ground_truth_or_text_only_extractor',
        sampler_source='https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices', note=PROTOCOL_NOTE)
    dry_plan = dict(sample=sample, n=min(args.limit or len(rows), len(rows)), inference_seed=args.seed,
        settings=settings, protocol=input_protocol, output=str(Path(args.out).resolve()), dry_run=True,
        weight_shards=len(set(weights.values())), model_loaded=False, output_created=False)
    if args.dry_run:
        print(json.dumps(dry_plan, ensure_ascii=False, indent=2))
        return dry_plan
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpu_count:
        raise RuntimeError(f'CUDA驱动不可用或可见GPU少于{args.gpu_count}张；未加载模型或创建实验输出')
    backend, provenance = load_backend()
    settings['backend'] = provenance
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    run = NativeExperiment(args, settings, tokenizer=processor.tokenizer)
    if len(run.done) == len(run.rows):
        return run.finish()
    for gpu in range(args.gpu_count):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
    model, loading = load_model(args, config, backend)
    write_json(run.out/'loading.json', dict(loading=loading, backend=provenance))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    groups = {}
    for row in run.rows:
        if row['id'] not in run.done:
            groups.setdefault(row['video_key'], []).append(row)
    for members in groups.values():
        record = prepare_record(members[0], args.visual_token_budget, fps=args.fps, max_frames=args.max_frames)
        started = time.perf_counter()
        pixels = torch.cat(preprocess(processor, record))
        preprocessing_seconds = time.perf_counter()-started
        for row in members:
            inputs = text_inputs(processor, row, record, args.context)
            input_tokens = inputs['input_ids'].shape[-1]
            if input_tokens+args.max_new_tokens > args.context:
                raise ValueError(f'完整输入及输出上限超出context，拒绝截断：{row["id"]}')
            inputs['pixel_values_videos'] = pixels
            seed = question_seed(args.seed, row['id'])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            synchronize(args.gpu_count)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **{key: value.to('cuda:0') for key, value in inputs.items()}, **parameters,
                    logits_processor=LogitsProcessorList([PresencePenalty(input_tokens)]),
                    return_dict_in_generate=True, output_logits=False, output_scores=False,
                    cache_implementation='dynamic')
            synchronize(args.gpu_count)
            elapsed = time.perf_counter()-started
            tokens = output.sequences[0, input_tokens:].tolist()
            truncated = len(tokens) >= args.max_new_tokens and (not tokens or tokens[-1] not in eos)
            run.append_response(row, processor.tokenizer.decode(tokens, skip_special_tokens=True), tokens,
                raw_response=processor.tokenizer.decode(tokens, skip_special_tokens=False), truncated=truncated,
                question_seed=seed, input_tokens=input_tokens, eos_token_ids=sorted(eos),
                preprocessing=record_audit(record), timing=dict(video_preprocessing_seconds=preprocessing_seconds,
                    question_inference_seconds=elapsed))
            del inputs, output
        del pixels
    metrics = run.finish()
    write_json(run.out/'multigpu_runtime.json', dict(per_gpu={str(gpu): dict(
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(gpu),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(gpu)) for gpu in range(args.gpu_count)}))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == '__main__':
    main()
