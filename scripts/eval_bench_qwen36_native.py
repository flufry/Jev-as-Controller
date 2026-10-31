"""Qwen3.6 benchmark screening with full response generation and option parsing."""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.bench_io import load_rows
from scripts.eval_longvideobench_qwen import ANSWER_INSTRUCTION, text_inputs
from scripts.eval_longvideobench_qwen36 import device_plan, load_backend, load_model, synchronize
from scripts.eval_longvideobench_qwen36_native import (
    PresencePenalty, generation_parameters,
)
from timejev.lvb_experiment import evaluation_parser, prepare_record, protocol, record_audit, write_json
from timejev.lvb_native_generation import NativeExperiment, PROTOCOL_NOTE, SEED_RULE
from timejev.rgb_media import preprocess


def main(argv=None):
    from transformers import AutoConfig, AutoProcessor, LogitsProcessorList

    parser = evaluation_parser(__doc__, 'models/Qwen3.6-35B-A3B')
    parser.set_defaults(fps=2., max_frames=2048)
    parser.add_argument('--gpu-count', type=int, choices=(4, 5, 6), default=6)
    parser.add_argument('--max-memory-gib', type=float, default=18.5)
    parser.add_argument('--max-new-tokens', type=int, default=32768)
    parser.add_argument('--limit', type=int, help='仅用于独立 smoke 输出目录')
    args = parser.parse_args(argv)
    if (args.device not in ('cuda', 'cuda:0') or not math.isfinite(args.max_memory_gib)
            or args.max_memory_gib <= 0 or args.limit is not None and args.limit < 1):
        raise ValueError('输入设备、每卡权重预算或 limit 无效')

    rows = load_rows(args.manifest)
    meta_path = Path(args.manifest).with_suffix('.meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('schema') == 'bench_subset_v1' and meta.get('seed') != args.seed:
            raise ValueError('Inference seed must match sample metadata seed')
    for row in rows:
        row.setdefault('duration_group', row.get('stratum', row.get('official_duration', 'all')))
    if args.limit is not None:
        rows = rows[:args.limit]
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    parameters = generation_parameters(args.max_new_tokens)
    input_protocol = protocol(args)
    checkpoint = Path(args.checkpoint)
    for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json',
                 'model.safetensors.index.json'):
        if not (checkpoint/name).is_file():
            raise ValueError(f'缺少本地模型/处理器文件：{name}')
    weight_map = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(weight_map.values()):
        if not (checkpoint/name).is_file():
            raise ValueError(f'缺少权重分片：{name}')

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    plan = device_plan(config, args.gpu_count)
    settings = dict(prompt=ANSWER_INSTRUCTION, thinking=False,
        scoring='native_generation_then_option_parsing', decoding=parameters,
        presence_penalty=1.5, presence_penalty_scope='generated_tokens_only_unique',
        presence_penalty_order='before_temperature_top_k_top_p', seed_rule=SEED_RULE,
        gpu_count=args.gpu_count, max_memory_gib=args.max_memory_gib, device_map=plan,
        dtype='bfloat16', quantized=False, cache='dynamic_without_offload',
        invalid_answers='count_as_incorrect', parser='deterministic_no_ground_truth_or_text_only_extractor',
        note=PROTOCOL_NOTE)
    backend, provenance = load_backend()
    settings['backend'] = provenance
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    code_paths = [ROOT/'timejev/lvb_native_generation.py', Path(__file__),
        ROOT/'scripts/eval_longvideobench_qwen36_native.py', ROOT/'scripts/eval_longvideobench_qwen36.py',
        ROOT/'scripts/eval_longvideobench_qwen.py', ROOT/'scripts/bench_io.py',
        ROOT/'timejev/lvb_experiment.py', ROOT/'timejev/rgb_media.py', ROOT/'timejev/video.py']
    run = NativeExperiment(args, settings, tokenizer=processor.tokenizer, rows=rows,
        code_paths=code_paths, method='qwen36_native_instruct')
    if len(run.done) == len(run.rows):
        return run.finish()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpu_count:
        raise RuntimeError(f'CUDA驱动不可用或可见GPU少于{args.gpu_count}张')
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
            if input_tokens+parameters['max_new_tokens'] > args.context:
                raise ValueError(f'完整输入及输出上限超出 context，拒绝截断：{row["id"]}')
            inputs['pixel_values_videos'] = pixels
            seed = int.from_bytes(__import__('hashlib').sha256(
                f'{args.seed}:{row["id"]}'.encode()).digest()[:8], 'big') % 2**63
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            synchronize(args.gpu_count)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**{key: value.to('cuda:0') for key, value in inputs.items()},
                    **parameters, logits_processor=LogitsProcessorList([PresencePenalty(input_tokens)]),
                    return_dict_in_generate=True, output_logits=False, output_scores=False,
                    cache_implementation='dynamic')
            synchronize(args.gpu_count)
            elapsed = time.perf_counter()-started
            tokens = output.sequences[0, input_tokens:].tolist()
            truncated = len(tokens) >= parameters['max_new_tokens'] and (not tokens or tokens[-1] not in eos)
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
