"""以未量化 BF16 Qwen3.6-35B-A3B 在 4–6 张 GPU 上评测 LongVideoBench。"""
from collections import Counter
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_longvideobench_qwen import (
    ANSWER_INSTRUCTION, candidate_scores, candidate_token_ids, text_inputs,
)
from timejev.lvb_experiment import (
    Experiment, evaluation_parser, prepare_record, record_audit, sha256, write_json,
)
from timejev.rgb_media import preprocess

GROUPED_REVISION = '172fada89fa7364fe5d026b3a0dfab58b591ffdd'
CUTLASS_REVISION = '8783c41851cd3582490e04e69e0cd756a8c1db7f'


def device_plan(config, gpu_count):
    """完整解码层连续分卡；视觉与词嵌入同卡，输出头放最后一卡。"""
    if (config.model_type != 'qwen3_5_moe'
            or config.architectures != ['Qwen3_5MoeForConditionalGeneration']
            or getattr(config, 'quantization_config', None) is not None):
        raise ValueError('需要官方、未量化的 Qwen3.6 Qwen3_5Moe checkpoint')
    if gpu_count not in (4, 5, 6):
        raise ValueError('gpu_count 必须为 4、5 或 6')
    text = config.text_config
    if (text.num_hidden_layers != 40 or text.num_experts != 256 or text.num_experts_per_tok != 8
            or config.tie_word_embeddings or text.tie_word_embeddings):
        raise ValueError('模型结构与官方 Qwen3.6-35B-A3B 不符')
    plan = {'model.visual': 'cuda:0', 'model.language_model.embed_tokens': 'cuda:0',
            'model.language_model.rotary_emb': 'cuda:0',
            'model.language_model.norm': f'cuda:{gpu_count-1}', 'lm_head': f'cuda:{gpu_count-1}'}
    for layer in range(text.num_hidden_layers):
        plan[f'model.language_model.layers.{layer}'] = f'cuda:{layer*gpu_count//text.num_hidden_layers}'
    return plan


def load_backend():
    """只接受本任务编译的 SM89 CUTLASS NN/NT 后端，不自动降级。"""
    overlay = ROOT/'.deps-qwen36'
    sys.path.insert(0, str(overlay))
    try:
        backend = importlib.import_module('grouped_gemm.backend')
        compiled = importlib.import_module('grouped_gemm_backend')
    except (ImportError, OSError) as error:
        raise RuntimeError('缺少 .deps-qwen36 的已编译 grouped GEMM / CUDA permutation 后端') from error
    binary = Path(compiled.__file__).resolve()
    if not binary.is_relative_to(overlay.resolve()) or not getattr(compiled, 'qwen36_cutlass_only', False):
        raise RuntimeError('拒绝使用未验证或会降级的 grouped GEMM 后端')
    source = overlay/'src/grouped_gemm'
    for path, revision in ((source, GROUPED_REVISION), (source/'third_party/cutlass', CUTLASS_REVISION)):
        if subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() != revision:
            raise ValueError(f'后端源码 revision 不匹配：{path}')
    provenance = dict(name='cutlass_sm89_nn_nt_cuda_permutation', grouped_gemm_version='1.1.4',
        grouped_gemm_revision=GROUPED_REVISION, cutlass_revision=CUTLASS_REVISION,
        binary=str(binary), binary_sha256=sha256(binary),
        backend_python_sha256=sha256(backend.__file__),
        patch_sha256=sha256(ROOT/'reports/setup-qwen36-runtime/sm89-cutlass.patch'),
        grouped_source_sha256=sha256(source/'csrc/grouped_gemm.cu'),
        ops_source_sha256=sha256(source/'csrc/ops.cu'),
        xtuner_revision=subprocess.check_output(
            ['git', '-C', str(ROOT/'.deps-xtuner'), 'rev-parse', 'HEAD'], text=True).strip(),
        accelerate_version=version('accelerate'),
        helper_sha256=sha256(ROOT/'scripts/eval_longvideobench_qwen.py'))
    return backend, provenance


def routed_experts(experts, hidden, indices, weights, backend):
    """先排列 token，再执行两次分组矩阵乘；不按 token 复制专家权重。"""
    if (hidden.dtype != torch.bfloat16 or experts.gate_up_proj.dtype != torch.bfloat16
            or experts.down_proj.dtype != torch.bfloat16):
        raise ValueError('专家输入及全部专家权重必须保持 BF16')
    if experts.gate_up_proj.device != hidden.device or experts.down_proj.device != hidden.device:
        raise ValueError('完整专家层与其输入必须位于同一设备')
    if not len(hidden):
        return hidden
    indices = indices.to(torch.int32).contiguous()
    counts = torch.bincount(indices.reshape(-1).long(), minlength=experts.num_experts).cpu()
    dispatched, row_map, _ = backend.permute(hidden.contiguous(), indices, 0, [], indices.numel())
    gate, up = backend.gmm(dispatched, experts.gate_up_proj, counts, trans_b=True).chunk(2, dim=-1)
    activated = (experts.act_fn(gate)*up).contiguous()
    output = backend.gmm(activated, experts.down_proj, counts, trans_b=True)
    return backend.unpermute(output, row_map, weights.float().contiguous(), len(hidden), indices.shape[1])


def install_backend(model, backend):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    def forward(self, hidden, indices, weights):
        if hidden.device.type != 'cuda':
            raise RuntimeError('分组专家仅支持 CUDA，禁止 CPU 或 Python 专家循环降级')
        # CUDA extension 使用当前设备的 stream；每个完整层可以位于不同 GPU。
        with torch.cuda.device(hidden.device):
            return routed_experts(self, hidden, indices, weights, backend)

    modules = [module for module in model.modules() if isinstance(module, Qwen3_5MoeExperts)]
    if len(modules) != model.config.text_config.num_hidden_layers:
        raise ValueError('没有完整覆盖官方模型的全部 MoE 专家层')
    for module in modules:
        module.forward = MethodType(forward, module)
    return len(modules)


def validate_loaded(model, expected_map, max_memory):
    if getattr(model, 'is_quantized', False) or getattr(model.config, 'quantization_config', None) is not None:
        raise ValueError('不允许量化模型')
    if model.hf_device_map != expected_map:
        raise ValueError('实际 device_map 与完整层分配不一致，禁止 CPU/disk offload')
    allocated, dtypes = Counter(), Counter()
    for name, parameter in model.named_parameters():
        if parameter.device.type != 'cuda' or parameter.dtype != torch.bfloat16:
            raise ValueError(f'模型参数不是 CUDA BF16：{name}')
        allocated[parameter.device.index] += parameter.numel()*parameter.element_size()
        dtypes[str(parameter.dtype)] += parameter.numel()
    if set(allocated) != set(max_memory) or any(allocated[gpu] > max_memory[gpu] for gpu in allocated):
        raise ValueError('权重分布未覆盖所选 GPU，或超出显式每卡权重预算')
    return dict(parameter_bytes_by_gpu=dict(allocated), parameter_dtypes=dict(dtypes),
                quantized=False, cpu_or_disk_offload=False, device_map=expected_map)


def load_model(args, config, backend, model_class=None):
    from transformers import Qwen3_5MoeForConditionalGeneration
    model_class = model_class or Qwen3_5MoeForConditionalGeneration
    plan = device_plan(config, args.gpu_count)
    max_memory = {gpu: int(args.max_memory_gib*2**30) for gpu in range(args.gpu_count)}
    model, loading = model_class.from_pretrained(
        args.checkpoint, config=config, local_files_only=True, dtype=torch.bfloat16,
        device_map=plan, max_memory=max_memory, attn_implementation='flash_attention_2',
        experts_implementation='eager', output_loading_info=True,
    )
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'模型加载不完整：{loading}')
    audit = validate_loaded(model, plan, max_memory)
    # Transformers 的加载诊断可返回 set；校验后稳定转为可保存的 JSON 列表。
    loading = {key: sorted(value) if isinstance(value, (set, frozenset)) else value
               for key, value in loading.items()}
    audit.update(loading_info=loading, max_memory_bytes=max_memory,
                 patched_expert_layers=install_backend(model, backend),
                 router='official_unchanged_bf16_linear_fp32_softmax_topk_normalize_bf16')
    return model.eval().requires_grad_(False), audit


def synchronize(gpu_count):
    for gpu in range(gpu_count):
        torch.cuda.synchronize(gpu)


def main(argv=None):
    from transformers import AutoConfig, AutoProcessor
    parser = evaluation_parser(__doc__, 'models/Qwen3.6-35B-A3B')
    parser.add_argument('--gpu-count', type=int, choices=(4, 5, 6), default=6)
    parser.add_argument('--max-memory-gib', type=float, default=18.5, help='每卡模型权重预算；额外保留激活空间')
    args = parser.parse_args(argv)
    if args.device not in ('cuda', 'cuda:0') or not math.isfinite(args.max_memory_gib) or args.max_memory_gib <= 0:
        raise ValueError('输入设备必须是逻辑 cuda:0，且每卡权重预算为有限正数；通过 CUDA_VISIBLE_DEVICES 选卡')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    backend, provenance = load_backend()
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    plan = device_plan(config, args.gpu_count)
    settings = dict(prompt=ANSWER_INSTRUCTION, thinking=False, scoring='first_decode_bare_letter_candidates',
        probabilities='softmax_over_candidate_letters_only', decoding=dict(do_sample=False, max_new_tokens=1),
        gpu_count=args.gpu_count, max_memory_gib=args.max_memory_gib, device_map=plan,
        dtype='bfloat16', quantized=False, backend=provenance, cache='dynamic_without_offload')
    run = Experiment(args, 'qwen36', settings)
    if len(run.done) == len(run.rows):
        run.finish()
        return
    if torch.cuda.device_count() < args.gpu_count:
        raise ValueError('可见 GPU 数量少于指定的完整层分布卡数')
    torch.manual_seed(args.seed)
    for gpu in range(args.gpu_count):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model, loading = load_model(args, config, backend)
    write_json(run.out/'loading.json', loading | dict(backend=provenance))
    groups = {}
    for row in run.rows:
        if row['id'] not in run.done:
            groups.setdefault(row['video_key'], []).append(row)
    for rows in groups.values():
        record = prepare_record(rows[0], args.visual_token_budget, fps=args.fps, max_frames=args.max_frames)
        started = time.perf_counter()
        pixels = torch.cat(preprocess(processor, record))
        preprocessing_seconds = time.perf_counter()-started
        for row in rows:
            ids = candidate_token_ids(processor.tokenizer, len(row['options']))
            inputs = text_inputs(processor, row, record, args.context)
            input_tokens = inputs['input_ids'].shape[-1]
            inputs['pixel_values_videos'] = pixels
            synchronize(args.gpu_count)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **{key: value.to('cuda:0') for key, value in inputs.items()},
                    do_sample=False, max_new_tokens=1, return_dict_in_generate=True,
                    output_logits=True, cache_implementation='dynamic',
                )
            synchronize(args.gpu_count)
            seconds = time.perf_counter()-started
            logits, probabilities = candidate_scores(output.logits[0][0], ids)
            first = int(output.sequences[0, input_tokens])
            run.append(row, probabilities, candidate_logits=logits, candidate_token_ids=ids,
                first_token_id=first, first_token_token=processor.tokenizer.convert_ids_to_tokens(first),
                first_token_raw_decode=processor.tokenizer.decode([first], skip_special_tokens=False),
                first_token_is_bare_candidate=first in ids, input_tokens=input_tokens,
                preprocessing=record_audit(record), timing=dict(video_preprocessing_seconds=preprocessing_seconds,
                question_inference_seconds=seconds))
            del output, inputs
        del pixels
    run.finish()
    write_json(run.out/'multigpu_runtime.json', dict(
        per_gpu={str(gpu): dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(gpu),
                               peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(gpu))
                 for gpu in range(args.gpu_count)}, loading=loading))


if __name__ == '__main__':
    main()
