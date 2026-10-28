"""在分层采样的 LongVideoBench 上评测 Qwen3.5-4B 首步候选字母概率。"""
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ANSWER_INSTRUCTION = 'Reply with only the letter of the correct option.'


def candidate_token_ids(tokenizer, count):
    """只接受无空格、且可独立往返解码的单 token 候选字母。"""
    if not 2 <= count <= 26:
        raise ValueError('候选数量必须在 2 到 26 之间。')
    result = []
    for index in range(count):
        letter = chr(ord('A') + index)
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != letter:
            raise ValueError(f'候选 {letter} 不是可往返解码的无空格单 token。')
        result.append(ids[0])
    if len(set(result)) != count:
        raise ValueError('不同候选字母映射到了相同 token。')
    return result


def candidate_scores(first_logits, token_ids):
    """提取首 decode 位置的原始 logits，仅在候选字母内归一化。"""
    if first_logits.ndim != 1:
        raise ValueError('需要单个首 decode 位置的全词表 logits。')
    selected = first_logits[token_ids].detach().float().cpu()
    if not torch.isfinite(selected).all():
        raise ValueError('候选字母 logits 存在非有限值。')
    return selected.tolist(), selected.softmax(dim=-1).tolist()


def text_inputs(processor, row, record, context):
    """保留共享采样时间线及几何，不调用原生视频重采样。"""
    import numpy as np
    from transformers.video_utils import VideoMetadata

    prompt = row['query'] + '\n' + '\n'.join(row['options']) + '\n' + ANSWER_INSTRUCTION
    text = processor.apply_chat_template([
        {'role': 'user', 'content': [
            {'type': 'video', 'video': 'provided_in_memory'},
            {'type': 'text', 'text': prompt},
        ]},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    metadata = VideoMetadata(
        total_num_frames=record['frames'], fps=record['effective_fps'],
        frames_indices=list(range(record['frames'])),
    )
    replacement = processor.replace_video_token({
        'video_grid_thw': np.asarray([record['grid_thw']]),
        'video_metadata': [metadata],
    }, video_idx=0)
    texts, _ = processor.get_text_with_replacements([text], [], [replacement], [])
    inputs = processor(text=texts, return_tensors='pt')
    if int((inputs['input_ids'] == processor.video_token_id).sum()) != record['tokens']:
        raise ValueError('视频占位 token 数量与共享几何不一致。')
    if inputs['input_ids'].shape[-1] + 1 > context:
        raise ValueError('完整视频及问题超出上下文预算，未做截断。')
    return dict(inputs) | {'video_grid_thw': torch.tensor([record['grid_thw']], dtype=torch.long)}


def main(argv=None):
    from timejev.lvb_experiment import (
        Experiment, evaluation_parser, prepare_record, record_audit,
    )
    from timejev.rgb_media import preprocess
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    args = evaluation_parser(__doc__, 'models/Qwen3.5-4B').parse_args(argv)
    settings = dict(
        prompt=ANSWER_INSTRUCTION, thinking=False,
        scoring='first_decode_bare_letter_candidates',
        probabilities='softmax_over_candidate_letters_only',
        decoding=dict(do_sample=False, max_new_tokens=1),
    )
    experiment = Experiment(args, 'qwen35', settings)
    if len(experiment.done) == len(experiment.rows):
        experiment.finish()
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('该评测使用 BF16 / FlashAttention 2，需要 CUDA 设备。')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': str(device)}, attn_implementation='flash_attention_2',
    ).eval()
    groups = {}
    for row in experiment.rows:
        if row['id'] not in experiment.done:
            groups.setdefault(row['video_key'], []).append(row)
    for rows in groups.values():
        record = prepare_record(rows[0], args.visual_token_budget,
                                fps=args.fps, max_frames=args.max_frames)
        audit = record_audit(record)
        # CPU patches 在同视频各题之间复用；不缓存或跨模型复用视觉特征。
        preprocessing_started = time.perf_counter()
        pixels = torch.cat(preprocess(processor, record), dim=0)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        for row in rows:
            ids = candidate_token_ids(processor.tokenizer, len(row['options']))
            inputs = text_inputs(processor, row, record, args.context)
            input_tokens = inputs['input_ids'].shape[-1]
            inputs['pixel_values_videos'] = pixels
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **{key: value.to(device) for key, value in inputs.items()},
                    do_sample=False, max_new_tokens=1,
                    return_dict_in_generate=True, output_logits=True,
                )
            torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            logits, probabilities = candidate_scores(output.logits[0][0], ids)
            first = int(output.sequences[0, input_tokens])
            experiment.append(
                row, probabilities, candidate_logits=logits, candidate_token_ids=ids,
                first_token_id=first,
                first_token_token=processor.tokenizer.convert_ids_to_tokens(first),
                first_token_raw_decode=processor.tokenizer.decode([first], skip_special_tokens=False),
                first_token_is_bare_candidate=first in ids,
                input_tokens=input_tokens, preprocessing=audit,
                timing=dict(video_preprocessing_seconds=preprocessing_seconds,
                            question_inference_seconds=inference_seconds),
            )
            del output, inputs
        del pixels
    experiment.finish()


if __name__ == '__main__':
    main()
