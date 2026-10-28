"""在同一 LongVideoBench 分层题目子集上评估本地 VideoJev-4B。"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timejev.lvb_experiment import (
    Experiment, evaluation_parser, prepare_record, record_audit,
)


def predict_group(model, processor, rows, record, context):
    import torch
    from timejev.rgb_media import preprocess

    features = []
    with torch.inference_mode():
        for chunk in preprocess(processor, record):
            h, w = record['grid_thw'][1:]
            grid = torch.tensor([[len(chunk)//(h*w), h, w]])
            features.append(model.encode_video(chunk, grid))
        features = torch.cat(features)
        questions = [{k: row[k] for k in ('id', 'task', 'query', 'options')} for row in rows]
        predictions = model.predict_features(processor, features, record, questions, context=context)
    return predictions


def main(argv=None):
    args = evaluation_parser(__doc__, 'models/VideoJev-4B').parse_args(argv)
    run = Experiment(args, 'videojev', dict(scoring='direct_candidate_softmax',
                     question_grouping='shared_video_prefix_without_answers'))
    if len(run.done) == len(run.rows):
        run.finish()
        return
    import torch
    from transformers import AutoModel, AutoProcessor
    from timejev.multiturn_inputs import group_memberships

    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model, loading = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, device_map={'': args.device},
        attn_implementation='flash_attention_2', output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'模型加载不完整：{loading}')
    model.prepare_for_inference()
    for group in group_memberships(run.rows):
        rows = [run.rows[i] for i in group['query_indices']]
        if all(row['id'] in run.done for row in rows):
            continue
        # 恢复时仍保留该视频全部抽中问题，避免改变共享前缀的因果上下文。
        record = prepare_record(rows[0], args.visual_token_budget,
                                fps=args.fps, max_frames=args.max_frames)
        torch.cuda.synchronize(args.device)
        group_started = time.perf_counter()
        predictions = predict_group(model, processor, rows, record, args.context)
        torch.cuda.synchronize(args.device)
        group_seconds = time.perf_counter() - group_started
        by_id = {p['id']: p for p in predictions}
        if len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
            raise ValueError('模型输出没有覆盖当前视频的全部抽中问题')
        for row in rows:
            if row['id'] not in run.done:
                run.append(row, by_id[row['id']]['probabilities'], preprocessing=record_audit(record),
                           timing=dict(video_group_total_seconds=group_seconds,
                                       question_count=len(rows)))
    run.finish()


if __name__ == '__main__':
    main()
