"""在同一 LongVideoBench 分层题目子集上评估本地 VideoJev-4B。"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timejev.lvb_experiment import (
    Experiment, evaluation_parser, prepare_record, record_audit,
)


def predict_group(model, processor, rows, record, context, *, save_logits=True, logits_only=False):
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
        if logits_only:
            from scripts.videojev_candidate_logits import predict_features_with_logits
            predictions = predict_features_with_logits(model, processor, features, record, questions, context,
                                                       logits_only=True)
        elif save_logits:
            # Match upstream's native prediction path, retaining scores across packed forwards.
            logits = []

            def retain_logits(module, inputs, output):
                for values, mask in zip(output.logits, output.score_mask):
                    logits.append(values[:int(mask.sum())].float().cpu().tolist())

            handle = model.register_forward_hook(retain_logits)
            try:
                predictions = model.predict_features(processor, features, record, questions, context=context)
            finally:
                handle.remove()
            if ([p['id'] for p in predictions] != [row['id'] for row in rows]
                    or len(logits) != len(predictions)):
                raise ValueError('VideoJev 原始 logits 与问题输出未完整对齐')
            for prediction, values in zip(predictions, logits):
                if len(values) != len(prediction['probabilities']):
                    raise ValueError('VideoJev 原始 logits 与候选概率数量不一致')
                prediction['logits'] = values
        else:
            predictions = model.predict_features(processor, features, record, questions, context=context)
    return predictions


def main(argv=None, *, save_logits=True, logits_only=False):
    if logits_only and not save_logits:
        raise ValueError('logits_only requires saving raw logits')
    description = 'VideoJev raw-logit evaluation for Video-MME / MLVU / LVBench.' if logits_only else __doc__
    args = evaluation_parser(description, 'models/VideoJev-4B').parse_args(argv)
    settings = dict(scoring='direct_candidate_softmax', question_grouping='shared_video_prefix_without_answers')
    if save_logits:
        from scripts.videojev_candidate_logits import validate_logits
        from timejev.lvb_experiment import sha256
        paths = ['scripts/videojev_candidate_logits.py', 'scripts/eval_bench_videojev.py',
                 'videojev_hf/processing_videojev.py']
        root = Path(__file__).resolve().parents[1]
        settings.update(scoring='candidate_logits_argmax' if logits_only else 'direct_candidate_softmax',
            candidate_scores='head_logits_before_candidate_softmax', output_scores='logits_only' if logits_only else 'logits_and_probabilities',
            logits_adapter_sha256={p: sha256(root/p) for p in paths})
    run = Experiment(args, 'videojev', settings)
    if save_logits:
        for name in paths:
            target = run.out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root/name).read_bytes())
    if save_logits and run.done:
        import json
        targets = {r['id']: r for r in run.rows}
        for line in run.predictions_path.read_text(encoding='utf-8').splitlines():
            value = json.loads(line)
            validate_logits(value, len(targets[value['id']]['options']))
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
        predictions = predict_group(model, processor, rows, record, args.context,
                                    save_logits=save_logits, logits_only=logits_only)
        torch.cuda.synchronize(args.device)
        group_seconds = time.perf_counter() - group_started
        by_id = {p['id']: p for p in predictions}
        if len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
            raise ValueError('模型输出没有覆盖当前视频的全部抽中问题')
        for row in rows:
            if row['id'] not in run.done:
                details = {}
                if save_logits:
                    details['logits'] = validate_logits(by_id[row['id']], len(row['options']))
                run.append(row, by_id[row['id']].get('probabilities'), **details, preprocessing=record_audit(record),
                           timing=dict(video_group_total_seconds=group_seconds,
                                       question_count=len(rows)))
    run.finish()


if __name__ == '__main__':
    main()
