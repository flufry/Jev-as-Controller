"""Read the actual VideoJev head scores, including from older local HF checkpoints."""
import math


def validate_logits(prediction, count):
    logits, probabilities = prediction.get('candidate_logits'), prediction.get('probabilities')
    if (not isinstance(logits, list) or len(logits) != count
            or not isinstance(probabilities, list) or len(probabilities) != count
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)):
        raise ValueError(f'缺少有效的原始候选 logits / probabilities：{prediction.get("id")}')
    weights = [math.exp(z-max(logits)) for z in logits]
    expected = [v/sum(weights) for v in weights]
    if any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6) for p, q in zip(expected, probabilities)):
        raise ValueError(f'候选 logits 与 softmax 概率不一致：{prediction.get("id")}')
    return logits


def predict_features_with_logits(model, processor, features, record, questions, context):
    # Use the unchanged model forward API. Older downloaded checkpoints need no
    # file replacement or re-export to expose their existing output.logits.
    import torch
    from videojev_hf.processing_videojev import prepare_query_groups, format_predictions

    predictions = []
    with torch.inference_mode():
        for batch, rows in prepare_query_groups(processor, record, questions, context):
            output = model(**batch, visual_features=features)
            values = format_predictions(output, rows, record, dict(model.config.decoder), return_logits=True)
            for row, value in zip(rows, values):
                validate_logits(value, len(row['options']))
            predictions.extend(values)
    return predictions
