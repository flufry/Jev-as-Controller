"""Read the actual VideoJev head scores, including from older local HF checkpoints."""
import math


def softmax(values):
    weights = [math.exp(v-max(values)) for v in values]
    return [v/sum(weights) for v in weights]


def validate_logits(prediction, count):
    logits = prediction.get('logits', prediction.get('candidate_logits'))
    probabilities = prediction.get('probabilities')
    if (not isinstance(logits, list) or len(logits) != count
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
        raise ValueError(f'缺少有效的原始候选 logits：{prediction.get("id")}')
    if 'logits' in prediction and 'candidate_logits' in prediction and prediction['candidate_logits'] != logits:
        raise ValueError('logits 与 candidate_logits 不一致')
    if 'probabilities' in prediction:
        if (not isinstance(probabilities, list) or len(probabilities) != count
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                or any(not math.isclose(p, q, rel_tol=1e-5, abs_tol=1e-6)
                       for p, q in zip(softmax(logits), probabilities))):
            raise ValueError(f'候选 logits 与 softmax 概率不一致：{prediction.get("id")}')
    return logits


def candidate_probabilities(prediction):
    """Normalize raw scores only at the consumer; accept historical probability runs."""
    if 'logits' in prediction or 'candidate_logits' in prediction:
        logits = validate_logits(prediction, len(prediction['options']))
        return prediction['probabilities'] if 'probabilities' in prediction else softmax(logits)
    probabilities = prediction.get('probabilities')
    if (not isinstance(probabilities, list) or len(probabilities) != len(prediction['options'])
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-5)):
        raise ValueError('候选概率无效或未归一化')
    return probabilities


def predict_features_with_logits(model, processor, features, record, questions, context, *, logits_only=False):
    # Use the unchanged model forward API. Older downloaded checkpoints need no
    # file replacement or re-export to expose their existing output.logits.
    import torch
    from videojev_hf.processing_videojev import prepare_query_groups, format_predictions

    predictions = []
    with torch.inference_mode():
        for batch, rows in prepare_query_groups(processor, record, questions, context):
            output = model(**batch, visual_features=features)
            values = format_predictions(output, rows, record, dict(model.config.decoder),
                                        return_logits=True, logits_only=logits_only)
            for row, value in zip(rows, values):
                validate_logits(value, len(row['options']))
            predictions.extend(values)
    return predictions
