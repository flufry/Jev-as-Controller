"""Compatibility alias for the unified upstream-aligned Controller evaluator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_bench_controller import control as _control, evaluate as _evaluate, main, softmax


def control(qwen, videojev, beta=2., *, score_source='raw_logits'):
    return _control(qwen, videojev, beta, fusion='loglinear', score_source=score_source)


def evaluate(manifest, qwen, videojev, out, beta=2., score_source='raw_logits'):
    return _evaluate(manifest, qwen, videojev, out, beta, fusion='loglinear', score_source=score_source)


if __name__ == '__main__':
    main()
