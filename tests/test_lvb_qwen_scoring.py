import math

import pytest
import torch

from scripts.eval_longvideobench_qwen import candidate_scores, candidate_token_ids, text_inputs


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(text) - ord('A') + 32]

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return chr(ids[0] - 32 + ord('A'))


def test_bare_letters_and_candidate_order():
    ids = candidate_token_ids(Tokenizer(), 5)
    assert ids == [32, 33, 34, 35, 36]
    logits = torch.full((500,), -10.)
    logits[ids] = torch.tensor([21.875, 20.75, 22.125, 22.25, 25.625])
    # 首个生成 token 即便是空格字母，也不改变事先固定的候选评分范围。
    logits[458] = 30.
    assert int(logits.argmax()) == 458
    selected, probabilities = candidate_scores(logits, ids)
    assert selected == [21.875, 20.75, 22.125, 22.25, 25.625]
    expected = [math.exp(value - 25.625) for value in selected]
    expected = [value / sum(expected) for value in expected]
    assert probabilities == pytest.approx(expected)
    assert sum(probabilities) == pytest.approx(1.)
    assert max(range(5), key=probabilities.__getitem__) == 4
    assert candidate_scores(logits + 100, ids)[1] == pytest.approx(probabilities)


@pytest.mark.parametrize('mode', ['multiple', 'space', 'duplicate'])
def test_candidate_token_validation(mode):
    class Invalid(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            if mode == 'multiple':
                return [198, 36]
            return [32] if mode == 'duplicate' else super().encode(text, add_special_tokens)

        def decode(self, ids, skip_special_tokens=False):
            if mode == 'space':
                return ' ' + super().decode(ids, skip_special_tokens)
            return super().decode(ids, skip_special_tokens)

    with pytest.raises(ValueError):
        candidate_token_ids(Invalid(), 5)


def test_nonfinite_logits_rejected():
    with pytest.raises(ValueError, match='非有限'):
        candidate_scores(torch.tensor([0., float('nan')]), [0, 1])


class Processor:
    video_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert 'answer_index' not in str(messages)
        assert messages[0]['content'][0] == dict(type='video', video='provided_in_memory')
        assert messages[0]['content'][1]['text'].endswith('Reply with only the letter of the correct option.')
        return '<video>Question'

    def replace_video_token(self, values, video_idx):
        assert video_idx == 0
        assert values['video_grid_thw'].tolist() == [[2, 2, 2]]
        meta = values['video_metadata'][0]
        assert meta.frames_indices == [0, 1, 2]
        assert meta.fps == 0.5
        assert meta.total_num_frames == 3
        return 'replacement'

    def get_text_with_replacements(self, texts, images, videos, audio):
        assert texts == ['<video>Question']
        assert not images and not audio and videos == ['replacement']
        return ['expanded text'], []

    def __call__(self, text, return_tensors):
        assert text == ['expanded text'] and return_tensors == 'pt'
        return dict(input_ids=torch.tensor([[1, 999, 999, 2]]), attention_mask=torch.ones((1, 4)))


def test_shared_geometry_timeline_and_context():
    row = dict(query='Question?', options=['A. Yes', 'B. No'], answer_index=1)
    record = dict(frames=3, effective_fps=.5, grid_thw=[2, 2, 2], tokens=2)
    inputs = text_inputs(Processor(), row, record, 5)
    assert inputs['video_grid_thw'].tolist() == [[2, 2, 2]]
    assert 'labels' not in inputs and 'answer_index' not in inputs
    with pytest.raises(ValueError, match='上下文'):
        text_inputs(Processor(), row, record, 4)
    with pytest.raises(ValueError, match='占位'):
        text_inputs(Processor(), row, dict(record, tokens=3), 5)
