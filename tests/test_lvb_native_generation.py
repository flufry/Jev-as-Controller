import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig

from timejev.lvb_native_generation import (
    NativeExperiment, PresencePenalty, audit_generated_tokens, extract_answer, generation_parameters,
    question_seed, read_generated, score_generated,
)


OPTIONS = ['A. red shirt', 'B. blue shirt', 'C. green shirt']


@pytest.mark.parametrize('response', ['B', ' B\n', '(B)', 'Answer: B',
    'The correct answer is B.', 'B. blue shirt', '{"answer": "B"}',
    'The clip shows a person in blue.\nAnswer: B'])
def test_parse_generated_answer(response):
    parsed = extract_answer(response, OPTIONS)
    assert parsed['valid'] is True
    assert parsed['answer_index'] == 1


@pytest.mark.parametrize('response', ['', 'A or B', 'Answer: A or B',
    'Answer: A\nAnswer: B', 'D', 'Answer: D', 'I cannot determine the answer.',
    'A. red shirt\nB. blue shirt', '<think>unfinished\nAnswer: B',
    'Answer: because', 'The answer is cannot determine',
    'The answer is a blue shirt.', 'Answer: a person in blue.',
    'The answer is a soldier saluting.', 'Answer: a or b', 'Answer: b or c',
    'Answer: A or b', 'Final answer: B or c'])
def test_invalid_or_ambiguous_answers_are_not_forced_to_an_option(response):
    parsed = extract_answer(response, OPTIONS)
    assert parsed['valid'] is False
    assert parsed['answer_index'] is None
    assert parsed['reason']


def test_truncated_answer_is_invalid_even_when_it_contains_a_letter():
    assert extract_answer('B', OPTIONS, truncated=True)['valid'] is False


def test_generated_tokens_and_both_decodes_must_match():
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            return 'B' if skip_special_tokens else 'B<|im_end|>'

    prediction = dict(generated_token_ids=[33, 248046], generated_tokens=2,
        response='B', raw_response='B<|im_end|>')
    audit_generated_tokens(prediction, Tokenizer())
    for key, value in [('response', 'A'), ('raw_response', 'A<|im_end|>'),
                       ('generated_tokens', 1), ('generated_token_ids', [True, 248046])]:
        changed = dict(prediction, **{key: value})
        with pytest.raises(ValueError):
            audit_generated_tokens(changed, Tokenizer())


def test_presence_penalty_only_applies_once_to_generated_tokens():
    # 前两个token是prompt；第一行生成过2两次，第二行还生成过3。
    tokens = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 3]])
    scores = torch.zeros((2, 5))
    actual = PresencePenalty(prompt_length=2, penalty=1.5)(tokens, scores)
    assert actual.tolist() == [[0., 0., -1.5, 0., 0.], [0., 0., -1.5, -1.5, 0.]]


def test_presence_penalty_does_not_penalize_prompt_at_first_decode():
    scores = torch.tensor([[.1, .2, .3]])
    actual = PresencePenalty(prompt_length=2, penalty=1.5)(torch.tensor([[0, 1]]), scores)
    assert actual.tolist() == scores.tolist()


def test_official_instruct_sampling_parameters_are_explicit():
    values = generation_parameters()
    assert {key: values[key] for key in ('do_sample', 'temperature', 'top_p', 'top_k',
        'min_p', 'repetition_penalty', 'max_new_tokens')} == dict(do_sample=True,
        temperature=.7, top_p=.8, top_k=20, min_p=0, repetition_penalty=1,
        max_new_tokens=32768)
    assert 'presence_penalty' not in values
    config = GenerationConfig(**values)
    assert config.max_new_tokens > 1
    assert config.do_sample is True
    assert generation_parameters(128)['max_new_tokens'] == 128


def test_presence_penalty_precedes_sampling_filters():
    # 验证真实Transformers处理器组合顺序，不能把惩罚值当成被忽略的kwargs。
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.logits_process import LogitsProcessorList

    class Model(GenerationMixin):
        device = torch.device('cpu')

    config = GenerationConfig(**generation_parameters())
    config._eos_token_tensor = None
    processors = Model()._get_logits_processor(config, input_ids_seq_length=2,
        encoder_input_ids=None, prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList([PresencePenalty(2)]), device='cpu', model_kwargs={})
    names = [type(processor).__name__ for processor in processors]
    assert names.index('PresencePenalty') < names.index('TemperatureLogitsWarper')
    assert names.index('PresencePenalty') < names.index('TopKLogitsWarper')
    assert names.index('PresencePenalty') < names.index('TopPLogitsWarper')


def test_per_question_seed_is_stable_and_resume_order_independent():
    qids = ['longvideobench/video_0', 'longvideobench/video_1']
    original = {qid: question_seed(42, qid) for qid in qids}
    resumed = {qid: question_seed(42, qid) for qid in reversed(qids)}
    assert original == resumed
    assert len(set(original.values())) == 2
    assert all(type(seed) is int and 0 <= seed < 2**63 for seed in original.values())
    assert question_seed(43, qids[0]) != question_seed(42, qids[0])


@pytest.fixture
def generated():
    rows = [dict(id=f'q{i}', task='choice', query=f'question {i}', options=OPTIONS,
        video_key=f'video{i}', answer_index=i, duration_group=group)
        for i, group in enumerate((15, 60, 3600))]
    by_id = {}
    for row, response, truncated in zip(rows, ('A', 'A', 'Answer: C'), (False, False, True)):
        parsed = extract_answer(response, row['options'], truncated=truncated)
        index = parsed['answer_index']
        by_id[row['id']] = dict(id=row['id'], query=row['query'], options=row['options'],
            response=response, truncated=truncated, parsed=parsed,
            answer_index=index, valid=parsed['valid'], invalid_reason=parsed['reason'] if not parsed['valid'] else None,
            answer=row['options'][index] if index is not None else None)
    return rows, by_id


def test_score_counts_invalid_answers_in_denominator(generated):
    metrics = score_generated(*generated)
    assert metrics['n'] == 3
    assert metrics['correct'] == 1
    assert metrics['invalid'] == 1
    assert metrics['accuracy'] == pytest.approx(1/3)
    assert metrics['by_duration_group']['3600']['accuracy'] == 0


def test_labels_only_affect_scoring(generated):
    rows, by_id = generated
    changed = copy.deepcopy(rows)
    changed[0]['answer_index'] = 1
    changed[1]['answer_index'] = 0
    assert score_generated(rows, by_id)['accuracy'] == score_generated(changed, by_id)['accuracy']
    assert by_id['q0']['answer_index'] == 0
    assert by_id['q1']['answer_index'] == 0
    assert by_id['q2']['answer_index'] is None


@pytest.mark.parametrize('kind', ['missing', 'extra', 'options', 'query', 'forced_answer'])
def test_score_rejects_mismatched_or_fabricated_answers(generated, kind):
    rows, by_id = generated
    if kind == 'missing':
        by_id.pop('q0')
    elif kind == 'extra':
        by_id['other'] = by_id['q0']
    elif kind == 'options':
        by_id['q0']['options'] = list(reversed(OPTIONS))
    elif kind == 'query':
        by_id['q0']['query'] = 'another question'
    else:
        by_id['q2'].update(answer_index=2, valid=True, answer=OPTIONS[2])
    with pytest.raises(ValueError):
        score_generated(rows, by_id)


@pytest.mark.parametrize('maximum', [0, -1, True, math.inf])
def test_invalid_output_limits_rejected(maximum):
    with pytest.raises(ValueError):
        generation_parameters(maximum)


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    from timejev.user_data import write_jsonl

    video = tmp_path/'video.mp4'
    video.write_bytes(b'local test input')
    rows = [dict(id=f'q{i}', task='choice', query=f'question {i}', options=OPTIONS,
        video_key='video', video_path='video.mp4', duration=10., answer_index=i,
        duration_group=group) for i, group in enumerate((15, 60, 3600))]
    manifest = tmp_path/'sample.jsonl'
    write_jsonl(rows, manifest)
    args = SimpleNamespace(manifest=str(manifest), out=str(tmp_path/'native'),
        checkpoint=str(tmp_path/'checkpoint'), seed=42, sample_seed=42, limit=None,
        context=262144, visual_token_budget=24576, fps=4., max_frames=4096,
        memory_fraction=.9, resume=False)
    monkeypatch.setattr('timejev.lvb_native_generation.checkpoint_identity',
        lambda path: dict(path=str(path), files={'config.json': {'sha256': 'mock', 'bytes': 1}}))
    return args, dict(decoding=generation_parameters()), rows


def test_native_run_persists_and_resumes_real_sampled_answers(native_run):
    args, settings, rows = native_run
    run = NativeExperiment(args, settings)
    run.append_response(rows[0], 'A', [32, 248046], raw_response='A<|im_end|>')
    assert run.done == {'q0'}
    with pytest.raises(FileExistsError):
        NativeExperiment(args, settings)
    args.resume = True
    resumed = NativeExperiment(args, settings)
    assert resumed.done == {'q0'}
    resumed.append_response(rows[1], 'B', [33, 248046])
    resumed.append_response(rows[2], 'I cannot determine.', [123, 248046])
    metrics = resumed.finish()
    assert metrics['accuracy'] == pytest.approx(2/3)
    assert metrics['invalid'] == 1
    assert len(read_generated(resumed.predictions_path, resumed.rows)) == 3
    saved = json.loads((resumed.out/'run.json').read_text())
    assert saved['status'] == 'completed'
    assert saved['identity']['selected_ids'] == ['q0', 'q1', 'q2']


@pytest.mark.parametrize('field', ['seed', 'max_new_tokens', 'limit', 'video_content'])
def test_resume_rejects_different_sampling_or_inputs(native_run, field):
    args, settings, _ = native_run
    NativeExperiment(args, settings)
    args.resume = True
    if field == 'seed':
        args.seed = 43
    elif field == 'max_new_tokens':
        settings['decoding'] = generation_parameters(128)
    elif field == 'limit':
        args.limit = 1
    else:
        from pathlib import Path
        Path(args.manifest).with_name('video.mp4').write_bytes(b'different input')
    with pytest.raises(ValueError, match='指纹'):
        NativeExperiment(args, settings)


def test_read_generated_rejects_duplicate_answers(native_run):
    args, settings, rows = native_run
    run = NativeExperiment(args, settings)
    run.append_response(rows[0], 'A', [32, 248046])
    first = run.predictions_path.read_text()
    run.predictions_path.write_text(first+first)
    with pytest.raises(ValueError, match='重复'):
        read_generated(run.predictions_path, run.rows, complete=False)


def test_evaluator_saves_generated_response_instead_of_candidate_logits(native_run, monkeypatch):
    from pathlib import Path
    import transformers
    from scripts import eval_longvideobench_qwen36_native as evaluator

    args, _, rows = native_run
    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir()
    for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json'):
        (checkpoint/name).write_text('{}')
    (checkpoint/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model.safetensors'}}))
    (checkpoint/'model.safetensors').write_bytes(b'only a test fixture')
    model_config = SimpleNamespace(model_type='qwen3_5_moe',
        architectures=['Qwen3_5MoeForConditionalGeneration'], tie_word_embeddings=False,
        text_config=SimpleNamespace(num_hidden_layers=40, num_experts=256,
            num_experts_per_tok=8, tie_word_embeddings=False))
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw: model_config)

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            assert tokens == [999, 33, 248046]
            return ' B\n' if skip_special_tokens else ' B<|im_end|>'

    processor = SimpleNamespace(tokenizer=Tokenizer())
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *a, **kw: processor)
    calls = []

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[248046, 248044])

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is True and kwargs['max_new_tokens'] == 32768
            assert kwargs['temperature'] == .7 and kwargs['top_p'] == .8
            assert kwargs['output_logits'] is False and kwargs['output_scores'] is False
            assert kwargs['logits_processor'][0].prompt_length == 2
            assert 'candidate_token_ids' not in kwargs
            calls.append(kwargs)
            return SimpleNamespace(sequences=torch.tensor([[9, 10, 999, 33, 248046]]))

    monkeypatch.setattr(evaluator, 'read_sample', lambda *a, **kw: (rows, {'seed': 42}))
    monkeypatch.setattr(evaluator, 'load_backend', lambda: (object(), {'mock': True}))
    monkeypatch.setattr(evaluator, 'load_model', lambda *a: (Model(), {'mock': True}))
    monkeypatch.setattr(evaluator, 'prepare_record', lambda *a, **kw: {'mock': True})
    monkeypatch.setattr(evaluator, 'record_audit', lambda *a: {'mock': True})
    monkeypatch.setattr(evaluator, 'preprocess', lambda *a: [torch.zeros((1, 2))])
    monkeypatch.setattr(evaluator, 'text_inputs', lambda *a: {'input_ids': torch.tensor([[9, 10]])})
    monkeypatch.setattr(evaluator, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 6)
    for name in ('set_per_process_memory_fraction', 'reset_peak_memory_stats', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    original_to = torch.Tensor.to

    def cpu_test_to(self, *a, **kw):
        return self if a and str(a[0]).startswith('cuda') else original_to(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, 'to', cpu_test_to)
    metrics = evaluator.main(['--manifest', args.manifest, '--out', args.out,
        '--checkpoint', args.checkpoint, '--seed', '42'])
    assert len(calls) == 3
    assert metrics['accuracy'] == pytest.approx(1/3)
    saved = [json.loads(line) for line in (Path(args.out)/'predictions.jsonl').read_text().splitlines()]
    assert all(row['answer_index'] == 1 and row['valid'] for row in saved)
    assert all(row['generated_token_ids'] == [999, 33, 248046] for row in saved)
    assert all('candidate_logits' not in row for row in saved)
