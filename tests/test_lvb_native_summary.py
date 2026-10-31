import json

import pytest

from scripts import summarize_longvideobench_native_baseline as summary
from scripts.sample_longvideobench_fixed_questions import METHOD_ROLES, simulate
from timejev.lvb_experiment import sha256, write_json
from timejev.lvb_native_generation import extract_answer, generation_parameters, question_seed, score_generated
from timejev.user_data import write_jsonl


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    rows = [dict(id=f'q{i}', task='choice', query=f'question {i}',
        options=['A. one', 'B. two', 'C. three'], video_key='video',
        duration_group=(15, 60, 600, 3600)[i % 4], answer_index=i % 3)
        for i in range(100)]
    manifest = tmp_path/'sample.jsonl'
    write_jsonl(rows, manifest)
    sample = dict(path=str(manifest), sha256=sha256(manifest), seed=42)
    protocol = dict(fps=4., max_frames=4096, context=262144, visual_token_budget=24576)
    checkpoint = dict(path='models/Qwen3.6-35B-A3B', files={'config.json': {'sha256': 'mock'}})

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            letter = chr(tokens[0]-32+65)
            return letter if skip_special_tokens else letter+'<|im_end|>'

    monkeypatch.setattr(summary.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    sources, source_predictions = {}, {}
    for method, role in METHOD_ROLES.items():
        directory = tmp_path/role
        directory.mkdir()
        predictions = {row['id']: dict(id=row['id'], query=row['query'], options=row['options'],
            probabilities=[.8, .1, .1], answer_index=0, answer=row['options'][0],
            preprocessing=dict(frames=4)) for row in rows}
        write_jsonl(list(predictions.values()), directory/'predictions.jsonl')
        sources[role] = dict(directory=str(directory), identity=dict(checkpoint=checkpoint))
        source_predictions[method] = predictions
    fixed, trials, predictions = simulate(rows, source_predictions)
    fixed.update(sample_seed=42, sample=sample, verified_sources={str(manifest): sha256(manifest)}, source_runs=sources)
    candidate_dir = tmp_path/'fixed_sampling'
    candidate_dir.mkdir()
    write_json(candidate_dir/'results.json', fixed)
    write_jsonl(trials, candidate_dir/'trials.jsonl')
    write_jsonl(predictions, candidate_dir/'predictions.jsonl')
    write_json(candidate_dir/'artifact_manifest.json', dict(files={name: sha256(candidate_dir/name)
        for name in ('results.json', 'trials.jsonl', 'predictions.jsonl')}))
    monkeypatch.setattr(summary, 'read_single', lambda *args: dict(rows=rows, sample=sample,
        report=dict(protocols=dict(baseline=protocol, controller=dict(protocol, fps=2., max_frames=2048)))))
    native = tmp_path/'native'
    for seed in (42, 43, 44):
        directory = native/f'seed{seed}'
        directory.mkdir(parents=True)
        settings = dict(decoding=generation_parameters(), thinking=False, presence_penalty=1.5,
            scoring='native_generation_then_option_parsing')
        identity = dict(method='qwen36_native_instruct', inference_seed=seed, sample_seed=42,
            manifest_sha256=sample['sha256'], limit=None, selected_ids=fixed['question_ids'],
            protocol=protocol, checkpoint=checkpoint, settings=settings)
        write_json(directory/'run.json', dict(identity=identity, status='completed'))
        generated = {}
        for row in rows:
            index = row['answer_index']
            response = chr(65+index)
            parsed = extract_answer(response, row['options'])
            generated[row['id']] = dict(id=row['id'], query=row['query'], options=row['options'],
                response=response, raw_response=response+'<|im_end|>', truncated=False,
                parsed=parsed, answer_index=index, valid=True,
                invalid_reason=None, answer=row['options'][index], generated_token_ids=[32+index, 248046],
                generated_tokens=2, question_seed=question_seed(seed, row['id']), preprocessing=dict(frames=4))
        write_jsonl(list(generated.values()), directory/'predictions.jsonl')
        metrics = score_generated(rows, generated)
        metrics.update(inference_seed=seed, method='qwen36_native_instruct', protocol=protocol)
        write_json(directory/'metrics.json', metrics)
    return native, candidate_dir, tmp_path/'summary', fixed


def test_summary_uses_native_answers_and_retains_other_sampling(experiment):
    native, candidate_dir, out, fixed = experiment
    result = summary.summarize(native, candidate_dir, out)
    assert result['methods']['qwen36_native_instruct']['accuracies'] == [1., 1., 1.]
    assert result['methods']['qwen36_native_instruct']['sample_variance_percentage_points_squared'] == 0
    assert result['qwen_invalid_answers'] == [0, 0, 0]
    for method in ('videojev', 'jev_as_controller'):
        assert result['methods'][method]['accuracies'] == fixed['methods'][method]['accuracies']
    assert (out/'results.json').is_file()
    assert 'Instruct生成' in (out/'results.md').read_text()
    with pytest.raises(FileExistsError):
        summary.summarize(native, candidate_dir, out)


@pytest.mark.parametrize('kind', ['incomplete', 'wrong_seed', 'direct_scoring', 'different_sampler', 'partial'])
def test_summary_rejects_mixed_or_incomplete_native_results(experiment, kind):
    native, candidate_dir, out, _ = experiment
    path = native/'seed43/run.json'
    value = json.loads(path.read_text())
    if kind == 'incomplete':
        value['status'] = 'running'
    elif kind == 'wrong_seed':
        value['identity']['inference_seed'] = 42
    elif kind == 'direct_scoring':
        value['identity']['settings']['scoring'] = 'first_decode_bare_letter_candidates'
    elif kind == 'different_sampler':
        value['identity']['settings']['decoding']['temperature'] = 1
    else:
        value['identity']['limit'] = 1
    write_json(path, value)
    with pytest.raises(ValueError):
        summary.summarize(native, candidate_dir, out)
    assert not out.exists()


def test_summary_rejects_changed_cached_predictions(experiment):
    native, candidate_dir, out, _ = experiment
    path = candidate_dir/'predictions.jsonl'
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    predictions[0]['answer_index'] = (predictions[0]['answer_index']+1) % 3
    write_jsonl(predictions, path)
    with pytest.raises(ValueError, match='SHA'):
        summary.summarize(native, candidate_dir, out)
    assert not out.exists()


def test_summary_rejects_tokens_that_do_not_decode_to_saved_answer(experiment):
    native, candidate_dir, out, _ = experiment
    path = native/'seed42/predictions.jsonl'
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    # 解析文本与选项保持不变，仅替换底层token，必须重decode后拒绝。
    predictions[0]['generated_token_ids'][0] = 33
    write_jsonl(predictions, path)
    with pytest.raises(ValueError):
        summary.summarize(native, candidate_dir, out)
    assert not out.exists()
