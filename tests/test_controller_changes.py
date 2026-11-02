import json

import pytest

from scripts.analyze_controller_changes import analyze, export, main
from scripts.bench_compat import generic_score
from scripts.bench_io import json_bytes, jsonl_bytes, sha256
from scripts.eval_bench_controller import evaluate
from scripts.eval_longvideobench_controller import softmax


@pytest.fixture
def controller_runs(tmp_path):
    # Agreement, rescue, harm, wrong-to-wrong, third-choice rescue, unchanged disagreement.
    cases = [([3., 0., 0.], [.8, .1, .1], 0),
        ([.3, 0., -2.], [0., 1., 0.], 1),
        ([.3, 0., -2.], [0., 1., 0.], 0),
        ([.3, 0., -2.], [0., 1., 0.], 2),
        ([1., 0., .9], softmax([0., 2., 1.8]), 2),
        ([4., 0., 0.], [0., 1., 0.], 0)]
    rows = [dict(id=f'videomme/{i}', task='choice', query='Which | answer?',
        options=['A. one', 'B. two', 'C. three'], video_key=f'v{i}',
        video_path='/not/local/video.mp4', duration=60., benchmark='videomme', stratum='short',
        answer_index=gold) for i, (_, _, gold) in enumerate(cases)]
    manifest = tmp_path/'sample_42.jsonl'
    manifest.write_bytes(jsonl_bytes(rows))
    protocol = dict(name='video_bench_screen_rgb_v1', fps=2., max_frames=2048,
        context=262144, visual_token_budget=24576, use_subtitles=False, use_audio=False)
    for name, method in (('qwen_logits', 'qwen36'), ('videojev', 'videojev')):
        directory = tmp_path/name
        directory.mkdir()
        predictions = []
        for row, (z, p, _) in zip(rows, cases):
            values = softmax(z) if method == 'qwen36' else p
            pred = dict(row, probabilities=values, answer_index=max(range(3), key=values.__getitem__),
                preprocessing=dict(frames=120, tokens=100))
            if method == 'qwen36':
                pred['candidate_logits'] = z
            predictions.append(pred)
        (directory/'predictions.jsonl').write_bytes(jsonl_bytes(predictions))
        (directory/'run.json').write_bytes(json_bytes(dict(identity=dict(method=method,
            manifest_sha256=sha256(manifest), protocol=protocol, seed=42))))
        (directory/'metrics.json').write_bytes(json_bytes(dict(
            generic_score(rows, {p['id']: p for p in predictions}), method=method, protocol=protocol)))
    # Controller creation checks real video paths, whereas post-hoc analysis must not.
    from unittest.mock import patch

    def offline_load(path):
        return [dict(r, duration_group=r['stratum']) for r in rows]

    with patch('scripts.bench_compat.generic_load_sample', offline_load):
        evaluate(manifest, tmp_path/'qwen_logits', tmp_path/'videojev', tmp_path/'controller')
    return dict(manifest=manifest, qwen=tmp_path/'qwen_logits',
        videojev=tmp_path/'videojev', controller=tmp_path/'controller')


def test_exports_all_changes_and_exact_formula_without_media(controller_runs, tmp_path):
    records, _ = analyze(**controller_runs)
    assert [r['change_type'] for r in records] == [
        'unchanged', 'rescued', 'harmed', 'changed_still_wrong', 'rescued', 'unchanged']
    assert records[4]['third_choice'] is True
    assert records[5]['disagreement'] and not records[5]['changed_vs_qwen']
    for row in records:
        scores = [p['fused_score'] for p in row['per_option']]
        assert softmax(scores) == pytest.approx([p['controller_probability'] for p in row['per_option']])
        for p in row['per_option']:
            assert p['correction'] == 2*p['videojev_probability']
            assert p['fused_score'] == p['qwen_logit']+p['correction']
    out = tmp_path/'analysis'
    result = export([controller_runs], out)
    assert result['counts']['disagreements'] == 5
    assert result['counts']['changed'] == 4
    assert result['counts']['rescued'] == 2
    assert result['counts']['harmed'] == 1
    assert result['counts']['changed_still_wrong'] == 1
    assert result['counts']['net_correct_gain_vs_qwen'] == 1
    assert result['counts']['controller_correct']-result['counts']['qwen_correct'] == 1
    for name, n in [('all', 6), ('rescued', 2), ('harmed', 1), ('disagreements', 5)]:
        assert len((out/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()) == n
    assert 'Which &#124; answer?' in (out/'changed.md').read_text(encoding='utf-8')
    assert json.loads((out/'summary.json').read_text(encoding='utf-8'))['counts'] == result['counts']
    with pytest.raises(FileExistsError):
        export([controller_runs], out)


def test_copied_run_directories_are_supported(controller_runs):
    path = controller_runs['controller']/'run.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    for source in meta['identity']['sources'].values():
        source['directory'] = '/old/server/location'
    path.write_bytes(json_bytes(meta))
    assert len(analyze(**controller_runs)[0]) == 6


@pytest.mark.parametrize('damage', ['source_hash', 'formula', 'embedded', 'duplicate', 'manifest'])
def test_rejects_modified_or_mismatched_inputs(controller_runs, tmp_path, damage):
    if damage == 'source_hash':
        path = controller_runs['qwen']/'run.json'
        path.write_bytes(path.read_bytes()+b'\n')
    elif damage in ('formula', 'embedded', 'duplicate'):
        path = controller_runs['controller']/'predictions.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        if damage == 'formula':
            rows[0]['probabilities'][0] -= .001
            rows[0]['probabilities'][1] += .001
        elif damage == 'embedded':
            rows[0]['qwen_probabilities'][0] = 0.
        else:
            rows.append(rows[0])
        path.write_bytes(jsonl_bytes(rows))
    else:
        path = controller_runs['manifest']
        path.write_bytes(path.read_bytes()+b'\n')
    out = tmp_path/'bad_analysis'
    with pytest.raises(ValueError):
        export([controller_runs], out)
    assert not out.exists()


def test_batch_cli_and_duplicate_seed_rejection(controller_runs, tmp_path):
    result = main(['--runs-root', str(tmp_path), '--seeds', '42', '--out', str(tmp_path/'batch')])
    assert result['counts']['all'] == 6
    with pytest.raises(ValueError, match='重复'):
        export([controller_runs, controller_runs], tmp_path/'duplicate')
