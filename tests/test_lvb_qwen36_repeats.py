import json
from pathlib import Path
import shutil

import pytest

from test_lvb_qwen36_summary import experiment, make_source
from scripts.eval_longvideobench_controller import evaluate
from scripts.sample_longvideobench import sample
from scripts.summarize_longvideobench_qwen36 import summarize as single_summary
from scripts.summarize_longvideobench_qwen36_repeats import difference_statistics, summarize
from timejev.lvb_experiment import sha256, write_json


def refresh(root, rebuild_controller=False):
    saved = json.loads((root/'summary/results.json').read_text())
    args = {role: Path(source['directory']) for role, source in saved['provenance']['runs'].items()}
    args.update(manifest=Path(saved['provenance']['sample']['path']), out=root/'summary', seed=saved['seed'])
    if rebuild_controller:
        shutil.rmtree(args['controller'])
        evaluate(args['manifest'], args['qwen_controller'], args['videojev_controller'],
                 args['controller'], qwen_method='qwen36')
    shutil.rmtree(args['out'])
    single_summary(**args)


@pytest.fixture
def runs(experiment, tmp_path):
    single_summary(**experiment)
    roots = [tmp_path]
    meta = json.loads(experiment['manifest'].with_suffix('.meta.json').read_text())
    for index, seed in enumerate((43, 44), 1):
        root = tmp_path/f'run{seed}'
        manifest = root/'sample.jsonl'
        sample(meta['source_manifest']['path'], meta['source_reference']['path'], manifest, seed=seed)
        args = dict(manifest=manifest, controller=root/'controller', out=root/'summary', seed=seed)
        for role, hits in dict(qwen_baseline=40+10*index, qwen_controller=50+5*index,
                               videojev_baseline=75-5*index, videojev_controller=72+2*index).items():
            identity = json.loads((experiment[role]/'run.json').read_text())['identity']
            args[role] = root/role
            make_source(args[role], manifest, Path(identity['checkpoint']['path']), identity['method'],
                        identity['protocol']['fps'], hits, seed=seed)
        evaluate(manifest, args['qwen_controller'], args['videojev_controller'], args['controller'], qwen_method='qwen36')
        single_summary(**args)
        roots.append(root)
    return roots


def test_three_repeats_statistics_gains_and_revalidation(runs, tmp_path):
    before = [sha256(root/'summary/results.json') for root in runs]
    result = summarize(runs[::-1], tmp_path/'repeat_summary')
    assert result['seeds'] == [42, 43, 44]
    qwen = result['methods']['qwen36']['overall']
    assert qwen['values'] == [.4, .5, .6]
    assert qwen['mean'] == pytest.approx(.5)
    assert qwen['sample_variance'] == pytest.approx(.01)
    assert qwen['population_variance'] == pytest.approx(.02/3)
    assert qwen['sample_std'] == pytest.approx(.1)
    controller = result['methods']['jev_as_controller']['overall']
    assert controller['values'] == [.72, .74, .76]
    assert controller['sample_variance'] == pytest.approx(.0004)
    gain = result['controller2_minus_qwen2']['overall']
    assert gain['values'] == pytest.approx([.22, .19, .16])
    assert gain['mean'] == pytest.approx(.19)
    assert gain['sample_variance'] == pytest.approx(.0009)
    assert gain['population_variance'] == pytest.approx(.0006)
    assert result['sample_variance_ddof'] == 1 and result['population_variance_ddof'] == 0
    assert set(result['methods']['videojev']['by_duration_group']) == {'15', '60', '600', '3600'}
    assert result['training_overlap_verified'] is False
    assert result['unique_questions'] < 300
    assert all(entry['shared_questions'] > 0 for entry in result['overlaps'])
    assert before == [sha256(root/'summary/results.json') for root in runs]
    assert json.loads((tmp_path/'repeat_summary/results.json').read_text()) == result
    markdown = (tmp_path/'repeat_summary/results.md').read_text()
    assert '50.00%' in markdown and '100.0000' in markdown
    assert '同协议融合增益' in markdown and 'ddof=1' in markdown
    with pytest.raises(FileExistsError):
        summarize(runs, tmp_path/'repeat_summary')


def test_negative_fusion_gain_is_valid():
    stats = difference_statistics([-.1, 0., .1])
    assert stats['mean'] == pytest.approx(0)
    assert stats['sample_variance'] == pytest.approx(.01)


@pytest.mark.parametrize('kind', ['forged_summary', 'missing_predictions', 'missing_metrics', 'duplicate',
                                 'model', 'settings', 'packages', 'code', 'media', 'source_annotations'])
def test_rejects_unverified_or_incompatible_repeats(runs, tmp_path, kind):
    root = runs[1]
    summary_path = root/'summary/results.json'
    saved = json.loads(summary_path.read_text())
    if kind == 'forged_summary':
        saved['methods']['qwen36']['accuracy'] = .99
        write_json(summary_path, saved)
    elif kind == 'duplicate':
        runs = [runs[0], runs[0], runs[2]]
    elif kind in ('missing_predictions', 'missing_metrics'):
        name = 'predictions.jsonl' if kind == 'missing_predictions' else 'metrics.json'
        (Path(saved['provenance']['runs']['qwen_baseline']['directory'])/name).unlink()
    elif kind == 'source_annotations':
        path = Path(saved['provenance']['sample']['path']).with_suffix('.meta.json')
        meta = json.loads(path.read_text())
        original = Path(meta['source_manifest']['path'])
        modified = original.with_name('modified_source_seed43.jsonl')
        modified.write_text(original.read_text()+'\n')
        meta['source_manifest'] = dict(path=str(modified), sha256=sha256(modified))
        write_json(path, meta)
        refresh(root)
    else:
        roles = ('qwen_baseline', 'qwen_controller', 'videojev_baseline', 'videojev_controller') if kind == 'media' else (
            'qwen_baseline', 'qwen_controller')
        for role in roles:
            path = Path(saved['provenance']['runs'][role]['directory'])/'run.json'
            run = json.loads(path.read_text())
            identity = run['identity']
            if kind == 'model':
                identity['checkpoint']['files']['weights.safetensors']['sha256'] = 'd'*64
            elif kind == 'media':
                identity['media'][next(iter(identity['media']))]['sha256'] = 'd'*64
            elif kind == 'code':
                identity['code_sha256']['mock.py'] = 'd'*64
            else:
                identity[kind]['changed'] = True
            write_json(path, run)
        refresh(root, rebuild_controller=True)
    with pytest.raises(ValueError):
        summarize(runs, tmp_path/'repeat_summary')
    assert not (tmp_path/'repeat_summary/results.json').exists()


def test_requires_three_runs(runs, tmp_path):
    with pytest.raises(ValueError):
        summarize(runs[:2], tmp_path/'repeat_summary')
