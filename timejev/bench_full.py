"""四个视频 benchmark 的全量清单、固定题集三 seed 采样及来源核验。"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics

from scripts.bench_io import json_bytes, jsonl_bytes, sha256, write_new

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ('longvideobench', 'videomme', 'mlvu', 'lvbench')
SEEDS = (42, 43, 44)
COUNTS = dict(longvideobench=1337, videomme=2700, mlvu=2174, lvbench=1549)
SPLITS = dict(longvideobench='validation', videomme='test', mlvu='dev_mcq', lvbench='test')
SCHEMA = 'video_bench_full_manifest_v1'
NAMES = dict(qwen_native='Qwen3.6-35B-A3B（自回归）', qwen_logits='Qwen3.6-35B-A3B（单步生成）',
    videojev='VideoJev-4B', controller_probability='Jev-as-Controller（旧概率融合）',
    controller_loglinear='Jev-as-Controller（log-linear）', baseline_or='Qwen3.6-35B-A3B（单步生成）or VideoJev-4B')
# 两版 Controller 使用同一均匀随机序列，沿用 latest 的比较规则。
RNG_METHODS = dict(qwen_logits='qwen36', videojev='videojev', controller_probability='jev_as_controller',
                   controller_loglinear='jev_as_controller')
BENCHMARK_NAMES = dict(longvideobench='LongVideoBench', videomme='Video-MME', mlvu='MLVU', lvbench='LVBench')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _official_rows(benchmark, reference):
    """只用官方标注构造题目身份；保留原始选项顺序和完整库存验证。"""
    from scripts.build_bench_manifest import read_annotation, mlvu_reference, validate_lvbench
    from timejev.qa_manifest import parse_choice
    reference = Path(reference).resolve()
    if benchmark == 'mlvu':
        official, files = mlvu_reference(reference)
    else:
        official, files = read_annotation(reference), [reference]
    expected = []
    if benchmark == 'longvideobench':
        from timejev.longvideobench_benchmark import validate_reference, formatted_options
        validate_reference(official)
        for r in official:
            expected.append(dict(id='longvideobench/'+r['id'], question_id=r['id'],
                video_key='longvideobench/'+r['video_id'], query=r['question'], options=formatted_options(r),
                answer_index=r['correct_choice'], duration=r['duration'], duration_group=r['duration_group'],
                stratum=str(r['duration_group']), **{k:r[k] for k in
                    ('level', 'question_category', 'topic_category') if k in r}))
    elif benchmark == 'videomme':
        from timejev.videomme_benchmark import validate_reference
        validate_reference(official)
        for r in official:
            expected.append(dict(id='videomme/'+str(r['question_id']), question_id=str(r['question_id']),
                video_key='videomme/'+r['videoID'], query=r['question'], options=r['options'],
                answer_index=ord(r['answer'])-65, stratum=r['duration'],
                **{k:r[k] for k in ('domain', 'sub_category', 'task_type')}))
    elif benchmark == 'mlvu':
        from timejev.mlvu_benchmark import formatted_options
        for r in official:
            expected.append(dict(id='mlvu/'+r['official_id'], question_id=r['official_id'],
                video_key='mlvu/'+Path(r['video']).stem, query=r['question'], options=formatted_options(r),
                answer_index=r['candidates'].index(r['answer']), stratum=r['task_type'], task_type=r['task_type']))
    elif benchmark == 'lvbench':
        validate_lvbench(official)
        for video in official:
            for qa in video['qa']:
                query, options, answer = parse_choice(qa['question'], qa['answer'])
                expected.append(dict(id='lvbench/'+str(qa['uid']), question_id=str(qa['uid']),
                    video_key='lvbench/'+video['key'], query=query, options=options, answer_index=answer,
                    question_type=qa['question_type'],
                    stratum=json.dumps(sorted(qa['question_type']), ensure_ascii=False, separators=(',', ':'))))
    else:
        raise ValueError('未知 benchmark')
    if len(expected) != COUNTS[benchmark] or len({r['id'] for r in expected}) != len(expected):
        raise ValueError('官方标注不完整或 ID 重复')
    return {r['id']:r for r in expected}, {str(p.resolve()):sha256(p) for p in files}


def _payload(benchmark, source_manifest, reference, out):
    source_manifest, reference, out = map(lambda p:Path(p).resolve(), (source_manifest, reference, out))
    before = sha256(source_manifest)
    official, reference_hashes = _official_rows(benchmark, reference)
    rows = read_jsonl(source_manifest)
    if (len(rows) != COUNTS[benchmark] or len({r['id'] for r in rows}) != len(rows)
            or {r['id'] for r in rows} != set(official)):
        raise ValueError('必须是完整官方题集，拒绝子集、重复题或未知 ID')
    result, media, owners = [], {}, {}
    for raw in rows:
        target = official[raw['id']]
        for key in ('id', 'video_key', 'query', 'options', 'answer_index'):
            if raw.get(key) != target[key]:
                raise ValueError(f'清单与官方题目/选项/标签不一致：{raw["id"]}/{key}')
        if raw.get('benchmark', benchmark) != benchmark or raw.get('task') != 'choice':
            raise ValueError('清单 benchmark/task 不一致')
        if (type(raw.get('answer_index')) is not int or any('<|' in s for s in [raw['query'], *raw['options']])
                or type(raw.get('duration')) not in (int, float) or not math.isfinite(raw['duration']) or raw['duration'] <= 0):
            raise ValueError('标签、题目或时长无效')
        if benchmark == 'longvideobench' and raw['duration'] != target['duration']:
            raise ValueError('LongVideoBench 时长与官方标注不一致')
        path = (source_manifest.parent/raw['video_path']).resolve()
        source = raw.get('media_source', dict(kind='video', path=raw['video_path']))
        if source.get('kind') != 'video' or (source_manifest.parent/source['path']).resolve() != path or not path.is_file():
            raise ValueError(f'需要完整原视频且路径一致：{path}')
        state = (str(path), raw['duration'])
        if media.setdefault(raw['video_key'], state) != state or owners.setdefault(str(path), raw['video_key']) != raw['video_key']:
            raise ValueError('视频分组、路径或时长不一致')
        member = dict(raw, **{k:v for k,v in target.items() if k not in raw or k in
            ('duration_group', 'stratum', 'question_type', 'question_id', 'task_type')})
        member.update(benchmark=benchmark, duration_group=target.get('duration_group', target['stratum']),
            video_path=os.path.relpath(path, out.parent),
            media_source=dict(kind='video', path=os.path.relpath(path, out.parent)))
        result.append(member)
    files = {str(source_manifest):before, **reference_hashes}
    verify_files(files)
    data = jsonl_bytes(result)
    import hashlib
    metadata = dict(schema=SCHEMA, benchmark=benchmark, split=SPLITS[benchmark], selection_seed=None,
        sampling='full_dataset_no_selection', n=len(rows), videos=len(media),
        manifest_sha256=hashlib.sha256(data).hexdigest(), selected_ids=[r['id'] for r in rows],
        strata=dict(Counter(r['stratum'] for r in result)), source_manifest=str(source_manifest),
        reference=str(reference), verified_inputs=files)
    return data, metadata


def verify_files(files):
    for path, digest in files.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError(f'文件 SHA 不匹配或文件已变化：{path}')


def prepare(benchmark, source_manifest, reference, out):
    if benchmark not in BENCHMARKS:
        raise ValueError('未知 benchmark')
    out = Path(out).resolve()
    if out.exists() or out.with_suffix('.meta.json').exists():
        raise FileExistsError('拒绝覆盖已有全量清单')
    data, meta = _payload(benchmark, source_manifest, reference, out)
    write_new(out, data)
    write_new(out.with_suffix('.meta.json'), json_bytes(meta))
    return meta


def read_full(manifest):
    manifest = Path(manifest).resolve()
    meta = read_json(manifest.with_suffix('.meta.json'))
    bench = meta.get('benchmark')
    if (meta.get('schema') != SCHEMA or bench not in BENCHMARKS
            or meta.get('selection_seed', 'missing') is not None or meta.get('n') != COUNTS[bench]):
        raise ValueError('需要已核验的全量固定题集，不能使用抽样清单')
    verify_files(meta['verified_inputs'])
    data, expected = _payload(bench, meta['source_manifest'], meta['reference'], manifest)
    if meta != expected or manifest.read_bytes() != data:
        raise ValueError('全量清单、顺序或元数据与官方来源不一致')
    rows = read_jsonl(manifest)
    for row in rows:
        row['video_path'] = str((manifest.parent/row['video_path']).resolve())
        row['media_source'] = dict(kind='video', path=row['video_path'])
    return rows, dict(meta, path=str(manifest), meta_sha256=sha256(manifest.with_suffix('.meta.json')))


def score_answers(rows, predictions, benchmark):
    """允许采样答案不等于 argmax；native 无效/截断计错且保留分母。"""
    ids = {r['id'] for r in rows}
    if not rows or len(ids) != len(rows) or set(predictions) != ids or benchmark not in BENCHMARKS:
        raise ValueError('计分必须完整覆盖非空互异题集')
    totals, hits, cats, cat_hits = Counter(), Counter(), Counter(), Counter()
    invalid = 0
    for row in rows:
        p = predictions[row['id']]
        if any(p.get(k) != row[k] for k in ('id', 'query', 'options')):
            raise ValueError('预测题目或选项不匹配')
        answer = p.get('answer_index')
        if answer is not None and (type(answer) is not int or not 0 <= answer < len(row['options'])):
            raise ValueError('预测答案索引无效')
        valid = answer is not None and p.get('valid', True) and not p.get('truncated', False)
        hit = bool(valid and answer == row['answer_index'])
        group = str(row['stratum'])
        totals[group] += 1; hits[group] += hit; invalid += not valid
        for category in row.get('question_type', []):
            cats[category] += 1; cat_hits[category] += hit
    groups = {g:dict(n=totals[g], correct=hits[g], accuracy=hits[g]/totals[g]) for g in sorted(totals)}
    result = dict(benchmark=benchmark, split=SPLITS[benchmark], n=len(rows), videos=len({r['video_key'] for r in rows}),
        correct=sum(hits.values()), invalid=invalid, accuracy=sum(hits.values())/len(rows),
        primary_metric='accuracy', by_stratum=groups, selection_seed=None)
    if benchmark == 'mlvu':
        from timejev.mlvu_benchmark import TASK_COUNTS
        if set(groups) != set(TASK_COUNTS):
            raise ValueError('MLVU M-Avg 必须覆盖七个任务')
        result.update(m_avg=statistics.mean(g['accuracy'] for g in groups.values()), primary_metric='m_avg', by_task=groups)
    if benchmark == 'longvideobench':
        result['by_duration_group'] = groups
    if cats:
        result['by_question_type'] = {g:dict(n=cats[g], correct=cat_hits[g], accuracy=cat_hits[g]/cats[g]) for g in sorted(cats)}
    return result


def code_files():
    names = ['timejev/bench_input_cache.py', 'scripts/run_bench_cached_source.py', 'timejev/bench_full.py', 'scripts/eval_bench_full_source.py', 'scripts/run_bench_full.py',
        'scripts/summarize_bench_full.py', 'scripts/prepare_bench_full.py', 'scripts/bench_io.py',
        'scripts/build_bench_manifest.py', 'scripts/bench_compat.py',
        'scripts/eval_longvideobench_qwen.py', 'scripts/eval_longvideobench_qwen36.py',
        'scripts/eval_longvideobench_videojev.py', 'scripts/eval_bench_qwen36_native.py',
        'scripts/eval_longvideobench_qwen36_native.py', 'scripts/eval_longvideobench_controller.py',
        'scripts/videojev_candidate_logits.py', 'timejev/lvb_experiment.py', 'timejev/lvb_native_generation.py',
        'timejev/rgb_media.py', 'timejev/rgb_data.py', 'timejev/video.py', 'timejev/temporal.py',
        'timejev/user_data.py', 'timejev/multiturn_inputs.py', 'timejev/decision_inputs.py',
        'videojev_hf/processing_videojev.py', 'videojev_hf/modeling_videojev.py',
        'timejev/longvideobench_benchmark.py', 'timejev/mlvu_benchmark.py', 'timejev/videomme_benchmark.py',
        'timejev/qa_manifest.py']
    return {name:sha256(ROOT/name) for name in names}


def snapshot(out, hashes):
    for name, digest in hashes.items():
        path = ROOT/name
        if sha256(path) != digest:
            raise ValueError('运行期间源码发生变化')
        target = Path(out)/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


@contextmanager
def install_sources(rows, full):
    """仅隔离导入的模型入口使用；保留原前向，替换全量输入和记分。"""
    from timejev import lvb_experiment as lvb
    from scripts.bench_compat import validate_prediction
    names = ('load_sample', 'protocol', 'score', 'Experiment', 'write_json')
    saved = {name:getattr(lvb, name) for name in names}
    hashes = code_files()
    def load(path):
        if Path(path).resolve() != Path(full['path']):
            raise ValueError('模型入口清单不一致')
        return rows
    def protocol(args):
        return dict(saved['protocol'](args), name='video_bench_full_rgb_v1')
    def score(targets, by_id):
        for row in targets:
            validate_prediction(row, by_id[row['id']])
        return score_answers(targets, by_id, full['benchmark'])
    def write(path, value):
        if Path(path).name == 'run.json':
            value = dict(value, schema='video_bench_full_source_run_v1', split=full['split'],
                selection_seed=None, sampling='full_dataset_no_selection')
        saved['write_json'](path, value)
    class Experiment(saved['Experiment']):
        def __init__(self, args, method, settings):
            super().__init__(args, method, dict(settings, full_dataset=full, full_code_sha256=hashes))
            snapshot(self.out, hashes)
    lvb.load_sample, lvb.protocol, lvb.score, lvb.Experiment, lvb.write_json = load, protocol, score, Experiment, write
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(lvb, name, value)


def sample_index(probabilities, rng):
    if (not probabilities or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-5)):
        raise ValueError('无效候选概率')
    point, cumulative = rng.random()*sum(probabilities), 0.
    for index, value in enumerate(probabilities):
        cumulative += value
        if point < cumulative:
            return index
    return max(i for i,p in enumerate(probabilities) if p > 0)


def sample_predictions(rows, distributions, seed, method):
    """不接收正确标签参与选择；固定原题序，每题消耗一个随机数。"""
    rng = random.Random(f'{seed}:{RNG_METHODS[method]}')
    result = {}
    if set(distributions) != {r['id'] for r in rows}:
        raise ValueError('候选分布题集不完整')
    for row in rows:
        p = distributions[row['id']]
        if any(p.get(k) != row[k] for k in ('id', 'query', 'options')) or len(p['probabilities']) != len(row['options']):
            raise ValueError('候选分布与题目不匹配')
        index = sample_index(p['probabilities'], rng)
        result[row['id']] = dict(id=row['id'], query=row['query'], options=row['options'], video_key=row['video_key'],
            stratum=row['stratum'], method=method, inference_seed=seed, probabilities=p['probabilities'],
            answer_index=index, answer=row['options'][index])
    return result


def _read_predictions(path, rows):
    values = read_jsonl(path)
    by_id = {p['id']:p for p in values}
    if len(values) != len(by_id) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('结果缺题、多题或 ID 重复')
    return by_id


def _check_metrics(saved, actual):
    if any(saved.get(k) != v for k,v in actual.items()):
        raise ValueError('保存指标与完整预测重算不一致')


def read_source(directory, rows, full, role, *, expected_seed=None):
    from scripts.bench_compat import validate_prediction
    from timejev.lvb_native_generation import validate_generated, generation_parameters
    from scripts.videojev_candidate_logits import validate_logits
    directory = Path(directory).resolve()
    run = read_json(directory/'run.json'); identity = run['identity']
    native = role.startswith('native_seed')
    if expected_seed is None or expected_seed not in SEEDS:
        raise ValueError('必须明确指定该来源对应的seed42/43/44')
    method = 'qwen36_native_instruct' if native else 'qwen36' if role == 'qwen_controller' else 'videojev'
    dataset = identity.get('full_dataset', identity.get('settings', {}).get('full_dataset'))
    if identity.get('method') != method or identity.get('manifest_sha256') != full['manifest_sha256'] or dataset != full:
        raise ValueError(f'来源不是同一完整题集/方法：{role}')
    fps, cap = (4., 4096) if native or role == 'videojev_baseline' else (2., 2048)
    protocol = identity['protocol']
    if any(protocol.get(k) != v for k,v in dict(name='video_bench_full_rgb_v1', fps=fps, max_frames=cap,
            use_subtitles=False, use_audio=False).items()):
        raise ValueError(f'来源视频协议错误：{role}')
    settings = identity.get('settings', {})
    if native:
        if run.get('status') != 'completed' or identity.get('inference_seed') != expected_seed or int(role.removeprefix('native_seed')) != expected_seed:
            raise ValueError('native 未完成或推理 seed 错误')
        decoding = settings.get('decoding', {})
        if (settings.get('thinking') is not False
                or decoding != generation_parameters(decoding.get('max_new_tokens', 0))
                or settings.get('presence_penalty') != 1.5
                or settings.get('scoring') != 'native_generation_then_option_parsing'):
            raise ValueError('native 来源与固定采样生成协议不一致')
    elif identity.get('seed') != expected_seed:
        raise ValueError(f'来源seed与当前评测seed不一致：{role}/seed{expected_seed}')
    elif role == 'qwen_controller' and (settings.get('thinking') is not False or settings.get('decoding') != dict(do_sample=False,max_new_tokens=1)):
        raise ValueError('Qwen 来源必须为确定性首步 logits')
    hashes = dict(identity.get('code_sha256', {}))
    hashes.update(settings.get('full_code_sha256', {}))
    if not hashes:
        raise ValueError('来源缺少封存源码指纹')
    archived_files = {}
    for relative, digest in hashes.items():
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('非法来源源码路径')
        archived_files[str(directory/'source'/relative)] = digest
        verify_files({str(directory/'source'/relative):digest})
    predictions = _read_predictions(directory/'predictions.jsonl', rows)
    for row in rows:
        pred = predictions[row['id']]
        if native:
            validate_generated(row, pred)
        else:
            validate_prediction(row, pred)
            if role == 'qwen_controller':
                if pred.get('candidate_token_ids') is None:
                    raise ValueError('Qwen 首步候选 token 审计缺失')
            validate_logits(pred, len(row['options']))
        audit = pred.get('preprocessing', {})
        if (not audit or audit.get('fps') != fps or type(audit.get('frames')) is not int
                or not 0 < audit['frames'] <= cap or audit.get('tokens', math.inf) > protocol['visual_token_budget']):
            raise ValueError('逐题预处理与协议不一致')
        h,w = audit['processed_hw']; groups = (audit['frames']+1)//2
        if audit['grid_thw'] != [groups,h//16,w//16] or audit['tokens'] != groups*(h//32)*(w//32):
            raise ValueError('视觉 token/尺寸记录不一致')
    actual = score_answers(rows, predictions, full['benchmark'])
    _check_metrics(read_json(directory/'metrics.json'), actual)
    return predictions, dict(directory=str(directory), identity=identity, files={**archived_files, **{str(directory/f):sha256(directory/f)
        for f in ('run.json','metrics.json','predictions.jsonl')}}), actual


def candidate_directories(base):
    """读取共享来源，或显式读取旧per-seed实验的seed42；不猜测混合两套来源。"""
    base=Path(base).resolve()
    shared=base/'sources'
    repeated=base/'sources/seed42'
    roles=('qwen_controller','videojev_controller','videojev_baseline')
    has_shared=any((shared/role).exists() for role in roles)
    has_repeated=any((repeated/role).exists() for role in roles)
    if has_shared and has_repeated:
        raise ValueError('来源同时含共享与per-seed两种布局，不能自动选择')
    directory=repeated if has_repeated else shared
    return {role:directory/role for role in roles}


def summarize(manifest, base, out=None, *, seeds=SEEDS, source_base=None):
    """共用seed42候选来源、三次答案采样；OR与latest公开表一致，使用单步Qwen+4FPS VideoJev。"""
    from scripts.eval_longvideobench_controller import control
    from scripts.videojev_candidate_logits import candidate_probabilities, validate_logits
    rows, full = read_full(manifest); base = Path(base).resolve()
    if out is not None and Path(out).exists():
        raise FileExistsError('拒绝覆盖汇总目录；只读重验使用 --verify')
    seeds=tuple(seeds)
    if seeds not in (SEEDS, *((seed,) for seed in SEEDS)):
        raise ValueError('只支持完整42/43/44或单个seed汇总')
    sources, predictions, source_metrics = {}, {}, {}
    roles={role:(directory,role,42) for role,directory in candidate_directories(source_base or base).items()}
    for seed in seeds:
        roles[f'native_seed{seed}']=(base/f'native/seed{seed}',f'native_seed{seed}',seed)
    for key,(directory,role,seed) in roles.items():
        predictions[key], sources[key], source_metrics[key] = read_source(directory, rows, full, role, expected_seed=seed)
    two = sources['qwen_controller']['identity']['protocol']
    for key, src in sources.items():
        expected = dict(two, fps=4., max_frames=4096) if key.startswith(('native','videojev_baseline')) else two
        if src['identity']['protocol'] != expected or src['identity']['media'] != sources['qwen_controller']['identity']['media']:
            raise ValueError('来源输入协议或完整媒体指纹不一致')
    first=sources[f'native_seed{seeds[0]}']['identity']
    for seed in seeds[1:]:
        other=sources[f'native_seed{seed}']['identity']
        if any(first.get(k)!=other.get(k) for k in ('checkpoint','settings','packages','code_sha256')):
            raise ValueError('三个native seed的来源设置、checkpoint、依赖或源码不同')
    for a,b in [('videojev_controller','videojev_baseline'), *[('qwen_controller',f'native_seed{seed}') for seed in seeds]]:
        if sources[a]['identity']['checkpoint'] != sources[b]['identity']['checkpoint']:
            raise ValueError('同模型的来源 checkpoint 指纹不同')
    trials, outputs = {method:[] for method in NAMES}, {}
    distributions = {method:{} for method in RNG_METHODS}
    for row in rows:
        key = row['id']; q = predictions['qwen_controller'][key]
        j = predictions['videojev_controller'][key]
        baseline = predictions['videojev_baseline'][key]
        if q['preprocessing'] != j['preprocessing']:
            raise ValueError('融合两源逐题视频预处理不一致')
        for seed in seeds:
            if predictions[f'native_seed{seed}'][key]['preprocessing'] != baseline['preprocessing']:
                raise ValueError('4FPS独立基线逐题视频预处理不一致')
        j = dict(j, probabilities=candidate_probabilities(j), logits=validate_logits(j,len(row['options'])))
        for fusion in ('probability','loglinear'):
            distributions['controller_'+fusion][key] = control(q,j,beta=2.,fusion=fusion)
        distributions['qwen_logits'][key] = dict(q, probabilities=candidate_probabilities(q))
        distributions['videojev'][key] = dict(baseline, probabilities=candidate_probabilities(baseline))
    argmax = {method:score_answers(rows,dist,full['benchmark']) for method,dist in distributions.items()}
    for seed in seeds:
        selected = {method:sample_predictions(rows, dist, seed, method) for method,dist in distributions.items()}
        selected['qwen_native'] = predictions[f'native_seed{seed}']
        oracle, overlap = {}, Counter()
        for row in rows:
            q = selected['qwen_logits'][row['id']]; j = selected['videojev'][row['id']]
            qhit = q['answer_index']==row['answer_index']
            jhit = j['answer_index']==row['answer_index']
            overlap['both_correct' if qhit and jhit else 'qwen_only' if qhit else 'videojev_only' if jhit else 'both_wrong'] += 1
            # 使用标签的 OR 仅用于事后统计，明确标记为 oracle，不反馈给模型或采样。
            answer = row['answer_index'] if qhit or jhit else None
            oracle[row['id']] = dict(id=row['id'],query=row['query'],options=row['options'],answer_index=answer,
                oracle=True, qwen_correct=bool(qhit), videojev_correct=bool(jhit),
                qwen_answer_index=q['answer_index'], videojev_answer_index=j['answer_index'])
        selected['baseline_or'] = oracle
        for method, by_id in selected.items():
            metrics = score_answers(rows,by_id,full['benchmark'])
            if method=='baseline_or':
                metrics.pop('invalid')
                metrics['overlap']=dict(overlap)
            trials[method].append(dict(metrics,inference_seed=seed))
            outputs[method,seed]=by_id
    methods = {}
    for method, records in trials.items():
        metric = records[0]['primary_metric']; values=[r[metric] for r in records]
        methods[method]=dict(primary_metric=metric,values=values,trials=records,mean=statistics.mean(values),
            sample_variance_fraction_squared=statistics.variance(values) if len(values)>1 else None,
            sample_variance_percentage_points_squared=statistics.variance(values)*10000 if len(values)>1 else None,
            sample_std_percentage_points=statistics.stdev(values)*100 if len(values)>1 else None,
            decoding='oracle' if method=='baseline_or' else
                'native_autoregressive_sampling' if method=='qwen_native' else 'categorical_sampling_temperature_1')
    files={str(Path(manifest).resolve()):full['manifest_sha256'],str(Path(manifest).resolve().with_suffix('.meta.json')):full['meta_sha256'],**full['verified_inputs']}
    for src in sources.values():files.update(src['files'])
    verify_files(files)
    report=dict(schema='video_bench_full_three_seed_summary_v4',benchmark=full['benchmark'],split=full['split'],n=len(rows),
        selection_seed=None,inference_seeds=list(seeds),source_seed=42,candidate_gpu_passes=3,native_gpu_passes=len(seeds),
        execution_protocol='shared_candidate_sources_three_sampling_seeds_v1',
        candidate_source_scores_reused=True,reference_latest_commit='cb0518a5ea12129384489ab8eabcaf57956e9d2e',
        primary_metric='m_avg' if full['benchmark']=='mlvu' else 'accuracy',methods=methods,argmax_diagnostics=argmax,
        full_dataset=full,protocols=dict(baseline=dict(two,fps=4.,max_frames=4096),controller=two),beta=2.,temperature=1.,
        sample_variance_ddof=1,rng_methods=dict(RNG_METHODS),
        rng="random.Random(f'{seed}:{method}'); 单步Qwen method=qwen36; VideoJev method=videojev; 两版Controller method=jev_as_controller; 固定原题序CDF采样",
        or_scope='same_seed_sampled_2fps_qwen_and_sampled_4fps_videojev',sources=sources,verified_sources=files,
        code_sha256=code_files(),note='固定全量题集；三份候选分数各计算一次，seed42/43/44只控制候选答案采样；native实际生成三遍。OR使用同seed的单步Qwen采样与4FPS VideoJev采样答案，非native OR。argmax另列诊断。')
    if out is not None:
        out=Path(out);out.mkdir(parents=True)
        for method,dist in distributions.items():
            write_new(out/f'{method}.distributions.jsonl',jsonl_bytes(dist.values()))
        for (method,seed),by_id in outputs.items():write_new(out/f'{method}.seed{seed}.predictions.jsonl',jsonl_bytes(by_id.values()))
        write_new(out/'results.json',json_bytes(report));write_new(out/'results.md',markdown(report).encode('utf-8'))
        snapshot(out,report['code_sha256'])
        write_new(out/'artifact_manifest.json',json_bytes(dict(files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()})))
        verify_files(files)
    return report


def markdown(report):
    mean_label = '平均 M-Avg' if report['primary_metric']=='m_avg' else '平均准确率'
    lines=[f'# {BENCHMARK_NAMES[report["benchmark"]]}','',
        f'固定 {report["n"]} 道题；selection_seed=None；主指标：{report["primary_metric"]}。','',
        '| 方法 | '+' | '.join(f'seed{seed}' for seed in report['inference_seeds'])+f' | {mean_label} | 样本方差 |',
        '|---|'+'---:|'*(len(report['inference_seeds'])+2)]
    best = max(s['mean'] for method,s in report['methods'].items() if method!='baseline_or')
    for method in NAMES:
        s = report['methods'][method]
        mean = f'{s["mean"]*100:.2f}%'
        if method!='baseline_or' and s['mean']==best:
            mean = '**'+mean+'**'
        variance = s['sample_variance_percentage_points_squared']
        variance_text = '—（单seed）' if variance is None else f'{variance:.4f}'
        lines.append('| '+NAMES[method]+' | '+' | '.join(f'{x*100:.2f}%' for x in s['values'])+
            f' | {mean} | {variance_text} |')
    lines += ['',report['note'],'',report['rng'],'',
        'native Qwen与VideoJev独立基线：4 FPS/4096帧；单步Qwen及融合双源：2 FPS/2048帧。单步生成指首步候选内采样，并非完整词表的自回归生成。两版Controller均β=2，候选采样temperature=1，无top-k/top-p。',
        '旧公式 softmax(z_Q + 2*p_J)；新公式 softmax(z_Q + 2*s_J)。OR对同seed的单步Qwen 2FPS采样答案与VideoJev 4FPS采样答案逐题取“至少一个正确”，与latest公开表定义一致；native不参与该OR。无效native答案在自身准确率中计错并保留分母。',
        'MLVU报告七任务宏平均M-Avg，各次trial同时保留总体accuracy。样本方差单位为百分点²（pp²），ddof=1，使用未舍入数值；标准差保留在JSON中。','',
        '## 确定性 argmax 诊断（不替代主表采样结果）','']
    for method,m in report['argmax_diagnostics'].items():
        lines.append(f'- {NAMES[method]}（共用来源）：{m[m["primary_metric"]]*100:.2f}%')
    lines+=['','## 分组结果','', '| 方法 | seed | 分组 | 正确/题数 | accuracy % |','|---|---:|---|---:|---:|']
    for method,s in report['methods'].items():
        for trial in s['trials']:
            for group,g in trial['by_stratum'].items():
                lines.append(f'| {NAMES[method]} | {trial["inference_seed"]} | {group} | {g["correct"]}/{g["n"]} | {g["accuracy"]*100:.2f} |')
    return '\n'.join(lines)+'\n'
