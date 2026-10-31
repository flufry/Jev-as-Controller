"""原生生成式 LongVideoBench QA：完整回答解析、独立计分及可恢复保存。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import torch
from transformers import LogitsProcessor

from .lvb_experiment import ROOT, checkpoint_identity, load_sample, protocol, sha256, write_json

SEED_RULE = 'SHA256(f"{inference_seed}:{question_id}")前8字节大端整数，模2**63；每题独立重设torch CPU/CUDA种子'
PROTOCOL_NOTE = ('使用官方Qwen模型卡推荐的Instruct采样设置，完整生成后确定性解析；'
    '未公开论文的完整baseline代码及text-only备用extractor，因此不声称逐项复现论文主表。'
    '解析失败、歧义和截断均计错，不以候选logits补选答案。')


def generation_parameters(max_new_tokens=32768):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens必须是正整数')
    return dict(do_sample=True, temperature=.7, top_p=.8, top_k=20, min_p=0.,
                repetition_penalty=1., max_new_tokens=max_new_tokens)


def question_seed(seed, qid):
    if type(seed) is not int or not isinstance(qid, str) or not qid:
        raise ValueError('需要整数推理种子和非空问题ID')
    return int.from_bytes(hashlib.sha256(f'{seed}:{qid}'.encode()).digest()[:8], 'big') % 2**63


class PresencePenalty(LogitsProcessor):
    """仅惩罚已生成的不同token；由HF在temperature/top-k/top-p之前调用。"""
    def __init__(self, prompt_length, penalty=1.5):
        if (type(prompt_length) is not int or prompt_length < 0
                or type(penalty) not in (int, float) or not math.isfinite(penalty)):
            raise ValueError('prompt_length或presence penalty无效')
        self.prompt_length, self.penalty = prompt_length, float(penalty)

    def __call__(self, input_ids, scores):
        result = scores.clone()
        for batch in range(input_ids.shape[0]):
            seen = input_ids[batch, self.prompt_length:].unique().to(result.device)
            result[batch, seen] -= self.penalty
        return result


def extract_answer(response, options, truncated=False):
    """只读取回答和原选项，不读取标签；对歧义采取保守计分。"""
    def result(index=None, reason='no_answer'):
        return dict(answer_index=index, valid=index is not None, reason='ok' if index is not None else reason)

    if truncated:
        return result(reason='truncated')
    if not isinstance(response, str) or not 2 <= len(options) <= 26:
        return result(reason='invalid_response')
    text = response.strip()
    if text.count('<think>') != text.count('</think>'):
        return result(reason='unclosed_thinking')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'^```(?:json|text)?\s*\n?(.*?)\n?```$', r'\1', text, flags=re.DOTALL).strip()
    text = text.replace('**', '')
    if text.startswith('{'):
        try:
            def unique_object(pairs):
                value = {}
                for key, member in pairs:
                    if key in value:
                        raise ValueError('duplicate JSON field')
                    value[key] = member
                return value
            value = json.loads(text, object_pairs_hook=unique_object)
            fields = [value[key] for key in ('answer', 'final_answer') if key in value]
            parsed = [extract_answer(member, options) for member in fields]
            if (parsed and all(member['valid'] for member in parsed)
                    and len({member['answer_index'] for member in parsed}) == 1):
                return parsed[0]
            return result(reason='ambiguous_or_invalid_json')
        except (ValueError, TypeError):
            return result(reason='invalid_json')
    def normalized(value):
        return value.strip().rstrip('.!').casefold().strip()

    option_texts = [normalized(re.sub(r'^[A-Z][.)]\s*', '', option)) for option in options]

    def parse_value(value):
        value = value.strip()
        alternatives = re.findall(r'\b([A-Z])\b\s*(?:or|and|/)\s*\b([A-Z])\b',
                                  value, flags=re.IGNORECASE)
        if any(first.upper() != second.upper() for first, second in alternatives):
            return result(reason='ambiguous')
        bare = re.fullmatch(r'(?:([A-Z])|\(([A-Z])\)|\[([A-Z])\])\s*[.!]?',
                            value, flags=re.IGNORECASE)
        if bare:
            index = ord(next(group for group in bare.groups() if group).upper())-ord('A')
            return result(index) if index < len(options) else result(reason='out_of_range')
        labeled = re.fullmatch(r'([A-Z])[.)：:\-]\s*(.+)', value, flags=re.IGNORECASE | re.DOTALL)
        if labeled:
            index = ord(labeled[1].upper())-ord('A')
            if index >= len(options):
                return result(reason='out_of_range')
            return result(index) if normalized(labeled[2]) == option_texts[index] else result(reason='invalid_option_text')
        exact = [index for index, option in enumerate(option_texts) if normalized(value) == option]
        return result(exact[0]) if len(exact) == 1 else result(reason='ambiguous' if exact else 'no_answer')

    patterns = (
        r'\bfinal\s+answer\s*(?::|is|=)\s*',
        r'\b(?:the\s+correct\s+answer|the\s+answer|answer)\s*(?::|is|=)\s*',
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            values = [parse_value(text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)])
                      for i, match in enumerate(matches)]
            if not all(value['valid'] for value in values):
                return next(value for value in values if not value['valid'])
            return values[0] if len({value['answer_index'] for value in values}) == 1 else result(reason='ambiguous')
    return parse_value(text)


def audit_generated_tokens(prediction, tokenizer):
    """从保存的完整token序列重解码，验证回答及原始特殊token文本。"""
    tokens = prediction.get('generated_token_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or type(prediction.get('generated_tokens')) is not int
            or prediction['generated_tokens'] != len(tokens)):
        raise ValueError(f'生成token IDs或数量无效：{prediction.get("id")}')
    try:
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
    except Exception as error:
        raise ValueError(f'生成token无法解码：{prediction.get("id")}') from error
    if response != prediction.get('response') or raw != prediction.get('raw_response'):
        raise ValueError(f'生成token、原始文本与回答不一致：{prediction.get("id")}')


def validate_generated(row, prediction):
    if (prediction.get('id') != row['id'] or prediction.get('query') != row['query']
            or prediction.get('options') != row['options'] or type(prediction.get('truncated')) is not bool
            or not isinstance(prediction.get('response'), str)):
        raise ValueError(f'生成预测的问题、选项或截断字段不匹配：{row["id"]}')
    parsed = extract_answer(prediction.get('response'), row['options'], prediction['truncated'])
    index = parsed['answer_index']
    if (prediction.get('parsed') != parsed or prediction.get('answer_index') != index
            or index is not None and type(prediction.get('answer_index')) is not int
            or prediction.get('valid') is not parsed['valid']
            or prediction.get('invalid_reason') != (None if parsed['valid'] else parsed['reason'])
            or prediction.get('answer') != (row['options'][index] if parsed['valid'] else None)):
        raise ValueError(f'生成预测的解析结果与原回答不一致：{row["id"]}')


def read_generated(path, rows, complete=True, tokenizer=None):
    targets = {row['id']: row for row in rows}
    predictions = {}
    path = Path(path)
    for line in path.read_text().splitlines() if path.exists() else []:
        prediction = json.loads(line)
        qid = prediction.get('id')
        if qid in predictions or qid not in targets:
            raise ValueError(f'重复或未知生成预测ID：{qid}')
        validate_generated(targets[qid], prediction)
        if tokenizer is not None:
            audit_generated_tokens(prediction, tokenizer)
        predictions[qid] = prediction
    if complete and set(predictions) != set(targets):
        raise ValueError('生成预测未完整覆盖题集')
    return predictions


def score_generated(rows, by_id):
    if not rows or len({row['id'] for row in rows}) != len(rows) or set(by_id) != {row['id'] for row in rows}:
        raise ValueError('计分需要非空互异题集及完整预测')
    totals, hits, invalid = Counter(), Counter(), 0
    for row in rows:
        if type(row.get('answer_index')) is not int or not 0 <= row['answer_index'] < len(row['options']):
            raise ValueError(f'真实标签超出选项范围：{row["id"]}')
        prediction = by_id[row['id']]
        validate_generated(row, prediction)
        group = str(row['duration_group'])
        totals[group] += 1
        hits[group] += prediction['valid'] and prediction['answer_index'] == row['answer_index']
        invalid += not prediction['valid']
    correct = sum(hits.values())
    return dict(n=len(rows), videos=len({row['video_key'] for row in rows}), correct=correct,
        invalid=invalid, accuracy=correct/len(rows), split='validation_subset',
        by_duration_group={g: dict(n=totals[g], correct=hits[g], accuracy=hits[g]/totals[g]) for g in totals})


class NativeExperiment:
    def __init__(self, args, settings, tokenizer=None):
        self.tokenizer = tokenizer
        rows = load_sample(args.manifest)
        self.rows = rows[:args.limit] if getattr(args, 'limit', None) else rows
        self.out = Path(args.out).resolve()
        self.predictions_path = self.out/'predictions.jsonl'
        meta_path = self.out/'run.json'
        if self.out.exists() and not getattr(args, 'resume', False):
            raise FileExistsError(f'拒绝覆盖已有输出：{self.out}')
        if getattr(args, 'resume', False) and not meta_path.is_file():
            raise ValueError('恢复需要已有完整run.json元数据')
        code = [Path(__file__), ROOT/'scripts/eval_longvideobench_qwen36_native.py',
            ROOT/'scripts/eval_longvideobench_qwen36.py', ROOT/'scripts/eval_longvideobench_qwen.py',
            ROOT/'timejev/lvb_experiment.py', ROOT/'timejev/rgb_media.py', ROOT/'timejev/video.py']
        media = {}
        for row in self.rows:
            path = Path(row['video_path'])
            if row['video_key'] not in media:
                media[row['video_key']] = dict(path=str(path), bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        packages = {}
        for name in ('torch', 'transformers', 'accelerate', 'triton', 'flash-attn'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        identity = dict(method='qwen36_native_instruct', manifest_sha256=sha256(args.manifest),
            checkpoint=checkpoint_identity(args.checkpoint), protocol=protocol(args), settings=settings,
            inference_seed=args.seed, seed_rule=SEED_RULE, sample_seed=getattr(args, 'sample_seed', 42),
            limit=getattr(args, 'limit', None), selected_ids=[row['id'] for row in self.rows],
            media=media, packages=packages,
            code_sha256={str(path.relative_to(ROOT)): sha256(path) for path in code})
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
            if self.metadata['identity'] != identity:
                raise ValueError('恢复指纹不匹配，不能复用其他题集、模型、设置或源码')
        else:
            self.out.mkdir(parents=True)
            self.metadata = dict(schema='longvideobench_native_generation_run_v1', identity=identity,
                status='running',
                manifest=str(Path(args.manifest).resolve()), command=[sys.executable, *sys.argv],
                started_at=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), pid=os.getpid(),
                python=sys.version, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True), note=PROTOCOL_NOTE)
            write_json(meta_path, self.metadata)
            for path in code:
                destination = self.out/'source'/path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        self.done = set(read_generated(self.predictions_path, self.rows, complete=False, tokenizer=self.tokenizer))

    def append_response(self, row, response, generated_token_ids, *, raw_response=None, truncated=False, **details):
        if row['id'] in self.done or row['id'] not in {member['id'] for member in self.rows}:
            raise ValueError(f'重复或未知题目：{row["id"]}')
        if (not isinstance(generated_token_ids, list) or not generated_token_ids
                or any(type(token) is not int or token < 0 for token in generated_token_ids)):
            raise ValueError('生成token IDs必须是非空非负整数列表')
        if raw_response is None and self.tokenizer is not None:
            raw_response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        parsed = extract_answer(response, row['options'], truncated)
        index = parsed['answer_index']
        prediction = dict(details, id=row['id'], query=row['query'], options=row['options'],
            video_key=row['video_key'], duration_group=row['duration_group'], response=response,
            raw_response=response if raw_response is None else raw_response,
            generated_token_ids=generated_token_ids, generated_tokens=len(generated_token_ids),
            truncated=truncated, parsed=parsed, answer_index=index, valid=parsed['valid'],
            invalid_reason=None if parsed['valid'] else parsed['reason'],
            answer=row['options'][index] if parsed['valid'] else None)
        validate_generated(row, prediction)
        if self.tokenizer is not None:
            audit_generated_tokens(prediction, self.tokenizer)
        with self.predictions_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.done.add(row['id'])
        print(json.dumps(dict(done=len(self.done), total=len(self.rows), id=row['id'], valid=parsed['valid'])), flush=True)

    def finish(self):
        metrics = score_generated(self.rows, read_generated(self.predictions_path, self.rows, tokenizer=self.tokenizer))
        metrics.update(method=self.metadata['identity']['method'], protocol=self.metadata['identity']['protocol'],
            inference_seed=self.metadata['identity']['inference_seed'],
            finished_at=datetime.now(timezone.utc).isoformat())
        write_json(self.out/'metrics.json', metrics)
        self.metadata.update(status='completed', finished_at=metrics['finished_at'])
        write_json(self.out/'run.json', self.metadata)
        return metrics
