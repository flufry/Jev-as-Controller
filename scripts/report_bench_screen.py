"""Combine audited benchmark screening summaries without rerunning inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import json_bytes, sha256, write_new


def report(summaries, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new report directory')
    rows, seen = [], set()
    for path in map(Path, summaries):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('schema') != 'bench_screen_summary_v1':
            raise ValueError('Expected summarize_bench results.json')
        identity = (value['benchmark'], value['seed'])
        if identity in seen:
            raise ValueError('Duplicate benchmark/seed summary')
        seen.add(identity)
        rows.append({k: value[k] for k in ('benchmark', 'seed', 'n', 'primary_metric', 'methods',
                     'delta_vs_qwen', 'or_minus_qwen', 'disagreement', 'decision')})
        rows[-1].update(source=str(path.resolve()), source_sha256=sha256(path))
    lines = ['# 跨 benchmark 粗筛', '', '| Benchmark | seed | n | 主指标 | Qwen % | VideoJev % | Controller % | Δ pp | OR − Qwen pp | 异答率 % |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        methods = r['methods']
        lines.append(f'| {r["benchmark"]} | {r["seed"]} | {r["n"]} | {r["primary_metric"]} | '
            + ' | '.join(f'{100*methods[m]["metric"]:.2f}' for m in ('qwen36', 'videojev', 'controller'))
            + f' | {100*r["delta_vs_qwen"]:+.2f} | {100*r["or_minus_qwen"]:+.2f} | {100*r["disagreement"]["rate"]:.2f} |')
    lines += ['', '各 benchmark 独立判断，不对不同指标/分布求总平均。粗筛子集、非官方全量、未算力对齐、训练重叠未核实。', '']
    write_new(out / 'results.json', json_bytes(dict(schema='bench_screen_report_v1', benchmarks=rows)))
    write_new(out / 'results.md', '\n'.join(lines).encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    report(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
