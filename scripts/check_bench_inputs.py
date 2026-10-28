"""CPU preflight: validate manifests and audit full-video geometry/token budgets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import json_bytes, load_rows, sha256, write_new


def check(manifest, out, budget=False):
    rows = load_rows(manifest)
    audits = {}
    if budget:
        from timejev.lvb_experiment import prepare_record, record_audit
        for row in rows:
            if row['video_key'] not in audits:
                record = prepare_record(row, 24576, fps=2., max_frames=2048)
                if abs(record['sampled_duration'] - row['duration']) > .1:
                    raise ValueError(f'Duration changed/truncated: {row["video_key"]}')
                audits[row['video_key']] = record_audit(record)
    result = dict(manifest_sha256=sha256(manifest), n=len(rows),
                  videos=len({r['video_key'] for r in rows}), budget_checked=budget, audits=audits)
    write_new(out, json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budget', action='store_true', help='Probe geometry and prepare_record for every sampled video')
    result = check(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != 'audits'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
