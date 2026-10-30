#!/usr/bin/env bash
# Arrange existing benchmark downloads using the same checks as the downloader.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bench=all
data_root="${BENCH_DATA_ROOT:-$ROOT/data/benchmarks}"
bench_python="${BENCH_PY:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_bench_data.sh [options]
  --bench all|videomme|mlvu|lvbench  Benchmark to arrange (default: all)
  --data-root PATH                 Existing benchmark root (default: data/benchmarks)
  --python PATH                   Download environment's Python interpreter
  -h, --help                      Show this help

Offline only. Reuses download_revision.json and download/ from
scripts/download_bench_data.py or scripts/stage_local_assets.py.
Extracts Video-MME ZIPs, arranges annotations and the MLVU video map,
and checks required video files. Preserves archives and existing inputs.
Does not build questions.jsonl; see explanation_here/09-三个Benchmark数据下载.md.
EOF
}

while (($#)); do
  case "$1" in
    --bench|--data-root|--python)
      if (($# < 2)) || [[ -z "$2" ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
      fi
      case "$1" in
        --bench) bench="$2" ;;
        --data-root) data_root="$2" ;;
        --python) bench_python="$2" ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$bench" in
  all) benches=(videomme mlvu lvbench) ;;
  videomme|mlvu|lvbench) benches=("$bench") ;;
  *) echo "ERROR: Unsupported benchmark: $bench" >&2; exit 2 ;;
esac

if [[ -z "$bench_python" ]]; then
  for candidate in "$ROOT/data/.venv-bench-data/bin/python" "$ROOT/.venv-bench-data/bin/python" python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      bench_python="$candidate"
      break
    fi
  done
fi
if [[ -z "$bench_python" ]] || ! command -v "$bench_python" >/dev/null 2>&1; then
  echo "ERROR: Python not found. Run inside the project container or pass --python PATH." >&2
  exit 2
fi

for current in "${benches[@]}"; do
  if [[ ! -f "$data_root/$current/download_revision.json" || ! -d "$data_root/$current/download" ]]; then
    echo "ERROR: Missing $data_root/$current/download_revision.json or download/ directory." >&2
    echo "Use the recorded directory from download_bench_data.py or stage_local_assets.py; this script does not download missing inputs." >&2
    exit 1
  fi
done

echo "Python: $bench_python"
echo "Benchmark root: $data_root"
for current in "${benches[@]}"; do
  echo "Preparing existing files: $current"
  "$bench_python" "$ROOT/scripts/download_bench_data.py" \
    --bench "$current" --data-root "$data_root" --prepare-local
done
echo "Benchmark files ready. Build and validate manifests as described in explanation_here/09-三个Benchmark数据下载.md."
