#!/usr/bin/env bash
# Linux host entry point: use the existing base image instead of host Python.
set -Eeuo pipefail
report_failure() {
  local status=$?
  echo "ERROR: wheel preparation failed at host line ${BASH_LINENO[0]} (exit $status)." >&2
  exit "$status"
}
trap report_failure ERR

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: bash scripts/prepare_bench_wheels.sh [download options]
No host Python, apt installation, GPU, or completed project image is required.
Environment:
  JEV_WHEEL_CACHE      Persistent bootstrap/Python/venv/cache directory
  JEV_PYPI_INDEX_URL   Bootstrap and ordinary package index (default: PyPI)
  JEV_DOWNLOAD_IMAGE  Base image (default: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel)
Options forwarded to prepare_bench_wheels.py:
  --index-url URL  --torch-index-url URL
  --flash-attn-wheel-url URL  --causal-conv1d-wheel-url URL
Output: docker/bench/wheels/ in this checkout (also the Docker build context).
HELP
  exit 0
fi

command -v docker >/dev/null || { echo 'Docker CLI is required on the Linux host.' >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'Run this entry point on the Linux x86_64 server.' >&2; exit 1;
}
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cache_root="${JEV_WHEEL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/jev-wheel-prep}"
mkdir -p -- "$cache_root" "$repo_root/docker/bench/wheels"
cache_root="$(cd -- "$cache_root" && pwd -P)"
mkdir -p -- "$cache_root/home" "$cache_root/venv" "$cache_root/tmp"
echo "Wheel output: $repo_root/docker/bench/wheels"
echo "Preparation environment and caches: $cache_root"

proxy_args=()
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  if [[ -v "$key" ]]; then
    proxy_args+=(--env "$key=${!key}")
  fi
done

docker run --rm --network host --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$repo_root,target=/workspace/Jev-as-Controller,readonly" \
  --mount "type=bind,source=$repo_root/docker/bench/wheels,target=/opt/jev-wheels" \
  --mount "type=bind,source=$cache_root,target=/cache" \
  --workdir /workspace/Jev-as-Controller \
  --env HOME=/cache/home --env TMPDIR=/cache/tmp --env PIP_CACHE_DIR=/cache/pip \
  --env PIP_CONFIG_FILE=/dev/null --env PIP_NO_INDEX=0 --env PIP_CONSTRAINT= \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env BASH_ENV=/dev/null \
  --env UV_CACHE_DIR=/cache/uv --env UV_PYTHON_INSTALL_DIR=/cache/python \
  --env "JEV_PYPI_INDEX_URL=${JEV_PYPI_INDEX_URL:-https://pypi.org/simple}" \
  "${proxy_args[@]}" \
  --entrypoint /bin/bash \
  "${JEV_DOWNLOAD_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel}" \
  --noprofile --norc -c '
    set -Eeuo pipefail
    phase=startup
    report_failure() {
      local status=$?
      echo "ERROR: preparation container failed during $phase (exit $status)." >&2
      exit "$status"
    }
    trap report_failure ERR
    echo "[1/4] Checking base image Python"
    /opt/conda/bin/python --version
    phase=bootstrap-venv
    echo "[2/4] Preparing bootstrap environment and uv"
    if [[ ! -x /cache/bootstrap/bin/python ]]; then
      /opt/conda/bin/python -m venv /cache/bootstrap
    fi
    /cache/bootstrap/bin/python -m pip install --disable-pip-version-check \
      --timeout 120 --retries 8 --index-url "$JEV_PYPI_INDEX_URL" "uv>=0.8,<1"
    phase=python-install
    echo "[3/4] Preparing Python 3.12.12"
    /cache/bootstrap/bin/uv python install 3.12.12
    bench_python="$(/cache/bootstrap/bin/uv python find 3.12.12)"
    phase=wheel-download
    echo "[4/4] Downloading and checking wheels"
    exec "$bench_python" scripts/prepare_bench_wheels.py \
      --wheel-dir /opt/jev-wheels --work-dir /cache/venv \
      --index-url "$JEV_PYPI_INDEX_URL" "$@"
  ' -- "$@"
