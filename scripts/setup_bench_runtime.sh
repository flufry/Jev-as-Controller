#!/usr/bin/env bash
# Run once inside the prepared Linux container, from an actual Git checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
PY
command -v nvcc
git rev-parse HEAD
grouped_source=.deps-qwen36/src/grouped_gemm
grouped_revision=172fada89fa7364fe5d026b3a0dfab58b591ffdd
cutlass_revision=8783c41851cd3582490e04e69e0cd756a8c1db7f
xtuner_revision=76e705134521eff867b409f3b3451df1c4d8dd36

checkout_pinned() {
  local directory="$1" url="$2" revision="$3"
  if [[ -e "$directory" ]]; then
    test "$(git -C "$directory" rev-parse HEAD)" = "$revision"
  else
    git init "$directory"
    git -C "$directory" remote add origin "$url"
    git -C "$directory" fetch --depth 1 origin "$revision"
    git -C "$directory" checkout --detach FETCH_HEAD
  fi
}
mkdir -p .deps-qwen36/src
checkout_pinned "$grouped_source" https://github.com/fanshiqing/grouped_gemm.git "$grouped_revision"
if [[ ! -f "$grouped_source/third_party/cutlass/include/cutlass/cutlass.h" ]]; then
  git -C "$grouped_source" submodule update --init third_party/cutlass
fi
test "$(git -C "$grouped_source/third_party/cutlass" rev-parse HEAD)" = "$cutlass_revision"
checkout_pinned .deps-xtuner https://github.com/InternLM/xtuner.git "$xtuner_revision"

# Prefer an existing complete ABI-matching runtime; importing it is mandatory.
if compgen -G '.deps-qwen36/grouped_gemm_backend*.so' > /dev/null; then
  source scripts/env_qwen36.sh
  python - <<'PY'
from scripts.eval_longvideobench_qwen36 import load_backend
print(load_backend()[1])
PY
  exit 0
fi
# Refuse to overwrite someone else's edits in the isolated third-party checkout.
test -z "$(git -C "$grouped_source" status --porcelain --untracked-files=no)"
mkdir -p reports/setup-qwen36-runtime
cp runtime/qwen36/grouped-gemm-v1.1.4-setup.py "$grouped_source/setup.py"
cp runtime/qwen36/grouped_gemm/{__init__,backend,ops}.py "$grouped_source/grouped_gemm/"
python runtime/qwen36/prepare_cutlass.py
cmp runtime/qwen36/sm89-cutlass.patch reports/setup-qwen36-runtime/sm89-cutlass.patch
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation --no-deps --target .deps-qwen36 "./$grouped_source"
source scripts/env_qwen36.sh
python - <<'PY'
from scripts.eval_longvideobench_qwen36 import load_backend
print(load_backend()[1])
PY
