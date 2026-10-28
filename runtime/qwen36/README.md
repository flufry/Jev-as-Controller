# Qwen3.6 的 SM89 原生专家运行时

此目录保存本次 `Qwen3.6-35B-A3B` 评测使用的小型源码与补丁，供新 checkout 重建被 `.gitignore` 排除的 `.deps-qwen36/`。不包含编译后的 `.so`、完整第三方 checkout、模型权重或运行日志。

适用的原实验环境为 Python 3.12、PyTorch `2.8.0+cu128`、CUDA Toolkit 12.8 和 RTX 4090（SM89）。当前评测入口要求这个带 `qwen36_cutlass_only` 标记的后端，不会退化为 Python 专家循环、量化或 CPU/disk offload。其他 GPU、CUDA/PyTorch ABI 的兼容性需另外验证。

## 文件与来源

| 文件 | 用途 |
| --- | --- |
| `grouped_gemm/{__init__,backend,ops}.py` | 原实验安装包的 Python 包装层，逐字节复制；`__init__.py` 会导入 `ops.py`，所以一并保留 |
| `grouped-gemm-v1.1.4-setup.py` | 原实验使用的 CUDA 扩展构建配置，与本地源码的 `setup.py` 一致 |
| `prepare_cutlass.py` | 从固定 Git HEAD 生成本次 NN/NT CUTLASS 修改，并写出兼容路径的补丁 |
| `sm89-cutlass.patch` | 原实验实际使用的补丁，包含 NN/NT 分组 GEMM 与扩展标记 |
| `SOURCE_MANIFEST.json` | 来源、文件大小、SHA256、固定 revision 和修改后 CUDA 源码 SHA256 |

复制文件的原始字节与 SHA256 已核对，并在临时的本地 Git checkout 重放了 `prepare_cutlass.py`：生成补丁及两份 CUDA 源码的 SHA256 均与原实验一致。本次发布没有重新联网下载、编译扩展、执行 GPU 数学验证或运行模型。下面的完整重建命令按固定源码和原环境构建记录整理，尚未在新的机器上重新执行；不要把源码校验当作新环境编译通过。

## 固定版本

| 依赖 | 来源 | revision |
| --- | --- | --- |
| grouped_gemm v1.1.4 | [fanshiqing/grouped_gemm](https://github.com/fanshiqing/grouped_gemm) | `172fada89fa7364fe5d026b3a0dfab58b591ffdd` |
| CUTLASS | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | `8783c41851cd3582490e04e69e0cd756a8c1db7f` |
| XTuner | [InternLM/xtuner](https://github.com/InternLM/xtuner) | `76e705134521eff867b409f3b3451df1c4d8dd36` |

XTuner checkout 在这里用于评测入口的 revision 来源记录；训练所需的 XTuner 安装另见项目训练文档。

## 从新 checkout 准备运行时

在仓库根目录用 **Bash** 执行。先按根 README 建好 `.venv`，安装匹配的 PyTorch/CUDA、FlashAttention 和 causal-conv1d；CUDA Toolkit 12.8 的 `nvcc` 与可用 C++ 编译器也必须存在。若已有完整且 revision/ABI 匹配的本地运行时，优先复用，不重复运行下载或构建步骤。

下面的下载与安装都使用临时代理环境变量。命令面向尚不存在 `.deps-qwen36/src/grouped_gemm`、`.deps-xtuner` 的新 checkout；已有目录时先核对来源，不用这些命令覆盖本地状态。

```bash
source .venv/bin/activate
(
  set -e
  export HTTP_PROXY=http://127.0.0.1:7890
  export HTTPS_PROXY=http://127.0.0.1:7890
  export ALL_PROXY=http://127.0.0.1:7890
  export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
  export NO_PROXY=localhost,127.0.0.1,::1
  export no_proxy="$NO_PROXY"

  test ! -e .deps-qwen36/src/grouped_gemm
  test ! -e .deps-xtuner
  mkdir -p .deps-qwen36/src
  git init .deps-qwen36/src/grouped_gemm
  git -C .deps-qwen36/src/grouped_gemm remote add origin https://github.com/fanshiqing/grouped_gemm.git
  git -C .deps-qwen36/src/grouped_gemm fetch --depth 1 origin 172fada89fa7364fe5d026b3a0dfab58b591ffdd
  git -C .deps-qwen36/src/grouped_gemm checkout --detach FETCH_HEAD
  git -C .deps-qwen36/src/grouped_gemm submodule update --init third_party/cutlass

  git init .deps-xtuner
  git -C .deps-xtuner remote add origin https://github.com/InternLM/xtuner.git
  git -C .deps-xtuner fetch --depth 1 origin 76e705134521eff867b409f3b3451df1c4d8dd36
  git -C .deps-xtuner checkout --detach FETCH_HEAD

  mkdir -p reports/setup-qwen36-runtime
  cp runtime/qwen36/grouped-gemm-v1.1.4-setup.py .deps-qwen36/src/grouped_gemm/setup.py
  cp runtime/qwen36/grouped_gemm/__init__.py .deps-qwen36/src/grouped_gemm/grouped_gemm/__init__.py
  cp runtime/qwen36/grouped_gemm/backend.py .deps-qwen36/src/grouped_gemm/grouped_gemm/backend.py
  cp runtime/qwen36/grouped_gemm/ops.py .deps-qwen36/src/grouped_gemm/grouped_gemm/ops.py

  # 脚本先核对 grouped_gemm/CUTLASS revision，再由 HEAD 生成两份 CUDA 源码。
  python runtime/qwen36/prepare_cutlass.py
  cmp runtime/qwen36/sm89-cutlass.patch reports/setup-qwen36-runtime/sm89-cutlass.patch

  uv pip install --python .venv/bin/python setuptools==78.1.0 wheel==0.48.0 packaging==26.3 ninja==1.13.2
  export TORCH_CUDA_ARCH_LIST=8.9
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
    --target .deps-qwen36 ./.deps-qwen36/src/grouped_gemm
)
```

`prepare_cutlass.py` 会写入 `reports/setup-qwen36-runtime/sm89-cutlass.patch`，该位置由现有评测入口读取并记录 SHA；不要省略。它从固定 Git HEAD 重建 `csrc/grouped_gemm.cu` 与 `csrc/ops.cu`，因此只在上述独立第三方 checkout 中运行，避免覆盖其他本地 CUDA 修改。

完成后检查源码版本与后端加载，再进行小规模真实推理。以下命令加载扩展并记录来源，不是 GPU 数学 parity 或完整评测：

```bash
source .venv/bin/activate
source scripts/env_qwen36.sh
python - <<'PY'
from scripts.eval_longvideobench_qwen36 import load_backend
backend, provenance = load_backend()
print(provenance)
PY
```

`.deps-qwen36` 保持独立，由 `scripts/env_qwen36.sh` 放在 `PYTHONPATH` 前部。安装一个未打补丁的同名 pip 包不能替代这个后端。

## 归属与许可证

Python 包装层与构建配置来自固定版本的 grouped_gemm，其许可证原文见 [LICENSE.grouped_gemm](LICENSE.grouped_gemm)。本次 `prepare_cutlass.py` 与补丁对该源码增加了 SM89 使用的 CUTLASS NN/NT 路径及扩展标记；修改内容以公开补丁为准。CUTLASS 的许可证原文见 [LICENSE.cutlass](LICENSE.cutlass)。此目录保留上游作者与来源，不代表上游维护者为本实验背书。
