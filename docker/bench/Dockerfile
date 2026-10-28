# syntax=docker/dockerfile:1
# Official Docker Hub CUDA development image; repository-specific extensions follow.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG PYPI_INDEX_URL=https://pypi.org/simple
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# The upstream image may use Python 3.11. Keep our 3.12 ABI separate and explicit.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    /opt/conda/bin/python -m pip install --index-url "$PYPI_INDEX_URL" 'uv>=0.8,<1' \
    && /opt/conda/bin/uv python install 3.12.12 \
    && /opt/conda/bin/uv venv --python 3.12.12 --seed /opt/jev
ENV PATH="/opt/jev/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    MAX_JOBS=4 \
    DS_BUILD_OPS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1
# Keep completed installs in separate layers; retain downloads if a later step fails.
# Torch must use the cu128 index, independently of the general Python package index.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" \
      setuptools==78.1.0 wheel==0.48.0 packaging==26.3 ninja==1.13.2
COPY requirements.txt requirements-overlay.txt /opt/jev-build/
COPY docker/bench/constraints.txt /opt/jev-build/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" \
      -c /opt/jev-build/constraints.txt -r /opt/jev-build/requirements.txt
# flash-attn 2.8.3.post2 uses os.rename() for its downloaded wheel. A pip cache
# mount puts the wheel output on another filesystem and causes EXDEV. Use pip's
# temporary wheel directory for this step; keep earlier dependency layers cached.
# As with causal-conv1d, prefer a local wheel or an explicit trusted wheel URL.
ARG FLASH_ATTN_WHEEL_URL=""
# Set to 1 to fail rather than fall back to an upstream source-package installer.
ARG REQUIRE_EXTENSION_WHEELS=0
RUN --mount=type=bind,source=docker/bench/wheels,target=/opt/jev-wheels \
    wheel_name="$(python -c 'import sys, sysconfig, torch; py="cp%d%d" % sys.version_info[:2]; tv=".".join(torch.__version__.split(".")[:2]); cu=torch.version.cuda.split(".")[0]; abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(); platform=sysconfig.get_platform().replace("-", "_"); print(f"flash_attn-2.8.3.post2+cu{cu}torch{tv}cxx11abi{abi}-{py}-{py}-{platform}.whl")')" \
    && echo "Expected FlashAttention wheel: $wheel_name" \
    && echo "Files visible in /opt/jev-wheels:" && ls -la /opt/jev-wheels \
    && if [[ -f "/opt/jev-wheels/$wheel_name" ]]; then \
      echo "Installing local wheel: $wheel_name"; \
      python -m pip install --no-cache-dir --no-index --no-deps "/opt/jev-wheels/$wheel_name"; \
    elif [[ -n "$FLASH_ATTN_WHEEL_URL" ]]; then \
      curl --fail --location --retry 5 --connect-timeout 30 --max-time 1800 \
        --output "/tmp/$wheel_name" "$FLASH_ATTN_WHEEL_URL" \
      && python -m pip install --no-cache-dir --no-index --no-deps "/tmp/$wheel_name" \
      && rm "/tmp/$wheel_name"; \
    elif [[ "$REQUIRE_EXTENSION_WHEELS" == "1" ]]; then \
      echo "ERROR: Matching FlashAttention wheel missing. Place $wheel_name in docker/bench/wheels/ or set FLASH_ATTN_WHEEL_URL." >&2; \
      exit 2; \
    else \
      echo "No matching local wheel ($wheel_name); using upstream installer (may download from GitHub)."; \
      python -m pip install -v --no-cache-dir --index-url "$PYPI_INDEX_URL" --no-build-isolation \
        -c /opt/jev-build/constraints.txt flash-attn==2.8.3.post2; \
    fi \
    && python -c 'from importlib.metadata import version; from packaging.version import Version; installed=version("flash-attn"); print("Installed flash-attn:", installed); assert Version(installed).public == "2.8.3.post2", f"Expected flash-attn 2.8.3.post2, got {installed}"'
# PyPI mirrors do not cover the GitHub release download in causal-conv1d's setup.py.
# Prefer a matching local wheel, then an explicit trusted wheel URL, then upstream.
ARG CAUSAL_CONV1D_WHEEL_URL=""
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=docker/bench/wheels,target=/opt/jev-wheels \
    wheel_name="$(python -c 'import sys, sysconfig, torch; py="cp%d%d" % sys.version_info[:2]; tv=".".join(torch.__version__.split(".")[:2]); cu=torch.version.cuda.split(".")[0]; abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(); platform=sysconfig.get_platform().replace("-", "_"); print(f"causal_conv1d-1.7.0+cu{cu}torch{tv}cxx11abi{abi}-{py}-{py}-{platform}.whl")')" \
    && echo "Expected causal-conv1d wheel: $wheel_name" \
    && echo "Files visible in /opt/jev-wheels:" && ls -la /opt/jev-wheels \
    && if [[ -f "/opt/jev-wheels/$wheel_name" ]]; then \
      echo "Installing local wheel: $wheel_name"; \
      python -m pip install --no-index --no-deps "/opt/jev-wheels/$wheel_name"; \
    elif [[ -n "$CAUSAL_CONV1D_WHEEL_URL" ]]; then \
      curl --fail --location --retry 5 --connect-timeout 30 --max-time 1800 \
        --output "/tmp/$wheel_name" "$CAUSAL_CONV1D_WHEEL_URL" \
      && python -m pip install --no-index --no-deps "/tmp/$wheel_name" \
      && rm "/tmp/$wheel_name"; \
    elif [[ "$REQUIRE_EXTENSION_WHEELS" == "1" ]]; then \
      echo "ERROR: Matching causal-conv1d wheel missing. Place $wheel_name in docker/bench/wheels/ or set CAUSAL_CONV1D_WHEEL_URL." >&2; \
      exit 2; \
    else \
      echo "No matching local wheel ($wheel_name); using upstream installer (may download from GitHub)."; \
      python -m pip install -v --index-url "$PYPI_INDEX_URL" --no-build-isolation \
        -c /opt/jev-build/constraints.txt causal-conv1d==1.7.0; \
    fi \
    && python -c 'from importlib.metadata import version; from packaging.version import Version; installed=version("causal-conv1d"); print("Installed causal-conv1d:", installed); assert Version(installed).public == "1.7.0", f"Expected causal-conv1d 1.7.0, got {installed}"'
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" \
      -c /opt/jev-build/constraints.txt pyarrow
RUN python -m pip check \
    && python -m pip freeze > /opt/jev-build/pip-freeze.txt
WORKDIR /workspace/Jev-as-Controller
CMD ["bash"]
