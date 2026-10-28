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
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" --no-build-isolation \
      -c /opt/jev-build/constraints.txt flash-attn==2.8.3.post2
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" --no-build-isolation \
      -c /opt/jev-build/constraints.txt causal-conv1d==1.7.0
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --index-url "$PYPI_INDEX_URL" \
      -c /opt/jev-build/constraints.txt pyarrow
RUN python -m pip check \
    && python -m pip freeze > /opt/jev-build/pip-freeze.txt
WORKDIR /workspace/Jev-as-Controller
CMD ["bash"]
