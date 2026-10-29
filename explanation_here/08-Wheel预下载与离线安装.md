# 08 · 新增扩展：宿主机预下载 wheel 与容器本地安装

服务器宿主机下载快，而 Docker build 内 pip 下载慢时，先在**宿主机**准备完整 wheel 目录，再构建镜像。本文覆盖 `docker/bench/Dockerfile` 中的全部 pip 安装，包括 Torch/cu128 依赖、requirements、构建工具、uv、pip、FlashAttention、causal-conv1d 和 pyarrow。依赖清单是 `docker/bench/requirements-wheels.txt`，沿用原 requirements 与 ABI constraints，不降低版本。

这一步只处理 Python wheel。基础镜像拉取、apt、uv 的 Python 3.12.12 解释器下载、后续 grouped-GEMM/CUTLASS/XTuner 的 Git 源码以及模型/视频仍按 [06](06-Docker与服务器环境.md) 准备。`PIP_NO_INDEX=1` 不代表整个 Docker build 都能断网；本次开发机也未实际运行 Linux/Docker/CUDA 下载与构建。

## 1. 在宿主机准备 Python 3.12

准备机器使用 **Linux x86_64、Python 3.12、glibc >= 2.35**，例如 Ubuntu 22.04/24.04。不要使用 Windows Python 或 Python 3.10/3.11 下载本机平台的包。脚本为容器的 Python 3.12 和 Ubuntu 22.04/glibc 2.35 指定下载平台，避免新宿主机选到容器不支持的高 glibc wheel。准备过程不需要 GPU，也不需要宿主机安装 CUDA Toolkit；会使用磁盘中的 CUDA Torch 包。

在**服务器仓库根目录**先确认：

```bash
python3.12 --version
ldd --version | head -n 1
curl --version
```

若没有 Python 3.12，但宿主机已有 uv，可用 uv 下载同版本解释器后运行脚本：

```bash
uv python install 3.12.12
uv run --no-project --python 3.12.12 python scripts/prepare_bench_wheels.py
```

尚未安装 uv 时，可在独立 bootstrap 虚拟环境安装，不改系统 Python；`.venv-bench-wheels` 留给脚本管理的 Python 3.12 环境：

```bash
python3 -m venv /tmp/jev-uv-bootstrap
/tmp/jev-uv-bootstrap/bin/python -m pip install 'uv>=0.8,<1'
/tmp/jev-uv-bootstrap/bin/uv python install 3.12.12
/tmp/jev-uv-bootstrap/bin/uv run --no-project --python 3.12.12 python scripts/prepare_bench_wheels.py
```

若系统缺少 `venv` 模块，先按服务器管理方式安装对应的 `python3-venv` 系统包。现有 Python 3.12 可直接使用下一节命令。

## 2. 一次准备完整依赖

```bash
# 宿主机，仓库根目录；使用宿主机网络，不经过 Docker。
python3.12 scripts/prepare_bench_wheels.py
```

普通包希望使用清华镜像时：

```bash
python3.12 scripts/prepare_bench_wheels.py \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

脚本会：

1. 在 `.venv-bench-wheels/` 创建隔离的 Python 3.12 准备环境，不修改宿主机系统包。
2. 从官方 cu128 索引取得 `torch==2.8.0+cu128`、`torchvision==0.23.0+cu128`，再从普通包源收集它们和构建工具的完整依赖；CUDA 库和 Triton 也包括在内。
3. 在准备环境安装构建所需工具和匹配的 Torch，设置 `DS_BUILD_OPS=0`，将 DeepSpeed 的源码发行包转为不含预编译算子的 Python wheel。只在构建此 wheel 的子进程选择 CPU accelerator，以免依赖宿主机 nvcc；该设置不带入 Docker 运行环境。
4. 复用已存在且 ZIP 完整性通过的两个原生扩展 wheel；缺失时下载官方匹配资产。下载先写 `.part`，完成并验证后再改为 `.whl`。
5. 读取仓库 requirements、constraints 和新增全量清单，收集全部传递依赖。对只有源码包的其它依赖会明确报缺少二进制包，不静默上传宿主机任意本地编译结果。
6. 用 `--no-index --ignore-installed --dry-run` 重新解析整个依赖集合，确认没有利用准备环境中已安装的包掩盖缺包或版本冲突。通过后写出固定版本清单 `resolved-constraints.txt`、解析报告 `resolution.json` 和文件校验和 `SHA256SUMS`。

下载 Torch/CUDA 包并在准备环境展开会占用数 GB，另有 wheel、pip 缓存和最终镜像空间，请按实际可用空间安排。中断后重跑同一条命令即可，已下载的 pip 文件/缓存及完整扩展 wheel 可复用，不用清空整个目录。只有命令最终打印 `Offline dependency resolution passed` 才表示本次依赖准备检查通过。

如果某个固定版本在镜像中未同步，使用默认官方源重试。两个扩展也支持匹配 wheel 的完整自定义 URL：

```bash
python3.12 scripts/prepare_bench_wheels.py \
  --flash-attn-wheel-url 'https://<可信文件服务>/<匹配的flash_attn文件名>.whl' \
  --causal-conv1d-wheel-url 'https://<可信文件服务>/<匹配的causal_conv1d文件名>.whl'
```

## 3. 从本地 wheel 构建

准备成功后，在同一仓库根目录运行：

```bash
docker build --network host --progress=plain \
  --build-arg PIP_NO_INDEX=1 \
  --build-arg PIP_CONSTRAINT=/opt/jev-wheels/resolved-constraints.txt \
  --build-arg REQUIRE_EXTENSION_WHEELS=1 \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
```

`PIP_NO_INDEX=1` 关闭构建期间 pip 索引访问，所有 pip 安装从只读挂载的 `/opt/jev-wheels` 查找 wheel。`PIP_CONSTRAINT` 固定为宿主机已经验证的解析结果，避免拆分安装时重新选不同版本。这些是 Docker **build ARG**，不会在最终容器中留下永久的 pip 离线配置。uv 创建虚拟环境时也会从本地目录安装 pip，避免 `--seed` 再发起包下载；Python 解释器本身仍由 `uv python install` 准备。

缺少包时会直接报 `No matching distribution`，不会转而访问 PyPI。日志应显示本地 `Processing /opt/jev-wheels/...whl`，pyarrow 也不再显示远程 `Downloading ...whl`。既然普通包全部本地安装，此时无需为 pip 再设置镜像源。更新 Dockerfile、改构建参数或修改 wheel 目录会影响层缓存；一次放齐后不加 `--no-cache`。

只有在准备阶段最后的离线解析检查失败时，才继续修复宿主机 wheel 目录后重建。不要用 `--no-deps` 跳过全部普通包依赖来掩盖缺包；只有已经收齐依赖的两个扩展采用直接 wheel 安装，构建末尾仍执行 `pip check`。

## 4. 迁移与运行

若在其它联网机器准备，连同 `docker/bench/wheels/` 下的全部 `.whl`、`resolved-constraints.txt` 和 `SHA256SUMS` 一起传到服务器。它们被 Git 忽略，不能靠 `git pull` 传输。可在传输完成后检查：

```bash
(cd docker/bench/wheels && sha256sum -c SHA256SUMS)
```

不传 `.venv-bench-wheels/`，不把 Windows 或另一 Python ABI 的环境目录当作 Linux 依赖。`resolution.json` 作为准备记录保留即可，不必加入 Docker build context。镜像完成后仍需执行 [06 的容器自检与后端编译](06-Docker与服务器环境.md#6-容器自检与专用后端编译)，以及真实模型 pilot；依赖解析通过不代表 GPU 算子或模型推理已经验证。

下载与本地构建机制参考 [pip download](https://pip.pypa.io/en/stable/cli/pip_download/)、[pip wheel](https://pip.pypa.io/en/stable/cli/pip_wheel/) 和 [DeepSpeed 安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)。
