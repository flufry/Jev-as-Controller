# 08 · 新增扩展：宿主机预下载 wheel 与容器本地安装

服务器宿主机下载快，而 Docker build 内 pip 下载慢时，先在**宿主机**准备完整 wheel 目录，再构建镜像。本文覆盖 `docker/bench/Dockerfile` 中的全部 pip 安装，包括 Torch/cu128 依赖、requirements、构建工具、uv、pip、FlashAttention、causal-conv1d 和 pyarrow。依赖清单是 `docker/bench/requirements-wheels.txt`，沿用原 requirements 与 ABI constraints，不降低版本。

这一步只处理 Python wheel。基础镜像拉取、apt、uv 的 Python 3.12.12 解释器下载、后续 grouped-GEMM/CUTLASS/XTuner 的 Git 源码以及模型/视频仍按 [06](06-Docker与服务器环境.md) 准备。`PIP_NO_INDEX=1` 不代表整个 Docker build 都能断网；本次开发机也未实际运行 Linux/Docker/CUDA 下载与构建。

若 apt 下载遇到 502 或代理错误，可在本文构建命令上叠加 `--build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu --build-arg APT_NO_PROXY=1`，具体条件见 [06 的 apt 故障处理](06-Docker与服务器环境.md#apt-下载-502ubuntu-镜像与代理绕过)。`APT_NO_PROXY=1` 适用于 Ubuntu 软件源能够直连的服务器。

## 1. 准备下载环境（优先 Bash 入口，宿主机无需 Python）

### 1.1 没有宿主机 Python：借用已有基础镜像

新增 `scripts/prepare_bench_wheels.sh`，在 Linux x86_64 宿主机执行。它使用官方 PyTorch 基础镜像启动临时准备容器，采用 host 网络；不需要先构建项目镜像，不执行 apt，也不需要 GPU。容器中准备隔离的 uv/Python 3.12.12，再复用原下载脚本的依赖解析、DeepSpeed 打包、原生 wheel 检查和离线完整性检查。

先同步新脚本，在服务器仓库根目录执行：

```bash
# 建议整个仓库也放在 /public 的数据盘上，wheel 会保存到当前仓库内。
export JEV_WHEEL_CACHE=/public/hyj/jev/wheel-prep-cache
export JEV_PYPI_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
bash scripts/prepare_bench_wheels.sh
```

宿主机需要 Bash、Docker 使用权限和可写的数据目录，无需 Python/pip/uv。基础镜像默认为 `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`，已有镜像可直接复用；未拉取时 Docker 会拉取，也可用 `JEV_DOWNLOAD_IMAGE` 指定兼容的镜像（需 `/opt/conda/bin/python`、venv 和 glibc >=2.35）。不要求准备镜像安装 curl：下载两个原生扩展时优先用 curl，缺少时改用 Python 标准库，仍有超时、重试和 wheel 校验。Docker 没有权限时，先按 06 配置当前用户的 Docker 访问权限，再以该用户运行下载入口。

环境、Python 解释器、准备虚拟环境及缓存保存到 `JEV_WHEEL_CACHE`，下次运行可以复用。wheel 和解析记录仍写入**当前仓库的 `docker/bench/wheels/`**，Dockerfile 可直接读取。目录以当前宿主机 UID/GID 写入；源码以只读方式挂载，wheel 单独挂到 `/opt/jev-wheels`，准备虚拟环境直接使用 `/cache/venv`，不在只读源码目录内创建嵌套挂载点。仓库若仍在已满的 `/home` 分区，单独把缓存放到 `/public` 不能解决 wheel 输出空间不足，应把仓库也放到大容量数据盘。

默认缓存路径为 `${XDG_CACHE_HOME:-$HOME/.cache}/jev-wheel-prep`。没有设置 `JEV_PYPI_INDEX_URL` 时使用官方 PyPI。脚本会转发已设置的 HTTP(S)/ALL/NO_PROXY 及小写变量；host 网络允许访问真实监听于宿主机的本地代理，但不会自动提高带宽。若需代理，先在宿主机 export 实际可用的代理地址，再运行脚本。Python 解释器及原生扩展仍需访问 GitHub；普通 pip 镜像不覆盖这些下载。

若日志显示缓存落到 `/root/.cache`，说明当前 HOME 为 `/root` 且没有生效的 `JEV_WHEEL_CACHE`，常见于 root/sudo 执行。可在同一条命令明确指定：`JEV_WHEEL_CACHE=/public/hyj/jev/wheel-prep-cache bash scripts/prepare_bench_wheels.sh`。sudo 可能清除之前 export 的变量，且 root 执行会产生 root 所有的文件，优先以有 Docker 权限的普通用户运行。

旧版入口若报 `.venv-bench-wheels ... read-only file system`，需同时同步 `scripts/prepare_bench_wheels.sh` 和 `scripts/prepare_bench_wheels.py`。新版将可写目录移出只读源码挂载，原 wheel 可复用。

若仅显示两条路径就退出 1，旧版的 `command -v curl >/dev/null` 在缺少 curl 时会静默触发退出。需同步上面两个文件；新版取消该强制检查，输出 `[1/4]` 至 `[4/4]` 阶段日志，并在 shell 失败时打印阶段及退出码。只看到路径不能判定一定缺少 curl，仍需新版错误日志确认。退出 0 且最后出现 `Offline dependency resolution passed` 才表示完成。

这是 Bash 操作入口，依赖解析仍在准备容器内由 Python/pip 完成，以保留完整的传递依赖和版本检查。后续构建命令不变。开发机完成了离线命令检查，完整下载和 Docker 运行仍需在服务器验证。

### 1.2 已有宿主机 Python：保留原入口（可选）

准备机器使用 **Linux x86_64、Python 3.12、glibc >= 2.35**，例如 Ubuntu 22.04/24.04。不要使用 Windows Python 或 Python 3.10/3.11 下载本机平台的包。脚本为容器的 Python 3.12 和 Ubuntu 22.04/glibc 2.35 指定下载平台，避免新宿主机选到容器不支持的高 glibc wheel。准备过程不需要 GPU，也不需要宿主机安装 CUDA Toolkit；会使用磁盘中的 CUDA Torch 包。

在**服务器仓库根目录**先确认：

```bash
python3.12 --version
ldd --version | head -n 1
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

推荐直接使用第 1.1 节的 Bash 命令，它会一次准备下面列出的全部依赖。只有选择第 1.2 节的宿主机 Python 方式时，才使用下面的 `python3.12` 命令；两种入口生成同一个 wheel 目录，不需要重复执行。

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

1. 创建隔离的 Python 3.12 准备环境，不修改宿主机系统包。直接 Python 入口默认使用仓库 `.venv-bench-wheels/`；Bash 入口使用 `JEV_WHEEL_CACHE/venv`，在准备容器中的路径为 `/cache/venv`。
2. 从官方 cu128 索引取得 `torch==2.8.0+cu128`、`torchvision==0.23.0+cu128`，再从普通包源收集它们和构建工具的完整依赖；CUDA 库和 Triton 也包括在内。
3. 在准备环境安装构建所需工具和匹配的 Torch，设置 `DS_BUILD_OPS=0`，将 DeepSpeed 的源码发行包转为不含预编译算子的 Python wheel。只在构建此 wheel 的子进程选择 CPU accelerator，以免依赖宿主机 nvcc；该设置不带入 Docker 运行环境。
4. 复用已存在且 ZIP 完整性通过的两个原生扩展 wheel；缺失时下载官方匹配资产。下载先写 `.part`，完成并验证后再改为 `.whl`。
5. 读取仓库 requirements、constraints 和新增全量清单，收集全部传递依赖。对只有源码包的其它依赖会明确报缺少二进制包，不静默上传宿主机任意本地编译结果。
6. 用 `--no-index --ignore-installed --dry-run` 重新解析整个依赖集合，确认没有利用准备环境中已安装的包掩盖缺包或版本冲突。通过后写出固定版本清单 `resolved-constraints.txt`、解析报告 `resolution.json` 和文件校验和 `SHA256SUMS`。

下载 Torch/CUDA 包并在准备环境展开会占用数 GB，另有 wheel、pip 缓存和最终镜像空间，请按实际可用空间安排。中断后重跑同一条命令即可，已下载的 pip 文件/缓存及完整扩展 wheel 可复用，不用清空整个目录。只有命令最终打印 `Offline dependency resolution passed` 才表示本次依赖准备检查通过。

如果某个固定版本在镜像中未同步，使用默认官方源重试。两个扩展也支持匹配 wheel 的完整自定义 URL：

```bash
bash scripts/prepare_bench_wheels.sh \
  --flash-attn-wheel-url 'https://<可信文件服务>/<匹配的flash_attn文件名>.whl' \
  --causal-conv1d-wheel-url 'https://<可信文件服务>/<匹配的causal_conv1d文件名>.whl'
```

原 Python 入口的对应命令：

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
