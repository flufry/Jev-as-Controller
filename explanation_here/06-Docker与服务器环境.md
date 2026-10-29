# 06 · 新增扩展：Docker 与服务器环境

本文只讲本次跨 benchmark 粗筛环境。评测流程见 [05-跨Benchmark粗筛运行手册.md](05-跨Benchmark粗筛运行手册.md)，文件清单见 [07-本次文件变更记录.md](07-本次文件变更记录.md)。原仓库 README 和 runtime 原说明保留。

## 1. 推荐镜像与适用硬件

优先拉取 Docker Hub 官方开发镜像：

```bash
docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
```

已核对 [Docker Hub 官方 tag](https://hub.docker.com/layers/pytorch/pytorch/2.8.0-cuda12.8-cudnn9-devel/images/sha256-a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6)：linux/amd64、CUDA 12.8.1、cuDNN 9、Ubuntu 22.04。请选 **devel**，因为 grouped-GEMM 和其它原生扩展需要 `nvcc`。基础镜像不是本仓库完整环境，不包含指定 VideoJev/Qwen 权重和自定义 CUTLASS 补丁。

当前新入口继续复用原 Qwen 实现，可靠的目标仍是 **RTX 4090 / SM89**，4–6 张放 Qwen，VideoJev 单卡；可复用同一组卡顺序执行。35B 权重本身约 67 GiB，此外还需激活显存。四卡能否在相同长视频协议下完成要实测，不承诺不 OOM。A100/H100/RTX 5090 等架构不能仅换镜像就视为后端已验证；需要额外适配和数学一致性验证。

推荐使用下面的 Dockerfile 在官方镜像上构建项目环境。没有核实到一个公开镜像已包含本仓库所需的**确切**补丁和完整 ABI，因此不提供未经验证的“拉完即可全流程运行”第三方镜像名。

## 2. 服务器宿主机前置

宿主机需要 Linux x86_64、可用 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。CUDA Toolkit 放容器里，宿主机无需再安装一套同版本 toolkit。先确认：

```bash
nvidia-smi
docker version
```

驱动需能支持 CUDA 12.8 运行时。为了减少新库兼容问题，推荐使用支持 CUDA 12.8 的 570 系列或更新驱动；是否可用最终以容器中的 CUDA 张量运算和后端测试为准，不能只看 `nvidia-smi` 的版本字样。服务器若由集群管理员配置，先使用现有配置。

Ubuntu/Debian 尚未安装 Docker 时按 [Docker 官方安装说明](https://docs.docker.com/engine/install/ubuntu/) 安装。NVIDIA Toolkit 的官方步骤见 [安装与 Docker 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。典型 rootful Docker 配置如下，均在**宿主机**执行：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`systemctl restart docker` 可能影响服务器上已有容器，应按服务器维护安排执行。非 root Docker 的配置路径不同，见上述官方说明。

## 3. 把代码放到服务器

推荐从 Git checkout 运行，保留 `.git`，原评测会记录 git HEAD、状态和 diff。本文没有替你执行远程 push、创建服务器或发布 Docker Hub 镜像。

本地提交并推送这次代码后，在服务器克隆对应分支，或在已有 checkout 拉取：

```bash
# 将分支名换成你实际推送的分支。
git clone --branch <你的分支> https://github.com/tangqh03/Jev-as-Controller.git
cd Jev-as-Controller
git rev-parse HEAD
git status --short
```

用 scp/rsync 上传也可以，但必须保留 Git 元数据或在服务器准备真实 checkout。不要上传 Windows 的 `.venv`、`.venv-bench-check`、已编译 `.deps*` 或半成品 runs 作为 Linux 环境。权重和视频放在服务器本地磁盘，推荐目录：

```text
/srv/jev/Jev-as-Controller/  # 代码 checkout
/srv/jev/data/              # 数据与原视频
/srv/jev/models/            # 两套权重
/srv/jev/runs/              # 实验结果，持久化写入
/srv/jev/hf-cache/          # 可选 HF 下载缓存
```

## 4. 构建项目环境镜像

**宿主机下载快、Docker 内 pip 下载慢时，优先使用 [08-Wheel预下载与离线安装.md](08-Wheel预下载与离线安装.md)**：一次收集全部 Python 依赖及传递依赖，再让构建从本地 wheel 目录安装，覆盖 pyarrow、Torch/CUDA 依赖和两个原生扩展。

在**服务器的仓库根目录**运行：

```bash
docker build --network host --progress=plain --pull \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
docker image inspect jev-bench:cu128 --format '{{.Id}}'
```
sha256:3b74968dfd5e26e85cf682e67d11e2bd7e4a5ee7b691544a7e238c2d7fed5f47

新增 `.dockerignore` 使用最小依赖文件白名单，避免把视频、权重、Git 历史和历史结果传入 build context。代码在运行时挂载，因此代码变更无需重新构建镜像；依赖变更需要重建。

这里给 **build 阶段**也指定 `--network host`，让 Dockerfile 内的 pip/apt 等 `RUN` 使用构建器的 host 网络。只给后面的 `docker run` 添加 host 网络，不会改变此前构建镜像时的 pip 网络。上述命令面向本机 Linux Docker Engine 默认构建器；远程构建器使用的是构建器所在机器的网络。参数作用见 [Docker build 网络说明](https://docs.docker.com/reference/cli/docker/buildx/build/#network)。

若宿主机已有 HTTP 代理，例如实际监听 `127.0.0.1:7890`，构建时还需显式传代理（端口按实际修改）：

```bash
docker build --network host --pull -f docker/bench/Dockerfile -t jev-bench:cu128 \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
  --build-arg NO_PROXY=localhost,127.0.0.1 .
```

这是 Docker [预定义的代理构建参数](https://docs.docker.com/build/building/variables/#proxy-arguments)，无需修改 Dockerfile 加入代理 ENV。只有代理确实在该地址运行时才添加。若自定义 BuildKit 报 `network.host is not allowed`，需让 builder 允许 `network.host` entitlement，并在 `docker buildx build` 中传 `--allow network.host --network host --load`。

host 网络有助于排查 bridge/NAT、DNS 或宿主机本地代理访问问题，**不保证单独提高 PyPI 下载速度**。`Building wheel` 也不能单独证明正在编译：FlashAttention/causal-conv1d 的构建钩子可能仍在从 GitHub 下载预编译 wheel。确认有 `nvcc`/`ninja` 编译进程后，换网络模式才不会缩短这部分编译耗时。

### 国内 pip 镜像与原生扩展下载停滞

普通 Python 包可使用 [清华 TUNA PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。在服务器仓库根目录执行：

```bash
docker build --network host --progress=plain \
  --build-arg PYPI_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
```

该参数覆盖普通 pip 安装，Torch 仍使用官方 cu128 源；不会改变 apt 源、uv 的 Python 解释器下载或 GitHub Release 地址。镜像若暂未同步某个固定版本，可恢复官方 PyPI 源，不直接修改版本锁定。切换 `PYPI_INDEX_URL` 会使使用此参数的层及后续层重新执行；已有 pip 下载缓存仍可复用。不要增加 `--no-cache`。

**当前 causal-conv1d 卡住的步骤要单独处理。** [上游 v1.7.0 安装代码](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py#L297-L318) 会用 `urllib.request.urlretrieve()` 访问 GitHub，普通 pip 源及 `PIP_DEFAULT_TIMEOUT` 不控制这次下载。新 Dockerfile 给默认安装增加 `-v`，便于看到 `Guessing wheel URL`；日志仍需结合下载/编译进程判断。

可以使用本节前面的 HTTP/HTTPS 代理构建命令；代理地址必须真实可用，`--network host` 本身不提供代理。如果服务器没有可用代理，推荐在可访问 GitHub 的机器上下载 wheel，再上传服务器，避免每次构建等待 GitHub。

当前 Python 3.12、Torch 2.8、CUDA 12、C++11 ABI TRUE、Linux x86_64 环境对应的文件名如下。**保存原文件名，不要把 cp311、torch2.7 或 ABI FALSE 的文件改名充当此文件。** 其它环境应从 [官方 Release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0) 选择真正匹配的资产。

```bash
# 在网络可用的机器的仓库根目录执行，直接保存到构建所读取的 wheels 目录。
mkdir -p docker/bench/wheels
wheel_name=causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
curl -fL --retry 5 --connect-timeout 30 --max-time 1800 \
  -o "docker/bench/wheels/$wheel_name" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/$wheel_name"
```

上述命令已将 `.whl` 保存到当前 checkout 的 `docker/bench/wheels/`。若直接在服务器仓库根目录下载，完成后用之前的构建命令重试即可；若在其它机器下载，把文件上传到**服务器 checkout 的同名目录**，保留 `.gitkeep`。wheel 被 Git 忽略，需要用 scp/rsync 等单独传输。Dockerfile 会按实际 Python/Torch/CUDA/ABI 查找准确文件名，找到后使用 `--no-index --no-deps` 安装该包，跳过其 GitHub 下载和源码编译。普通依赖仍需完成，因此这是该扩展的离线安装入口，不是整个镜像的离线构建。构建末尾仍执行 `pip check`，容器内按第 6 节验证原生扩展导入和 GPU 环境。

本地 wheel 通过只读 build bind mount 使用，不将原始 wheel 复制到镜像层。FlashAttention 和 causal-conv1d 共用 `docker/bench/wheels/`，建议将两份文件一次放齐；改变目录内容可能使两个安装层及后续层重新执行。保持原来的 pip 源和前面依赖文件时，此前成功的 Torch、构建工具、requirements 层仍可复用。

如果已有可信的内部 wheel 镜像/文件服务，也可直接传入这个**匹配 wheel 文件的完整 HTTPS 地址**（替换下面的占位符），不需要先上传本地文件：

```bash
docker build --network host --progress=plain \
  --build-arg CAUSAL_CONV1D_WHEEL_URL='https://<可信文件服务>/<匹配的完整wheel文件名>.whl' \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
```

优先级为：匹配的本地 wheel → 显式指定的 wheel URL → 上游安装器。显式下载失败会报错，不再静默转入源码编译；curl 有下载进度、重试和超时。这里的 URL 参数只作用于 causal-conv1d，不是通用 GitHub 镜像开关，不填含访问密钥的地址。

### FlashAttention 同样使用本地 wheel 或指定下载地址

FlashAttention 的上游安装器也会从 GitHub Release 下载预编译 wheel，换 pip 镜像不能覆盖这次下载。Dockerfile 支持与 causal-conv1d 相同的优先级：**匹配的本地 wheel → `FLASH_ATTN_WHEEL_URL` → 上游安装器**，并继续保留默认安装分支的 `--no-cache-dir` 和不挂载 pip 缓存的修正。

当前 Python 3.12、Torch 2.8、CUDA 12、C++11 ABI TRUE、Linux x86_64 对应以下 [官方 Release 资产](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)。在网络可用的机器的仓库根目录执行，命令会直接保存到 `docker/bench/wheels/`，与 causal-conv1d wheel 一起放置；若下载机器不是服务器，再将该文件上传到服务器 checkout 的同名目录：

```bash
mkdir -p docker/bench/wheels
wheel_name=flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
curl -fL --retry 5 --connect-timeout 30 --max-time 1800 \
  -o "docker/bench/wheels/$wheel_name" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/$wheel_name"
```

服务器目录示例：

```text
docker/bench/wheels/
  .gitkeep
  flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

上传最新 Dockerfile 和 wheel 目录后，使用原构建命令重试；两个包匹配成功时均使用 `--no-index --no-deps` 安装，不运行它们的 GitHub 下载或源码编译。wheel 文件被 Git 忽略，必须单独上传，且不能将不匹配 ABI/版本的 wheel 改名。未放匹配 wheel 且未传地址时，日志会打印预期文件名并回到上游安装流程。

已有可信 wheel 镜像/文件服务时，可同时指定两个完整下载地址。将下面占位符换成真正匹配的 HTTPS wheel 地址；若某包已有本地 wheel，其本地文件优先：

```bash
docker build --network host --progress=plain \
  --build-arg FLASH_ATTN_WHEEL_URL='https://<可信文件服务>/<匹配的flash_attn文件名>.whl' \
  --build-arg CAUSAL_CONV1D_WHEEL_URL='https://<可信文件服务>/<匹配的causal_conv1d文件名>.whl' \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
```

显式 URL 下载有进度、重试和超时，失败即报错；可叠加本节前面的普通包 `PYPI_INDEX_URL` 参数或真实可用的 HTTP/HTTPS 代理。构建后仍按第 6 节检查 `import flash_attn, causal_conv1d` 和 GPU 运行环境，不能仅凭 wheel 安装成功判断所有原生算子可用。

### 已准备 wheel，却仍然出现 `Downloading flash_attn-*.tar.gz`

`.tar.gz` 是源码发行包。这条日志本身不代表安装失败，也不能证明正在 CUDA 编译：上游构建钩子随后仍可能下载 GitHub wheel。但它说明该安装步骤没有走本地 wheel 或显式 wheel URL 分支。先在**服务器、实际 build context 的仓库根目录**检查：

```bash
pwd
grep -nE 'FLASH_ATTN_WHEEL_URL|REQUIRE_EXTENSION_WHEELS|Expected FlashAttention' docker/bench/Dockerfile
ls -lh docker/bench/wheels/
```

Dockerfile 应有上述参数/日志代码，wheel 目录应有完整、未改名的 `.whl` 文件。只有 `.gitkeep` 表示尚未把 wheel 传到服务器；wheel 被 Git 忽略，`git pull` 不会把它带过去。不要放到容器中的仓库目录、其它 checkout 或仓库根目录的 `wheels/`，构建读取的是末尾 `.` 对应 checkout 的 `docker/bench/wheels/`。如果使用自定义 `.dockerignore` 或 `Dockerfile.dockerignore`，也应确认没有排除该目录；仓库提供的白名单已允许该目录。

更新后的 Dockerfile 在两个扩展的安装前都会打印 `Expected ... wheel:` 和 `/opt/jev-wheels` 中的文件列表。准备好两份 wheel 后，建议启用严格模式构建：

```bash
docker build --network host --progress=plain \
  --build-arg REQUIRE_EXTENSION_WHEELS=1 \
  -f docker/bench/Dockerfile -t jev-bench:cu128 .
```

如之前使用了 `PYPI_INDEX_URL`，继续添加相同值。`REQUIRE_EXTENSION_WHEELS=1` 同时约束 FlashAttention 与 causal-conv1d：必须找到匹配的本地 wheel 或设置对应的完整 wheel URL；否则立即报错，不转入源码包安装流程。默认值 `0` 保留上游安装方式。此参数只约束这两个扩展，不限制其它依赖是否使用源码包。

成功选择本地 FlashAttention 时日志应出现 `Installing local wheel: flash_attn-...whl`，随后 pip 显示 `Processing /opt/jev-wheels/...whl`。严格模式下若同一扩展步骤仍显示 `Collecting flash-attn`/下载 `.tar.gz`，应核对实际使用的 Dockerfile、构建参数及这段日志所属步骤，而非反复清空缓存。若仍无法判断，保留该步骤顶部的 RUN 命令、预期 wheel 名称和文件列表一起排查。

### 本地 wheel 已成功安装，随后版本校验 `AssertionError`

如果日志已显示 `Installing local wheel`、`Processing /opt/jev-wheels/...whl` 和 `Successfully installed flash-attn-2.8.3.post2`，随后在版本断言处退出，这次不是下载失败，也不需要重新下载 wheel。早期 Dockerfile 的校验误用了 `Version(...).base_version`，会把 `2.8.3.post2` 变成 `2.8.3`，因此与目标字符串 `2.8.3.post2` 比较必然失败。

当前 Dockerfile 已改用 `.public`，保留 `.post2` 等公开版本后缀，只去掉 `+cu12...` 这类本地版本标签；causal-conv1d 也统一使用 `.public`，避免把预发行或其它 post 版本误当作目标版本，见 [packaging 版本 API](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version.public)。两个校验都会打印实际安装版本，失败时说明目标与实际值。

若手工修改服务器上的旧 Dockerfile，FlashAttention 的判断应为：

```python
Version(version("flash-attn")).public == "2.8.3.post2"
```

建议上传仓库最新 Dockerfile，保留现有 `docker/bench/wheels/` 文件和原构建参数重试；不加 `--no-cache`，不清理 builder 缓存。失败的安装层会从本地 wheel 重新安装，前面已成功且输入未变的依赖层可以复用。这只修复版本校验，原生扩展导入和 GPU 自检仍按第 6 节执行。

### pip 报 `No matching distribution found for transformers==5.14.1`

已核对 [PyPI 的 5.14.1 发行页](https://pypi.org/project/transformers/5.14.1/)：该版本存在，有通用 Python wheel，要求 Python >=3.10，与目标 Python 3.12 相容。仅有 `from versions: none` 不能证明版本不存在，也可能是网络、索引、代理或源未同步。保留其前面的重试、HTTP/TLS 和索引日志，不直接降级 Transformers。

更新后的 Dockerfile 给普通 Python 包显式指定 `https://pypi.org/simple`，并支持覆盖 `PYPI_INDEX_URL`；Torch 独立保持 CUDA 12.8 专用源。各阶段拆成独立 RUN，普通依赖的 pip 下载使用 BuildKit cache mount，超时设为 120 秒、重试 8 次；FlashAttention 的安装单独禁用 pip 缓存挂载，原因见下一节。后续安装失败时，已成功的安装层和适用的下载缓存可以重用，见 [Docker 缓存说明](https://docs.docker.com/build/cache/optimize/#use-cache-mounts)。缓存位于同一 builder 内，不会自动跨服务器迁移。

将最新 Dockerfile 上传到服务器后，在仓库根目录重试，并保存完整日志：

```bash
set -o pipefail
docker build --network host --progress=plain \
  --build-arg PYPI_INDEX_URL=https://pypi.org/simple \
  -f docker/bench/Dockerfile -t jev-bench:cu128 . \
  2>&1 | tee docker-build.log
```

不加 `--no-cache`，以便复用已成功的层。第一次改用拆分后的 Dockerfile 时，旧失败 RUN 内已完成的 Torch 安装无法恢复为缓存层，需要重装一次；以后才能逐步复用。

若仍失败，可以先用已拉取的基础镜像单独测试同一网络下是否能取到该版本（不需要 GPU）：

```bash
docker run --rm --network host pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  python -m pip download --no-deps --only-binary=:all: \
  --index-url https://pypi.org/simple --dest /tmp/package-check transformers==5.14.1
```

若需要宿主机代理，这条 `docker run` 也要增加实际的 `-e HTTP_PROXY=... -e HTTPS_PROXY=...`。单包测试失败先修网络/代理/源，成功而构建失败再检查 builder 与测试容器的网络差异及完整构建日志。可以将 `PYPI_INDEX_URL` 指向已确认含全部固定版本的可信镜像，但不要把通用 PyPI 源替换到 Torch 的 cu128 安装行。

### FlashAttention 报 `[Errno 18] Invalid cross-device link`

若日志已打印 `Guessing wheel URL` 和 `Raw wheel path`，随后报从 `.whl` 移动到 `/root/.cache/pip/wheels/...` 的 `Invalid cross-device link`，这是文件跨挂载点移动失败。[FlashAttention 2.8.3.post2 的安装代码](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/setup.py#L447-L468) 下载预编译 wheel 后使用 `os.rename()`；源码临时目录与 BuildKit 的 pip 缓存挂载属于不同文件系统，触发 EXDEV。构建日志中的 `No CUDA runtime is found` 和 setuptools 警告并非这次退出的原因。

修正为：FlashAttention 这一条 RUN **移除** `--mount=type=cache,target=/root/.cache/pip,...`，并加 `--no-cache-dir`，让 pip 使用普通临时 wheel 目录。当前 Dockerfile 的默认联网安装分支保留这一设置，其 pip 命令如下；本地/显式 URL 分支则直接安装 wheel：

```bash
python -m pip install -v --no-cache-dir --index-url "$PYPI_INDEX_URL" --no-build-isolation \
  -c /opt/jev-build/constraints.txt flash-attn==2.8.3.post2
```

仓库 Dockerfile 已包含该修正。更新服务器上的文件后，使用原构建命令重试即可，不加 `--no-cache`，不清理 builder 缓存；此前成功的 Torch、构建工具和 requirements 层可复用。FlashAttention 此步可能重新下载 wheel。也核对了 [causal-conv1d 1.7.0 的安装代码](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py#L297-L318)：它使用支持跨文件系统的 `shutil.move()`，保留其缓存挂载。

Dockerfile 完成：

1. 从官方 PyTorch devel 镜像继承 CUDA 12.8、cuDNN 与编译工具链。
2. 独立创建 `/opt/jev` 的 **Python 3.12.12**。不假设基础镜像 Python 小版本与原实验相同；官方 Dockerfile 的默认 Python 是 3.11，可参见 [上游 v2.8 Dockerfile](https://github.com/pytorch/pytorch/blob/v2.8.0/Dockerfile)。
3. 安装 PyTorch 2.8.0+cu128、torchvision 0.23.0、Triton 3.4.0，以及仓库 requirements；用 `constraints.txt` 固定关键 ABI 与 accelerate 1.15.0、NumPy 1.26.4。
4. 安装 FlashAttention 2.8.3.post2、causal-conv1d 1.7.0 和读取 Parquet 的 pyarrow；原生扩展使用 `--no-build-isolation`，以当前 Torch 构建；未匹配到 wheel 时会源码编译，可能耗时较长。
5. 以 `DS_BUILD_OPS=0` 避免为纯评测预编译不必要的 DeepSpeed 算子。运行 `pip check` 并保存 `/opt/jev-build/pip-freeze.txt`。

构建时无需加载模型或提供 GPU；原生编译使用 `TORCH_CUDA_ARCH_LIST=8.9`。镜像内暂不包含项目专用 grouped-GEMM，它在下一步编译进持久化 checkout。

若希望进一步固定基础镜像，先保存 `docker image inspect pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` 的 RepoDigests，再以确认过的 digest 传入：

```bash
docker build --network host -f docker/bench/Dockerfile -t jev-bench:cu128 \
  --build-arg BASE_IMAGE='pytorch/pytorch@sha256:<你实际核验的digest>' .
```

联网构建失败应检查服务器能否访问 Docker Hub、PyPI、PyTorch wheel 站和 GitHub。不要关闭 TLS 校验。只给安装/下载阶段设置实际可达的代理；不要将 token 写入 Dockerfile。

## 5. 创建与进入容器

先创建宿主机数据目录，再启动容器（按实际服务器路径修改 `JEV_HOST_ROOT`）：

```bash
JEV_HOST_ROOT="$HOME/srv/jev"
mkdir -p "$JEV_HOST_ROOT"/{data,models,runs,hf-cache}

docker run -d --name jev-bench --gpus all --shm-size=32g \
  --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="$JEV_HOST_ROOT/Jev-as-Controller",dst=/workspace/Jev-as-Controller \
  --mount type=bind,src="$JEV_HOST_ROOT/data",dst=/workspace/Jev-as-Controller/data \
  --mount type=bind,src="$JEV_HOST_ROOT/models",dst=/workspace/Jev-as-Controller/models \
  --mount type=bind,src="$JEV_HOST_ROOT/runs",dst=/workspace/Jev-as-Controller/runs \
  --mount type=bind,src="$JEV_HOST_ROOT/hf-cache",dst=/root/.cache/huggingface \
  --workdir /workspace/Jev-as-Controller \
  jev-bench:cu128 tail -f /dev/null

docker exec -it jev-bench bash
```

GPU 编号：`--gpus all` 暴露所有卡；评测命令内再用 `CUDA_VISIBLE_DEVICES` 选择。若改为 `--gpus '"device=..."'` 限制卡，容器内需重新检查编号，不照抄宿主机编号。`--shm-size=32g` 是上限，按机器内存调整。默认 root 创建的结果在宿主机可能归 root 所有，可按服务器惯例配置用户/缓存目录。

Linux 上 `--network host`（等价于 `--net=host`，注意是两个横杠）让容器共享宿主机网络，容器内的 `127.0.0.1` 可以访问宿主机本地代理；不再需要 `-p` 端口映射。见 [Docker host 网络说明](https://docs.docker.com/engine/network/drivers/host/)。该选项不会自动配置 pip 代理；如果需要使用宿主机已有的 HTTP 代理，在容器内设置：

```bash
# 仅在宿主机确实有此代理时设置，地址/端口按实际修改。
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1
```

已创建的 bridge 容器不能通过 `docker exec` 或重启改成 host 网络；需要用上述挂载重新创建容器，可以改用新名字（如 `jev-bench-host`）保留原容器，后续 `docker exec` 也使用新名字。

这里不使用 `--rm`，便于重入。退出交互 shell 不会删除容器。结果、权重、数据以及 checkout 中 `.deps-qwen36`/`.deps-xtuner`/reports 都在宿主机持久化；它们**不是**容器根文件系统的一次性内容。

```bash
# 宿主机：重入或停/启。
docker exec -it jev-bench bash
docker stop jev-bench
docker start jev-bench
```

## 6. 容器自检与专用后端编译

以下在**容器内**执行：

```bash
which python
python --version
nvcc --version
nvidia-smi
python - <<'PY'
import torch, torchvision, transformers, triton
print(torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, triton.__version__)
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    assert torch.cuda.get_device_capability(i) == (8, 9), '该预设仅针对 SM89'
    x = torch.ones((64, 64), device=f'cuda:{i}', dtype=torch.bfloat16)
    assert (x @ x).float().mean().item() == 64.
import flash_attn, causal_conv1d
print('CUDA/BF16/native imports OK')
PY

bash scripts/setup_bench_runtime.sh
source scripts/env_qwen36.sh
python scripts/check_bench_runtime.py --out reports/bench-runtime-check.json
```

`setup_bench_runtime.sh` 复用 runtime 原说明的固定 revision 和本地补丁：grouped_gemm v1.1.4、CUTLASS、XTuner 来源 checkout；检查 revision，再生成并比对补丁，编译到 `.deps-qwen36`。它不安装通用同名 grouped_gemm 包来替代补丁。已有完整 `.so` 时尝试加载并检查来源，不覆盖后端；ABI 不匹配直接报错。

脚本只会修改自己创建/核验的独立第三方 checkout。编译中断后若留下源码修改，再次执行会拒绝覆盖；保留日志并对照 [原 runtime 重建说明](../runtime/qwen36/README.md) 检查，确认已有补丁正确后可手工重跑其中的安装步骤，不要对陌生目录盲目 `git reset`。

`check_bench_runtime.py` 在可见 GPU 上逐卡执行小规模 BF16 NN/NT、含空专家组的重排/加权专家前向，与独立 PyTorch 运算对比（atol/rtol 均为 0.02），成功后保存误差和来源。可用 `--devices 0,1` 限定逻辑卡号。它仍不代表完整模型、长上下文或性能已验证；下一步必须跑 pilot 和（若有原数据）LVB 复现闸门。非 4090 的新 GPU 不应跳过该限制强行沿用结果。

## 7. 权重、数据与运行

在联网阶段下载原仓库所固定的权重版本：

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir models/Qwen3.6-35B-A3B
hf download ZhuYuhan/VideoJev-4B \
  --revision 1c1a253bf90dfa66176124d25a48ffb791d517d5 --local-dir models/VideoJev-4B
```

然后按 05 文档依次执行：完整清单 → 抽样 → `check_bench_inputs.py --budget` → VideoJev pilot → 正式两模型 → CPU 融合 → 汇总。下载完成后再设置：

```bash
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

容器内用 `/opt/jev/bin/python`，**无需也不要激活上传的 Windows `.venv`**。原 env 脚本只补环境变量与运行时搜索路径。

长任务建议在宿主机 tmux 中执行 `docker exec`。重连不会丢掉 tmux 内的前台任务。也可以在容器内部保存日志；使用管道时打开 `pipefail`，保留真实退出码：

```bash
set -o pipefail
# 把 05 文档的某条完整评测命令放在下面，路径选择新日志文件。
python scripts/eval_bench_videojev.py \
  --manifest runs/screen/videomme/sample.jsonl --out runs/screen/videomme/videojev \
  --checkpoint models/VideoJev-4B --device cuda:0 --seed 42 \
  --fps 2 --max-frames 2048 --context 262144 --visual-token-budget 24576 \
  2>&1 | tee runs/screen/videomme/videojev-console.log
```

重入容器需要重新 source 环境；中断后同配置加 `--resume`，不要删除旧结果。完成后把 `runs/screen/*/summary/results.*` 和总表取回本地即可查看；原视频和权重无需回传。

## 8. 保存可复用环境与验证范围

镜像不含运行时挂载的数据/代码/权重/编译后端。可保存并在另一个兼容服务器加载：

```bash
docker save jev-bench:cu128 -o jev-bench-cu128.tar
# 另一台服务器：docker load -i jev-bench-cu128.tar
```

如以后要发布到自己的 Docker Hub，需要单独给镜像打上你账号下的 tag 并执行 `docker push`；本文没有发布镜像，也没有可用的个人 Hub 地址。专用 `.so` 仍应在相同 Python/Torch/CUDA/SM89 环境重建，或连同已验证的独立后端保存来源证明后迁移。

本次开发机是 Windows CPU 环境，无 Docker daemon、无 CUDA/模型/正式 benchmark 视频。已验证项目代码的 CPU 测试，不代表 Dockerfile 在服务器已成功构建，也不代表正式三 bench 已执行。服务器还需完成：镜像构建、`pip check`、GPU 张量自检、原生扩展加载/数学验证、真实视频预算预检、pilot 和正式推理。这些执行状态不得提前写成通过。
