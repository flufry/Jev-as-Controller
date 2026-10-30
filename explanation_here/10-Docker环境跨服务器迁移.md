# 10 · 新增扩展：Docker 环境跨服务器迁移

已在其它服务器装好项目环境时，可以保存镜像并导入目标服务器，跳过 wheel 预下载和重新构建。只安装好了 Docker Engine 或只拉取了 PyTorch 基础镜像，不等于已配置好项目依赖，迁移基础镜像仍需安装本项目环境。

以下在两台服务器的**宿主机**执行。示例使用 sudo；已有 Docker 权限时可省略。目标数据目录使用 `/public/hyj/jev`。命令中的源路径、容器名和目标 IP 按实际替换。两台机器应为 Linux x86_64，目标需要 NVIDIA 驱动及 NVIDIA Container Toolkit；镜像不会迁移宿主机驱动。本项目后端当前针对 SM89，迁到其它 GPU 架构仍需适配与验证。

## 1. 源服务器：保存完整项目镜像

先查看已有镜像、容器和挂载，确认哪个容器实际可运行项目：

```bash
sudo docker images
sudo docker ps -a
sudo docker inspect jev-bench --format '{{json .Mounts}}'
```

选择下面一种方式，统一生成 `jev-bench:migrated`：

```bash
# A：项目依赖已全部固化在构建后的镜像中。
sudo docker tag jev-bench:cu128 jev-bench:migrated
```

如果是在启动容器后手工 pip/apt 安装依赖，则改用 B，保存该容器可写层的变更：

```bash
# B：源容器名按实际替换；在没有评测任务运行时操作。
sudo docker commit jev-bench jev-bench:migrated
```

`docker commit` 默认会短暂暂停容器；它不包含 bind mount/volume 中的文件。尤其是仓库、`.deps-qwen36`、模型、数据、缓存和结果，需要单独复制。Python 环境若装在挂载的目录中，该目录也必须一起复制并保持容器内原路径。见 [Docker commit 说明](https://docs.docker.com/reference/cli/docker/container/commit/)。

然后在源服务器有空间的磁盘保存镜像：

```bash
TRANSFER=/$HOME/hyj/jev/transfer
mkdir -p "$TRANSFER"
set -o pipefail
sudo docker save jev-bench:migrated | gzip -1 > "$TRANSFER/jev-bench.tar.gz"
```

确认上条命令成功后生成校验和：

```bash
(cd "$TRANSFER" && sha256sum jev-bench.tar.gz > jev-bench.tar.gz.sha256)
```

使用 [docker save](https://docs.docker.com/reference/cli/docker/image/save/) 和 load 保留镜像层与配置；不要用容器 export/import 替代此流程。

## 2. 从源服务器传输

```bash
TARGET=hyj@目标服务器IP
ssh "$TARGET" 'mkdir -p /public/hyj/jev/transfer'
rsync -a --partial --info=progress2 \
  "$TRANSFER/jev-bench.tar.gz" "$TRANSFER/jev-bench.tar.gz.sha256" \
  "$TARGET:/$HOME/hyj/jev/transfer/"
```

单独复制仓库与持久化数据。源目录以源服务器实际挂载为准；下面假设为 `~/srv/jev`，包含 `.git`、代码与后端目录。目标已有不同代码修改时，先选新的目标目录，避免覆盖。

```bash
SOURCE_ROOT="$HOME/srv/jev"
rsync -a --partial --info=progress2 \
  --exclude='.venv*/' --exclude='docker/bench/wheels/' \
  "$SOURCE_ROOT/Jev-as-Controller" "$TARGET:/public/hyj/jev/"

# 如需要迁移已下载的模型、视频、缓存和结果，再执行这条。
rsync -a --partial --info=progress2 \
  "$SOURCE_ROOT/models" "$SOURCE_ROOT/data" \
  "$SOURCE_ROOT/hf-cache" "$SOURCE_ROOT/runs" \
  "$TARGET:/public/hyj/jev/"
```

不需要为已经完整配置的镜像重新传 wheel。上述仓库复制会排除普通准备虚拟环境；如果实际运行依赖就装在某个 `.venv` 挂载中，则应另行完整迁移该 Linux 环境，保持路径并验收。源文件为 root 所有且普通用户无读取权限时，需要由服务器管理者授权读取再传输。

## 3. 目标服务器：导入并创建新容器

```bash
cd /public/hyj/jev/transfer
sha256sum -c jev-bench.tar.gz.sha256
sudo docker load -i jev-bench.tar.gz
sudo docker image inspect jev-bench:migrated --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

校验成功后继续；创建前确认迁移后的仓库目录真实存在。使用新容器名保留原有容器：

```bash
JEV_HOST_ROOT=/public/hyj/jev
test -d "$JEV_HOST_ROOT/Jev-as-Controller/scripts" || exit 1
mkdir -p "$JEV_HOST_ROOT"/{data,models,runs,hf-cache}

sudo docker run -d --name jev-bench-migrated \
  --gpus all --network host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="$JEV_HOST_ROOT/Jev-as-Controller",dst=/workspace/Jev-as-Controller \
  --mount type=bind,src="$JEV_HOST_ROOT/models",dst=/workspace/Jev-as-Controller/models \
  --mount type=bind,src="$JEV_HOST_ROOT/data",dst=/workspace/Jev-as-Controller/data \
  --mount type=bind,src="$JEV_HOST_ROOT/runs",dst=/workspace/Jev-as-Controller/runs \
  --mount type=bind,src="$JEV_HOST_ROOT/hf-cache",dst=/root/.cache/huggingface \
  --workdir /workspace/Jev-as-Controller \
  jev-bench:migrated tail -f /dev/null

sudo docker exec -it jev-bench-migrated bash
```

若源服务器使用不同容器内路径、用户、环境变量或额外挂载，应按源运行配置调整；镜像归档不会自动重新创建原容器的 GPU、网络和宿主机目录挂载。旧服务器的 `127.0.0.1:7890` 代理也不自动迁移，不要将其视为新服务器已有服务。

## 4. 在新容器中验收

下面在新容器内执行，先验证 Python、依赖和 GPU：

```bash
cd /workspace/Jev-as-Controller
python --version
python -m pip check
nvidia-smi
python -c 'import torch, flash_attn, causal_conv1d; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])'

git config --global --add safe.directory /workspace/Jev-as-Controller
bash scripts/setup_bench_runtime.sh
source scripts/env_qwen36.sh
python scripts/check_bench_runtime.py \
  --out "reports/bench-runtime-migrated-$(date +%Y%m%d-%H%M%S).json"
```

后端 `.so` 可能依赖原 Python/Torch/CUDA ABI 和 GPU 架构。迁移后必须通过新服务器的加载和数学检查；不兼容时按 06/runtime 说明在新环境重建，不能只凭 docker load 成功认定可推理。检查通过后继续 05 的数据预检和真实模型 pilot。

本文只提供迁移流程，未替你执行远程打包、传输、加载或 GPU 验证。

## 5. 只预下载 CUTLASS 源码再传输

目标服务器卡在 `Cloning into .../third_party/cutlass` 时，可在另一台有 Git、tar 和 SSH 的 Linux 服务器预下载，不需要 Python、Docker 或 GPU。必须使用本项目固定的 CUTLASS commit，并把 `.git` 一起打包；GitHub 源码 ZIP 不包含安装脚本核验 revision 所需的 Git 元数据。

### 下载服务器宿主机

创建一个新临时目录，浅拉取固定提交，避免下载完整历史。确认该服务器 HOME 有足够空间，或把下面 mktemp 模板换到数据盘。

```bash
CUTLASS_BUNDLE_ROOT="$(mktemp -d "$HOME/jev-cutlass.XXXXXX")"
(
  set -e
  git init "$CUTLASS_BUNDLE_ROOT/cutlass"
  git -C "$CUTLASS_BUNDLE_ROOT/cutlass" remote add origin https://github.com/NVIDIA/cutlass.git
  git -C "$CUTLASS_BUNDLE_ROOT/cutlass" fetch --depth 1 --progress origin \
    8783c41851cd3582490e04e69e0cd756a8c1db7f
  git -C "$CUTLASS_BUNDLE_ROOT/cutlass" checkout --detach FETCH_HEAD
  test "$(git -C "$CUTLASS_BUNDLE_ROOT/cutlass" rev-parse HEAD)" = \
    8783c41851cd3582490e04e69e0cd756a8c1db7f
  tar -C "$CUTLASS_BUNDLE_ROOT" -czf "$CUTLASS_BUNDLE_ROOT/cutlass-src.tar.gz" cutlass
  cd "$CUTLASS_BUNDLE_ROOT"
  sha256sum cutlass-src.tar.gz > cutlass-src.tar.gz.sha256
)
```

上面的下载命令成功后再传输；TARGET 替换为目标服务器的账户和地址：

```bash
TARGET=hyj@目标服务器IP
ssh "$TARGET" 'mkdir -p "$HOME/jev-transfer"'
scp "$CUTLASS_BUNDLE_ROOT/cutlass-src.tar.gz" \
  "$CUTLASS_BUNDLE_ROOT/cutlass-src.tar.gz.sha256" "$TARGET:jev-transfer/"
```

### 目标服务器宿主机

先在原编译终端 Ctrl+C，等 setup/Git 命令返回后再操作，避免与下载进程同时写同一目录。下面对应当前实际挂载的 `/home/hyj/srv/jev/Jev-as-Controller`；若挂载源不同，修改 CODE_ROOT。保留旧的未完成目录，不直接删除它。

```bash
CODE_ROOT="$HOME/srv/jev/Jev-as-Controller"
(
  set -e
  cd "$HOME/jev-transfer"
  sha256sum -c cutlass-src.tar.gz.sha256
  CUTLASS_PARENT="$CODE_ROOT/.deps-qwen36/src/grouped_gemm/third_party"
  sudo mkdir -p "$CUTLASS_PARENT"
  if [[ -e "$CUTLASS_PARENT/cutlass" || -L "$CUTLASS_PARENT/cutlass" ]]; then
    sudo mv -- "$CUTLASS_PARENT/cutlass" "$CUTLASS_PARENT/cutlass.partial-$(date +%Y%m%d-%H%M%S)"
  fi
  sudo tar --no-same-owner -xzf cutlass-src.tar.gz -C "$CUTLASS_PARENT"
  test "$(sudo git -C "$CUTLASS_PARENT/cutlass" rev-parse HEAD)" = \
    8783c41851cd3582490e04e69e0cd756a8c1db7f
  test -f "$CUTLASS_PARENT/cutlass/include/cutlass/cutlass.h"
)
```

该命令按容器内 root 编译的当前配置解压，独立 Git 仓库不依赖源服务器的路径。旧 submodule 在父仓库 `.git/modules` 内的未完成对象不会被本次独立 checkout 使用；保留它们不影响当前 setup 的 header/commit 核验。

然后进入当前目标容器（名称按实际修改）继续：

```bash
sudo docker exec -it jev-bench-migrated-v2 bash
```

```bash
cd /workspace/Jev-as-Controller
bash scripts/setup_bench_runtime.sh
```

脚本发现正确的 CUTLASS header 和 commit 后会跳过该 submodule 下载，继续准备 XTuner、应用补丁和编译。此流程只预下载 CUTLASS，XTuner 等尚未准备的依赖仍会联网。若 setup 已进入补丁/编译阶段并修改了 grouped_gemm 源码，不能将它当作纯下载失败直接重跑，按 06 的本地修改说明处理。
