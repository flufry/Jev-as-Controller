# 09 · 新增扩展：三个 benchmark 的命令行下载

本节为 [05](05-跨Benchmark粗筛运行手册.md) 准备 Video-MME v1、MLVU-Dev 七任务 MCQ 和 LVBench。下载入口 `scripts/download_bench_data.py` 会下载、整理目录并检查官方题目库存；不加载模型、不截取视频、不改变题序或候选选项。真实下载、YouTube 可用性和视频解码仍需在服务器验收。

若改为 Windows 本地下载、上传后清理以节省本地空间，使用 [11-Windows本地分批下载与SCP上传.md](11-Windows本地分批下载与SCP上传.md)。该流程逐文件上传原始数据，然后在服务器使用 `--prepare-local` 整理，避免本地同时存储完整视频库和压缩包。

## 1. 宿主机准备（只需一次）

宿主机下载快时，推荐直接在**宿主机**执行。以下路径对应 06 中的 `~/srv/jev` 布局；进入实际 checkout 后运行：

```bash
cd "$HOME/srv/jev/Jev-as-Controller"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python3 -m venv .venv-bench-data
.venv-bench-data/bin/python -m pip install -U huggingface_hub pyarrow 'yt-dlp[default]'
BENCH_DATA_ROOT="$HOME/srv/jev/data/benchmarks"
mkdir -p "$BENCH_DATA_ROOT"
.venv-bench-data/bin/hf auth login
```

下载环境需要 Python >=3.10；宿主机缺少 venv 模块时先按服务器管理方式安装对应的 `python3-venv`。它独立于容器中固定的推理依赖。普通包源慢时可给 pip 命令增加 `--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。

**MLVU 当前需要 HF 数据集访问权限**：先登录浏览器打开 [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU)，取得访问权限，再在运行脚本的同一用户下用 `hf auth login` 输入具有相应读取权限的 token。只有 CLI 登录不能代替网页访问申请。不要把 token 写进下载脚本。

若直接在容器中下载，先 `cd /workspace/Jev-as-Controller`，用容器的 `python` 创建同名虚拟环境，并改用 `BENCH_DATA_ROOT=/workspace/Jev-as-Controller/data/benchmarks`。宿主机示例使用 `~/srv/jev/data/benchmarks`，与 06 的 data 挂载对应；不要默认写到宿主机 checkout 内的 data，否则可能被容器的独立挂载遮住。

### 宿主机没有 Python：使用已有项目容器

无需在宿主机安装 Python，也无需重新构建镜像。下面使用已迁移的项目容器 `jev-bench-migrated-v2`；实际容器名不同时替换。先在**宿主机**执行：

```bash
sudo docker inspect jev-bench-migrated-v2 --format '{{.HostConfig.NetworkMode}} {{json .Mounts}}'
sudo docker exec -it jev-bench-migrated-v2 bash
```

确认网络为 `host`，并且 `/public/hyj/jev/data` 挂载到 `/workspace/Jev-as-Controller/data`。数据会直接保存到该宿主机目录，不占容器可写层。`docker exec` 使用现有容器的网络，不能再给它添加 `--network host`。

以下命令均在**容器内**执行。将下载工具安装到单独的虚拟环境，不调整推理环境依赖：

```bash
cd /workspace/Jev-as-Controller
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
BENCH_DATA_ROOT=/workspace/Jev-as-Controller/data/benchmarks
mkdir -p "$BENCH_DATA_ROOT"
python -m venv /workspace/Jev-as-Controller/data/.venv-bench-data
BENCH_PY=/workspace/Jev-as-Controller/data/.venv-bench-data/bin/python
"$BENCH_PY" -m pip install -U huggingface_hub pyarrow 'yt-dlp[default]' \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

MLVU 仍需先在其官方 HF 页面取得访问权限，再在这个下载环境登录；交互输入读取 token，不把 token 写入命令：

```bash
/workspace/Jev-as-Controller/data/.venv-bench-data/bin/hf auth login
```

为 LVBench 安装 Deno。以下适用于本项目的 Linux x86_64 服务器，利用容器 Python 解压，不依赖宿主机 Python 或额外的 unzip 包：

```bash
export DENO_INSTALL=/workspace/Jev-as-Controller/data/.bench-deno
mkdir -p "$DENO_INSTALL/bin"
curl -fL --retry 5 --connect-timeout 30 \
  https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
  -o "$DENO_INSTALL/deno.zip" \
  && "$BENCH_PY" -m zipfile -e "$DENO_INSTALL/deno.zip" "$DENO_INSTALL/bin" \
  && chmod +x "$DENO_INSTALL/bin/deno"
export PATH="$DENO_INSTALL/bin:$PATH"
deno --version
```

然后依次下载三个 benchmark：

```bash
"$BENCH_PY" scripts/download_bench_data.py --bench videomme --data-root "$BENCH_DATA_ROOT"
"$BENCH_PY" scripts/download_bench_data.py --bench mlvu --data-root "$BENCH_DATA_ROOT"
"$BENCH_PY" scripts/download_bench_data.py --bench lvbench --data-root "$BENCH_DATA_ROOT"
```

也可以用 `--bench all` 按上述顺序执行，遇到一个失败就停止。镜像源只作用于 HF 数据，无法加速 LVBench 的 YouTube 视频。下载不使用 GPU；与权重同时下载会共享网络和磁盘带宽。Video-MME 会同时保留 ZIP 和解压视频，需为两者预留空间。重新进入容器后，需重新设置上述变量和 Deno 的 PATH，但不必重建虚拟环境、重复安装工具。

## 2. Video-MME v1

```bash
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench videomme --data-root "$BENCH_DATA_ROOT"
```

脚本使用 [当前 HF 仓库 `lmms-eval/Video-MME`](https://huggingface.co/datasets/lmms-eval/Video-MME/tree/main)（原 `lmms-lab/Video-MME` 已重定向），下载 `videomme/test-*.parquet` 和 `videos_chunked_*.zip`。20 个 ZIP 是独立归档，无需拼接；脚本只提取标注中对应的原视频 MP4，整理为：

```text
videomme/official.json
videomme/videos/<videoID>.mp4
videomme/download_revision.json
videomme/download/  # 原 Parquet、ZIP 和 HF 下载缓存
```

当前视频归档约 100 GB，解压另需视频空间；不会下载字幕包。题目检查要求 2700 题、900 视频、short/medium/long 各 900 题。重复执行会复用下载缓存和大小一致的已解压文件，视频真正可解码及其时长由后续 manifest 构建检查。

## 3. MLVU-Dev MCQ

```bash
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench mlvu --data-root "$BENCH_DATA_ROOT"
```

从官方 [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU) 下载 `MLVU/json/[1-7]_*.json` 和 `MLVU/video/` 下对应编号任务目录。脚本保留标注内容及行序，忽略 8–9 生成式任务和 MLVU-Test。生成目录：

```text
mlvu/json/1_plotQA.json
mlvu/json/2_needle.json
...                         # 共七任务
mlvu/download/MLVU/video/1_plotQA/<video字段>
mlvu/video-map.json          # 原 video 字段到上述视频根目录的相对路径
mlvu/download_revision.json
```

检查要求 2174 个 MCQ、1122 个视频字段、七个任务的固定题数。映射只尝试官方字段与对应任务目录的精确路径，不递归猜测同名文件；有冲突时拒绝选一个文件冒充。

## 4. LVBench

[LVBench 官方下载说明](https://github.com/zai-org/LVBench#download) 指定按 metadata 的 `key` 下载 YouTube 视频；[HF 仓库](https://huggingface.co/datasets/zai-org/LVBench/tree/main) 只有标注，没有完整视频包。这里使用 yt-dlp 下载完整 MP4 视频流，命名为 `<key>.mp4`，对应本次无音频/无字幕协议。

当前 [yt-dlp YouTube 支持需要 JS runtime](https://github.com/yt-dlp/yt-dlp/wiki/EJS)。已有 Deno 时可直接执行下载；没有时按 [官方安装方式](https://docs.deno.com/runtime/getting_started/installation/) 安装到独立下载目录：

```bash
curl -fsSL https://deno.land/install.sh \
  | DENO_INSTALL="$PWD/.venv-bench-data/deno" sh
export PATH="$PWD/.venv-bench-data/deno/bin:$PATH"
deno --version

.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench lvbench --data-root "$BENCH_DATA_ROOT"
```

已有 Node >=22 时，也可以给下载命令增加 `--js-runtime node`，使用 Node 替代 Deno。重新登录 shell 后需重新设置 `BENCH_DATA_ROOT` 和对应 JS runtime 的 PATH。

```text
lvbench/video_info.meta.jsonl
lvbench/videos/<key>.mp4
lvbench/youtube-urls.txt
lvbench/youtube-completed.txt
lvbench/missing-videos.json
lvbench/download_revision.json
```

YouTube 的网络、地区/登录限制和视频下架可能导致缺失。脚本会写 `missing-videos.json` 并以失败状态结束，不自动删题或换视频。重跑相同命令可以继续下载；若需要登录，可按 yt-dlp 官方方法准备 cookies 文件，并给脚本添加 `--cookies <文件路径>`。这些文件不提交到 Git。标注库存仍要求 1549 个唯一 UID，通过文件检查也不等于真实解码或推理已通过。

## 5. 只取标注、一次下载三个与固定版本

先小规模检查网络、权限和题目库存：

```bash
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench all --annotations-only --data-root "$BENCH_DATA_ROOT"
```

访问权限与 JS runtime 准备好后，可一次依次执行三个 bench：

```bash
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench all --data-root "$BENCH_DATA_ROOT"
```

首次下载会将 HF commit SHA 写入各 bench 的 `download_revision.json`；中断重跑、从仅标注切换到完整下载时复用该 SHA，不重新追踪 main。支持给单个 bench 显式传入 `--revision <完整commit>`；与已有记录不一致时使用新的 `--data-root`。已有标注内容不同时拒绝覆盖。下载线程数默认 2，可用 `--workers 4` 调整 HF 文件下载并发。

### 已下载数据：自动解压与整理

前面的 `download_bench_data.py` 在下载结束后已自动完成这一步。若下载中断后需要单独整理，或采用 11 的分批上传方式，可以在**容器内**运行新入口；它仅操作已有文件，不访问 HF 或 YouTube：

```bash
cd /workspace/Jev-as-Controller
bash scripts/prepare_bench_data.sh \
  --data-root /workspace/Jev-as-Controller/data/benchmarks
```

脚本优先使用 `data/.venv-bench-data/bin/python`，其次是仓库 `.venv-bench-data/bin/python`、`python3`、`python`；也可以显式指定 `--python /opt/jev/bin/python` 或通过 `BENCH_PY` 指定下载环境。Video-MME 整理需要该 Python 环境已安装 `pyarrow`。若只下载了其中一个 benchmark，用 `--bench videomme`、`--bench mlvu` 或 `--bench lvbench` 单独整理。

在**宿主机**没有 Python 时，可直接调用已运行的容器执行：

```bash
sudo docker exec -it -w /workspace/Jev-as-Controller jev-bench-migrated-v2 \
  bash scripts/prepare_bench_data.sh \
  --data-root /workspace/Jev-as-Controller/data/benchmarks
```

输入必须是本节下载入口或 11 上传入口的标准布局：每个 benchmark 包含 `download_revision.json` 和 `download/`。不伪造缺失的版本记录，也不自动猜测其它来源压缩包的目录。输出与评测需要的路径对应：

```text
data/benchmarks/
├── videomme/
│   ├── official.json                  # 由 Parquet 转换
│   ├── videos/<videoID>.mp4            # 从 download/videos_chunked_*.zip 解压
│   └── download/                      # 保留原始文件和 ZIP
├── mlvu/
│   ├── json/[1-7]_*.json              # 官方七任务标注
│   ├── video-map.json                 # 标注中的视频名到实际视频路径的映射
│   └── download/MLVU/video/...         # 沿用原视频位置，不复制一份
└── lvbench/
    ├── video_info.meta.jsonl
    ├── videos/<key>.mp4               # 沿用已下载/上传视频，不需要解压
    └── missing-videos.json             # 缺失视频清单
```

如果 data 挂载与前面的示例一致，上述目录在宿主机就是 `/public/hyj/jev/data/benchmarks`。可重复运行；已解压且大小一致的 Video-MME 视频会跳过，不删除 ZIP、缓存或原始视频。缺失视频、冲突标注或损坏 ZIP 会报错，不把不完整数据标记为完成。整理成功后仍需下一节构建清单，检查视频几何信息与时长。

## 6. 回到容器构建清单

按 06 配置好 data 挂载后，在容器的 `/workspace/Jev-as-Controller` 执行：

```bash
python scripts/build_bench_manifest.py --bench videomme \
  --base data/benchmarks/videomme

python scripts/build_bench_manifest.py --bench mlvu \
  --base data/benchmarks/mlvu --annotation data/benchmarks/mlvu/json \
  --videos data/benchmarks/mlvu/download/MLVU/video \
  --video-map data/benchmarks/mlvu/video-map.json

python scripts/build_bench_manifest.py --bench lvbench \
  --base data/benchmarks/lvbench \
  --annotation data/benchmarks/lvbench/video_info.meta.jsonl
```

然后继续 05 的抽样与预算预检。已存在的清单输出不能覆盖，需选择新的 `--out`；不要将缺视频的数据集按可下载性筛成另一个库存。
