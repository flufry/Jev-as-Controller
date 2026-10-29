# 09 · 新增扩展：三个 benchmark 的命令行下载

本节为 [05](05-跨Benchmark粗筛运行手册.md) 准备 Video-MME v1、MLVU-Dev 七任务 MCQ 和 LVBench。下载入口 `scripts/download_bench_data.py` 会下载、整理目录并检查官方题目库存；不加载模型、不截取视频、不改变题序或候选选项。真实下载、YouTube 可用性和视频解码仍需在服务器验收。

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
