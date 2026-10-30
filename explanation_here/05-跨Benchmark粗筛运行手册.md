# 05 · 新增扩展：跨视频 benchmark 粗筛运行手册

本文件描述本次新增实现，与原仓库的 LongVideoBench 实验说明分开。以 `04-待办与执行计划.md` 的小样本路线为目标：Video-MME 150 题、MLVU-Dev MCQ 140 题、LVBench 150 题；不追加 LVB 全量实验。原有 `timejev/`、模型评测脚本、融合公式和已发布结果均未修改。

服务器先按 [06-Docker与服务器环境.md](06-Docker与服务器环境.md) 配好容器。除第 2 节可在宿主机下载外，本文命令在容器内、仓库根目录的 Bash 中执行。

## 1. 实现与计划的补充

- 两模型统一使用 **2 FPS / 最多 2048 帧 / context 262144 / 视觉 token 24576 / 无字幕无音频**；主实验 β 固定为 2。β=1/2/3 只作事后诊断，不据此调参。
- 兼容层不仅替换加载和计分，也处理旧 `Experiment.append()`、Controller 对 `duration_group` 的直接索引。内部用真实 `stratum` 作别名，不把新数据集强行映射为 LVB 四档。原脚本仅在当前调用期间打补丁，结束后恢复。
- 新增 `bench_io.py` 负责不加载模型的清单校验。媒体路径必须指向同一个原视频，拒绝帧缓存；同视频多题共享一致的路径和时长。
- **LVBench 的 `question_type` 是多标签列表**。同一题可能属于多个类别，无法把六类当作互斥人口直接分配 150 个名额。实现按排序后的“标签组合”建立互斥分层，再用整数最大余数法按比例抽题；汇总同时报告每个官方类别的重叠统计。类别计数之和可以超过样本总数。
- MLVU 以七任务 **M-Avg** 为主指标；同时保留微平均 accuracy，便于核对题数。等量抽样时二者相等，不等量时不会混用。
- **OR 不是该融合公式的严格上界**：完整候选分布融合可能选择两个模型 top-1 以外的第三个选项。报告保留 OR oracle 参照、异答率，并统计第三选项、两基线均错但融合答对的题数；低 OR 空间/低异答率仅作启发式筛选。
- 置信区间采用**按分层、问题级配对 bootstrap**，两方法共享重采样题目；不会破坏 MLVU 的宏平均结构。没有建模同视频多题相关性，不能将小样本 CI 或一次 McNemar p 值当作强结论。McNemar 检验的是合并后逐题正确性，不是 M-Avg 的显著性。

## 2. 准备官方标注与原视频

权重、数据和实际输出均不入 Git。新增下载入口负责整理官方数据，清单/评测脚本不替换失效视频；视频不齐时直接报错，避免静默丢题改变分布。

官方入口：

| Benchmark    | 标注/视频来源                                                                                                  | 本实现读取的标注                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Video-MME v1 | [官方仓库](https://github.com/BradyFU/Video-MME)、[HF 数据](https://huggingface.co/datasets/lmms-eval/Video-MME) | HF Parquet，或其逐题 JSON/JSONL 导出；字段 `question_id/videoID/duration/question/options/answer` 等 |
| MLVU-Dev     | [官方仓库](https://github.com/JUNJIE99/MLVU)、[HF：MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU)         | 原发行版七个任务 JSON，或已合并且有 `official_id` 的 JSON                                            |
| LVBench      | [官方仓库](https://github.com/zai-org/LVBench)、[HF 数据](https://huggingface.co/datasets/zai-org/LVBench)       | `video_info.meta.jsonl`，每视频一行，含 `qa`                                                       |

具体工具安装、HF 登录、Deno/Node、下载目录及失败重试见 [09-三个Benchmark数据下载.md](09-三个Benchmark数据下载.md)。推荐在下载速度正常的宿主机执行，进入实际 checkout 后准备独立工具环境：

```bash
cd "$HOME/srv/jev/Jev-as-Controller"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python3 -m venv .venv-bench-data
.venv-bench-data/bin/python -m pip install -U huggingface_hub pyarrow 'yt-dlp[default]'
BENCH_DATA_ROOT="$HOME/srv/jev/data/benchmarks"
.venv-bench-data/bin/hf auth login
```

先在 [MLVU 数据页](https://huggingface.co/datasets/MLVU/MVLU)取得访问权限，再使用相应账号的 token 登录。只登录 CLI 不能代替访问申请。下载目录与 06 的宿主机 data 挂载对应；如果在容器中执行，应进入 `/workspace/Jev-as-Controller`，用 `python` 建环境，并把 `BENCH_DATA_ROOT` 改为 `/workspace/Jev-as-Controller/data/benchmarks`。

三个 benchmark 的完整下载命令：

```bash
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench videomme --data-root "$BENCH_DATA_ROOT"

.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench mlvu --data-root "$BENCH_DATA_ROOT"

# LVBench 从 YouTube 下载；准备 JS runtime，安装到独立的下载工具目录。
curl -fsSL https://deno.land/install.sh \
  | DENO_INSTALL="$PWD/.venv-bench-data/deno" sh
export PATH="$PWD/.venv-bench-data/deno/bin:$PATH"
.venv-bench-data/bin/python scripts/download_bench_data.py \
  --bench lvbench --data-root "$BENCH_DATA_ROOT"
```

Video-MME 视频 ZIP 约 100 GB，解压另需空间；脚本会导出 `official.json` 并整理 900 个原视频。MLVU 只下载 Dev 的七个 MCQ 任务，保留原任务文件行序，生成 `video-map.json`。LVBench 的 HF 仓库仅含标注；YouTube 网络/登录限制或视频下架可能导致无法补齐，失败 ID 会写入 `missing-videos.json`，不会删题冒充完整数据集。

下载首次固定并记录 HF commit；相同命令重跑会复用记录、缓存与已有文件。可给单个 bench 添加 `--revision <完整commit>`，或先用 `--bench all --annotations-only` 检查三个标注的下载权限与库存。不要把 Video-MME-v2 或 MLVU-Test 当成本次三个库存。联网下载前不要设置离线环境变量，代理使用实际可达的地址。

建议整理为：

```text
data/benchmarks/
  videomme/
    official.json        # 或直接用 --annotation 指向下载的 .parquet
    videos/<videoID>.mp4
  mlvu/
    json/1_plotQA.json   # 七个编号任务 JSON；保留文件内容和原始行序
    json/2_needle.json   # 具体编号/文件名以所下载发行版为准
    ...
    download/MLVU/video/1_plotQA/<video字段>
    video-map.json      # 原 video 字段到实际视频子目录的精确映射
  lvbench/
    video_info.meta.jsonl
    videos/<key>.mp4
```

MLVU 七任务名称须为 `plotQA / needle / ego / count / order / anomaly_reco / topic_reasoning`；目录读取只选这些任务，忽略生成式任务。原文件缺少 `official_id` 时使用 `<task_type>/<该任务文件内0起始行号>`，它是本适配器的稳定定位 ID，不宣称是官方发布的 UID。不得重新排序源题。重复候选答案保留，按现有官方对齐规则取第一个匹配项。

若媒体文件布局不同，用 JSON 映射文件，避免模糊搜索同名视频：

```json
{
  "official-video-id": "subdir/actual-file.mp4"
}
```

Video-MME 映射键为 `videoID`，LVBench 为 `key`，MLVU 为原标注 `video` 完整字段。值相对 `--videos`，不能越出该目录。LVBench 可直接从官方 metadata 构建，无需重复准备 TSV；问题和选项沿用仓库既有 `parse_choice()` 的官方格式解析。

## 3. 构建全库存清单（CPU，不加载权重）

```bash
source scripts/env.sh

# 若标注尚为 Parquet，将 --annotation 改成真实的单个 Parquet 文件路径。
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

可选参数：`--videos /mnt/实际视频根目录 --video-map data/映射.json --out 新目录/questions.jsonl`。

构建时校验官方库存：Video-MME 2700 题/900 视频、三档各 900 题；MLVU 2174 题/1122 视频及七任务题数；LVBench 1549 个唯一 UID、合法选项及类别。每个原视频只探测一次真实解码时长，写入秒数 `duration`，**不把 short/medium/long 当秒数**。MLVU 原始时长另保留为 `official_duration`。产物是 `questions.jsonl` 与 `questions.meta.json`，含来源 SHA、分层题数及清单 SHA。

所有视频必须先到位，所以全库存探测耗时可能较长。仅有标注不能完成本步骤；这也是避免抽样后才发现缺视频的检查。缺视频应先补齐，不要按可下载性筛选样本。

## 4. 抽样和预算预检

```bash
python scripts/sample_bench_subset.py --bench videomme --seed 42 --n 150 \
  --out runs/screen/videomme/sample.jsonl
python scripts/sample_bench_subset.py --bench mlvu --seed 42 --per-task 20 \
  --out runs/screen/mlvu/sample.jsonl
python scripts/sample_bench_subset.py --bench lvbench --seed 42 --n 150 --proportional \
  --out runs/screen/lvbench/sample.jsonl

python scripts/check_bench_inputs.py --manifest runs/screen/videomme/sample.jsonl \
  --out runs/screen/videomme/input_audit.json --budget
python scripts/check_bench_inputs.py --manifest runs/screen/mlvu/sample.jsonl \
  --out runs/screen/mlvu/input_audit.json --budget
python scripts/check_bench_inputs.py --manifest runs/screen/lvbench/sample.jsonl \
  --out runs/screen/lvbench/input_audit.json --budget
```

`--manifest` 可覆盖默认完整清单位置。每份 `sample.meta.json` 包含 seed、源清单 SHA、人口、配额、余数并列顺序、选中 ID 列表、样本 SHA。问题按源题序输出，不改选项/标签，同视频可选部分题。所有输出拒绝覆盖；复现检查应输出到同一层级的另一个目录，再比较文件 SHA。相对路径编码保证整个仓库移动到服务器后仍可读取，但不同深度输出目录的相对路径不同，文件 SHA 可能不同，这是预期行为。

`check_bench_inputs.py --budget` 对样本全部视频执行真实 `prepare_record()`，保存 frames/grid/tokens/时间范围。省略 `--budget` 只做清单、路径、SHA 校验，无需导入模型框架。预算预检不验证 GPU 推理、长问题文本预算或模型加载。

旧 `evaluate_hf.py --dry-run` 若另作交叉检查，应使用 `--benchmark custom --manifest <样本>`。直接传官方 bench 名会要求全量库存和对应 `official.json`，也不能替代本预检；MLVU 原任务 JSON、LVBench 新文件名不会自动被旧入口识别。

## 5. 两模型推理及复现闸门

先用新抽样文件在独立 pilot 目录跑 20–30 题，VideoJev 通过后再跑正式 Qwen。不要用 `head` 截取 JSONL 却沿用原 `sample.meta.json`，SHA 校验会拒绝。Video-MME 的 pilot 可以通过 `--n 24` 生成三档各 8 题；MLVU 用 `--per-task 3` 生成 21 题。

```bash
python scripts/sample_bench_subset.py --bench videomme --seed 42 --n 24 \
  --out runs/pilot/videomme/sample.jsonl
```

以下正式命令按原实验 6 张 4090 放 Qwen、另 1 张放 VideoJev 演示。可依实际卡号改 `CUDA_VISIBLE_DEVICES`。顺序执行即可，不需要两模型并行。Qwen 入口支持 `--gpu-count 4/5/6`，尚未适配单张大显存 GPU 或其它架构后端。

```bash
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 依次将 BENCH 改为 videomme / mlvu / lvbench；首次可先使用 pilot 的路径。
BENCH=videomme
M=runs/screen/$BENCH/sample.jsonl

CUDA_VISIBLE_DEVICES=7 python scripts/eval_bench_videojev.py \
  --manifest "$M" --out "runs/screen/$BENCH/videojev" --checkpoint models/VideoJev-4B \
  --seed 42 --device cuda:0 --fps 2 --max-frames 2048 \
  --context 262144 --visual-token-budget 24576 --memory-fraction .9

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 python scripts/eval_bench_qwen36.py \
  --manifest "$M" --out "runs/screen/$BENCH/qwen" --checkpoint models/Qwen3.6-35B-A3B \
  --seed 42 --device cuda:0 --gpu-count 6 --max-memory-gib 18.5 \
  --fps 2 --max-frames 2048 --context 262144 --visual-token-budget 24576 --memory-fraction .9
```

原来有 LVB 样本和历史结果时，可以用新 VideoJev 入口在新目录重跑同一题集作复现闸门。原 LVB 模型脚本和媒体处理代码没有变，但新报告字段不同；比较答案和候选概率，不要求新旧 metrics JSON 逐字相同。

每个模型目录写 `run.json / predictions.jsonl / metrics.json / runtime.json / source/`。完成标准：预测覆盖全部 ID，概率有效且 answer_index 为 argmax、每题 preprocessing 非空；两模型 protocol 和逐视频 preprocessing 完全一致，才能融合。

中断恢复：原命令只增加 `--resume`，不改源码、清单、模型、媒体、设备协议或参数。新增兼容代码和原模型脚本的 SHA 纳入 settings 指纹，保留原断点校验。首次计算全部权重/媒体 SHA 可能很慢。不要把 Windows 上运行到一半的 `run.json` 当成能在 Linux 接着跑的断点：绝对路径、媒体时间戳和环境指纹会变；应在服务器新建输出目录。

## 6. CPU 融合、诊断与跨 bench 总表

```bash
BENCH=videomme
M=runs/screen/$BENCH/sample.jsonl

python scripts/eval_bench_controller.py --manifest "$M" \
  --qwen "runs/screen/$BENCH/qwen" --videojev "runs/screen/$BENCH/videojev" \
  --out "runs/screen/$BENCH/controller" --beta 2 --qwen-method qwen36

python scripts/summarize_bench.py --manifest "$M" \
  --qwen "runs/screen/$BENCH/qwen" --videojev "runs/screen/$BENCH/videojev" \
  --controller "runs/screen/$BENCH/controller" --out "runs/screen/$BENCH/summary" \
  --beta-sweep 1,2,3 --bootstrap 10000 --random-repeats 50 --seed 42

# 三个 bench 分别完成上面流程后：
python scripts/report_bench_screen.py --summaries \
  runs/screen/videomme/summary/results.json \
  runs/screen/mlvu/summary/results.json \
  runs/screen/lvbench/summary/results.json --out runs/screen/report
```

汇总不仅读取 metrics，还重验完整预测、run/预测/metrics 的来源 SHA、清单 SHA、两模型 seed 和固定协议、逐题预处理，以及 β=2 的融合公式。任何一份来源被改过都拒绝。随机选边模拟 50 次，同时给出其精确期望 `(Qwen指标 + VideoJev指标)/2`；该 50 次波动不是样本总体准确率的置信区间。

输出 `results.md / results.json` 包含三方法、随机选边、OR 参照、异答率、分层、β 扫描、bootstrap CI、配对纠正/损害、精确双侧 McNemar。MLVU 的统计、随机选边与 β 扫描均用 M-Avg。总表不会对三个不同分布/指标简单平均。

继续沿用计划的方向性阈值：Δ≥+2pp 且 β 诊断同向可列为 go；±2pp 内优先看分层并考虑 seed43；≤−2pp 如实记录。OR 参照空间<1pp 或异答率<5%时先排查实现，再考虑停止该 bench；由于第三选项存在，这些不是理论不可能性判据。

所有结果固定声明：**粗筛子集、非官方全量口径、未算力对齐、训练重叠未核实**。先从第一个 bench 的真实 `timing` 和 `runtime.json` 外推预算，不承诺服务器运行时长。

## 7. 本地验证与服务器验收

本次本地验证：**28 项新增测试通过**；加上 `test_lvb_experiment.py / test_lvb_controller.py / test_lvb_qwen36.py`，合计 **81 项通过**。同时通过新增 Python 文件语法和安装脚本 Bash 语法检查。新增测试包含完整合成标注库存、确定性抽样、MLVU 非等量宏平均、融合公式/来源校验、第三选项统计、断点恢复和代码指纹拒绝。这里的合成视频占位文件不用于证明真实解码或真实推理正确。

开发机为 Windows、Python 3.12.7、CPU Torch 2.12.1、Transformers 5.14.1、NumPy 1.26.4；不是服务器目标 CUDA 栈。临时 `.venv-bench-check` 和 runs 下测试产物已被忽略，不需要上传。

```bash
python -X utf8 -m pytest tests/test_bench_screen.py tests/test_lvb_experiment.py \
  tests/test_lvb_controller.py tests/test_lvb_qwen36.py -q
```

另行扩大到旧 LVB 比例抽样与汇总测试时，Windows 下出现旧代码的 GBK 默认读取及 CRLF 精确字节比较问题（原测试运行统计：133 通过、4 失败、10 fixture 错误）。核心组改用 `-X utf8` 后 81 项通过；原比例抽样脚本按文本模式写 CRLF、按 LF 重建后比较字节的问题仍保留，未为此修改原实验文件。服务器 Linux 应再跑相关比例测试；本地不能宣称所有旧测试全部通过。

本次没有正式数据、模型权重或 GPU，以下仍属于服务器待验收：Docker 镜像构建及 pip check、后端逐卡数学检查、三个 benchmark 官方标注/原视频全库存校验、真实预算审计、LVB 复现（有原样本时）、新 bench pilot、正式两模型推理及结果汇总。不要把代码测试通过写成正式实验已完成。
