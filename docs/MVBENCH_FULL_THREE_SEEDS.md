# MVBench 全量任务三 seed 对照

本入口接入通用全量队列，默认顺序：Qwen logits 2FPS → Qwen native 4FPS seed42/43/44 → VideoJev 2FPS → VideoJev 4FPS → 三seed汇总与六行总表。首次校验输入内容SHA，后续阶段及恢复复用大文件缓存；默认不并行。

## 两种覆盖范围

- `--mvbench-scope public19`（默认）：19个公开任务，每任务200题，共3800题，明确标为“MVBench 19个公开任务”。不含fine_grained_pose，不称为4000题官方全量。
- `--mvbench-scope full20`：20任务，每任务200题，共4000题。需要自行准备有授权的fine_grained_pose视频及标注，缺任务或缺素材直接报错。

全量准备不接受任意任务子集。标注目录即使包含第20任务，public19也只读取指定19个任务文件，但源manifest必须正好3800题；不要用4000题清单配public19或反过来。每份任务JSON的全部题目、原选项顺序和标签均校验。

沿用已有MVBench适配：有时间边界的视频使用已裁剪MP4，episodic_reasoning使用已按边界物化的3FPS帧序列视频。全量准备会核对clip_start、clip_end、media_transform、source_video和派生视频路径。不会重新裁剪或把整部原片当作片段输入。提示词及2/4FPS模型采样协议属于本项目，不等同于官方VideoChat2评测协议。

## 公开19任务：从已有数据开始

以下命令在服务器容器内仓库根目录执行。数据下载和素材转换见 `explanation_here/15-MVBench公开任务完整评测.md`；已有完整数据时不要重复下载。

```bash
cd /workspace/Jev-as-Controller
export PATH="/opt/jev/bin:$PATH"
export XTUNER_ROOT="$PWD/.deps-xtuner"
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
```

如尚未建立3800题源清单，先执行（已有文件则跳过，不覆盖）：

```bash
python scripts/build_bench_manifest.py --bench mvbench \
  --base data/benchmarks/mvbench \
  --annotation data/benchmarks/mvbench/json \
  --out data/benchmarks/mvbench/questions_public19.jsonl
```

该构建器会读取目录中的全部任务JSON；若目录还包含fine_grained_pose而你只准备了19任务，请为这一步指定仅含19个任务标注的目录，不能用缺视频的20任务清单冒充19任务。

准备带官方题目逐项校验和来源SHA的固定全量清单：

```bash
python scripts/prepare_bench_full.py --bench mvbench \
  --source-manifest data/benchmarks/mvbench/questions_public19.jsonl \
  --reference data/benchmarks/mvbench/json \
  --mvbench-scope public19 \
  --out runs/full/mvbench_public19/questions.jsonl
```

检查计划，然后正式运行（按实际GPU修改）：

```bash
COMMON=(--qwen-gpus 0,1,2,3,4,5 --videojev-gpu 7 \
  --qwen-python /opt/jev/bin/python --videojev-python /opt/jev/bin/python)

python scripts/run_bench_full.py --bench mvbench \
  --base runs/full/mvbench_public19 "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench mvbench \
  --base runs/full/mvbench_public19 "${COMMON[@]}"
```

默认每卡权重预算18.5GiB；若用4/5卡或非对称显存，按实际情况增加`--max-memory-gib-per-gpu`。可选`--parallel-models`须保证VideoJev GPU不在Qwen列表中；默认先Qwen后VideoJev。

## 完整20任务

已准备好全部20任务素材后，构建`questions_full20.jsonl`并改用：

```bash
python scripts/prepare_bench_full.py --bench mvbench \
  --source-manifest data/benchmarks/mvbench/questions_full20.jsonl \
  --reference data/benchmarks/mvbench/json \
  --mvbench-scope full20 \
  --out runs/full/mvbench_full20/questions.jsonl

python scripts/run_bench_full.py --bench mvbench \
  --base runs/full/mvbench_full20 "${COMMON[@]}"
```

19与20任务必须使用不同输出目录；scope固化在全量清单，不通过runner偷偷切换。

## 报告、采样与恢复

结果在 `runs/full/mvbench_public19/summary/results.md` 和 `results.json`，或相应full20目录。

| 行 | 输入/取得答案 |
|---|---|
| Qwen native | 4FPS/4096帧，三个seed各实际完整生成一次 |
| Qwen单步 | 2FPS/2048帧，一次候选logits来源，三个seed从候选softmax采样 |
| VideoJev-4B | 4FPS/4096帧，一次分数来源，三个seed采样 |
| 旧概率Controller | 双2FPS，softmax(z_Q+2*p_J)，三个seed采样 |
| log-linear Controller | 双2FPS，softmax(z_Q+2*s_J)，三个seed采样 |
| 单步Qwen OR VideoJev | 同seed的单步Qwen2FPS采样与VideoJev4FPS采样任一正确；使用标签的事后参照 |

候选采样T=1，无top-k/top-p，两版Controller共用随机数序列。固定相同全量题集，seed不抽题。每任务200题，所以总体Accuracy等于任务准确率的宏平均；同时保存`task_average_accuracy`、逐任务正确数、coverage和完整预测。native无效/截断计错，分母不变。总表使用三次未舍入准确率计算均值、ddof=1样本方差（pp²）。

恢复时源码、清单、模型和参数保持一致：

```bash
python scripts/run_bench_full.py --bench mvbench \
  --base runs/full/mvbench_public19 "${COMMON[@]}" --resume
```

更新代码会改变原实验的源码指纹。此新增功能仅用于后续新实验；不要同步到仍在运行的旧实验目录后强行resume。没有真实GPU测验前，不应将CPU测试通过描述为已经完成MVBench全量实验。
