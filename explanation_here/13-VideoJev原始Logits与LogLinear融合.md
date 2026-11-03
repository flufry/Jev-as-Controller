# 13 · 对齐官方：VideoJev 原始 logits 与两种 Controller 融合

已对齐官方 main 提交 `3a26780` 的融合实现，统一由 `eval_bench_controller.py` 调用。MLVU 和 LVBench 可以使用同一套命令。

## 当前入口与分数定义

| 入口 | 当前行为 |
|---|---|
| `eval_bench_videojev.py` | 保存 softmax 前的 `logits`，直接 argmax 计分，不保存归一化概率 |
| `eval_bench_qwen36_logits.py` | 保存 Qwen 首步裸候选字母 `candidate_logits`，作为 Controller 输入 |
| `eval_bench_controller.py` | 默认 `--fusion loglinear --beta 2` |
| `eval_bench_qwen36.py` / `eval_bench_qwen36_native.py` | 完整生成后解析；是独立基线，不是 Controller 的 logits 来源 |

`eval_bench_videojev_logits.py` 和 `eval_bench_controller_loglinear.py` 保留为兼容入口，共享主实现；不用再维护两套不同的融合代码。

统一记号：`z_Q` 是 Qwen 原始候选 logits，`s_J` 是 VideoJev 原始候选 logits，`p_J=softmax(s_J)`。

```text
默认新方法：pi = softmax(z_Q + 2 * s_J)           --fusion loglinear
显式旧方法：pi = softmax(z_Q + 2 * softmax(s_J))  --fusion probability
```

旧方法在 CPU 融合时单独计算 VideoJev 的 softmax；只需一次 Jev GPU 推理即可评测两种公式。Jev 单模型用 `argmax(s_J)`，与理想精度下 `argmax(softmax(s_J))` 相同。新方法不会把概率当原始 logits 加入 Qwen。

新预测字段名采用官方的 `logits`。读取端兼容之前新增接口保存的 `candidate_logits`。旧结果只有概率时，官方的兼容规则是：各概率严格大于0才可用 `log(p_J)` 恢复等效分差；含0时拒绝 log-linear，不裁剪。可传 `--score-source raw_logits` 强制要求真实原始分数。

已有旧 checkpoint 可直接使用，不需替换 `models/VideoJev-4B` 中的源码或重下权重。跨 bench 入口直接读取已有模型 forward 的原始 head 输出。模型内部可能仍生成概率，但本入口不读取它、不把它作为评测或默认融合分数。

## MLVU / LVBench 完整测试命令

以下在服务器容器内、仓库根目录 Bash 执行。数据下载和完整清单构建沿用 05/09；如果已有 `questions.jsonl`，不要重复构建覆盖。

```bash
python scripts/build_bench_manifest.py --bench mlvu --base data/benchmarks/mlvu
python scripts/build_bench_manifest.py --bench lvbench --base data/benchmarks/lvbench \
  --annotation data/benchmarks/lvbench/video_info.meta.jsonl
```

下面先测 MLVU；把 BENCH 改为 lvbench 即可测另一数据集。每个实验 seed 分别抽取对应题集并运行；如果要复用同一题集做多次随机推理，也可以只抽一次，然后改变评测命令的 `--seed`。抽题 seed 与推理 seed 不要求相同。GPU 与每卡预算按实际空闲资源修改。首次使用全新 RUN_ROOT；每次运行使用独立目录，结果不会覆盖。

```bash
set -euo pipefail
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

BENCH=mlvu
RUN_ROOT=runs/screen/$BENCH/aligned_seed_runs_v3
QWEN_GPUS=0,1,3,4
QWEN_MEMORY=40,20,40,40
JEV_GPU=7
for SEED in 42 43 44; do
  SAMPLE_SEED=$SEED
  M=$RUN_ROOT/sample$SAMPLE_SEED.jsonl
  R=$RUN_ROOT/sample$SAMPLE_SEED/inference$SEED

  if [ "$BENCH" = mlvu ]; then
    python scripts/sample_bench_subset.py --bench mlvu --seed "$SAMPLE_SEED" \
      --per-task 20 --out "$M"
  else
    python scripts/sample_bench_subset.py --bench lvbench --seed "$SAMPLE_SEED" \
      --n 150 --proportional --out "$M"
  fi
  python scripts/check_bench_inputs.py --manifest "$M" --budget --out "$RUN_ROOT/sample$SAMPLE_SEED/input_audit.json"

  CUDA_VISIBLE_DEVICES=$JEV_GPU python scripts/eval_bench_videojev.py \
    --manifest "$M" --out "$R/videojev_logits" --checkpoint models/VideoJev-4B \
    --seed "$SEED" --device cuda:0 --fps 2 --max-frames 2048 \
    --context 262144 --visual-token-budget 24576 --memory-fraction .9

  CUDA_VISIBLE_DEVICES=$QWEN_GPUS python scripts/eval_bench_qwen36_logits.py \
    --manifest "$M" --out "$R/qwen_logits" --checkpoint models/Qwen3.6-35B-A3B \
    --seed "$SEED" --device cuda:0 --gpu-count 4 \
    --max-memory-gib-per-gpu "$QWEN_MEMORY" \
    --fps 2 --max-frames 2048 --context 262144 \
    --visual-token-budget 24576 --memory-fraction .9

  # CPU：两种公式复用完全相同的模型结果，各自保存结果和诊断。
  for FUSION in loglinear probability; do
    python scripts/eval_bench_controller.py --manifest "$M" \
      --qwen "$R/qwen_logits" --videojev "$R/videojev_logits" \
      --fusion "$FUSION" --beta 2 --out "$R/controller_$FUSION"

    python scripts/summarize_bench.py --manifest "$M" \
      --qwen "$R/qwen_logits" --videojev "$R/videojev_logits" \
      --controller "$R/controller_$FUSION" --out "$R/summary_$FUSION" \
      --beta-sweep 1,2,3 --bootstrap 10000 --random-repeats 50 --seed "$SEED"
  done
done
```

需要不同抽题题集时，为每个抽题 seed 生成一份 manifest，并分别运行 Qwen、VideoJev 和 Controller；同一个 Controller 的两路模型必须使用同一题集和推理 seed。也可固定抽题 seed，单独更改模型命令的 `--seed` 以比较推理随机性。代码不要求抽题 seed 与推理 seed 相同。已有 Qwen logits 可复用。新 Jev 输出统一保存 raw logits；不要将 native Qwen 文本生成目录传给融合。

中断恢复时，不要从头重跑上面的整个循环：用当前 seed 的同一条模型评测命令增加 `--resume`；抽样、预算预检、CPU 融合及汇总都拒绝覆盖已有文件。此次源码与输出格式均已变化，旧模型目录不能强制 `--resume`，应另选新目录。

## 独立 4 FPS 基线

如果要延续既有报告的独立基线，另跑 Qwen native 和 VideoJev 4 FPS，不替换上面的两条 2 FPS Controller 输入。此示例固定使用抽题 seed42 的清单，若文件尚不存在，先用抽样命令生成；GPU 变量沿用上文：

```bash
M=$RUN_ROOT/sample42.jsonl
for SEED in 42 43 44; do
  R=$RUN_ROOT/seed$SEED
  CUDA_VISIBLE_DEVICES=$QWEN_GPUS python scripts/eval_bench_qwen36_native.py \
    --manifest "$M" --out "$R/qwen_native_4fps" --checkpoint models/Qwen3.6-35B-A3B \
    --seed "$SEED" --device cuda:0 --gpu-count 4 --max-memory-gib-per-gpu "$QWEN_MEMORY" \
    --fps 4 --max-frames 4096 --context 262144 --visual-token-budget 24576 --memory-fraction .9

  CUDA_VISIBLE_DEVICES=$JEV_GPU python scripts/eval_bench_videojev.py \
    --manifest "$M" --out "$R/videojev_4fps" --checkpoint models/VideoJev-4B \
    --seed "$SEED" --device cuda:0 --fps 4 --max-frames 4096 \
    --context 262144 --visual-token-budget 24576 --memory-fraction .9
done
```

原生 Qwen 的完整生成、解析和默认 temperature/top-p/top-k/presence penalty 沿用现有实现。三个独立基线运行仍读取同一份 M，而不是重新抽题。

## 结果与核验

每个模型/Controller 目录保存 `run.json / predictions.jsonl / metrics.json / source/`。Jev 逐题为 `logits` 与 `answer_index`；Controller 保存实际 `fusion`、β、`videojev_scores`、融合前后概率、`fused_logits` 和来源 SHA。无 `fusion` 的历史 Controller 仅按旧 probability 公式解释。

`summarize_bench.py` 已支持两种融合并按实际公式重算，β扫描跟随对应模式；MLVU 主指标是七任务 M-Avg，LVBench 同时报重叠类别。`analyze_controller_changes.py` 支持两种模式，逐选项同时列出 Jev 概率和实际参与融合的分数；`report_bench_screen.py` 显示模式并拒绝混合两种模式。

```bash
python scripts/analyze_controller_changes.py --manifest "$M" \
  --qwen "$R/qwen_logits" --videojev "$R/videojev_logits" \
  --controller "$R/controller_loglinear" --out "$R/changes_loglinear"
```

OR 诊断仍针对 Controller 的两条同协议输入。它是只在两个 top-1 间选边的 router 上限，不是可选第三个选项的完整分布融合的严格上限。这组三次实验固定题集，差异来自推理 seed；Qwen logits 和 Jev argmax 是确定性取分，三次可能相同，不能把这种重复解释为三份独立抽题证据。Qwen native 使用逐题生成 seed，才会产生真正的回答采样波动。

所有 GPU 输入协议、清单 SHA、两模型 seed、逐题选项/预处理、分数与来源指纹均校验。新旧公式分别评测和记录，不覆盖旧结果；完整 GPU 推理需在服务器验收。
