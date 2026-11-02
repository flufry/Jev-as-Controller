# 13 · VideoJev 原始 logits 与 log-linear 融合

新增独立接口，公式为：

```text
z0 = Qwen 首步候选字母 logits
Fs = VideoJev head 的原始候选分数（softmax 前）
ps = softmax(Fs)
旧：pi1 = softmax(z0 + beta * ps)
新：pi1 = softmax(z0 + beta * Fs)
等效：pi1_i ∝ pi0_i * ps_i**beta
```

新方法标识为 `jev_as_controller_loglinear`。原 `eval_bench_controller.py` 保持概率融合；不要把新方法的结果目录覆盖到旧 Controller 目录。β=2 只作为本次固定起点，两套分数尺度不同，不保证新方法更好。

## 1. 新增 VideoJev 原始分数接口

在服务器容器内、仓库根目录执行。视频采样、分组、checkpoint 和 seed 应与 Qwen logits 输入完全一致：

```bash
BENCH=videomme
M=runs/screen/$BENCH/sample_42.jsonl

CUDA_VISIBLE_DEVICES=7 python scripts/eval_bench_videojev_logits.py \
  --manifest "$M" --out "runs/screen/$BENCH/videojev_logits" \
  --checkpoint models/VideoJev-4B --seed 42 --device cuda:0 \
  --fps 2 --max-frames 2048 --context 262144 \
  --visual-token-budget 24576 --memory-fraction .9
```

新增预测字段 `candidate_logits` 保存真实 head logits，同时保留原 `probabilities`、选项、preprocessing 等。method 仍为 `videojev`，`settings.candidate_scores=head_logits_before_candidate_softmax` 区分输出类型。保存前和恢复时检查 `softmax(candidate_logits)` 与概率一致。新增适配器源码 SHA 纳入指纹并保存源码副本；首次使用新输出目录，之后恢复可原命令加 `--resume`。

这个入口直接调用已有模型 forward 返回的 `output.logits`，不修改 head、不从概率猜测分数。它使用当前仓库的输入分组/格式化适配器，因此已经下载的旧 VideoJev checkpoint 也可使用，**不需要手动覆盖模型目录的 Python 文件或重新导出权重**。

HF 源码接口也增加可选 `return_logits=True`：

```python
answers = model.predict_features(processor, features, record, questions,
    context=262144, return_logits=True)
# 或 model.predict_video(..., return_logits=True)
```

该可选参数需要加载含本次源码的新导出模型；已有旧模型用上面的评测入口即可。默认 False 保持原预测格式，选择题只导出有效选项分数，不包含 padding；grounding 逻辑不变。

## 2. 用真实原始分数融合（CPU 即可）

Qwen 仍使用 `eval_bench_qwen36_logits.py` 的结果，不能使用 native 文本生成结果：

```bash
python scripts/eval_bench_controller_loglinear.py \
  --manifest "$M" \
  --qwen "runs/screen/$BENCH/qwen_logits" \
  --videojev "runs/screen/$BENCH/videojev_logits" \
  --score-source raw_logits --beta 2 \
  --out "runs/screen/$BENCH/controller_loglinear"
```

默认 `--score-source raw_logits`，缺少原始分数时明确报错，不自动切换算法。融合接口可对同协议的2 FPS或4 FPS结果运行，不强行要求2 FPS，但两输入的完整协议、清单 SHA、seed、逐题预处理必须一致。清单中的媒体路径在本地无需存在，可分析复制下来的结果。

输出 `run.json / metrics.json / comparison.json / predictions.jsonl / source/`。逐题记录 `videojev_scores`、`videojev_score_source`、`fused_logits`、两专家概率、融合概率和最终选项；metrics 包含相对 Qwen 的改对/改错/改答，MLVU 用七任务 M-Avg，LVBench 保留多标签类别统计。记录来源 SHA 和实际 β，拒绝覆盖已有输出。

原 `summarize_bench.py`、`analyze_controller_changes.py` 目前专用于旧 β·ps 融合，会拒绝新方法。新方法先查看自己的 `metrics.json / comparison.json` 和逐题预测，不要混用旧汇总脚本。

## 3. 复用已经保存的 VideoJev 概率

若 `ps=softmax(Fs)` 且全部 ps>0，则 `log(ps)=Fs-logsumexp(Fs)`。后项对同一道题所有选项相同，因此：

```text
softmax(z0 + beta * log(ps)) = softmax(z0 + beta * Fs)
```

这使现有结果也能验证新融合，**无需重跑 VideoJev 或 Qwen**。显式指定 `--score-source log_probabilities`；记录将标为归一化分数，不冒充原始 head logits。保存概率的浮点舍入可能带来微小数值误差。若任何候选概率为0则拒绝，不使用 epsilon 裁剪替代；应运行第1节重新导出原始分数。

当前本地已有结果，PowerShell 批量三个 seed：

```powershell
foreach ($s in 42,43,44) {
    $suffix = if ($s -eq 42) { '' } else { "_$s" }
    python -X utf8 scripts/eval_bench_controller_loglinear.py `
      --manifest "D:/cmder/runs/sample_$s.jsonl" `
      --qwen "D:/cmder/runs/qwen_logits$suffix" `
      --videojev "D:/cmder/runs/videojev$suffix" `
      --score-source log_probabilities --beta 2 `
      --out "runs/analysis/videomme_loglinear_beta2_seed$s"
}
```

上面输出目录在本次验证中已生成；重复执行请改新目录名。本次仅固定 β=2，没有按子集测试结果选 β：

| seed | Qwen logits | 旧 β·ps 融合 | 新 β·Fs 等效融合 | 新方法相对Qwen改对/改错 |
|---|---:|---:|---:|---|
| 42 | 117/150 | 120/150 | 124/150 | 10/3 |
| 43 | 123/150 | 122/150 | 122/150 | 6/7 |
| 44 | 119/150 | 118/150 | 115/150 | 6/10 |
| 合计题次 | 359/450（79.78%） | 360/450（80.00%） | 361/450（80.22%） | 22/20 |

新方法比旧方法合计只多对1题，seed44反而下降，尚不足以说明稳定提升；不同seed包含重复题，450为题次。新公式消除了概率修正幅度的上限，但也可能放大错误且自信的VideoJev分数，β和分数校准仍需独立开发集确定。

## 4. 验证范围

CPU回归覆盖原始分数导出和padding排除、旧checkpoint forward适配器、原始/归一化分数等效性、公共常数不变性、β=0、概率/logits一致性、零概率拒绝、清单与预处理校验以及旧融合流程。真实GPU评测接口仍需在服务器做小样本验收；本地三seed新方法结果来自已保存概率的等效复算。
