# Video-MME 实验报告

## 结论

本报告整理 Video-MME 结果。每个抽题 seed 含 150 道题，long、medium、short 三个时长组各 50 题；seed 42、43、44 合计 450 题次。

Qwen 的独立基线是完整生成后解析的 `qwen_native`，三 seed 合计 **76.89%（346/450）**。Controller **旧方法（概率融合，β=2）为 80.00%（360/450）**，**新方法（log-linear 融合，β=2）为 80.22%（361/450）**。新方法相对旧方法净增 1 题，即 **+0.22 个百分点**；三个 seed 的变化分别为 +4、0、−3 题，样本方差从 **1.7778 pp²** 增至 **9.9259 pp²**，不能据此宣称稳定提升。

两种 Controller 的同协议 Qwen logits source 为 **79.78%（359/450）**，旧方法与新方法相对它分别为 **+0.22 pp**、**+0.44 pp**。两输入模型至少一个 top-1 答对的 OR 上界参照为 **85.33%（384/450）**。native Qwen 使用 4 FPS / 4096 帧，Controller 两源使用 2 FPS / 2048 帧；相对 native 的差异同时包含视频协议与答案取得方式的变化。

## 方法与协议

| 方法 | 输出方式 | 视频协议 | 目录 |
|---|---|---|---|
| Qwen native（主基线） | 完整自回归生成后解析最终选项 | 4 FPS / 4096 帧 | `qwen_native_4fps`, `qwen_native_43_4fps`, `qwen_native_44_4fps` |
| Qwen36 logits source | 首步候选字母 logits / 候选内 softmax，仅供 Controller 使用 | 2 FPS / 2048 帧 | `qwen_logits`, `qwen_logits_43`, `qwen_logits_44` |
| VideoJev | 候选概率 | 2 FPS / 2048 帧 | `videojev`, `videojev_43`, `videojev_44` |
| Controller（旧方法） | `softmax(z_Q + 2 * p_J)`，概率融合 | 双源均为 2 FPS / 2048 帧 | `controller`, `controller_43`, `controller_44` |
| Controller（新方法，β=2） | `softmax(z_Q + 2 * log(p_J))`，log-linear 融合 | 双源均为 2 FPS / 2048 帧 | `controller_loglinear`, `controller_loglinear_43`, `controller_loglinear_44` |
| Qwen logits OR VideoJev（上界参照） | 用真实标签判断任一输入模型的 top-1 是否正确 | 双源均为 2 FPS / 2048 帧 | `summary`, `summary_43`, `summary_44` 中的 `or_reference` |
| VideoJev 4 FPS | 候选概率 | 4 FPS / 4096 帧 | `videojev_4fps`, `videojev_43_4fps`, `videojev_44_4fps` |

Qwen native 使用 `temperature=0.7`、`top_p=0.8`、`top_k=20`、`presence_penalty=1.5`、`repetition_penalty=1`，完整生成后解析；三次运行均有 0 个无效回答。它是独立的 4 FPS 生成式基线，不能作为 Controller 的 logits 输入，也不能与 2 FPS logits 结果直接作严格因果比较。

本批新方法的运行记录明确为 `fusion=loglinear`、`beta=2.0`、`videojev_score_source=log_probabilities`，即使用已有候选概率的对数。这里记录实际运行配置，不将其写成另一次原始 VideoJev head logits 实验。

## 三 seed 主结果

| 方法 | seed 42 | seed 43 | seed 44 | 合计 450 题次 | seed 均值 | seed 样本方差（pp²） | seed 样本标准差（pp） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen native，4 FPS | 117/150 = 78.00% | 114/150 = 76.00% | 115/150 = 76.67% | 346/450 = 76.89% | 76.89% | 1.0370 | 1.02 |
| Qwen36 logits source，2 FPS | 117/150 = 78.00% | 123/150 = 82.00% | 119/150 = 79.33% | 359/450 = 79.78% | 79.78% | 4.1481 | 2.04 |
| VideoJev，2 FPS | 114/150 = 76.00% | 110/150 = 73.33% | 111/150 = 74.00% | 335/450 = 74.44% | 74.44% | 1.9259 | 1.39 |
| Controller（旧方法），β=2 | 120/150 = 80.00% | 122/150 = 81.33% | 118/150 = 78.67% | 360/450 = 80.00% | 80.00% | 1.7778 | 1.33 |
| Controller（新方法），β=2 | 124/150 = 82.67% | 122/150 = 81.33% | 115/150 = 76.67% | 361/450 = 80.22% | 80.22% | 9.9259 | 3.15 |
| Qwen logits OR VideoJev（上界参照） | 128/150 = 85.33% | 130/150 = 86.67% | 126/150 = 84.00% | 384/450 = 85.33% | 85.33% | 1.7778 | 1.33 |
| VideoJev，4 FPS | 112/150 = 74.67% | 109/150 = 72.67% | 106/150 = 70.67% | 327/450 = 72.67% | 72.67% | 4.0000 | 2.00 |

合计准确率按 450 题次的正确数计算。由于每个 seed 的题数相同，合计准确率等于 seed 准确率均值。方差使用三个 seed 的未四舍五入准确率计算：`s² = Σ(a_s − ā)² / (3 − 1)`，其中 `a_s = 100 × correct_s / 150`，即 **ddof=1 的样本方差，单位为百分点²（pp²）**；标准差为其平方根，单位为 pp。这里衡量的是不同抽题 seed 间的结果波动，不是固定题集上的生成采样方差。Qwen native 是主基线；Qwen logits 保留为 Controller 的同协议技术参照。

## 新旧方法配对汇总

| 指标 | seed 42 | seed 43 | seed 44 | 合计 |
|---|---:|---:|---:|---:|
| 新方法 − 旧方法：正确数 | +4 | 0 | −3 | **+1** |
| 新方法 − 旧方法：准确率 | +2.67 pp | 0.00 pp | −2.00 pp | **+0.22 pp** |
| 新方法 − Qwen logits source | +4.67 pp | −0.67 pp | −2.67 pp | **+0.44 pp** |
| 新方法纠正（Qwen 错、新方法对） | 10 | 6 | 6 | **22** |
| 新方法损害（Qwen 对、新方法错） | 3 | 7 | 10 | **20** |

新方法平均准确率略高，但跨 seed 样本标准差为 3.15 pp，高于旧方法的 1.33 pp。上述净变化和方差不等同于新旧方法差异的显著性检验；后文已有的置信区间、McNemar 检验及 beta 扫描均属于旧方法，不套用于新方法。

## OR 上界参照与互补性

截图中的 OR 数据已与本地三份 `summary*/results.json` 的 `methods.or_reference` 核对：seed 42、43、44 分别为 **128/150、130/150、126/150**，合计 **384/450 = 85.33%**，跨 seed 样本方差为 **1.7778 pp²**。

| 两输入模型的正确性组合 | 450 题次中的数量 |
|---|---:|
| 两者都对 | 310 |
| 仅 Qwen logits 对 | 49 |
| 仅 VideoJev 对 | 25 |
| 两者都错 | 66 |

上述合计满足：`310 + 49 = 359`（Qwen 正确数）、`310 + 25 = 335`（VideoJev 正确数）、`310 + 49 + 25 = 384`（OR 正确数）。这与用户提供的截图一致。OR 使用真实标签，是两个输入模型 top-1 的事后上界参照，不是可部署方法，也不是允许选择第三个选项的融合算法的严格数学上界。

## Controller 旧方法诊断（相对 logits source）

| 指标 | seed 42 | seed 43 | seed 44 | 合计 |
|---|---:|---:|---:|---:|
| Controller 旧方法 − Qwen logits source | +2.00 pp | -0.67 pp | -0.67 pp | **+0.22 pp** |
| Qwen / VideoJev 不同答案 | 30/150 | 36/150 | 26/150 | **92/450 = 20.44%** |
| 纠正（Qwen 错、旧方法对） | 5 | 3 | 5 | **13** |
| 损害（Qwen 对、旧方法错） | 2 | 4 | 6 | **12** |
| 两个基线都错但旧方法对 | 0 | 0 | 0 | **0** |
| 选择第三个选项 | 0 | 0 | 1 | **1** |

使用真实标签构造的 OR 参照（Qwen logits source 或 VideoJev 任一 top-1 正确）为 **384/450 = 85.33%**，比 logits source 高 5.56 pp。这说明两个 Controller 输入专家之间仍有可利用的互补空间，但旧方法固定融合只回收了其中很小的一部分。OR 是事后上界参照，不是可部署方法的成绩。Controller 旧方法相对主基线 Qwen native 的合计差值为 360-346 = 14 题，即 **+3.11 pp**，但该比较混合了协议和答案取得方式，不能作为严格的 paired Controller gain。

按 seed 的 OR 参照分别为 85.33%、86.67%、84.00%。Controller 旧方法的 paired bootstrap 95% CI（相对 Qwen）分别为 `[-1.33, +5.33]` pp、`[-4.00, +2.67]` pp、`[-4.67, +3.33]` pp；均包含 0。McNemar 精确双侧 p 值分别为 0.453125、1.0、1.0。

## 时长分层

下表合并三个 seed，每组 150 道题。

| 时长组 | Qwen native 4 FPS（主基线） | Qwen logits 2 FPS source | VideoJev 2 FPS | Controller 旧方法 2 FPS | VideoJev 4 FPS |
|---|---:|---:|---:|---:|---:|
| long | 99/150 = 66.00% | 104/150 = 69.33% | 95/150 = 63.33% | **105/150 = 70.00%** | 89/150 = 59.33% |
| medium | 118/150 = 78.67% | 123/150 = 82.00% | 118/150 = 78.67% | **124/150 = 82.67%** | 113/150 = 75.33% |
| short | 129/150 = 86.00% | **132/150 = 88.00%** | 122/150 = 81.33% | 131/150 = 87.33% | 125/150 = 83.33% |

相对 logits source，Controller 旧方法的净收益主要出现在 seed 42；seed 43 只在 long 组高于 logits source，seed 44 在 long 组反而低于 logits source。long 组不是简单的“VideoJev 更强”：三 seed 合并后 Qwen native 主基线为 66.00%，Qwen logits source 为 69.33%，VideoJev 为 63.33%，Controller 旧方法为 70.00%。

## 旧方法 beta 诊断

旧概率融合公式下，固定 beta 的三 seed 合计诊断结果如下。这里的 beta 扫描使用同一批结果，仅用于描述敏感性，不能视为独立测试集选择。

| beta | 合计正确数 | 准确率 | 相对 Qwen logits source |
|---:|---:|---:|---:|
| 1 | 362/450 | 80.44% | +0.67 pp |
| 2 | 360/450 | 80.00% | +0.22 pp |
| 3 | 363/450 | 80.67% | +0.89 pp |

不同 seed 的最优 beta 不一致：seed 42 为 beta=3，seed 43 为 beta=3，seed 44 为 beta=1。不能据此把 beta=3 宣称为最终改进；需要独立开发集校准后，再在未见过的测试集上锁定参数。

## 数据完整性与限制

- 每个 seed 的 sample meta 均为 150 题，long/medium/short 各 50 题；原始 Video-MME source SHA256 均为 `52b3416e...d94bea0`。
- sample SHA256 分别为 seed 42 `faec01f0...d5254a1`、seed 43 `b6a3cacb...15cf62`、seed 44 `eaae3cc1...dda1209`。
- `qwen_native_*` 的实际输入 manifest SHA 与 Video-MME sample 一致，但其历史 `run.json` protocol name 仍为 `lvb_equal_duration_rgb_subset_v1`，metrics 中 benchmark 字段也未写入 `videomme`，split 误标为 `validation_subset`。其 FPS、帧数、样本 ID 和预测结果与 Video-MME 清单相符，因此本报告按核验后的清单归类。已修复跨 bench native 入口的元数据与分组计分：新运行使用 `video_bench_screen_rgb_v1`、实际 benchmark 和 `screening_subset`；生成与解析算法不变。本报告的历史产物保持原样，标签修复不改变这里重算的正确数与准确率。
- 4 FPS 与 2 FPS 的独立基线不是同协议比较；native Qwen 还改变了答案取得方式，不能用于证明 4 FPS 本身带来的收益。
- Controller 旧方法相对主基线 Qwen native 的 +3.11 pp、新方法的 +3.33 pp 均混合了协议和解码方式。相对同协议 logits source，旧方法为 +0.22 pp、新方法为 +0.44 pp；新方法相对旧方法仅净增 1 题，且跨 seed 方差更大，不能据此声称稳定或统计显著提升。
- 本次新增数据来自用户提供的两张截图，并核对本地 `D:/cmder/runs` 下的 `controller_loglinear{,_43,_44}/metrics.json`、`run.json`、`predictions.jsonl` 和 `summary{,_43,_44}/results.json`。新方法三个正确数为 124、122、115，OR 为 128、130、126，与截图一致。
- 已核验每个 seed 的新旧 Controller 与两模型来源具有相同 manifest SHA、输入协议及 150 个题目 ID；新方法逐题保留的 Qwen logits、VideoJev probabilities 与对应来源一致，融合 logits 满足 `z_Q + 2 * videojev_scores`，预测为融合分数 argmax。本次未重新运行模型；OR 与正确数使用已保存的评测汇总，未重新加载完整官方标签重算。
