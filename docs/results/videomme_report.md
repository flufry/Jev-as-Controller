# Video-MME 实验报告

## 结论

本报告整理 Video-MME 结果。每个抽题 seed 含 150 道题，long、medium、short 三个时长组各 50 题；seed 42、43、44 合计 450 道题。

Qwen 的独立基线是完整生成后解析的 `qwen_native`，三 seed 合计 **76.89%（346/450）**；Jev-as-Controller 为 **80.00%（360/450）**，表面差值为 **+3.11 个百分点**。但两者协议和答案取得方式不同：native Qwen 是 4 FPS / 4096 帧的完整生成，Controller 使用 2 FPS / 2048 帧的 Qwen 候选 logits 与 VideoJev 概率。Controller 内部的同协议参照是 Qwen logits source（79.78%），相对仅 **+0.22 个百分点**，三个 seed 的方向也不一致。

## 方法与协议

| 方法 | 输出方式 | 视频协议 | 目录 |
|---|---|---|---|
| Qwen native（主基线） | 完整自回归生成后解析最终选项 | 4 FPS / 4096 帧 | `qwen_native_4fps`, `qwen_native_43_4fps`, `qwen_native_44_4fps` |
| Qwen36 logits source | 首步候选字母 logits / 候选内 softmax，仅供 Controller 使用 | 2 FPS / 2048 帧 | `qwen_logits`, `qwen_logits_43`, `qwen_logits_44` |
| VideoJev | 候选概率 | 2 FPS / 2048 帧 | `videojev`, `videojev_43`, `videojev_44` |
| Jev-as-Controller | `softmax(Qwen logits + beta * VideoJev probabilities)`，beta=2 | 双源均为 2 FPS / 2048 帧 | `controller`, `controller_43`, `controller_44` |
| VideoJev 4 FPS | 候选概率 | 4 FPS / 4096 帧 | `videojev_4fps`, `videojev_43_4fps`, `videojev_44_4fps` |

Qwen native 使用 `temperature=0.7`、`top_p=0.8`、`top_k=20`、`presence_penalty=1.5`、`repetition_penalty=1`，完整生成后解析；三次运行均有 0 个无效回答。它是独立的 4 FPS 生成式基线，不能作为 Controller 的 logits 输入，也不能与 2 FPS logits 结果直接作严格因果比较。

## 三 seed 主结果

| 方法 | seed 42 | seed 43 | seed 44 | 合计 450 题 | seed 均值 | seed 样本标准差 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen native，4 FPS | 117/150 = 78.00% | 114/150 = 76.00% | 115/150 = 76.67% | 346/450 = 76.89% | 76.89% | 1.02 pp |
| Qwen36 logits source，2 FPS | 117/150 = 78.00% | 123/150 = 82.00% | 119/150 = 79.33% | 359/450 = 79.78% | 79.78% | 2.04 pp |
| VideoJev，2 FPS | 114/150 = 76.00% | 110/150 = 73.33% | 111/150 = 74.00% | 335/450 = 74.44% | 74.44% | 1.39 pp |
| Jev-as-Controller，beta=2 | 120/150 = 80.00% | 122/150 = 81.33% | 118/150 = 78.67% | **360/450 = 80.00%** | 80.00% | 1.33 pp |
| VideoJev，4 FPS | 112/150 = 74.67% | 109/150 = 72.67% | 106/150 = 70.67% | 327/450 = 72.67% | 72.67% | 2.00 pp |

合计准确率按 450 道题的正确数计算；由于三个 seed 的题集不同，报告同时保留 seed 均值和 seed 间样本标准差。Qwen native 是主基线；Qwen logits 仅是 Controller 的技术输入，不能替代主基线。

## Controller 诊断（相对 logits source）

| 指标 | seed 42 | seed 43 | seed 44 | 合计 |
|---|---:|---:|---:|---:|
| Controller - Qwen logits source | +2.00 pp | -0.67 pp | -0.67 pp | **+0.22 pp** |
| Qwen / VideoJev 不同答案 | 30/150 | 36/150 | 26/150 | **92/450 = 20.44%** |
| 纠正（Qwen 错、Controller 对） | 5 | 3 | 5 | **13** |
| 损害（Qwen 对、Controller 错） | 2 | 4 | 6 | **12** |
| 两个基线都错但 Controller 对 | 0 | 0 | 0 | **0** |
| 选择第三个选项 | 0 | 0 | 1 | **1** |

使用真实标签构造的 OR 参照（Qwen logits source 或 VideoJev 任一 top-1 正确）为 **384/450 = 85.33%**，比 logits source 高 5.56 pp。这说明两个 Controller 输入专家之间仍有可利用的互补空间，但当前固定融合只回收了其中很小的一部分。OR 是事后上界参照，不是可部署方法的成绩。Controller 相对主基线 Qwen native 的合计差值为 360-346 = 14 题，即 **+3.11 pp**，但该比较混合了协议和答案取得方式，不能作为严格的 paired Controller gain。

按 seed 的 OR 参照分别为 85.33%、86.67%、84.00%。Controller 的 paired bootstrap 95% CI（相对 Qwen）分别为 `[-1.33, +5.33]` pp、`[-4.00, +2.67]` pp、`[-4.67, +3.33]` pp；均包含 0。McNemar 精确双侧 p 值分别为 0.453125、1.0、1.0。

## 时长分层

下表合并三个 seed，每组 150 道题。

| 时长组 | Qwen native 4 FPS（主基线） | Qwen logits 2 FPS source | VideoJev 2 FPS | Controller 2 FPS | VideoJev 4 FPS |
|---|---:|---:|---:|---:|---:|
| long | 99/150 = 66.00% | 104/150 = 69.33% | 95/150 = 63.33% | **105/150 = 70.00%** | 89/150 = 59.33% |
| medium | 118/150 = 78.67% | 123/150 = 82.00% | 118/150 = 78.67% | **124/150 = 82.67%** | 113/150 = 75.33% |
| short | 129/150 = 86.00% | **132/150 = 88.00%** | 122/150 = 81.33% | 131/150 = 87.33% | 125/150 = 83.33% |

相对 logits source，Controller 的净收益主要出现在 seed 42；seed 43 只在 long 组高于 logits source，seed 44 在 long 组反而低于 logits source。long 组不是简单的“VideoJev 更强”：三 seed 合并后 Qwen native 主基线为 66.00%，Qwen logits source 为 69.33%，VideoJev 为 63.33%，Controller 为 70.00%。

## beta 诊断

固定 beta 的三 seed 合计诊断结果如下。这里的 beta 扫描使用同一批结果，仅用于描述敏感性，不能视为独立测试集选择。

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
- Controller 相对主基线 Qwen native 的 +3.11 pp 混合了协议和解码方式，Controller 相对同协议 logits source 的 +0.22 pp 远小于当前不确定性，不能声称统计显著提升。
