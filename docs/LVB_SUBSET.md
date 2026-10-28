# LongVideoBench 100 题对照实验

从本地完整 validation 的官方 `duration_group=15/60/600/3600` 各抽 25 道题，默认 seed=42。
抽样单位为问题，同一视频可以只抽中部分问题；保留抽中问题的原始顺序、候选顺序和标签。
这不是完整官方 1337 题的评估结果，也不按原始四组占比加权。

在仓库根目录执行，复用现有 `uv` 环境与本地权重：

```bash
source .venv/bin/activate
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/sample_longvideobench.py \
  --out runs/lvb100_equal_seed42/sample.jsonl --per-group 25 --seed 42

CUDA_VISIBLE_DEVICES=1 python scripts/eval_longvideobench_qwen.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl \
  --checkpoint models/Qwen3.5-4B --out runs/lvb100_equal_seed42/qwen

CUDA_VISIBLE_DEVICES=2 python scripts/eval_longvideobench_videojev.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl \
  --checkpoint models/VideoJev-4B --out runs/lvb100_equal_seed42/videojev

python scripts/eval_longvideobench_controller.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl \
  --qwen runs/lvb100_equal_seed42/qwen \
  --videojev runs/lvb100_equal_seed42/videojev \
  --out runs/lvb100_equal_seed42/controller --beta 2
```

采样和控制器拒绝覆盖已有输出。两条基线命令可以在不同空闲 GPU 上并行运行；
基线异常中断后，在原命令末尾加 `--resume`，只有样本、媒体内容、权重、代码、依赖和设置指纹完全匹配时才恢复。
基线逐题保存 `predictions.jsonl`，完整覆盖样本后才写 `metrics.json`。
每个运行目录保存来源、配置、代码快照及权重 SHA256；控制器的 `comparison.json` 汇总三种方法的总体和分组准确率。

默认配置为 **2 FPS、最多 2048 帧、文本上下文 262144、视觉 token 上限 24576**，不读取字幕或音频。
两种基线均匀采样完整视频，使用相同的原始帧时间线及空间网格；超过帧数上限时覆盖完整时长。
空间网格在视觉预算内取最大原生网格，每帧面积至多 409600 像素。
视觉预算小于论文正式评估的 `262144 × 80%`，用于 24GB 显存上的此次比较，不能当作论文配置复现。
可用 `--visual-token-budget` 明确修改预算，两个基线必须一致；所有设置均写入结果。

Qwen 用首个生成位置的裸候选字母（A、B、C……）原始 logits，在该题候选内做 softmax，
并保存实际首 token 以供核对。首 token 若不是裸字母，仍按预先定义的候选评分计入全部题目。
VideoJev 直接使用候选 softmax 概率。控制器复用这两个完整结果，不加载第三套模型：

```text
π₀ = softmax(Qwen 裸字母 logits)
FS = VideoJev 候选概率
π₁ = softmax(log π₀ + β × FS)，β=2
```

`FS` 不取 log，也不再次 softmax。控制器核对两源的题目、候选顺序、样本哈希、协议和逐视频几何。
β 在评测前固定，不根据这 100 题的答案调整。准确率分母始终为全部抽中题数，预测失败不会被当作成功样本静默剔除。

无需 GPU 或联网的自动测试：

```bash
python -m pytest -q tests/test_sample_longvideobench.py tests/test_lvb_experiment.py \
  tests/test_lvb_qwen_scoring.py tests/test_lvb_controller.py tests/test_longvideobench_benchmark.py
```

## 三次独立抽样的均值与方差

重复实验使用 seed=42、43、44，每次四组各 25 题。保留已完成的 seed42，
另外两次使用同一套脚本和推理设置；采样及两条基线的 `--seed` 都要对应修改。
例如 seed43 的输出根目录使用 `runs/lvb100_equal_seed43`，每次分别完成控制器评估后运行：

```bash
python scripts/summarize_longvideobench_repeats.py \
  --runs runs/lvb100_equal_seed42 runs/lvb100_equal_seed43 runs/lvb100_equal_seed44 \
  --out runs/lvb100_repeat3_seed42_43_44/summary
```

汇总脚本从逐题预测重新计分，检查三次模型、推理协议、依赖和代码指纹一致，
保存各方法总体及分时长组的三次准确率、均值、方差、标准差和样本交集。
主报告采用样本方差 `Σ(accuracyᵢ − mean)² / (3 − 1)`，Markdown 的单位为百分点²；
JSON 使用比例准确率及其平方单位，并同时保存分母为 3 的总体方差。
这里衡量跨抽样的准确率波动；三次样本可以重叠，不代表 300 道互异题目。

训练隔离目前未经验证：论文将 LongVideoBench val 用于评测，但当前仓库和发布权重
没有提供与该 checkpoint 绑定的完整 QA 训练清单，无法按视频和问题核对训练重叠。
报告保留 `training_overlap_verified=false`，不能将结果称为“已验证无训练污染”。

重复统计的 CPU 测试为 `python -m pytest -q tests/test_lvb_repeats.py`。

## 1 FPS、4096 帧基线与原控制器的比较

这次比较直接读取上述 seed=42、43、44 的三个原始 `sample.jsonl`，不重新抽样。
仅将 Qwen3.5-4B 和 VideoJev-4B 改为 `--fps 1 --max-frames 4096`；
上下文 262144、视觉 token 预算 24576、候选打分和同视频问题分组保持原设置。
Jev-as-Controller 复用原 `2 FPS / 2048 帧 / β=2` 的三个完整结果，不重新融合新基线。
输出使用独立目录，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_longvideobench_qwen.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl --seed 42 \
  --fps 1 --max-frames 4096 \
  --out runs/lvb100_baselines_fps1_frames4096_seed42_43_44/seed42/qwen

CUDA_VISIBLE_DEVICES=1 python scripts/eval_longvideobench_videojev.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl --seed 42 \
  --fps 1 --max-frames 4096 \
  --out runs/lvb100_baselines_fps1_frames4096_seed42_43_44/seed42/videojev

# seed43、44 同样读取各自原清单，完成六次基线推理后汇总。
python scripts/compare_longvideobench_sampling.py \
  --baseline-root runs/lvb100_baselines_fps1_frames4096_seed42_43_44 \
  --reference-root-pattern 'runs/lvb100_equal_seed{seed}' --seeds 42 43 44 \
  --out runs/lvb100_baselines_fps1_frames4096_seed42_43_44/summary
```

参数变化不能直接证明计算开销相同。长视频原先受到 2048 帧上限约束，
新设置在 2048 秒以上的视频上可能采到更多帧；固定视觉 token 预算还会改变空间网格。
逐题结果保留实际帧数、有效 FPS、网格和 token 数。新 Qwen 计时分别记录同视频 CPU
预处理和每题推理（含向 GPU 传输）；VideoJev 记录同视频预处理、编码和问题评分总时间。
共享视频的预处理或分组时间在多题行重复保存，统计时按视频去重。
`runtime.json` 同时记录显存峰值和运行时间范围；旧结果缺少同口径的分阶段计时，
因此不能仅用新旧耗时宣称已经严格对齐 FLOPs 或硬件开销。

## 4 FPS、4096 帧基线重新评估

用户纠正采样意图后，两条基线改为 `--fps 4 --max-frames 4096`，
仍直接读取原 seed=42、43、44 三份清单，各 100 题、四组各 25 题。
Controller 继续复用原 2 FPS / 2048 帧 / β=2 的结果。
4 FPS 表示每隔 0.25 秒采样，超过 4096 帧时均匀覆盖完整视频。
上下文 262144、视觉 token 上限 24576、空间几何规则和评分方式保持此前设置。

新运行目录为 `runs/lvb100_baselines_fps4_frames4096_seed42_43_44`，
此前的 1 FPS 和 2 FPS 结果均保留。例如 seed42：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_longvideobench_qwen.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl --seed 42 \
  --fps 4 --max-frames 4096 \
  --out runs/lvb100_baselines_fps4_frames4096_seed42_43_44/seed42/qwen

CUDA_VISIBLE_DEVICES=1 python scripts/eval_longvideobench_videojev.py \
  --manifest runs/lvb100_equal_seed42/sample.jsonl --seed 42 \
  --fps 4 --max-frames 4096 \
  --out runs/lvb100_baselines_fps4_frames4096_seed42_43_44/seed42/videojev

python scripts/compare_longvideobench_sampling.py \
  --baseline-root runs/lvb100_baselines_fps4_frames4096_seed42_43_44 \
  --reference-root-pattern 'runs/lvb100_equal_seed{seed}' --seeds 42 43 44 \
  --fps 4 --max-frames 4096 \
  --out runs/lvb100_baselines_fps4_frames4096_seed42_43_44/summary
```

FPS 与帧数上限提高后，实际输入帧数约为原 2 FPS / 2048 帧的两倍。
视觉 token 上限仍相同，空间网格可能相应缩小；实际 token 和 FLOPs 不保证翻倍，
报告保存帧数、有效 FPS、视觉 token、耗时和显存，不能仅凭 FPS 宣称严格等算力。
