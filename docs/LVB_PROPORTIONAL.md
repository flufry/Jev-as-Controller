# LongVideoBench 按题量比例抽样实验

此实验从完整 validation 的 1337 道题中，按官方四个时长组的题目数量比例抽取 100 题，分别使用 seed=42、43、44。
整数配额使用最大余数法；余数相同时按下表组序分配。

| 官方时长组 | 完整 val 题数 | 每个 seed 抽题数 |
| --- | ---: | ---: |
| 8–15 秒 | 189 | 14 |
| 15–60 秒 | 172 | 13 |
| 180–600 秒 | 412 | 31 |
| 900–3600 秒 | 564 | 42 |

抽样单位是问题，并非独立视频。每个 seed 从完整 val 独立抽样，允许与此前实验及其它 seed 重叠；不根据答案或模型表现选题。原始题序、选项顺序、标签和视频完整保留。
本次重新运行全部模型；同一视频的问题组合可能变化，不按问题 ID 复用旧模型分数。

两条基线为 Qwen3.6-35B-A3B 和 VideoJev-4B，使用 4 FPS、最多 4096 帧。Controller 使用两模型各自的 2 FPS、最多 2048 帧结果，固定 β=2，按 `softmax(Qwen候选logits + 2 × VideoJev候选概率)` 融合。
全部运行保持上下文 262144、视觉 token 上限 24576，不使用字幕或音频。
Qwen 使用六张 GPU、BF16、无量化；VideoJev 在另一张 GPU 上依次执行两个采样协议。

每个 seed 报告两基线、Controller、随机选边及 OR 上界。随机选边在两基线同答时保留答案，异答时逐题以 50% 概率选边，独立重复 50 次后取准确率均值。OR 使用真实标签，只代表两基线至少一个答对的上界。
总表报告三个 seed 的准确率、平均准确率和样本方差（分母为 2，单位为百分点²）。随机方法先在各 seed 内平均 50 次，再计算跨 seed 方差。

在仓库根目录加载已有环境后抽样：

```bash
source .venv/bin/activate
source scripts/env_qwen36.sh
for seed in 42 43 44; do
  python scripts/sample_longvideobench_proportional.py \
    --seed "$seed" --n 100 \
    --out "runs/lvb100_proportional_seed42_43_44/seed$seed/sample.jsonl"
done
```

准备并只读检查运行计划：

```bash
python scripts/run_longvideobench_proportional.py --phase prepare
python scripts/run_longvideobench_proportional.py --phase full --check-only
```

每次运行保留独立输出，默认根目录为 `runs/lvb100_proportional_seed42_43_44/`。
正式推理前复核已有四协议真实小规模验证及其源码、环境和权重来源；本次采样器、控制器元数据适配和统计汇总另外执行 CPU 回归测试。
每个模型任务执行前检查所用 GPU 的空闲显存，外部任务占用时等待，失败时保留现场。
长任务由指定的 `gpt-6-luna`、`max` 推理强度监控；需先启动该监控，再使用 `scripts/launch_longvideobench_proportional.sh` 启动持久任务。启动脚本在所有子进程真实成功退出且监控交接完成后执行最终验收；最终验收日志位于实验目录之外，避免修改封存产物。

正式结果写入 `summary/results.json`、`summary/results.md` 和 `summary/random_trials.jsonl`；逐题模型预测及审计信息保存在对应 seed 的方法目录。
最终验收重算指标、核验来源和全部步骤退出状态，再保存完整性清单。运行中或验收失败时不能视为实验完成。

这是按官方题量比例抽样的子集实验，不代表完整官方评测；视觉 token 预算也小于论文正式协议。抽样波动不能解释为训练随机性。训练与验证是否存在重叠仍未获得完整训练清单验证，不能声称已验证无训练污染。
