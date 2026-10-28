# Jev-as-Controller 项目整理与扩展评测建议

本目录前半部分是对原仓库的阅读笔记；新增扩展的实现、Docker 环境和运行说明单独放在 05–07。01–03 保留原有分析；本轮实际采用 04 的小样本粗筛路线，具体命令和修正以 05–06 为准。

| 文档 | 内容 |
| --- | --- |
| [01-架构总览.md](01-架构总览.md) | 项目定位、目录分层、模块职责、数据流、协议常量、产物格式 |
| [02-使用说明.md](02-使用说明.md) | 从零到跑通：环境、模型/数据下载、LVB 实验复现、HF 单模型评测、测试与常见坑 |
| [03-其他benchmark测试建议.md](03-其他benchmark测试建议.md) | 换到别的视频 benchmark：推荐哪些、走哪条路线、要改什么、怎么设计实验 |
| [04-待办与执行计划.md](04-待办与执行计划.md) | 本轮小样本粗筛计划 |
| [05-跨Benchmark粗筛运行手册.md](05-跨Benchmark粗筛运行手册.md) | **新增扩展**：标注适配、抽样、预算预检、两模型、融合和统计汇总 |
| [06-Docker与服务器环境.md](06-Docker与服务器环境.md) | **新增扩展**：官方 Docker Hub 基础镜像、服务器容器、依赖/后端和运行方式 |
| [07-本次文件变更记录.md](07-本次文件变更记录.md) | 本轮新增/修改文件名 |

## 一句话概括

本仓库把 **VideoJev-4B 当作候选答案评分器 `F_s`**，去修正 **Qwen3.6-35B-A3B 的候选答案分布 `π₀`**：

```text
π₀ = softmax(Qwen 第一个 decode 位置上裸候选字母 A/B/C… 的原始 logits)
F_s = VideoJev 输出的候选 softmax 概率（不取 log、不再 softmax）
π₁ = softmax(log π₀ + β·F_s) = softmax(z + β·F_s),  β = 2
answer = argmax π₁
```

目前公开的实验只在 **LongVideoBench validation 的分层子集**上做（100 题/seed、seed 42/43/44、等量 300 题/seed 两种抽样口径），主结果：Controller 平均 72.67%（比例抽样）/ 75.11%（等量抽样），比同协议 2 FPS Qwen 基线高约 2.7–3.7 个百分点。

## 三分钟读懂代码结构

```text
scripts/            入口：抽样 → 两模型推理 → CPU 融合 → 汇总（每步独立进程、指纹可核验）
  └ timejev/        实验框架 + 计分 + 训练库（lvb_experiment / *_benchmark / model / rgb_media …）
      └ videojev_hf/  发布用 Hugging Face 推理包（AutoModel + trust_remote_code）
runtime/qwen36/     35B 的 grouped-GEMM + CUTLASS SM89 后端源码（评测的前置硬依赖）
configs/            DeepSpeed / 两阶段训练 / qa_evaluation 注册表
docs/               上游 VideoJev 的数据/训练/评估/导出说明 + 已发布结果摘要
tests/              48 个测试文件，多数可在无 GPU、无权重的情况下跑
studio/             Gradio demo（上传短视频，问 grounding / QA）
```

## 换 benchmark 的结论先行

1. **最省事**：`scripts/evaluate_hf.py --benchmark <name>` 已支持 4 个选择题 QA benchmark（`videomme` 2700 / `mlvu` 2174 / `lvbench` 1549 / `longvideobench` 1337）+ 3 个时序定位 benchmark（`charades` / `activitynet` / `qvhighlights`）。但这只测 **F_s 单模型**，不测 Controller。
2. **要验证方法本身**：必须让 **Qwen 和 VideoJev 在同一份清单、同一 fps/帧数/上下文/视觉预算** 下各跑一遍，再融合。Controller 的融合脚本 `scripts/eval_longvideobench_controller.py` 是**清单驱动、通用**的，可以复用到新 benchmark。
3. **必须先解决的耦合**：`timejev/lvb_experiment.py` 强制每行有 `duration_group ∈ {15,60,600,3600}`；比例抽样适配器把 LVB 官方标注、seed∈{42,43,44}、n=100 写死了。新 benchmark 要么把时长映射进这四档，要么写一个对等的 `*_experiment` 变体（推荐后者）。
4. **选哪个 bench**：优先复用已写好官方严格计分器的 4 个（Video-MME / MLVU-Dev / LVBench / LongVideoBench-val），并且**跑全量而不是抽子集**——现有 100 题子集的跨 seed 方差高达 30.3 个百分点²，噪声会吃掉 2–3 个点的增益。

详见 [03-其他benchmark测试建议.md](03-其他benchmark测试建议.md)。
