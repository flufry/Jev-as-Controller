# 四个视频 benchmark：全量固定题集、三个答案采样 seed

本流程在当前仓库实现。**每个seed独立执行四组完整GPU评测，不跨seed复用候选分数**；候选答案采样沿用latest的temperature=1、固定题序CDF规则。无需等待历史100题实验，不使用latest中的本机运行目录。**全量指有公开标签的评测集，不是训练集或隐藏标签测试集。**

| benchmark | 完整题数 | 主指标 |
|---|---:|---|
| LongVideoBench validation | 1337 | Accuracy |
| Video-MME | 2700 | Accuracy |
| MLVU-Dev 七类选择题 | 2174 | 七任务宏平均 M-Avg |
| LVBench | 1549 | Accuracy |

## 采样与方法口径

三个seed **42、43、44使用完全相同的全量问题**、原选项顺序和原题序，`selection_seed=None`。seed不用于抽题。

| 报告方法 | 视频输入 | 三seed的含义 |
|---|---|---|
| Qwen3.6 自回归 | 4 FPS / 4096帧 | 实际执行三遍完整采样生成，逐题保存输出token和解析 |
| 单步 Qwen logits（融合来源） | 2 FPS / 2048帧 | 每个seed重新执行全量首步前向，再从该seed的softmax(z_Q)候选分布采样 |
| VideoJev-4B 独立基线 | 4 FPS / 4096帧 | 每个seed重新执行全量4FPS前向，再从该seed的候选分布采样 |
| Controller 旧方法 | 两源均2 FPS / 2048帧 | `softmax(z_Q + 2*p_J)`，只融合并采样当前seed的两份2FPS来源 |
| Controller 新方法 | 两源均2 FPS / 2048帧 | `softmax(z_Q + 2*s_J)`，只融合并采样当前seed的两份2FPS来源 |
| Qwen OR VideoJev-4B | **两独立基线均4 FPS** | 同seed的native Qwen与采样VideoJev至少一个答对；使用真实标签的事后参照 |

`z_Q`是Qwen首decode位置裸字母候选logits，`s_J`是VideoJev原始head logits，`p_J=softmax(s_J)`。融合还需要一份 **2 FPS VideoJev** 来源，不能拿4 FPS基线与2 FPS Qwen混用。

单步Qwen、VideoJev和两版Controller采用temperature=1、无top-k/top-p的候选CDF采样，沿用latest的固定原题序。单步Qwen使用`random.Random(f'{seed}:qwen36')`；VideoJev使用`random.Random(f'{seed}:videojev')`；两版Controller各自重置`random.Random(f'{seed}:jev_as_controller')`，使用相同均匀随机序列，以便比较公式。

native沿用关闭thinking的原设置：temperature=.7、top_p=.8、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1，默认最多32768个输出token。每题使用`SHA256(f'{seed}:{question_id}')`派生torch CPU/CUDA种子。无效、歧义、截断均计错，保留在分母中。

默认context=262144、visual-token-budget=24576、不使用字幕/音频。全量各方法的同视频问题组从头计算，不拼接旧子集分数。每个seed分别执行Qwen logits 2FPS、VideoJev 2FPS、VideoJev 4FPS、Qwen native 4FPS，共**12个GPU阶段**。每完成一个seed的四组模型评测，立即执行该seed的CPU采样、融合和OR汇总，最后计算三个seed总表（总计16个队列阶段）。

Qwen logits的GPU阶段仍为确定性首步取分，最终“单步生成”成绩从对应seed的候选分布采样。各模型入口都初始化Python/NumPy/Torch随机种子。独立重跑不保证分数一定不同；不人为制造方差。

## 公共环境和GPU参数

以下命令在服务器的**当前代码仓库根目录**执行，先激活已验证的环境；这里不安装依赖、不下载权重。根据服务器实际位置调整模型/数据路径。

```bash
source .venv/bin/activate
export XTUNER_ROOT="$PWD/.deps-xtuner"
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1

# 默认示例：Qwen使用六卡；VideoJev阶段使用单卡0，阶段按顺序执行。
COMMON=(--qwen-gpus 0,1,2,3,4,5 --videojev-gpu 0)
```

若Qwen用四张卡且需逐卡预算，可改成下面的形式；数值仅示范参数格式，须按实际设备配置：

```bash
COMMON=(--qwen-gpus 0,1,2,3 --videojev-gpu 0 \
  --max-memory-gib-per-gpu 30,45,45,45)
```

支持4/5/6张Qwen卡，预算按上述逻辑GPU顺序传递。若两模型使用独立环境，添加解释器**完整路径**：

```bash
COMMON+=(--qwen-python /path/to/qwen-env/bin/python \
         --videojev-python /path/to/videojev-env/bin/python)
```

runner继承当前环境变量，分别设置各阶段的`CUDA_VISIBLE_DEVICES`。请在有正确CUDA/grouped-GEMM运行环境的shell启动。它不会终止其他任务或无限重试；显存不足时保留真实失败日志。

## 全量源清单前置条件

需要各数据集完整的`questions.jsonl`和官方标注。`prepare_bench_full.py`会逐题核对官方ID、问题、选项、标签、视频分组以及完整库存，并生成带来源SHA的全量清单，不能用同题数伪造子集。

- LongVideoBench：使用原README的数据准备步骤，官方文件可为`official.json`或`lvb_val.json`；新入口会补齐时长分组，不要求原清单已经包含stratum。
- Video-MME：使用`build_bench_manifest.py --bench videomme`生成完整清单。
- MLVU：使用`build_bench_manifest.py --bench mlvu`，下文默认标注在`data/benchmarks/mlvu/json/`，这是本仓库下载器复制的七任务标注目录。
- LVBench：使用`build_bench_manifest.py --bench lvbench`，官方标注默认为`official.jsonl`。

数据迁移到另一台机器后，先确保源清单中视频路径可解析到新的数据位置；不要直接复用旧实验的绝对路径或元数据冒充新运行。

## LongVideoBench：单独执行

```bash
python scripts/prepare_bench_full.py --bench longvideobench \
  --source-manifest data/benchmarks/longvideobench/questions.jsonl \
  --reference data/benchmarks/longvideobench/official.json \
  --out runs/full/longvideobench/questions.jsonl

python scripts/run_bench_full.py --bench longvideobench \
  --base runs/full/longvideobench "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench longvideobench \
  --base runs/full/longvideobench "${COMMON[@]}"
```

## Video-MME：单独执行

```bash
python scripts/prepare_bench_full.py --bench videomme \
  --source-manifest data/benchmarks/videomme/questions.jsonl \
  --reference data/benchmarks/videomme/official.json \
  --out runs/full/videomme/questions.jsonl

python scripts/run_bench_full.py --bench videomme \
  --base runs/full/videomme "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench videomme \
  --base runs/full/videomme "${COMMON[@]}"
```

## MLVU：单独执行

```bash
python scripts/prepare_bench_full.py --bench mlvu \
  --source-manifest data/benchmarks/mlvu/questions.jsonl \
  --reference data/benchmarks/mlvu/json \
  --out runs/full/mlvu/questions.jsonl

python scripts/run_bench_full.py --bench mlvu \
  --base runs/full/mlvu "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench mlvu \
  --base runs/full/mlvu "${COMMON[@]}"
```

MLVU全量七任务题数不相同，**M-Avg不等于总体accuracy**。主表均值/方差使用每个seed的M-Avg，同时保留总体accuracy、正确数和逐任务指标。

## LVBench：单独执行

```bash
python scripts/prepare_bench_full.py --bench lvbench \
  --source-manifest data/benchmarks/lvbench/questions.jsonl \
  --reference data/benchmarks/lvbench/official.jsonl \
  --out runs/full/lvbench/questions.jsonl

python scripts/run_bench_full.py --bench lvbench \
  --base runs/full/lvbench "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench lvbench \
  --base runs/full/lvbench "${COMMON[@]}"
```

四个区块可以独立执行；相同GPU不应同时启动四份队列。`prepare`拒绝覆盖已有清单；清单准备成功后无需重复执行该步。`--dry-run`验证题集和计划、打印全部命令，不启动GPU、不创建模型输出；它不验证CUDA内核能运行，也不加载checkpoint。

## 输出与恢复

```text
runs/full/<benchmark>/
  questions.jsonl / questions.meta.json
  experiment.json                 # 阶段状态、真实退出码、命令和源码指纹
  source/                         # 本次封存源码
  logs/<stage>.attemptN.log
  logs/<stage>.attemptN.exitcode
  sources/seed42/qwen_controller/     # seed42独立2FPS Qwen前向
  sources/seed42/videojev_controller/ # seed42独立2FPS VideoJev前向
  sources/seed42/videojev_baseline/   # seed42独立4FPS VideoJev前向
  sources/seed43/                    # 同样三份来源，实际重新执行
  sources/seed44/                    # 同样三份来源，实际重新执行
  native/seed42/                  # 实际完整自回归生成
  native/seed43/
  native/seed44/
  per_seed/seed42/results.md       # 四组完成后立即生成该seed六方法表
  per_seed/seed43/results.md
  per_seed/seed44/results.md
  summary/results.md             # 六行六列：方法、三seed、均值、样本方差
  summary/results.json           # 精确值、分组指标、来源SHA、OR定义
  summary/*.seed42.distributions.jsonl # 同样保存seed43/44，各自分布
  summary/*.seed42.predictions.jsonl   # 同样保存seed43/44
  summary/artifact_manifest.json
```

方差使用三次未舍入主指标的**样本方差，ddof=1**，显示单位为pp²。主表按“方法、seed42、seed43、seed44、平均准确率、样本方差”六列展示；MLVU均值列写作“平均M-Avg”，JSON额外保留标准差。报告额外保留单步Qwen、VideoJev和两版Controller的确定性argmax诊断，不拿argmax结果替代采样结果。

中断或失败后，保留原命令的参数并添加`--resume`，例如：

```bash
python scripts/run_bench_full.py --bench mlvu \
  --base runs/full/mlvu "${COMMON[@]}" --resume
```

已完成阶段先验证文件SHA，再跳过；未完成模型阶段通过原评测器的`--resume`继续。恢复要求源码、题集、模型路径和运行参数一致；源码改变会拒绝混合结果。历史子集结果不能直接作为本流程的来源。若仅CPU汇总因中断留下不完整目录，保留并移走该`summary/`目录后恢复，不需重跑已完成模型阶段。

单独重算汇总（首次目录不能存在）：

```bash
python scripts/summarize_bench_full.py \
  --manifest runs/full/mlvu/questions.jsonl \
  --base runs/full/mlvu --out runs/full/mlvu/summary
```

只读核验已完成汇总：

```bash
python scripts/summarize_bench_full.py \
  --manifest runs/full/mlvu/questions.jsonl \
  --base runs/full/mlvu --out runs/full/mlvu/summary --verify
```

把命令中的`mlvu`替换为对应数据集即可。队列正常完成前也会执行一次独立汇总重验。

## 旧队列与新版队列不可混用

当前队列schema为`video_bench_full_queue_v2`，执行协议为`four_gpu_runs_per_seed_v1`，三seed总表schema为`video_bench_full_three_seed_summary_v3`。旧版只执行一次候选来源的队列不满足本协议，不能以改写seed或复制来源目录方式冒充三次独立GPU运行，也不能直接用旧experiment.json恢复新队列。

若已有旧实验，先停止并保留旧目录，同步当前代码后使用新base；完整题集清单可直接复用。例如Video-MME：

```bash
python scripts/run_bench_full.py --bench videomme \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_per_seed4 "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench videomme \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_per_seed4 "${COMMON[@]}"
```

这是新队列，第一次不加`--resume`。此后中断可对这个新base以相同参数加`--resume`继续。不要删除原runs。新版先后完成：

```text
qwen_controller_seed42 → videojev_controller_seed42 → videojev_baseline_seed42 → native_seed42 → summary_seed42
qwen_controller_seed43 → videojev_controller_seed43 → videojev_baseline_seed43 → native_seed43 → summary_seed43
qwen_controller_seed44 → videojev_controller_seed44 → videojev_baseline_seed44 → native_seed44 → summary_seed44
summary（最终六行、三seed、均值、样本方差）
```

单个seed汇总只有一个值，样本方差标为“—（单seed）”，不伪造为0。总表使用三个seed主指标计算ddof=1方差。单步Qwen、VideoJev和两版Controller每次只使用同seed的分数。OR始终是同seed的4FPS native Qwen与4FPS VideoJev采样答案，不采用2FPS融合来源。

三seed的成绩允许偶然相同、方差为0；截图仅是表格样式参考，所有数值均从实验预测重算。
