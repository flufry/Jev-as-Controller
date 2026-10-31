# Jev-as-Controller

将 **VideoJev-4B** 作为候选答案评分器 `F_s`，修正 **Qwen3.6-35B-A3B** 的候选答案分布 `π₀`。本仓库保存方法实现、LongVideoBench 分层抽样评测、随机选边对照和实验结果。

代码基于 [VideoJev](https://github.com/MCG-NJU/VideoJev)，保留上游 `timejev/` 与训练工具。这里报告的是验证集子集上的控制器实验，不是 VideoJev 论文的完整官方评测。

## 方法

对原题的候选答案 A、B、C……，取 Qwen **第一个 decode 位置、不带空格的裸字母 token** 的原始 logits，记为 `z`：

```math
\pi_0=\mathrm{softmax}(z),\qquad
F_s=\mathrm{softmax}(\text{VideoJev candidate scores})
```

```math
\pi_1=\mathrm{softmax}(\log\pi_0+\beta F_s)
=\mathrm{softmax}(z+\beta F_s),\qquad \beta=2.
```

历史确定性实验选择 `argmax π₁`；固定题集的重复采样实验从 `π₁` 中随机抽取一个候选答案。`F_s` 直接使用 VideoJev 输出的候选概率，不取 log，也不再次 softmax。β 在评测前固定。

Qwen 的概率只在该题真实候选内归一化，不是字母在全词表中的概率。每题保留实际首 token 供审计；即使首 token 不是候选字母，也按预定义的候选 logits 打分，不剔除该题。原始问题、3/4/5 个选项及其顺序均保留。

Controller 需要两模型各自的推理。已保存两份候选分数后，融合与计分只需 CPU；融合步骤本身不再次加载模型。

## 当前基线：Qwen 原生生成后解析

Qwen 对照入口现改为 [eval_longvideobench_qwen36_native.py](scripts/eval_longvideobench_qwen36_native.py)：关闭 Thinking，使用官方模型卡推荐的 Instruct 设置，完整自回归采样后解析最终选项。逐题保存生成 token、原始回答、EOS/截断信息和解析结果；无效、歧义或截断回答计错，不从候选 logits 补答案。

本轮固定比例抽样 seed42 的 100 题，推理采样种子为 42、43、44；三次使用相同的问题和选项。

| 方法 | 答案取得方式 | 视频输入 |
| --- | --- | --- |
| Qwen3.6-35B-A3B | Instruct 完整自回归采样，解析最终答案 | 4 FPS / 最多4096帧 |
| VideoJev-4B | 从已有候选概率中采样，temperature=1 | 4 FPS / 最多4096帧 |
| Jev-as-Controller | 从融合后的候选概率中采样，temperature=1 | 双源2 FPS / 各最多2048帧 |

Qwen 设置为 `temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1`，最多生成32768个新token。Controller 使用的首步候选分布与独立 Qwen 生成式基线分别保留，汇总时不会拿历史 Qwen 候选采样结果代替原生生成结果。

**状态（2026-10-31）：106项相关CPU测试、四次输入检查及原生后端导入检查通过；最近一次GPU预检显存不足，单题GPU验证和三次完整原生生成均未启动，目前没有新基线准确率。** 运行、恢复、来源限制和汇总步骤见 [原生生成基线说明](docs/LVB_NATIVE_BASELINE.md)。

## 评测设置

| 项目 | 当前设置 |
| --- | --- |
| 数据 | LongVideoBench validation：1337 题、753 个视频 |
| π₀ | Qwen3.6-35B-A3B，BF16，无量化、无 CPU/disk offload |
| Fₛ | VideoJev-4B，候选 softmax 概率 |
| 两条独立基线 | 4 FPS，最多 4096 帧 |
| Controller 的两模型来源 | 2 FPS，最多 2048 帧 |
| 上下文 / 视觉 token 预算 | 262144 / 24576 |
| 模态 | 原始视频，不使用字幕或音频 |
| 抽题种子 | 历史实验使用 42、43、44；固定题集实验使用同一份题目 |
| 推理采样种子 | 原生生成基线使用 42、43、44，各采样一次 |
| 实际硬件布局 | 6 张 RTX 4090 放置 Qwen，另 1 张放置 VideoJev |

超过帧数上限时均匀覆盖完整视频。空间网格在固定视觉预算内选择。两模型同一协议的逐视频时间线与几何必须一致。

**4 FPS 基线与 2 FPS Controller 并不构成严格的 FLOPs 或耗时对齐。** 相同视觉 token 预算也不能证明计算开销相同。结果差异同时包含输入采样协议与融合的影响。

## 实验结果

本节已完成实验的 **Qwen 均使用首步裸字母候选打分或采样**；这些数字不能作为新原生生成基线的评测结果。

准确率单位为 %；样本方差的单位为**百分点²**，分母为 `3−1=2`。

### 固定seed42题集的候选概率采样（已完成）

三次固定使用同一份比例抽样100题，temperature=1，不使用top-k/top-p过滤，各方法从自己的候选分布重新抽取答案。复用已校验的首步概率，不重复执行GPU前向。这一实验的方差只反映答案采样的波动。

| 方法 | 采样seed42 | 采样seed43 | 采样seed44 | 平均准确率 | 样本方差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B（候选采样） | 61.00 | 66.00 | 66.00 | 64.33 | 8.3333 |
| VideoJev-4B | 61.00 | 57.00 | 62.00 | 60.00 | 7.0000 |
| Jev-as-Controller | 65.00 | 68.00 | 63.00 | 65.33 | 6.3333 |

完整的精确统计、采样规则与来源SHA见 [固定题集采样摘要](docs/results/longvideobench_fixed_seed42_sampling.json)。全部900条逐题采样已在本地独立回放验证。新的原生生成对照将保留此处VideoJev和Controller结果，重新执行Qwen逐token生成。

### 历史不同题集的比较

下述实验的三个seed分别重新抽题，因此方差同时反映抽题差异；它与上面固定题集的采样方差属于不同统计口径。

- `Qwen + VideoJev`：两基线同答则保留，异答则以 50% 概率随机选一个答案；每个 seed 重复 50 次，先求该 seed 的均值，再计算跨 seed 方差。
- `Qwen or VideoJev`：使用标签判断至少一个基线是否答对，是 oracle 上界，不能作为无需标签的推理方法。

#### 按完整 val 的题量比例抽样：每个 seed 100 题

采用整数最大余数法分配四个官方时长组的配额：

| 官方时长组   | val 题数 | 每个 seed 抽题数 |
| ------------ | -------: | ---------------: |
| 8–15 秒     |      189 |               14 |
| 15–60 秒    |      172 |               13 |
| 180–600 秒  |      412 |               31 |
| 900–3600 秒 |      564 |               42 |

| 方法                           | seed42 | seed43 | seed44 |      平均准确率 | 样本方差 |
| ------------------------------ | -----: | -----: | -----: | --------------: | -------: |
| Qwen3.6-35B-A3B                |  67.00 |  69.00 |  71.00 |           69.00 |   4.0000 |
| VideoJev-4B                    |  64.00 |  68.00 |  68.00 |           66.67 |   5.3333 |
| Qwen3.6-35B-A3B + VideoJev-4B  |  65.46 |  68.20 |  69.38 |           67.68 |   4.0444 |
| Jev-as-Controller              |  67.00 |  73.00 |  78.00 | **72.67** |  30.3333 |
| Qwen3.6-35B-A3B or VideoJev-4B |  74.00 |  78.00 |  80.00 |           77.33 |   9.3333 |

共 300 题次，实际为 **283 道互异题、251 个视频**；三个 seed 分别涉及 95、96、97 个视频。抽样单位是问题，不保证每份 100 题来自 100 个不同视频；不同 seed 允许重叠，也不排除历史实验抽过的问题。

Controller 平均比 4 FPS Qwen 高 3.67 个百分点，比 4 FPS VideoJev 高 6.00 个百分点，但波动也更大。作为同协议参考，Qwen 的 2 FPS 来源准确率为 66%、67%、77%，Controller 相对它的平均提升为 **2.67 个百分点**。

#### 四个时长组等量抽样：每个 seed 累计 300 题

先做一组每时长组各 25 题的 100 题实验，再各新增两组 100 题。同一个 seed 的三组问题互斥，合并后四组各 75 题。跨 seed 合计 900 题次，实际为 **671 道互异题、570 个视频**。

| 方法                           | seed42 | seed43 | seed44 |      平均准确率 | 样本方差 |
| ------------------------------ | -----: | -----: | -----: | --------------: | -------: |
| Qwen3.6-35B-A3B                |  73.33 |  73.33 |  73.33 |           73.33 |   0.0000 |
| VideoJev-4B                    |  71.33 |  72.00 |  72.00 |           71.78 |   0.1481 |
| Qwen3.6-35B-A3B + VideoJev-4B  |  72.16 |  72.45 |  72.59 |           72.40 |   0.0489 |
| Jev-as-Controller              |  74.67 |  75.67 |  75.00 | **75.11** |   0.2593 |
| Qwen3.6-35B-A3B or VideoJev-4B |  81.33 |  81.67 |  81.00 |           81.33 |   0.1111 |

各 100 题组的完整表、未四舍五入的数值和来源 SHA 保存在 [结果摘要](docs/results/README.md) 与 [机器可读 JSON](docs/results/longvideobench_summary.json)。比例实验的三份题目 ID 见 [proportional_samples.json](docs/results/proportional_samples.json)。

两种抽样分布不同，不能将上面两个均值直接比较来判断方法退步或进步。只有三个 seed，且题集可能重叠，不能据此宣称统计显著性。当前也没有与该 VideoJev checkpoint 绑定的完整 QA 训练清单，**训练与验证集是否重叠尚未核实**。

## 环境与模型

当前实验实际使用 Python 3.12.12、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、Triton 3.4.0、transformers 5.14.1、accelerate 1.15.0、FlashAttention 2.8.3.post2、causal-conv1d 1.7.0、av 18.1.0。

使用 `uv` 管理环境。下载依赖、权重或数据时，在当前 shell 设置代理；不要关闭 TLS 校验：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"

git clone https://github.com/tangqh03/Jev-as-Controller.git
cd Jev-as-Controller

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
```

还需安装与 Python、PyTorch、CUDA ABI 匹配的 FlashAttention 和 causal-conv1d。`requirements.txt` 不包含完整 CUDA 栈锁文件，不能仅凭这一条安装命令假定所有原生扩展可用。

**35B 的 grouped-GEMM 后端是必需前置。** 本实现针对 RTX 4090/SM89 使用固定源码和 CUTLASS 补丁，拒绝静默回退到 Python 专家循环。可发布的源码、版本、许可证与构建步骤见 [runtime/qwen36](runtime/qwen36/README.md)。该说明从实际构建源整理；本次文档发布没有在新机器上重新编译，其他 GPU 架构需要单独验证。

| 模型            | Hugging Face 仓库                                                  | 本次 revision                                |
| --------------- | ------------------------------------------------------------------ | -------------------------------------------- |
| Qwen3.6-35B-A3B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| VideoJev-4B     | [ZhuYuhan/VideoJev-4B](https://huggingface.co/ZhuYuhan/VideoJev-4B) | `1c1a253bf90dfa66176124d25a48ffb791d517d5` |

在已配置上述代理的 shell 下载固定版本：

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir models/Qwen3.6-35B-A3B
hf download ZhuYuhan/VideoJev-4B \
  --revision 1c1a253bf90dfa66176124d25a48ffb791d517d5 \
  --local-dir models/VideoJev-4B
```

权重、数据、`.deps*` 和实际运行产物不放入 Git。Qwen 权重本身约 67 GiB，还需足够的激活空间。

## 准备 LongVideoBench

本次视频来自 [Jialuo21/LongVideoBench 镜像](https://huggingface.co/datasets/Jialuo21/LongVideoBench)，固定 revision `0c30edb990835a4ffd48059cd84d0680c62887a3`。标注与官方 [LongVideoBench-Meta](https://huggingface.co/datasets/longvideobench/LongVideoBench-Meta) 的公开 validation 共有字段做过核对；没有宣称镜像视频与官方大 tar 逐文件相同。

以下下载完整 val 的原视频与标注，约 26.76 GiB 视频；继续使用前面的代理设置：

```bash
hf download Jialuo21/LongVideoBench --repo-type dataset \
  --revision 0c30edb990835a4ffd48059cd84d0680c62887a3 \
  --include 'lvb_val.json' --include 'videos/*.mp4' \
  --local-dir data/benchmarks/longvideobench
```

离线转换为评测所需的清单，保留官方 `video_path`，不要用 `video_id + ".mp4"` 猜文件名：

```bash
source .venv/bin/activate
source scripts/env.sh
python - <<'PY'
import json
from pathlib import Path
from timejev.longvideobench_benchmark import formatted_options, validate_reference

base = Path('data/benchmarks/longvideobench')
raw = (base / 'lvb_val.json').read_bytes()
official = json.loads(raw)
validate_reference(official)
rows = [dict(
    id='longvideobench/' + r['id'], question_id=r['id'],
    video_key='longvideobench/' + r['video_id'],
    video_path='videos/' + r['video_path'], duration=r['duration'],
    task='choice', query=r['question'], options=formatted_options(r),
    answer_index=r['correct_choice'],
) for r in official]
assert all((base / row['video_path']).is_file() for row in rows)
with (base / 'official.json').open('xb') as stream:
    stream.write(raw)
with (base / 'questions.jsonl').open('x', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
PY
```

目标文件已存在时转换会拒绝覆盖；完整官方库存校验要求 1337 题、753 视频及原始选项数量分布。

## 复现历史候选打分实验

完成模型、数据与 native 后端准备后，在 Bash 中运行下列分步入口。GPU 编号需要对应自己的空闲卡；这里沿用本次六卡 Qwen 加单卡 VideoJev 的布局。每个 seed 先做四份模型推理，再做 CPU 融合和汇总。示例按顺序执行以便复现，不会自动调度共享 GPU。

```bash
source .venv/bin/activate
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LVB_RUN=runs/lvb_proportional_reproduce

for seed in 42 43 44; do
  cell="$LVB_RUN/seed$seed"
  python scripts/sample_longvideobench_proportional.py \
    --seed "$seed" --n 100 --out "$cell/sample.jsonl"

  for protocol in baseline controller; do
    if [ "$protocol" = baseline ]; then
      fps=4
      frames=4096
    else
      fps=2
      frames=2048
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 \
      python scripts/eval_longvideobench_qwen36.py \
      --manifest "$cell/sample.jsonl" --out "$cell/qwen_$protocol" \
      --checkpoint models/Qwen3.6-35B-A3B --seed "$seed" \
      --device cuda:0 --gpu-count 6 --max-memory-gib 18.5 \
      --fps "$fps" --max-frames "$frames" \
      --context 262144 --visual-token-budget 24576 --memory-fraction .9

    CUDA_VISIBLE_DEVICES=7 \
      python scripts/eval_longvideobench_videojev.py \
      --manifest "$cell/sample.jsonl" --out "$cell/videojev_$protocol" \
      --checkpoint models/VideoJev-4B --seed "$seed" \
      --device cuda:0 --fps "$fps" --max-frames "$frames" \
      --context 262144 --visual-token-budget 24576 --memory-fraction .9
  done

  python scripts/eval_longvideobench_proportional_controller.py \
    --manifest "$cell/sample.jsonl" --seed "$seed" \
    --qwen "$cell/qwen_controller" --videojev "$cell/videojev_controller" \
    --out "$cell/controller" --beta 2 --qwen-method qwen36

  python scripts/summarize_longvideobench_proportional.py \
    --mode single --manifest "$cell/sample.jsonl" --seed "$seed" \
    --qwen-baseline "$cell/qwen_baseline" --qwen-controller "$cell/qwen_controller" \
    --videojev-baseline "$cell/videojev_baseline" \
    --videojev-controller "$cell/videojev_controller" \
    --controller "$cell/controller" --out "$cell/summary"
done

python scripts/summarize_longvideobench_proportional.py \
  --mode all --base "$LVB_RUN" --out "$LVB_RUN/summary"
```

输出目录必须是新的。自动汇总会重验采样来源、两模型同协议的预处理、概率、融合公式、三 seed 的模型和协议指纹，并生成：

```text
runs/lvb_proportional_reproduce/
  seed42/  seed43/  seed44/
    sample.jsonl
    sample.meta.json
    qwen_baseline/  qwen_controller/
    videojev_baseline/  videojev_controller/
    controller/  summary/
  summary/
    results.json
    results.md
    random_trials.jsonl
```

每个模型目录保存 `run.json`、`predictions.jsonl`、`metrics.json` 和相关加载/资源审计。每行预测对应一个问题，ID 唯一，`answer_index` 从 0 开始。

若只复现等量 100 题，改用 `scripts/sample_longvideobench.py --per-group 25` 和等量对应的 Controller/汇总入口，不要给比例汇总器伪造等量元数据。[历史等量协议说明](docs/LVB_SUBSET.md) 保留早期 4B 与 1/2/4 FPS 对照的背景。

`scripts/run_longvideobench_*`、`launch_*` 及恢复脚本保留了本次实验的冻结、显存门禁和审计过程，但依赖本机历史 `runs/`、pilot 或验收文件；它们不是新 checkout 的通用一键运行入口。

## 验证与结果来源

无需下载权重或启动 GPU 的相关 CPU 回归测试：

```bash
source .venv/bin/activate
source scripts/env.sh
python -m pytest -q \
  tests/test_sample_longvideobench.py \
  tests/test_lvb_experiment.py \
  tests/test_lvb_qwen_scoring.py \
  tests/test_lvb_qwen36.py \
  tests/test_lvb_controller.py \
  tests/test_lvb_qwen36_summary.py \
  tests/test_lvb_proportional_sampling.py \
  tests/test_lvb_proportional_summary.py
```

历史结果首次发布前，复用已有 `uv` 解释器，在不含 `models/`、`data/`、`runs/`、`reports/` 的临时干净 checkout 中执行了 17 个相关测试文件，**366 项测试通过**。这不代表重新安装或编译了完整 GPU 环境，也未重新运行模型评测。

上述366项对应历史实验发布验证。本轮原生生成与固定题集采样相关检查为 **106项CPU测试通过**，可执行：

```bash
python -m pytest -q \
  tests/test_lvb_native_generation.py \
  tests/test_lvb_native_summary.py \
  tests/test_lvb_qwen36.py \
  tests/test_lvb_fixed_question_sampling.py
```

测试覆盖原生生成参数、presence penalty执行顺序、无效答案计错、token与回答重解码一致性、恢复指纹和三次汇总；完整多卡GPU生成仍待验证。

历史比例实验持久保存了命令、配置、模型和数据指纹、逐题预测、退出码及完整性清单，共完成 **12 个模型步骤和 7 个 CPU 步骤**；其中一个已完成模型结果经完整来源核验后复用，剩余任务从空输出重跑。

该历史实验的原外层 wrapper/监控器最终退出记录缺失，因此最终封存采用了**19 个子步骤真实退出 0 + 新独立 CPU 重验进程真实退出 0**的证据，明确保留原外层退出码未知的事实，没有补造 `full.exitcode`。公开摘要记录来源 SHA；完整本机 `runs/` 与 `reports/` 未上传。

## 代码导航与上游

| 路径 | 内容 |
| --- | --- |
| `scripts/sample_longvideobench*.py` | 等量、比例及追加抽样 |
| `scripts/eval_longvideobench_qwen36.py` | Qwen 首步候选字母打分（历史基线） |
| `scripts/eval_longvideobench_qwen36_native.py` | 当前 Qwen 基线：完整 Instruct 生成后解析 |
| `scripts/eval_longvideobench_videojev.py` | VideoJev 候选概率 |
| `scripts/eval_longvideobench_proportional_controller.py` | 比例样本的双源融合 |
| `scripts/sample_longvideobench_fixed_questions.py` | 固定题集的候选概率重复采样 |
| `scripts/summarize_longvideobench_native_baseline.py` | 原生 Qwen 与保留的两方法采样结果汇总 |
| `scripts/summarize_longvideobench_proportional.py` | 三 seed、随机选边和 OR 统计 |
| `timejev/lvb_experiment.py` | 视频预处理、指纹、断点和计分 |
| `timejev/lvb_native_generation.py` | 原生生成解析、采样惩罚、token审计与恢复 |
| `runtime/qwen36/` | 固定 native 后端的小型源码与构建说明 |
| `docs/results/` | 可公开的结果、样本 ID 与来源摘要 |

原 VideoJev 的训练、checkpoint 导出和完整 benchmark 接口仍见 [数据与训练](docs/DATA.md)、[评估说明](docs/EVALUATION.md)、[checkpoint 导出](docs/CHECKPOINTS.md) 及 [上游仓库](https://github.com/MCG-NJU/VideoJev)。模型和数据的使用同时遵守各自发布方的许可证；仓库代码许可证见 [LICENSE](LICENSE)。
