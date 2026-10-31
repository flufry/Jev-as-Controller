# LongVideoBench 原生生成基线

本轮仅替换 Qwen3.6-35B-A3B 的答案取得方式：使用模型正常自回归生成，再解析最终选项。VideoJev-4B 和 Jev-as-Controller 按实验要求，继续对各自保存的候选概率进行 temperature=1 采样。三者固定使用同一份比例抽样 seed42 的 100 道题；重复的采样种子为 42、43、44。

## 配置与来源

| 项目 | 设置 |
| --- | --- |
| 固定题集 | `runs/lvb100_proportional_seed42_43_44/seed42/sample.jsonl`，SHA256 `b4890d7a63295625957fe2bc13400c3964861e05512b2988e63f229d8d5093b5` |
| Qwen 模式 | Instruct，`enable_thinking=False` |
| Qwen 采样 | `temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1` |
| Qwen 输出 | 全词表逐 token 生成，到 EOS 或最多 32768 个新 token |
| Qwen 基线输入 | 4 FPS，最多4096帧，context262144，视觉预算24576 |
| VideoJev 基线 | 原4 FPS候选概率，temperature=1采样 |
| Controller | 原双2 FPS来源、β=2的候选概率，temperature=1采样 |
| 模型与放置 | 既有BF16权重，6卡完整层分配，无量化和CPU/disk offload |

采样参数来自固定版本的 [Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md)。当前 Transformers 未直接提供 `presence_penalty` 参数，本实现通过 `LogitsProcessor` 对已经生成的不同 token 各减1.5，排除prompt，并在 temperature/top-k/top-p 前执行。

[VideoJev论文附录A.2](https://arxiv.org/html/2610.18615v1)的生成式QA基线采用正常生成后解析，附录A.4另有单token Direct基线。上游仓库没有公开完整的生成基线及备用文本提取器实现；这里实现保守的确定性解析，未解析成功、歧义、未闭合思考块和截断均计错。不会按候选logits强制补选答案，因此不声称逐项复现论文主表代码。

为使恢复前后的采样一致，每题用 `SHA256(f'{inference_seed}:{question_id}')` 的前8字节派生独立torch种子。问题顺序变化或跳过已完成题不会改变后续题的随机数流。这里的三次种子控制生成随机性，题目集合不变。

## 本机产物前置条件

下面的命令用于复用本机已有完整实验目录。`runs/`、权重和视频不随Git发布，新checkout不会自带这些输入；公开JSON摘要也不能代替原始候选概率、模型指纹及完整性记录。

新环境先按 [README的数据准备与历史候选打分步骤](../README.md#复现历史候选打分实验) 准备官方标注、视频、模型和原生后端。将历史示例的 `LVB_RUN` 设为 `runs/lvb100_proportional_seed42_43_44`，并将抽题循环设为仅 `for seed in 42`，完成该seed的四份模型分数、Controller及单次汇总；跳过需要三个抽题seed的末尾 `--mode all` 命令。随后生成下面汇总所需的候选采样缓存：

```bash
python scripts/sample_longvideobench_fixed_questions.py \
  --base runs/lvb100_proportional_seed42_43_44 \
  --sample-seed 42 --inference-seeds 42 43 44 \
  --out runs/lvb100_proportional_seed42_fixed_questions_sampling3_T1
```

已有完整产物时直接复用，无需重复执行这些前置步骤。使用其它目录时同步修改后续 `--manifest`、`--fixed-sampling` 等路径；来源校验不会因目录变更而关闭。

## 先做CPU检查

复用现有 `uv` 环境与权重，不联网下载：

```bash
source .venv/bin/activate
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/eval_longvideobench_qwen36_native.py \
  --manifest runs/lvb100_proportional_seed42_43_44/seed42/sample.jsonl \
  --out runs/lvb_native_dry_run --seed 42 --sample-seed 42 --dry-run
```

检查会重验原100题的抽样来源和本地模型结构、处理器及26个权重分片存在性，不加载模型、不计算全权重SHA，也不创建输出目录。

## GPU验证与完整运行

先确认六张卡有足够空闲显存，再将 `CUDA_VISIBLE_DEVICES` 替换成实际选定的六张卡。单题smoke使用独立目录，保留相同生成配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6
LVB_NATIVE=runs/lvb100_seed42_native_instruct_repeat3

python scripts/eval_longvideobench_qwen36_native.py \
  --manifest runs/lvb100_proportional_seed42_43_44/seed42/sample.jsonl \
  --out "$LVB_NATIVE/smoke" --seed 42 --sample-seed 42 --limit 1
```

smoke完成后再运行三次，并将日志与真实退出状态持久化。以下命令在Bash中执行：

```bash
mkdir -p "$LVB_NATIVE/logs"
for seed in 42 43 44; do
  if python scripts/eval_longvideobench_qwen36_native.py \
    --manifest runs/lvb100_proportional_seed42_43_44/seed42/sample.jsonl \
    --out "$LVB_NATIVE/seed$seed" --seed "$seed" --sample-seed 42 \
    > "$LVB_NATIVE/logs/seed$seed.log" 2>&1; then
    printf '0\n' > "$LVB_NATIVE/logs/seed$seed.exitcode"
  else
    trial_exit=$?
    printf '%s\n' "$trial_exit" > "$LVB_NATIVE/logs/seed$seed.exitcode"
    break
  fi
done

python scripts/summarize_longvideobench_native_baseline.py \
  --native-base "$LVB_NATIVE" \
  --fixed-sampling runs/lvb100_proportional_seed42_fixed_questions_sampling3_T1 \
  --out "$LVB_NATIVE/summary"
```

失败后只对对应seed增加 `--resume` 继续，保持其余参数和源码不变。恢复会重验模型、媒体内容、题目、代码和采样参数；smoke输出不能用于完整100题结果。汇总要求三次各100条真实生成记录、完成状态及一致的模型和输入协议，并重新回放另外两方法原有的候选采样。

每题输出保存完整生成token IDs、包含特殊token的原始decode、去特殊token的回答、派生种子、解析状态、是否截断及预处理信息。方差为固定题集上的三次准确率样本方差，分母为2，Markdown显示单位为百分点²。

## 本次验证范围

当前已经完成CPU输入检查和相关回归测试，GPU完整生成仍待运行。2026-10-31的资源预检显示8张RTX4090均有其它进程占用；即使空闲最多的一张卡，也低于历史同配置六卡运行中最低的单卡峰值，无法安全启动本轮smoke及三次完整评测。原结果和当前占卡进程保留。
