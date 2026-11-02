# 12 · Controller 逐题变化与概率导出

脚本 `scripts/analyze_controller_changes.py` 读取已经完成的模型与 Controller 结果，核验来源 SHA、题集、seed、协议、逐题预处理和融合公式，然后导出统计与逐题概率。只用 Python 标准库，无需 GPU、模型、视频或重新推理；从服务器复制到本地的结果也可以分析，原运行中保存的 Linux 路径不必存在。

## 当前 Video-MME 三 seed 本地命令

在仓库根目录执行，路径中有空格时保留引号。下面命令在 PowerShell 和 Bash 均可作为一整行执行：

```text
python -X utf8 scripts/analyze_controller_changes.py --runs-root D:/cmder/runs --seeds 42 43 44 --out runs/analysis/videomme_controller_changes
```

批量模式使用 `sample_42.jsonl`、`sample_43.jsonl`、`sample_44.jsonl`；seed 42 的清单不存在时才尝试 `sample.jsonl`。seed 42 对应无后缀的 `qwen_logits`、`videojev`、`controller`，其他 seed 使用 `_43`、`_44` 后缀。最终仍会校验实际 seed 与清单 SHA，不能靠重命名混用结果。

在服务器容器内，相应命令为：

```bash
python scripts/analyze_controller_changes.py \
  --runs-root runs/screen/videomme --seeds 42 43 44 \
  --out runs/analysis/videomme_controller_changes
```

脚本拒绝覆盖已有输出；重复执行时换一个新的 `--out`。

## 单个 seed 或其他目录命名

```bash
python scripts/analyze_controller_changes.py \
  --manifest runs/screen/videomme/sample_42.jsonl \
  --qwen runs/screen/videomme/qwen_logits \
  --videojev runs/screen/videomme/videojev \
  --controller runs/screen/videomme/controller \
  --out runs/analysis/videomme_controller_seed42
```

`--qwen` 必须是此次 Controller 实际使用的候选 logits 输入目录，不能传 `qwen_native` 或不同 FPS 的独立基线。输入为三个完整目录，各需 `run.json`、`predictions.jsonl`、`metrics.json`，另需原 sample 清单；已有 sample meta 也会核验。

## 统计口径与输出

“修改答案”指 Controller 的 top-1 不等于输入 Qwen 的 top-1；“改对”指 Qwen 错而 Controller 对；“改错”指 Qwen 对而 Controller 错；“改后仍错”指两者都错但选了不同选项。因此 `改答 = 改对 + 改错 + 改后仍错`，`净增正确 = 改对 − 改错`。JSONL 另存相对 VideoJev 的修改、纠正和损害标记。

`summary.md` 和 `summary.json` 提供每个 seed 和合计统计。合计单位是题次，跨 seed 可能重复出现同一道题，逐题记录同时保留 seed 与 ID。

每一组均生成同名 `.md` 和 `.jsonl`，Markdown 适合阅读，JSONL 保留原始数值精度供后续分析：

| 文件名（不含扩展名） | 内容 |
|---|---|
| `all` | 所有题目 |
| `disagreements` | Qwen 与 VideoJev 异答题 |
| `changed` | Controller 修改 Qwen 答案的题目 |
| `rescued` | 改对 |
| `harmed` | 改错 |
| `changed_still_wrong` | 修改后仍错 |
| `third_choice` | Controller 选了两个专家 top-1 之外的选项 |

每题列出问题、全部选项、正确选项、两专家与 Controller 的 top-1，以及每个选项的：

| 字段 | 含义 |
|---|---|
| `qwen_logit` | 输入 Qwen 候选字母原始分数 z |
| `qwen_probability` | π₀ = softmax(z) |
| `videojev_probability` | VideoJev 的 Fₛ，即实际融合所用的候选概率 |
| `correction` | β × Fₛ |
| `fused_score` | z + β × Fₛ |
| `controller_probability` | π₁ = softmax(z + β × Fₛ) |

`z + βFₛ` 与 `log(π₀) + βFₛ` 相差每题所有选项共享的常数，两者 softmax 结果相同。所有概率是原始候选选项内的归一化概率。脚本不对 VideoJev 概率取 log，也不再次对它做 softmax。

本次 `D:/cmder/runs` 实际导出：450 题次、异答 92、改答 32、改对 13、改错 12、改后仍错 7、第三选项 1。逐 seed 的改对/改错分别为 42：5/2，43：3/4，44：5/6；合计净增正确 1。
