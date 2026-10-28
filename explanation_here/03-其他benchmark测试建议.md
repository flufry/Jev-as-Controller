# 03 · 在其他视频 benchmark 上测试：建议与操作方案

目标分两种，先想清楚你要哪一种，路线完全不同：

| 目标 | 需要跑什么 | 成本 |
| --- | --- | --- |
| **A. 复现 / 报告 F_s（VideoJev 单模型）** | 只跑 `scripts/evaluate_hf.py` | 低，代码已就绪 |
| **B. 验证方法本身（Controller）** | Qwen + VideoJev 同协议各跑一遍 + CPU 融合 | 高（35B 要 6 卡），需少量适配 |
| **C. 评你自己训练的新 checkpoint** | 特征缓存 + `scripts/evaluate.py` | 最高，只在训练后需要 |

下面按这个顺序展开。

---

## 1. 现状：已经有什么、缺什么

| Benchmark | 官方严格计分器 | 题数 | 需要的标注 | 现状 |
| --- | --- | ---: | --- | --- |
| `longvideobench` (val) | `timejev/longvideobench_benchmark.py` | 1337 | `official.json`（含正确选项、四时长组、3/4/5 选项分布） | ✅ 全链路已跑通（README 有转换脚本） |
| `videomme` | `timejev/videomme_benchmark.py` | 2700 | `official.json`（short/medium/long 各 900） | ✅ 计分器就绪，**缺**清单构建脚本 |
| `mlvu` (dev MCQ) | `timejev/mlvu_benchmark.py` | 2174 | `official.json`（7 任务，1122 视频） | ✅ 计分器就绪（含 M-Avg 宏平均），**缺**清单构建脚本 |
| `lvbench` | `timejev/qa_benchmark.py` | 1549 | `official.jsonl`（UID 全清单） | ✅ 计分器就绪，`qa_manifest.lvbench_rows` 可把官方 TSV 转成行 |
| `charades` / `activitynet` / `qvhighlights` | `timejev/benchmark.py` | 3363 / 4500 / 1541 | 官方 JSON | ✅ 时序定位（mIoU、R1@k），与 Controller 无关 |
| 任意自定义选择题 | 无（只做通用 accuracy） | — | `answer_index` 即可 | ✅ `--manifest` 路径 |

三条硬事实（决定了适配成本）：

1. **逐模型评测是清单驱动的**：`eval_longvideobench_qwen36.py` / `_videojev.py` 只认 `5.1 那种 JSONL + 协议参数，**没有**写死 LVB 的题目内容。
2. **但实验框架对 LVB 有耦合**：`timejev/lvb_experiment.load_sample()` 强制 `duration_group ∈ {15, 60, 600, 3600}`，`score()` 按这个字段分组统计。
3. **融合脚本是通用的**：`scripts/eval_longvideobench_controller.py` 只要清单里两源预测齐全、协议一致即可；而 `eval_longvideobench_proportional_controller.py` 把 LVB 官方标注、seed∈{42,43,44}、n=100 全部写死，**不能**用于别的数据集。

---

## 2. 推荐测哪些 benchmark（按优先级）

### 第一优先级：先把「已有计分器」的 4 个跑全量

理由：不用写计分代码；题量大（1337–2700），抽样噪声可忽略；而且能和 VideoJev 论文/上游 README 的公开数字比对，先确认你的权重与后端没跑歪。

| Benchmark | 为什么先跑 | 关键指标 |
| --- | --- | --- |
| **Video-MME** | 唯一带 short/medium/long 三档（各 900 题）的宽覆盖集，正好检验「时长越长，小模型评分器帮助越大」这个假设 | Accuracy + 三档分解 |
| **MLVU-Dev** | 7 类任务（plotQA/needle/ego/count/order/anomaly/topic），能看出 Controller 在**哪类能力**上补了 Qwen | M-Avg（宏平均，别报微平均） |
| **LVBench** | 超长视频（很多 >1 小时），是 Controller 增益最可能出现也最容易翻车的地方 | Accuracy + 分类 |
| **LongVideoBench val** | 已跑子集（100 题/seed），把它扩到全量 1337 才能和上面的数字同口径 | Accuracy + 四时长组 |

> 强烈建议**跑全量而不是子集**：现有 100 题子集的跨 seed 样本方差达 30.3 个百分点²（比例口径），意味着 ±5pp 级别的噪声；而 Controller 相对同协议 Qwen 的增益只有 2.67pp。子集实验无法分辨方法优劣。

### 第二优先级：新增 1–2 个「细粒度时序 / 动作」选择题集

用来回答「增益从哪来」：VideoJev 的强项是**时序定位**，若 Controller 的增益真来自时序感知，那么时序密集的 benchmark 上增益应更大。

| 候选 | 特点 | 适配工作量 |
| --- | --- | --- |
| **MVBench** | 20 个任务、大量动作/时序子任务，题量适中，社区常用 | 中：写一个官方标注→清单的转换器；指标为总平均 |
| **TempCompass** | 专门考时序（方向/速度/顺序），多选题 + 成对比较 | 中：同上 |
| **NExT-QA** | 因果/时序/描述三类，问题短、视频短 | 中：同上；可作为「短视频下限」对照 |
| **EgoSchema** | 3 分钟级长视频、只有 test 标签不公开 → 只能自建 val 子集 | 中高：需说明「非官方划分」 |
| **VideoMMMU** | 多学科、难度高，能看上限 | 中：同上 |

不建议：`LongVideoBench` test（无公开标签）、需要字幕/音频才能公平比较的赛道（本协议明确 `use_subtitles=False, use_audio=False`）。

### 第三优先级：只测 F_s 的定位能力

Controller 公式只对**选择题**成立（`F_s` 是候选概率）。若要展示 VideoJev 的另一半能力，用 `charades` / `activitynet` / `qvhighlights`（已内置，`--benchmark vtg`），但这与 Controller 结论是两件事，别混在同一张表里。

---

## 3. 路线 A：只测 F_s（VideoJev 单模型）

### 3.1 准备目录

```text
data/benchmarks/<name>/
  questions.jsonl     # 题数必须 == 官方全量（2700 / 2174 / 1549 / 1337）
  videos/...          # 或 videos/<videoID>.mp4
  official.json       # lvbench 用 official.jsonl
```

### 3.2 Video-MME 清单构建（示例）

⚠️ 官方标注里 `duration` 是 `short/medium/long` **标签**，不是秒数；清单的 `duration` 必须是真实时长，否则 `rgb_media.metadata()` 会按 `min(清单时长, 视频真实时长)` 截断采样时间线。

```python
import json
from pathlib import Path
from timejev.rgb_media import probe_video_geometry
from timejev.videomme_benchmark import validate_reference

base = Path('data/benchmarks/videomme')
official = json.loads((base / 'official.json').read_text())
validate_reference(official)                      # 2700 题 / 900 视频 / 三档各 900

rows = []
for r in official:
    video = base / 'videos' / f"{r['videoID']}.mp4"
    _, _, seconds = probe_video_geometry(video)   # 探测真实时长，不要用标签
    rows.append(dict(
        id='videomme/' + r['question_id'],
        question_id=r['question_id'],
        video_key='videomme/' + r['videoID'],
        video_path=str(video.relative_to(base)),
        duration=seconds,
        task='choice',
        query=r['question'],
        options=r['options'],                     # 保持官方原样，顺序即字母顺序
        answer_index=ord(r['answer']) - 65,
        duration_group={'short': 60, 'medium': 600, 'long': 3600}[r['duration']],
        task_type=r['task_type'],
    ))

with (base / 'questions.jsonl').open('x', encoding='utf-8') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
```

`duration_group` 那一行是**为了迁就 `lvb_experiment.load_sample`**（详见第 5 节）；`task_type` 供以后接官方计分器用。

### 3.3 跑评测

```bash
source .venv/bin/activate && source scripts/env.sh
python scripts/evaluate_hf.py --checkpoint models/VideoJev-4B \
  --benchmark videomme --data-root data/benchmarks --out runs/hf_videomme --dry-run
python scripts/evaluate_hf.py --checkpoint models/VideoJev-4B \
  --benchmark videomme --data-root data/benchmarks --out runs/hf_videomme
```

输出 `metrics.json` 里有 `accuracy` 与 `by_duration`；这条路径不写 `run.json` 指纹，**不适合作为 Controller 的输入源**（Controller 需要 `Experiment` 写的 `run.json`）。要喂 Controller，见路线 B。

---

## 4. 路线 B：把 Controller 扩展到新 benchmark（推荐）

五步，全部沿用现有脚本，只在「清单构建」和「汇总」两处写少量新代码。

### Step 1 · 构建清单（`questions.jsonl`）

字段定义见 [01-架构总览](01-架构总览.md) 第 5.1 节。要点：

- `options` 必须与官方字符串**逐字**相同（`score_videomme` 用 `row['options'] != target['options']` 比对）。
- MLVU 例外：官方计分器用 `formatted_options()` 给每条加 `"A. "` 前缀，清单也要加。
- MLVU 还需 `video_key = 'mlvu/' + Path(target['video']).stem`、`official_duration`、`task_type`。
- 同一视频的多题共享 `video_key`，且 `video_path`/`duration` 必须一致（`group_memberships` 会校验）。

### Step 2 · 处理 `duration_group` 耦合（三选一）

| 做法 | 代价 | 适用 |
| --- | --- | --- |
| ① 映射进 `{15,60,600,3600}`（Video-MME：short→60、medium→600、long→3600） | 零代码，但 `by_duration_group` 报表语义变成「分档」而非「官方组」 | 想最快出数 |
| ② 写一个 `<bench>_experiment.py` 变体，复用 `Experiment` 但换 `load_sample`/`score` 的分组键 | 约 100 行，最干净 | 要发表级产物（推荐） |
| ③ 给 `lvb_experiment` 加参数（`groups=`、`stratum_key=`） | 改动核心模块，需回归 8 个 LVB 测试 | 长期维护 |

做法 ② 的最小骨架（**建议新增文件，不要改 `lvb_experiment.py`**，避免影响 LVB 结果的可复现性）：

```python
# scripts/nb_experiment.py（示例骨架，未在仓库中）
from timejev.lvb_experiment import Experiment, score, read_predictions, write_json
from timejev.user_data import load_manifest

STRATUM = 'task_type'          # 或 'duration'

def load_sample(path):
    rows = load_manifest(path)
    if not rows or any(r['task'] != 'choice' for r in rows):
        raise ValueError('新 benchmark 清单需要非空选择题')
    return rows

def score_rows(rows, by_id):
    hits = sum(by_id[r['id']]['answer_index'] == r['answer_index'] for r in rows)
    return dict(n=len(rows), correct=hits, accuracy=hits / len(rows))
```

> 注意 `Experiment.__init__` 内部直接调用 `lvb_experiment.load_sample` 与 `score`，所以做法 ② 实际是在**运行时替换模块级函数**（像 `eval_longvideobench_proportional_controller.py` 用 `legacy.read_source = ...` 那样做猴子补丁），而不是继承。若觉得 hack，直接复制一份框架文件改名更稳。

### Step 3 · 两模型同协议推理

```bash
source .venv/bin/activate && source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=data/benchmarks/videomme/subset_or_full.jsonl

# 控制器输入协议：2 FPS / 2048 帧 / 262144 / 24576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 python scripts/eval_longvideobench_qwen36.py \
  --manifest $M --out runs/videomme/qwen_ctrl --checkpoint models/Qwen3.6-35B-A3B \
  --seed 42 --device cuda:0 --gpu-count 6 --max-memory-gib 18.5 \
  --fps 2 --max-frames 2048 --context 262144 --visual-token-budget 24576 --memory-fraction .9

CUDA_VISIBLE_DEVICES=7 python scripts/eval_longvideobench_videojev.py \
  --manifest $M --out runs/videomme/videojev_ctrl --checkpoint models/VideoJev-4B \
  --seed 42 --device cuda:0 --fps 2 --max-frames 2048 \
  --context 262144 --visual-token-budget 24576 --memory-fraction .9
```

两个 `--fps/--max-frames/--context/--visual-token-budget` **必须逐字相同**，否则融合直接报错。另外建议再跑一套「算力更强的参考基线」（如 4 FPS / 4096 帧）用于论文对照，但要明确写「未与 Controller 做算力对齐」。

### Step 4 · 融合（纯 CPU）

```bash
python scripts/eval_longvideobench_controller.py \
  --manifest $M --qwen runs/videomme/qwen_ctrl --videojev runs/videomme/videojev_ctrl \
  --out runs/videomme/controller --beta 2 --qwen-method qwen36
```

输出 `comparison.json`（qwen / videojev / controller 三方）与 `predictions.jsonl`（含 `candidate_logits`、两源概率、β）。

### Step 5 · 用官方计分器交叉校验

`comparison.json` 里的准确率来自 `lvb_experiment.score`（只比 `answer_index`）。要在论文里报「官方口径」，再把 controller 的预测交给官方计分器：

```python
import json
from pathlib import Path
from timejev.videomme_benchmark import score_videomme

rows = [json.loads(l) for l in Path(M).read_text().splitlines()]
preds = [json.loads(l) for l in Path('runs/videomme/controller/predictions.jsonl').read_text().splitlines()]
official = json.loads(Path('data/benchmarks/videomme/official.json').read_text())
metrics, answers, scored, per_video = score_videomme(preds, rows, official)
print(metrics['accuracy'], metrics['by_duration'])
```

`score_videomme` 会核对官方全量、选项顺序、答案字母，是「防伪」的最后一道闸。

### Step 6 · 汇总

`summarize_longvideobench_proportional.py` 会重验 LVB 抽样元数据（`SAMPLING` / 配额 / 官方 reference SHA），**不能直接用于新数据集**。两个选项：

- 快速：直接从各方法 `predictions.jsonl` 重算准确率、随机选边、OR 上界（`simulate_longvideobench_random_choice.py` 的算法可照抄）。
- 规范：仿 `summarize_longvideobench_proportional.py` 写一个 `summarize_<bench>.py`，保留同样的「三方法 + 随机 + OR + 分档 + 指纹校验」结构。

---

## 5. 需要改动 / 新增的代码点清单

| # | 位置 | 现状 | 建议 |
| --- | --- | --- | --- |
| 1 | `timejev/lvb_experiment.load_sample` | 强制 `duration_group ∈ GROUPS` 且 `media_source.kind == 'video'` | 新增 `<bench>_experiment.py`（做法②）或加参数化（做法③） |
| 2 | `timejev/lvb_experiment.score` | 按 `duration_group` 分组 | 同上；或允许 `stratum` 键 |
| 3 | `lvb_experiment.prepare_record` | `fps==2 and max_frames==2048` 时以 `benchmark='longvideobench'` 调 `rgb_media.metadata` → 可能命中 `data/frames/longvideobench/<stem>` 帧缓存 | 新数据集给每行显式 `media_source`，或确认 `data/frames/longvideobench/` 下没有同名 stem |
| 4 | `scripts/sample_longvideobench_proportional.py` | 只用 LVB 官方标注、最大余数法配额 | 抽新数据集时写同构的 `sample_<bench>_proportional.py`（算法可照抄 `allocate_counts`） |
| 5 | `scripts/eval_longvideobench_proportional_controller.py` | 硬编码 seed∈{42,43,44}、n=100、LVB meta 校验 | 新数据集**改用** `eval_longvideobench_controller.py`，不要用这个适配器 |
| 6 | `summarize_longvideobench*.py` | 校验 LVB 抽样协议 | 新写 `summarize_<bench>.py` |
| 7 | `timejev/hf_evaluation.evaluation_inputs` | 官方 benchmark 强制全量题数（`COUNTS`） | 只想跑子集时用 `--manifest`（走 `custom` 分支，跳过计数校验） |
| 8 | 各 `*_benchmark.py` | 已有 4 个 QA 计分器 | 新数据集需要新计分器时，照 `score_videomme` 的「先 validate_reference、再逐题比对、最后分组统计」结构写 |
| 9 | `configs/qa_evaluation.json` | 只对 4 个 QA benchmark + 特征缓存注册 | 只有走路线 C（训练 checkpoint）才需要补条目 |

---

## 6. 实验设计建议（比"跑哪个 bench"更重要）

### 6.1 必做的三个诊断（先做，再决定是否全量）

1. **OR 上界**（`Qwen ∪ VideoJev` 至少一个答对）：这是 Controller 的绝对天花板。LongVideoBench 上 OR=77.33% vs Qwen=69.00%，理论上限只有 +8.3pp；如果某 benchmark 上 OR 只比 Qwen 高 1–2pp，**别投算力**。
2. **异答率**（两模型答案不同的题目占比）：Controller 只在异答样本上改变结果，异答越少，增益上限越小。
3. **β 敏感性**（CPU 零成本，只需重跑融合）：β∈{0.5,1,2,3,5}。若结论对 β 极敏感，说明增益不稳。

### 6.2 统计口径

- **配对设计优于多样本重复**：同一批题上比 Qwen vs Controller，用 McNemar 检验或 bootstrap（10k 次重采样）给 95% CI。现有 3-seed 方差法在小样本上太不稳定（30.3 pp² 就是证据）。
- **题量**：单格 100 题 → 95% CI 约 ±9pp；要分辨 +2–3pp 需要 ≥600–1000 题/格，或直接用全量。
- **多 seed 仍要报**：seed 只影响抽样，不影响模型；全量实验里 seed 只剩「无意义」——那时应该报 bootstrap，而不是跨 seed 方差。
- 报告里必须写清：视觉 token 预算（24576 ≠ 论文 262144×80%）、无字幕/音频、Controller 与 4 FPS 基线**未做算力对齐**。

### 6.3 预算估算方法

先跑 100 题 pilot，然后：

- Qwen 侧：`runs/.../qwen/predictions.jsonl` 每题带 `timing.question_inference_seconds` 与 `video_preprocessing_seconds`；`runtime.json` 有 `wall_seconds`。全量时间 ≈ 单题推理 × 题数 + 视频数 × 预处理（同视频多题共享一次预处理，但每题一次完整前向）。
- VideoJev 侧：按视频分组，`timing.video_group_total_seconds` 已包含预处理 + 编码 + 同组全部题目。
- **35B 是瓶颈**：6 卡 2048 帧/题的单次前向是秒级到十几秒量级，2700 题 ≈ 十几个小时的量级，先 pilot 再外推，别拍脑袋。
- 融合与统计全部 CPU、秒级，可以随便扫 β。

### 6.4 建议的表格结构（每个 benchmark 一张）

| 方法 | 协议 | n | 准确率 | 95% CI | 与 Qwen(2fps) 差值 | McNemar p |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | 2 FPS / 2048 | | | | — | — |
| VideoJev-4B | 2 FPS / 2048 | | | | | |
| **Jev-as-Controller (β=2)** | 2 FPS / 2048 | | | | | |
| Qwen3.6-35B-A3B（算力更强参考） | 4 FPS / 4096 | | | | | |
| VideoJev-4B（算力更强参考） | 4 FPS / 4096 | | | | | |
| 随机选边（50 次均值） | — | | | | | |
| OR 上界（需标签，不可作方法） | — | | | | | |

再附一张「按时长/任务类型」的分组表和一张「β 敏感性」表。

---

## 7. 陷阱清单（换数据集时最容易踩）

| 风险 | 说明 |
| --- | --- |
| **时长字段填错** | `duration` 会被 `min(清单值, 真实时长)` 夹住：填小了会**截断视频**（后半段根本没采样）。必须用 `probe_video_geometry()` 探测真实秒数，不要用官方标签或元数据估算 |
| **字幕/音频不对等** | 本协议强制 `use_subtitles=False, use_audio=False`。Video-MME 官方榜常用字幕，**你的数字天然低于榜单**，报告里要写明 "video-only" |
| **选项文本/顺序被改写** | 官方计分器逐字比对；MLVU 还有 5 题存在重复候选文本，官方规则是取**第一个**匹配项 |
| **帧数/上下文上限** | 超长视频（LVBench 1h+）在 2048 帧 + 24576 视觉 token 下被大幅降采样；`prepare_record` 会在预算内取最大网格（最小边 1024 起）并在放不下时直接抛错。要提升预算必须两模型同时提 |
| **帧缓存路由串台** | `prepare_record` 在 2FPS/2048 下用固定名字 `'longvideobench'` 去查 `data/frames/longvideobench/<stem>`；新数据集的视频若与 LVB 视频同名 stem，会读到别人的帧。用显式 `media_source` 或改这段路由 |
| **官方计分器拒绝子集** | `validate_reference` 要求 2700/2174/1549/1337 全量。跑子集只能走 `--manifest` 的通用 accuracy，并在报告里标注 "subset, 非官方口径" |
| **训练污染未知** | 本仓库无法核实 VideoJev 训练清单，任何 benchmark 都可能已被训练见过。保留 `training_overlap_verified=false` 的表述，不要写「已验证无污染」 |
| **许可证** | Video-MME / MLVU / LVBench / LongVideoBench 各自许可不同；视频与题目全文不要提交进仓库（现有 `docs/results/` 只存 ID 与摘要，照此办理） |
| **多机/多卡混用** | 35B 脚本要求 `--gpu-count ∈ {4,5,6}` 且 `--device` 是逻辑 `cuda:0`；VideoJev 单卡即可。两者可并行但要留足显存（`--memory-fraction .9` 是 per-process 上限） |
| **重跑覆盖** | 所有目录都拒绝非空；想重跑就换目录，别 `rm` 已有结果——它们带指纹，是复现证据 |

---

## 8. 执行顺序 checklist

```text
[ ] 1. 选 1 个 benchmark（建议 Video-MME）+ 写清单构建脚本 + validate_reference 通过
[ ] 2. evaluate_hf.py --dry-run 校验清单（不加载权重）
[ ] 3. 抽样 100 题 pilot：Qwen(2fps) + VideoJev(2fps) 各一次
[ ] 4. CPU 融合，看 异答率 / OR 上界 / β 敏感性
        ├─ OR 上界提升 < 1pp → 放弃该 benchmark 或先查数据/协议
        └─ 有空间 → 继续
[ ] 5. 全量 F_s（evaluate_hf.py --benchmark <name>），与上游公开数字比对
[ ] 6. 全量 Controller（Qwen + VideoJev 同协议 + 融合）
[ ] 7. 官方计分器交叉校验（score_videomme / score_mlvu / score_lvbench）
[ ] 8. 汇总：主表 + 分档表 + β 敏感性 + bootstrap CI + McNemar
[ ] 9. 另 1–2 个 benchmark 做同样的 3–8（不同任务类型才算复现）
```

若只想先出一版可信的公开结果，**最小可行方案**是：Video-MME 全量（2700）跑完整 Controller + 同协议 Qwen 基线，并用官方计分器出 Accuracy 三档分解。它一次就能同时回答「跨数据集是否有效」和「长视频是否增益更大」两个问题，且不需要任何新的计分代码。
