# 四个视频 benchmark：全量固定题集、三个答案采样 seed

本流程在当前仓库实现，按本地latest提交`cb0518a5ea12129384489ab8eabcaf57956e9d2e`对齐。**三份候选来源各执行一次GPU前向，seed42/43/44复用分布采样；仅native实际生成三遍**。无需等待历史100题实验，不使用latest中的本机运行目录。**全量指有公开标签的评测集，不是训练集或隐藏标签测试集。**

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
| 单步 Qwen logits（融合来源） | 2 FPS / 2048帧 | seed42计算一次全量首步logits；三个seed分别从同一softmax(z_Q)候选分布采样 |
| VideoJev-4B 独立基线 | 4 FPS / 4096帧 | seed42计算一次全量4FPS分数；三个seed分别从同一候选分布采样 |
| Controller 旧方法 | 两源均2 FPS / 2048帧 | `softmax(z_Q + 2*p_J)`，共用两份2FPS来源，各seed重新采样融合分布 |
| Controller 新方法 | 两源均2 FPS / 2048帧 | `softmax(z_Q + 2*s_J)`，共用两份2FPS来源，各seed重新采样融合分布 |
| 单步 Qwen OR VideoJev-4B | **Qwen 2 FPS + VideoJev 4 FPS** | 同seed的单步Qwen采样与VideoJev采样至少一个答对；与latest公开表定义一致，native不参与 |

`z_Q`是Qwen首decode位置裸字母候选logits，`s_J`是VideoJev原始head logits，`p_J=softmax(s_J)`。融合还需要一份 **2 FPS VideoJev** 来源，不能拿4 FPS基线与2 FPS Qwen混用。

单步Qwen、VideoJev和两版Controller采用temperature=1、无top-k/top-p的候选CDF采样，沿用latest的固定原题序。单步Qwen使用`random.Random(f'{seed}:qwen36')`；VideoJev使用`random.Random(f'{seed}:videojev')`；两版Controller各自重置`random.Random(f'{seed}:jev_as_controller')`，使用相同均匀随机序列，以便比较公式。

native沿用关闭thinking的原设置：temperature=.7、top_p=.8、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1，默认最多32768个输出token。每题使用`SHA256(f'{seed}:{question_id}')`派生torch CPU/CUDA种子。无效、歧义、截断均计错，保留在分母中。

默认context=262144、visual-token-budget=24576、不使用字幕/音频。首次全量候选分数仍需覆盖全部问题和完整视频问题组，不拼接旧子集分数。GPU阶段为Qwen logits 2FPS一次、VideoJev 2FPS一次、VideoJev 4FPS一次、native三次，共**6个GPU阶段**。默认顺序为**Qwen logits → native42 → native43 → native44 → VideoJev 2FPS → VideoJev 4FPS → 三份单seed汇总 → 总表**（共10个队列阶段，其中4个是CPU）。由于单seed表需要VideoJev来源，所有CPU汇总都在模型阶段结束后执行。

原始候选来源的seed固定为42，最终报告的seed42/43/44是答案采样seed。其差异不依赖重新执行确定性GPU前向。单步Qwen的argmax与采样成绩分别保存。

### 与latest的对齐边界

- 执行方式、VideoJev/Controller的CDF采样及两版融合共用均匀随机序列，与latest已公开函数一致。
- latest完整六行公开表的OR明确是**单步Qwen 2FPS OR VideoJev 4FPS**。本版改成同一定义，不再把native双4FPS OR放进该行。
- latest未提供直接生成完整六行表的专用脚本。单步Qwen的随机数命名沿用其已有`qwen36`候选采样约定，并明确写入JSON；不能仅凭口径对齐保证复现公开表的每个数值。题集原序、checkpoint、实际分数、环境及随机数约定都需相同。
- 当前仍扩展到四个benchmark，MLVU使用M-Avg；不存在把LongVideoBench公开成绩复制到新报告的行为。

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
  input_hash_cache.json            # 默认首次校验的视频及两套模型指纹，后续阶段/恢复共用
  logs/<stage>.attemptN.log
  logs/<stage>.attemptN.exitcode
  sources/qwen_controller/        # 一次2FPS Qwen首步前向，来源seed42
  sources/videojev_controller/    # 一次2FPS VideoJev前向，来源seed42
  sources/videojev_baseline/      # 一次4FPS VideoJev前向，来源seed42
  native/seed42/                  # 实际完整自回归生成
  native/seed43/
  native/seed44/
  per_seed/seed42/results.md       # 两模型全部完成后生成该seed六方法表
  per_seed/seed43/results.md
  per_seed/seed44/results.md
  summary/results.md             # 六行六列：方法、三seed、均值、样本方差
  summary/results.json           # 精确值、分组指标、来源SHA、OR定义
  summary/*.distributions.jsonl    # 三个seed共用的候选分布
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

## 旧运行的安全复用与迁移

当前队列schema为`video_bench_full_queue_v4`，执行协议为`shared_candidate_sources_three_sampling_seeds_v1`，总表schema为`video_bench_full_three_seed_summary_v4`。旧队列的`experiment.json`不能直接用新代码`--resume`；不要修改其中的源码SHA或状态绕过校验。停止旧队列再同步代码，旧目录保留。

### 三份候选来源均已完整完成，native尚未完成

用新base并只读引用旧实验的三份来源，可免去三次GPU前向。支持旧共享布局`OLD/sources/<role>`及旧per-seed布局`OLD/sources/seed42/<role>`。同时存在两种布局会拒绝自动选择。每份来源必须完整覆盖全量题集、seed为42，模型/协议匹配、源码快照SHA及保存指标可验证；缺题/未完成来源不可复用。旧native不会被这个选项复用，本次仍实际执行三个native seed。

```bash
python scripts/run_bench_full.py --bench videomme \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_latest_aligned \
  --reuse-sources-from runs/full/videomme_per_seed4 \
  "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench videomme \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_latest_aligned \
  --reuse-sources-from runs/full/videomme_per_seed4 \
  "${COMMON[@]}"
```

`--dry-run`在指定复用时也会读取并校验候选来源（CPU），不会占用GPU或创建新队列。若旧目录叫`videomme_fixed_python`，替换`--reuse-sources-from`即可；若三份来源尚未全部完成，不能使用该选项，保留旧目录后用新base完整运行。首次新队列不加`--resume`，后续恢复保留同一复用参数再加`--resume`。

### 六组所需来源均已完成，只重新生成表格

旧候选来源及三个native均完整时，可以直接从原实验目录生成新的CPU报告，不重新运行任何GPU阶段：

```bash
python scripts/summarize_bench_full.py \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_fixed_python \
  --out runs/full/videomme_fixed_python/summary_latest_aligned
```

如果native与三份候选来源位于不同实验目录，使用`--base`指定包含`native/seed42,43,44`的目录，`--source-base`指定包含候选来源的目录。来自旧per-seed完整实验时只读取seed42候选来源，seed43/44的重复分数不参与latest口径报告。所有来源仍保留其真实路径、配置、原始SHA记录与封存源码，不修改历史产物。三次native的配置、checkpoint、依赖和源码必须一致。

### 新队列阶段顺序

```text
首次输入内容哈希（原视频+两套模型）
qwen_controller → native_seed42 → native_seed43 → native_seed44
videojev_controller → videojev_baseline
summary_seed42 → summary_seed43 → summary_seed44
summary（最终六行、三seed、均值、样本方差）
```

复用来源时前三个阶段标记为`reused`，其余完成阶段标记为`completed`。单个seed的样本方差标为“—（单seed）”；总表用三个seed主指标计算ddof=1方差。三seed的成绩允许偶然相同；截图仅作表格样式参考，所有数值均由实际预测重算。

## 旧版本辅助入口：恢复时显式复用输入哈希

下面两个helper主要用于尚未接入实验级缓存的旧冻结队列。新版runner默认使用下一节的一次哈希缓存，无需手动调用这些helper。

当已确认视频和权重没有变化，且不想在失败重试时再次完整读取大文件，可使用独立入口`resume_bench_fast.py`。它仍调用原`run_bench_full.py`，保留队列源码、设置、完整清单和预测校验；仅对存在`run.json`的恢复模型子进程复用历史视频/权重SHA256。新模型阶段没有历史记录时仍正常计算全部输入哈希。

```bash
/opt/jev/bin/python scripts/resume_bench_fast.py \
  --bench videomme \
  --manifest runs/full/videomme/questions.jsonl \
  --base runs/full/videomme_latest_aligned_restart \
  --qwen-gpus 0,1,2,3,4,5 --videojev-gpu 7 \
  --qwen-python /opt/jev/bin/python --videojev-python /opt/jev/bin/python \
  --resume
```

除入口名称外，必须使用原队列的全部参数。该入口不允许首次运行，不能绕过源码/参数不一致，也不消除模型加载和实际视频解码所需的读取。

- 视频：检查存在、路径、字节大小及`mtime_ns`，随后复用已记录SHA256。
- `.safetensors`：检查存在、路径和字节大小；旧记录有mtime时也检查。当前历史checkpoint记录通常不含mtime，因此不能识别同大小的权重内容变更。
- 小配置、tokenizer、源码、官方标注、清单和预测文件：沿用原校验，不复用其内容hash。
- 若文件大小或已记录的mtime变化，则拒绝快速恢复；应使用原runner执行完整校验，不要手动改记录。

**这不是完整内容一致性验证。**同大小且保留mtime的内容变化可能漏检，用户需明确确认输入未修改。脚本在模型目录新增`input_hash_reuse.jsonl`记录helper源码SHA、先前run.json SHA及本次检查强度，不改写历史identity。日志会打印`input_hash_reuse`事件。

对正在使用冻结源码的旧实验，只上传新增的`scripts/resume_bench_fast.py`即可；不要为这个功能更新已纳入原队列指纹的文件。它也不会修复其他既有源码差异。缺少run.json时没有可复用的已完成哈希，仍需首次初始化。

## 已有队列先执行 Qwen native

若Qwen logits已完成、希望优先使用空闲的多张GPU完成native，可只上传新增`scripts/run_bench_native_first.py`，不修改冻结评测源码。先停止原runner并确认其子进程退出，进入容器tmux执行：

```bash
/opt/jev/bin/python scripts/run_bench_native_first.py \
  --base runs/full/videomme_latest_aligned_restart --dry-run
/opt/jev/bin/python scripts/run_bench_native_first.py \
  --base runs/full/videomme_latest_aligned_restart
```

它读取原experiment.json中的native三组原命令、设备及seed，不重新运行Qwen logits、不运行VideoJev。每次原模型进程成功退出并核验完整结果后，才记录该阶段completed。三个native完成后整体状态为awaiting_remaining_stages，不宣称整个实验完成。仍会执行各native阶段原有的输入哈希和模型加载。

之后在VideoJev所需单卡可用时，用原run_bench_full.py、原参数和--resume继续；完成的native会核验后跳过，VideoJev及CPU汇总继续执行。原identity和已完成其他阶段记录不变，新增入口仅改变实际执行顺序；日志路径仍为logs/native_seedXX.attemptN.log。若该入口中断，重新执行同一命令会按已有run.json恢复native。不能与原runner同时运行；Linux下会检查同一base的活动runner/模型进程并拒绝并发。

### 三个native都复用已完成Qwen来源的输入哈希

已确认输入未变化时，可显式指定原Qwen来源的run.json：

```bash
/opt/jev/bin/python scripts/run_bench_native_first.py \
  --base runs/full/videomme_latest_aligned_restart \
  --trust-input-hashes-from runs/full/videomme_latest_aligned_restart/sources/qwen_controller/run.json
```

需同步更新两个独立helper：`scripts/run_bench_native_first.py`和`scripts/resume_bench_fast.py`。它们未加入冻结评测源码列表，不需要修改原identity；停止旧进程后再替换helper。此模式覆盖第一次启动的seed和中断恢复的seed，三次都复用所指定来源的视频、safetensors哈希。source run.json SHA会固定并逐次核验，题集SHA、视频集合及路径、checkpoint文件库存、小配置内容仍核验，旧来源不会被改写。

这是显式信任旧内容哈希，不是完整内容校验：视频仅检查路径、大小和mtime；历史权重没有mtime时仅检查路径和大小。同大小内容修改可能漏检。每次运行单独保存`input_hash_reuse.jsonl`并在队列attempt中标明来源和策略。模型仍须各加载三遍，首次视频实际解码仍须读取文件。缺少完整旧输入指纹时不能用此模式。完成后原runner照常--resume继续VideoJev及汇总。

## 后续实验可选：Qwen与VideoJev不同GPU并行

仅在新实验启动时添加`--parallel-models`。默认串行不变；不要为了使用本功能更新正在运行的Video-MME源码，等该实验结束后再同步。

```bash
/opt/jev/bin/python scripts/run_bench_full.py --bench mlvu \
  --manifest runs/full/mlvu/questions.jsonl \
  --base runs/full/mlvu_parallel \
  --qwen-gpus 0,1,2,3,4,5 --videojev-gpu 7 \
  --qwen-python /opt/jev/bin/python --videojev-python /opt/jev/bin/python \
  --parallel-models --dry-run
```

计划确认后去掉`--dry-run`。LongVideoBench/LVBench等使用同一选项，替换`--bench`、清单及输出目录。恢复保留全部参数（包括`--parallel-models`）再加`--resume`。

```text
Qwen通道：Qwen logits 2FPS → native42 → native43 → native44
VideoJev通道：VideoJev 2FPS → VideoJev 4FPS
两个通道全部完成 → seed42/43/44 CPU汇总 → 最终总表
```

最多同时存在一个Qwen模型进程和一个VideoJev模型进程。VideoJev卡不得属于Qwen卡列表（例如默认VideoJev GPU0与Qwen0..5重叠会被拒绝）。**只启动这一份runner，不额外启动native-first或另一份原队列**。CPU汇总推迟到全部模型阶段结束，避免读取尚未完成的来源。

此选项仅改变调度顺序，不改变checkpoint、帧率/帧数、seed、采样或融合公式。不同GPU仍共享CPU、RAM和磁盘带宽，并行不保证总时间一定更短，也不自动预留或检测他人GPU占用。新实验默认在主进程先完成一次输入哈希缓存，再启动两通道，子进程只复用视频/权重SHA；不会由两通道并发重复建立缓存。

所有状态仅由主调度进程写入：`experiment.json`增加`active_stages`列表，同时运行两阶段时旧`active_stage`字段为null，**此时不代表空闲**。可用下面命令在宿主机查看（替换数据集目录）：

```bash
docker exec jev-bench-full /opt/jev/bin/python -c '
import json
from pathlib import Path
s=json.loads(Path("/workspace/Jev-as-Controller/runs/full/mlvu_parallel/experiment.json").read_text())
print("整体:",s["status"],"活动阶段:",s.get("active_stages",[]))
for name in s.get("active_stages",[]):
    a=s["steps"][name]["attempts"][-1]
    print(name,"PID:",a.get("pid"),"日志:",a["log"])
'
```

每个阶段保持独立`attemptN.log`和真实退出码。一个通道失败/校验失败或按Ctrl+C时，调度器只终止自己启动的另一通道子进程，保留两边的完整预测供恢复；不会杀服务器其他用户任务，不自动重试。源码、参数、来源和已完成产物的指纹保护保持不变。

## 新版默认：Qwen优先，一次输入哈希

不加额外参数即可使用新默认。它不改变帧率、帧数、模型、采样或融合方法，只改变队列顺序和输入校验成本。

- 首次启动：在第一个模型加载前，对本实验全部原视频及Qwen/VideoJev两套checkpoint文件计算SHA256，写入`BASE/input_hash_cache.json`。每个唯一文件只计算一次（同视频多题不重复）。控制台输出`hash_input_start`、文件名及完成数；整个阶段状态为`checking_input_hash_cache`。
- 后续阶段与`--resume`：复用该缓存中的视频、`.safetensors`哈希，仅检查路径、库存、大小、纳秒级修改时间。两套权重都记录mtime，避免旧helper中缺少权重mtime的限制。缓存文件自身仍核对SHA。
- 模型JSON配置、tokenizer/处理器文件、源码、清单、官方标注、预测和产物指纹仍按原有规则校验。模型加载和视频解码仍须实际读取数据，不会跳过。
- 首次哈希中断：已完成文件逐个原子持久化，`--resume`时只计算剩余文件；文件发生变化则拒绝复用。
- 每个模型阶段写入`input_hash_cache_use.jsonl`，记录缓存路径、SHA和实际复用数量。队列attempt记录实际包装命令和原模型命令`model_command`。GPU模型入口的`sys.argv`和原预测身份保持一致。

**“一次校验”只针对视频/权重等大输入，不是关闭所有校验。**大小和mtime匹配不能保证内容完全未变；同大小且保留mtime的修改可能漏检。默认策略以用户保证输入不变为前提。缓存较小，存储的是文件指纹而非模型/视频副本。

如果需要每个模型阶段完整重算内容哈希，新实验启动时加：

```bash
--input-hash-mode always
```

默认值是`--input-hash-mode once`。模式写入队列身份，恢复不得更换；缓存缺失或变化也不会静默重建。输入/源码或校验模式改变，请选择新实验目录，不改写旧experiment.json。

新版改变了队列源码和调度身份，**不要中途更新正在运行的旧Video-MME，也不要直接用旧base尝试新版resume**。当前运行继续使用冻结旧代码；等旧实验完成后同步，用新base启动后续benchmark。若旧来源已完成，可按前文显式复用；新队列仍建立自己的输入缓存。
