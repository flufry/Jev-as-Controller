# MVBench 公开可下载任务完整评测

当前默认选择 MVBench 的 19 个公开可下载、单正确答案任务，共 3,800 题，使用同一份 manifest 一起评测 Qwen 候选 logits、VideoJev 原始候选 logits 和 Controller。第 20 个 fine_grained_pose 的 NTU RGB+D 视频已被官方从公开仓库移除，需按原数据集授权自行获取；所以默认结果必须标为「MVBench 19 个公开可下载任务」，不能标为 4,000 题的官方全量结果。各任务可有 2～8 个候选项，但每题恰有一个正确项。

官方数据：https://huggingface.co/datasets/OpenGVLab/MVBench

任务的视频来源及 start/end 边界以官方评测 notebook 为准：https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb 。当前代码将有边界的原视频裁为单独 MP4，将 episodic_reasoning 的 3 FPS JPEG 帧序列按边界编码为 MP4。这样三个评测器读到同一段素材；但模型提示词、采帧与官方 VideoChat2 协议不同，不能把分数直接当作官方榜单成绩。

## 服务器下载与准备

在已配置模型和 GPU 运行环境的容器内、仓库根目录执行。先确认 data 挂载到有足够空间的宿主机目录，且可运行 ffmpeg。公开 ZIP 合计约 17 GB；解压后的大小、派生裁剪视频和 HF 缓存还需要额外空间。perception.zip 占其中约 11.2 GB。下载支持复用已完成的文件和固定的 HF revision；十个 ZIP 只拉一次，共用的素材只解压一次。

~~~bash
cd /workspace/Jev-as-Controller
df -h data /root/.cache/huggingface
ffmpeg -version | head -n 1
export HF_ENDPOINT=https://hf-mirror.com  # 官方站更快时可取消

python scripts/download_bench_data.py --bench mvbench \
  --data-root data/benchmarks --workers 2

python scripts/build_bench_manifest.py --bench mvbench \
  --base data/benchmarks/mvbench \
  --annotation data/benchmarks/mvbench/json \
  --out data/benchmarks/mvbench/questions_public19.jsonl

python scripts/check_bench_inputs.py \
  --manifest data/benchmarks/mvbench/questions_public19.jsonl \
  --out reports/mvbench-public19-inputs.json
~~~

预期 manifest 为 3,800 题、19 个 task_type；对应的 questions_public19.meta.json 中 coverage.official_full_benchmark 为 false。构建过程会校验每个任务 200 道官方题，逐个检查视频或裁剪文件存在并探测时长。已有同名 manifest 不会被覆盖；重建时换一个 --out 文件名。若只先拉标注，加 --annotations-only；再次去掉该参数会补视频。若 download/ 与 download_revision.json 已被上传，使用 --prepare-local 可离线整理。中断后重跑同一条下载命令，已解压、已裁剪的文件会跳过。不要在运行中清理 download/ ZIP 或 videos/。

## 同一清单运行三个方法

下面给出 seed 42 的完整 3,800 题运行命令；按机器实际可用卡号和显存修改 GPU 参数。要和先前的 4 FPS / 4096 帧实验对照，保留下面的采帧设置，并确保两源模型读同一个 M。若先用 2 FPS / 2048 帧，需要两者一起修改，并把结果另标为不同协议。每个 --out 必须是新目录；中断后恢复可加 --resume，前提是清单与参数指纹一致。

~~~bash
M=data/benchmarks/mvbench/questions_public19.jsonl

CUDA_VISIBLE_DEVICES=0,1,3,4 python scripts/eval_bench_qwen36_logits.py \
  --manifest "$M" --out runs/mvbench_public19/qwen_logits_42 \
  --checkpoint models/Qwen3.6-35B-A3B --seed 42 --device cuda:0 \
  --gpu-count 4 --max-memory-gib-per-gpu 40,20,40,40 \
  --fps 4 --max-frames 4096 --context 262144 \
  --visual-token-budget 24576 --memory-fraction .9

CUDA_VISIBLE_DEVICES=7 python scripts/eval_bench_videojev_logits.py \
  --manifest "$M" --out runs/mvbench_public19/videojev_logits_42 \
  --checkpoint models/VideoJev-4B --seed 42 --device cuda:0 \
  --fps 4 --max-frames 4096 --context 262144 \
  --visual-token-budget 24576 --memory-fraction .9

python scripts/eval_bench_controller_loglinear.py \
  --manifest "$M" \
  --qwen runs/mvbench_public19/qwen_logits_42 \
  --videojev runs/mvbench_public19/videojev_logits_42 \
  --out runs/mvbench_public19/controller_42 \
  --beta 2 --fusion loglinear --score-source raw_logits
~~~

每个运行目录产生 metrics.json 和 predictions.jsonl；Controller 还写出 comparison.json。关注每个任务的准确率及 19 任务汇总。若要测 seed 43、44，用相同完整清单、相同采帧参数，分别改 --seed 与三个输出目录后重复执行。完整 3,800 题无需先调用 sample_bench_subset.py。

## 有授权时补齐 NTU 任务

1. 到 ROSE Lab 注册研究账号：https://rose1.ntu.edu.sg/register/ 。登录后在 https://rose1.ntu.edu.sg/dataset/actionRecognition/request 提交申请、接受使用协议，等待批准。申请时说明用于 MVBench 的 fine_grained_pose，索取 RGB 视频，不需要骨架、深度图或 IR。MVBench 提供参考文件名清单：https://huggingface.co/datasets/OpenGVLab/MVBench/blob/main/video/MVBench_videos_ntu.txt 。清单含 NTU RGB+D 60 和 120 的编号；若官方能提供所需子集，可避免下载整个 RGB 视频库。未经批准无法给出可用的直链。
2. 在容器内先下载公开标注、列出当前 pinned revision 真正需要的视频名：

~~~bash
cd /workspace/Jev-as-Controller
python scripts/download_bench_data.py --bench mvbench --annotations-only \
  --mvbench-tasks fine_grained_pose --data-root data/benchmarks
python - <<'PY'
import json
from pathlib import Path
base = Path('data/benchmarks/mvbench')
rows = json.loads((base/'json/fine_grained_pose.json').read_text(encoding='utf-8'))
names = sorted({row['video'] for row in rows})
(base/'required-ntu.txt').write_text(''.join(name+'\n' for name in names), encoding='utf-8')
print('questions:', len(rows), 'unique RGB videos:', len(names))
print('first names:', names[:5])
PY
~~~

3. 从获批的官方页面下载相应 RGB 视频；站点在登录后才显示具体下载入口。把文件放在 data/benchmarks/mvbench/videos/fine_grained_pose/ 下，保留 required-ntu.txt 中的相对路径和原始 AVI 文件名。若官方交付的是完整压缩包，解压或复制时只挑 required-ntu.txt 所列文件，避免占用整套 RGB 库的空间。容器的 data 是宿主机 bind mount，也可先在宿主机目标目录摆放。
4. 检查缺失后完成第 20 任务：

~~~bash
python - <<'PY'
from pathlib import Path
base = Path('data/benchmarks/mvbench')
root = base/'videos/fine_grained_pose'
names = (base/'required-ntu.txt').read_text(encoding='utf-8').splitlines()
missing = [name for name in names if not (root/name).is_file() or not (root/name).stat().st_size]
print('missing:', len(missing), 'examples:', missing[:10])
raise SystemExit(bool(missing))
PY
python scripts/download_bench_data.py --bench mvbench --prepare-local \
  --mvbench-tasks fine_grained_pose --data-root data/benchmarks
python scripts/build_bench_manifest.py --bench mvbench \
  --base data/benchmarks/mvbench --annotation data/benchmarks/mvbench/json \
  --out data/benchmarks/mvbench/questions_full20.jsonl
~~~

此时只有成功生成 4,000 题、20 个任务的 questions_full20.meta.json，才可称为完整 MVBench 清单。若授权视频命名或目录不同，按标注相对路径整理后再构建，不要跳过缺失视频。

## 本轮文件记录

- 新增：timejev/mvbench_benchmark.py、tests/test_mvbench_benchmark.py、explanation_here/14-MVBench小规模评测.md、explanation_here/15-MVBench公开任务完整评测.md
- 修改：scripts/bench_compat.py、scripts/bench_io.py、scripts/build_bench_manifest.py、scripts/download_bench_data.py、scripts/eval_bench_controller.py、scripts/sample_bench_subset.py、scripts/prepare_bench_data.sh、explanation_here/README.md

## 全量三 seed 六方法队列（新增）

上述命令保留原单来源/Controller直接评测方式。若需Qwen native 4FPS、单步Qwen2FPS、VideoJev4FPS、两版Controller及采样OR的完整三seed表，使用 [MVBench全量三seed说明](../docs/MVBENCH_FULL_THREE_SEEDS.md)。统一入口为`prepare_bench_full.py --bench mvbench --mvbench-scope public19`及`run_bench_full.py --bench mvbench`，默认先Qwen后VideoJev、首次输入哈希一次。默认19任务3800题，完整20任务需显式选择full20并具备全部素材。
