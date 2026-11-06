# MVBench 小规模评测

这是历史小规模试跑方案；当前适配器默认下载全部 19 个公开可获取任务，不再默认只下载 egocentric_navigation。若只需复现本页试跑，务必显式指定 --mvbench-tasks egocentric_navigation。完整可下载任务的流程见 [15-MVBench公开任务完整评测.md](15-MVBench公开任务完整评测.md)。

官方数据页：https://huggingface.co/datasets/OpenGVLab/MVBench 。任务与视频根目录的对应关系取自官方 video_chat2/mvbench.ipynb。完整 MVBench 另含需自行取得的 NTU RGB+D 视频、按帧保存的任务，以及带 start/end 裁剪边界的任务；这次没有把这些任务误当作完整视频输入。

## 下载与构建

先把本次代码同步到服务器，然后在 Docker 容器的仓库根目录执行。确保 data 仍映射到有空间的宿主机目录。脚本固定并记录 Hugging Face revision，重试会复用下载；只下载选定任务的标注和视频 ZIP，并仅解压标注所引用的视频。

~~~bash
cd /workspace/Jev-as-Controller
export HF_ENDPOINT=https://hf-mirror.com  # 官方站速度正常可去掉
python scripts/download_bench_data.py --bench mvbench \
  --data-root data/benchmarks --workers 2 \
  --mvbench-tasks egocentric_navigation
python scripts/build_bench_manifest.py --bench mvbench \
  --base data/benchmarks/mvbench \
  --annotation data/benchmarks/mvbench/json/egocentric_navigation.json
python scripts/check_bench_inputs.py \
  --manifest data/benchmarks/mvbench/questions.jsonl \
  --out reports/mvbench-egonav-inputs.json
~~~

正常应得到 200 题；questions.meta.json 的 coverage.official_full_benchmark 应为 false。若首次只下载标注，可加 --annotations-only，之后去掉再运行。若已经上传 download/ 和 download_revision.json，可加 --prepare-local 离线解压。更换任务时，用 --out 指定新的 manifest 文件名；脚本不会覆盖原有 manifest。

可选的三选一任务：

~~~bash
python scripts/download_bench_data.py --bench mvbench \
  --mvbench-tasks action_antonym --data-root data/benchmarks
python scripts/build_bench_manifest.py --bench mvbench \
  --base data/benchmarks/mvbench \
  --annotation data/benchmarks/mvbench/json/action_antonym.json \
  --out data/benchmarks/mvbench/action_antonym.jsonl
~~~

如两个任务都要下载，传 --mvbench-tasks egocentric_navigation action_antonym；之后用包含两个标注的 json/ 目录和新 --out 构建两任务 manifest。结果须注明“两任务子集”。

## 运行现有评测器

下例沿用现有 raw-logits Qwen、VideoJev 和 log-linear Controller。按已验证的服务器配置修改 CUDA_VISIBLE_DEVICES 和每卡预算。两次源模型运行必须使用相同 manifest、推理 seed、fps 和 max-frames。2 fps / 2048 帧是快速试跑配置，不能与此前的 4 fps / 4096 帧当成同配置比较。本仓库的提示词和采帧协议也不是官方 VideoChat2 的完整评测协议，因此不要直接与 MVBench 官方榜单分数并列。

~~~bash
M=data/benchmarks/mvbench/questions.jsonl
CUDA_VISIBLE_DEVICES=0,1,3,4 python scripts/eval_bench_qwen36_logits.py \
  --manifest "$M" --out runs/mvbench_egonav/qwen_logits_42 \
  --checkpoint models/Qwen3.6-35B-A3B --seed 42 --device cuda:0 \
  --gpu-count 4 --max-memory-gib-per-gpu 40,20,40,40 \
  --fps 2 --max-frames 2048 --context 262144 \
  --visual-token-budget 24576 --memory-fraction .9

CUDA_VISIBLE_DEVICES=7 python scripts/eval_bench_videojev_logits.py \
  --manifest "$M" --out runs/mvbench_egonav/videojev_logits_42 \
  --checkpoint models/VideoJev-4B --seed 42 --device cuda:0 \
  --fps 2 --max-frames 2048 --context 262144 \
  --visual-token-budget 24576 --memory-fraction .9

python scripts/eval_bench_controller_loglinear.py \
  --manifest "$M" --qwen runs/mvbench_egonav/qwen_logits_42 \
  --videojev runs/mvbench_egonav/videojev_logits_42 \
  --out runs/mvbench_egonav/controller_42 \
  --beta 2 --fusion loglinear --score-source raw_logits
~~~

各运行目录有 metrics.json 和 predictions.jsonl；Controller 目录还有 comparison.json。如需固定抽题，可先运行以下命令，再将三个评测命令的 M 统一指向该 sample 文件。抽题 seed 写在样本 metadata 中；--seed 是推理 seed。

~~~bash
python scripts/sample_bench_subset.py --bench mvbench \
  --manifest "$M" --out data/benchmarks/mvbench/sample_seed42.jsonl \
  --seed 42 --n 60
~~~

## 本次文件

- 新增：timejev/mvbench_benchmark.py、tests/test_mvbench_benchmark.py、explanation_here/14-MVBench小规模评测.md
- 修改：scripts/bench_io.py、scripts/build_bench_manifest.py、scripts/download_bench_data.py、scripts/sample_bench_subset.py、scripts/prepare_bench_data.sh
