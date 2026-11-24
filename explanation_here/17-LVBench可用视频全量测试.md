# LVBench：修复标注路径并测试全部可用视频

下载器保存的官方标注叫`data/benchmarks/lvbench/video_info.meta.jsonl`，不存在`official.jsonl`不是下载失败。不要改名或复制视频，直接修正`--reference`。

`questions.available.jsonl`和`questions.available.meta.json`是已构建的可用题集。完整官方标注含1549题；下架或无法下载的视频会整体排除。新增`prepare_bench_full.py --allow-incomplete-population`显式接受全部可用题目，不抽题，并逐项核验官方身份、选项、答案及排除的视频/题目记录。清单、原始标注及元数据均封存SHA，运行时重验；任意抽样子集不能通过此开关冒充可用全集。

报告写明实际题数、覆盖率及排除题数，split为`test_available_videos`。Qwen logits、native、VideoJev、两版Controller及OR使用相同题集和分母。不能将该结果标成官方1549题全量成绩。

## 上传源码更新包

本地已准备`reports/transfer/lvbench-full-update-20261124.tar.gz`及同名`.sha256`。包内保留`scripts/`、`timejev/`和`videojev_hf/`的目录结构，含当前全量评测所需源码及依赖，另附说明和相关测试；不含数据、权重、结果、虚拟环境或编译后端。服务器trace对应较旧版本，因此将依赖一起同步，避免只替换入口导致导入失败。

在本地PowerShell仓库根目录执行；将Server设为实际目标（下面按迁移后的10.168.100.128举例）：

```powershell
$Server = 'hyj@10.168.100.128'
scp .\reports\transfer\lvbench-full-update-20261124.tar.gz .\reports\transfer\lvbench-full-update-20261124.tar.gz.sha256 "${Server}:~/"
```

在服务器宿主机执行。先从`docker ps`确认容器名，再设置`CONTAINER`。通过`docker cp`进入容器，不需要猜测宿主机仓库路径，也不需要修改项目文件所有权。请在相关评测启动前同步源码；恢复中的旧运行会检查源码指纹。

```bash
cd "$HOME"
sha256sum -c lvbench-full-update-20261124.tar.gz.sha256
sudo docker ps --format 'table {{.Names}}\t{{.Status}}'
CONTAINER=jev-bench-migrated-v2  # 按上一步实际名称修改

sudo docker cp lvbench-full-update-20261124.tar.gz "$CONTAINER:/tmp/lvbench-full-update-20261124.tar.gz"
sudo docker exec -w /workspace/Jev-as-Controller "$CONTAINER" \
  tar --exclude='__pycache__' -czf /tmp/jev-code-before-lvbench-update-20261124.tar.gz scripts timejev videojev_hf
sudo docker cp "$CONTAINER:/tmp/jev-code-before-lvbench-update-20261124.tar.gz" ./jev-code-before-lvbench-update-20261124.tar.gz
sudo docker exec "$CONTAINER" tar -xzf /tmp/lvbench-full-update-20261124.tar.gz -C /workspace/Jev-as-Controller
sudo docker exec -it "$CONTAINER" bash
```

## 准备可用全集

以下在容器内、已配置好Qwen/VideoJev依赖及后端的Python环境执行。沿用之前能运行评测的解释器和环境设置；若两模型环境分开，按原全量手册传入`--qwen-python`和`--videojev-python`。

```bash
cd /workspace/Jev-as-Controller
python scripts/prepare_bench_full.py --help

python scripts/prepare_bench_full.py --bench lvbench \
  --source-manifest data/benchmarks/lvbench/questions.available.jsonl \
  --reference data/benchmarks/lvbench/video_info.meta.jsonl \
  --allow-incomplete-population \
  --out runs/full/lvbench_available/questions.jsonl
```

不需要重跑数据下载和视频探测。源`questions.available.meta.json`中引用的标注/映射必须仍在原位置且内容匹配；校验不通过应检查来源，而不是删除哈希记录。准备过程打印`n`和`coverage`以核对实际覆盖。输出拒绝覆盖；准备成功后直接执行下一步即可，首次失败且没有生成文件的目录可以继续使用。

## 三seed运行及输出

下例Qwen使用4张卡、VideoJev使用第0张卡，各模型阶段默认串行。GPU编号和逐卡预算按实际空闲显存调整：

```bash
COMMON=(--qwen-gpus 0,1,2,3 --videojev-gpu 0 \
  --max-memory-gib-per-gpu 40,20,40,40)

python scripts/run_bench_full.py --bench lvbench \
  --base runs/full/lvbench_available "${COMMON[@]}" --dry-run

python scripts/run_bench_full.py --bench lvbench \
  --base runs/full/lvbench_available "${COMMON[@]}"
```

三份候选来源各计算一次，复用分布分别按42/43/44采样；native实际生成三遍。默认native和独立VideoJev基线4FPS/4096帧，Controller双源2FPS/2048帧。方法、概率公式、采样和方差口径见`docs/BENCH_FULL_THREE_SEEDS.md`。

最终报告为`runs/full/lvbench_available/summary/results.md`及`results.json`；阶段日志在同目录的`logs/`；每个seed的汇总在`per_seed/seed42/`、`seed43/`、`seed44/`。中断后保持命令、源码和输入不变，在运行命令末尾加`--resume`。`--dry-run`只验证清单与调度计划，不验证权重及CUDA内核能否运行。

## 本次修改文件

- `scripts/prepare_bench_full.py`
- `scripts/build_bench_manifest.py`
- `timejev/bench_full.py`
- `timejev/lvb_native_generation.py`
- `tests/test_lvbench_available.py`
- `docs/BENCH_FULL_THREE_SEEDS.md`
- `explanation_here/README.md`
- `explanation_here/07-本次文件变更记录.md`

## 本次新增文件

- `explanation_here/17-LVBench可用视频全量测试.md`
