# 06 · 新增扩展：Docker 与服务器环境

本文只讲本次跨 benchmark 粗筛环境。评测流程见 [05-跨Benchmark粗筛运行手册.md](05-跨Benchmark粗筛运行手册.md)，文件清单见 [07-本次文件变更记录.md](07-本次文件变更记录.md)。原仓库 README 和 runtime 原说明保留。

## 1. 推荐镜像与适用硬件

优先拉取 Docker Hub 官方开发镜像：

```bash
docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
```

已核对 [Docker Hub 官方 tag](https://hub.docker.com/layers/pytorch/pytorch/2.8.0-cuda12.8-cudnn9-devel/images/sha256-a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6)：linux/amd64、CUDA 12.8.1、cuDNN 9、Ubuntu 22.04。请选 **devel**，因为 grouped-GEMM 和其它原生扩展需要 `nvcc`。基础镜像不是本仓库完整环境，不包含指定 VideoJev/Qwen 权重和自定义 CUTLASS 补丁。

当前新入口继续复用原 Qwen 实现，可靠的目标仍是 **RTX 4090 / SM89**，4–6 张放 Qwen，VideoJev 单卡；可复用同一组卡顺序执行。35B 权重本身约 67 GiB，此外还需激活显存。四卡能否在相同长视频协议下完成要实测，不承诺不 OOM。A100/H100/RTX 5090 等架构不能仅换镜像就视为后端已验证；需要额外适配和数学一致性验证。

推荐使用下面的 Dockerfile 在官方镜像上构建项目环境。没有核实到一个公开镜像已包含本仓库所需的**确切**补丁和完整 ABI，因此不提供未经验证的“拉完即可全流程运行”第三方镜像名。

## 2. 服务器宿主机前置

宿主机需要 Linux x86_64、可用 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。CUDA Toolkit 放容器里，宿主机无需再安装一套同版本 toolkit。先确认：

```bash
nvidia-smi
docker version
```

驱动需能支持 CUDA 12.8 运行时。为了减少新库兼容问题，推荐使用支持 CUDA 12.8 的 570 系列或更新驱动；是否可用最终以容器中的 CUDA 张量运算和后端测试为准，不能只看 `nvidia-smi` 的版本字样。服务器若由集群管理员配置，先使用现有配置。

Ubuntu/Debian 尚未安装 Docker 时按 [Docker 官方安装说明](https://docs.docker.com/engine/install/ubuntu/) 安装。NVIDIA Toolkit 的官方步骤见 [安装与 Docker 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。典型 rootful Docker 配置如下，均在**宿主机**执行：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`systemctl restart docker` 可能影响服务器上已有容器，应按服务器维护安排执行。非 root Docker 的配置路径不同，见上述官方说明。

## 3. 把代码放到服务器

推荐从 Git checkout 运行，保留 `.git`，原评测会记录 git HEAD、状态和 diff。本文没有替你执行远程 push、创建服务器或发布 Docker Hub 镜像。

本地提交并推送这次代码后，在服务器克隆对应分支，或在已有 checkout 拉取：

```bash
# 将分支名换成你实际推送的分支。
git clone --branch <你的分支> https://github.com/tangqh03/Jev-as-Controller.git
cd Jev-as-Controller
git rev-parse HEAD
git status --short
```

用 scp/rsync 上传也可以，但必须保留 Git 元数据或在服务器准备真实 checkout。不要上传 Windows 的 `.venv`、`.venv-bench-check`、已编译 `.deps*` 或半成品 runs 作为 Linux 环境。权重和视频放在服务器本地磁盘，推荐目录：

```text
/srv/jev/Jev-as-Controller/  # 代码 checkout
/srv/jev/data/              # 数据与原视频
/srv/jev/models/            # 两套权重
/srv/jev/runs/              # 实验结果，持久化写入
/srv/jev/hf-cache/          # 可选 HF 下载缓存
```

## 4. 构建项目环境镜像

在**服务器的仓库根目录**运行：

```bash
docker build --pull -f docker/bench/Dockerfile -t jev-bench:cu128 .
docker image inspect jev-bench:cu128 --format '{{.Id}}'
```

新增 `.dockerignore` 使用最小依赖文件白名单，避免把视频、权重、Git 历史和历史结果传入 build context。代码在运行时挂载，因此代码变更无需重新构建镜像；依赖变更需要重建。

Dockerfile 完成：

1. 从官方 PyTorch devel 镜像继承 CUDA 12.8、cuDNN 与编译工具链。
2. 独立创建 `/opt/jev` 的 **Python 3.12.12**。不假设基础镜像 Python 小版本与原实验相同；官方 Dockerfile 的默认 Python 是 3.11，可参见 [上游 v2.8 Dockerfile](https://github.com/pytorch/pytorch/blob/v2.8.0/Dockerfile)。
3. 安装 PyTorch 2.8.0+cu128、torchvision 0.23.0、Triton 3.4.0，以及仓库 requirements；用 `constraints.txt` 固定关键 ABI 与 accelerate 1.15.0、NumPy 1.26.4。
4. 安装 FlashAttention 2.8.3.post2、causal-conv1d 1.7.0 和读取 Parquet 的 pyarrow；原生扩展使用 `--no-build-isolation`，以当前 Torch 构建；未匹配到 wheel 时会源码编译，可能耗时较长。
5. 以 `DS_BUILD_OPS=0` 避免为纯评测预编译不必要的 DeepSpeed 算子。运行 `pip check` 并保存 `/opt/jev-build/pip-freeze.txt`。

构建时无需加载模型或提供 GPU；原生编译使用 `TORCH_CUDA_ARCH_LIST=8.9`。镜像内暂不包含项目专用 grouped-GEMM，它在下一步编译进持久化 checkout。

若希望进一步固定基础镜像，先保存 `docker image inspect pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` 的 RepoDigests，再以确认过的 digest 传入：

```bash
docker build -f docker/bench/Dockerfile -t jev-bench:cu128 \
  --build-arg BASE_IMAGE='pytorch/pytorch@sha256:<你实际核验的digest>' .
```

联网构建失败应检查服务器能否访问 Docker Hub、PyPI、PyTorch wheel 站和 GitHub。不要关闭 TLS 校验。只给安装/下载阶段设置实际可达的代理；不要将 token 写入 Dockerfile。

## 5. 创建与进入容器

先创建宿主机数据目录，再启动容器（按实际服务器路径修改 `JEV_HOST_ROOT`）：

```bash
JEV_HOST_ROOT=/srv/jev
mkdir -p "$JEV_HOST_ROOT"/{data,models,runs,hf-cache}

docker run -d --name jev-bench --gpus all --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,src="$JEV_HOST_ROOT/Jev-as-Controller",dst=/workspace/Jev-as-Controller \
  --mount type=bind,src="$JEV_HOST_ROOT/data",dst=/workspace/Jev-as-Controller/data \
  --mount type=bind,src="$JEV_HOST_ROOT/models",dst=/workspace/Jev-as-Controller/models \
  --mount type=bind,src="$JEV_HOST_ROOT/runs",dst=/workspace/Jev-as-Controller/runs \
  --mount type=bind,src="$JEV_HOST_ROOT/hf-cache",dst=/root/.cache/huggingface \
  --workdir /workspace/Jev-as-Controller \
  jev-bench:cu128 tail -f /dev/null

docker exec -it jev-bench bash
```

GPU 编号：`--gpus all` 暴露所有卡；评测命令内再用 `CUDA_VISIBLE_DEVICES` 选择。若改为 `--gpus '"device=..."'` 限制卡，容器内需重新检查编号，不照抄宿主机编号。`--shm-size=32g` 是上限，按机器内存调整。默认 root 创建的结果在宿主机可能归 root 所有，可按服务器惯例配置用户/缓存目录。

这里不使用 `--rm`，便于重入。退出交互 shell 不会删除容器。结果、权重、数据以及 checkout 中 `.deps-qwen36`/`.deps-xtuner`/reports 都在宿主机持久化；它们**不是**容器根文件系统的一次性内容。

```bash
# 宿主机：重入或停/启。
docker exec -it jev-bench bash
docker stop jev-bench
docker start jev-bench
```

## 6. 容器自检与专用后端编译

以下在**容器内**执行：

```bash
which python
python --version
nvcc --version
nvidia-smi
python - <<'PY'
import torch, torchvision, transformers, triton
print(torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, triton.__version__)
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    assert torch.cuda.get_device_capability(i) == (8, 9), '该预设仅针对 SM89'
    x = torch.ones((64, 64), device=f'cuda:{i}', dtype=torch.bfloat16)
    assert (x @ x).float().mean().item() == 64.
import flash_attn, causal_conv1d
print('CUDA/BF16/native imports OK')
PY

bash scripts/setup_bench_runtime.sh
source scripts/env_qwen36.sh
python scripts/check_bench_runtime.py --out reports/bench-runtime-check.json
```

`setup_bench_runtime.sh` 复用 runtime 原说明的固定 revision 和本地补丁：grouped_gemm v1.1.4、CUTLASS、XTuner 来源 checkout；检查 revision，再生成并比对补丁，编译到 `.deps-qwen36`。它不安装通用同名 grouped_gemm 包来替代补丁。已有完整 `.so` 时尝试加载并检查来源，不覆盖后端；ABI 不匹配直接报错。

脚本只会修改自己创建/核验的独立第三方 checkout。编译中断后若留下源码修改，再次执行会拒绝覆盖；保留日志并对照 [原 runtime 重建说明](../runtime/qwen36/README.md) 检查，确认已有补丁正确后可手工重跑其中的安装步骤，不要对陌生目录盲目 `git reset`。

`check_bench_runtime.py` 在可见 GPU 上逐卡执行小规模 BF16 NN/NT、含空专家组的重排/加权专家前向，与独立 PyTorch 运算对比（atol/rtol 均为 0.02），成功后保存误差和来源。可用 `--devices 0,1` 限定逻辑卡号。它仍不代表完整模型、长上下文或性能已验证；下一步必须跑 pilot 和（若有原数据）LVB 复现闸门。非 4090 的新 GPU 不应跳过该限制强行沿用结果。

## 7. 权重、数据与运行

在联网阶段下载原仓库所固定的权重版本：

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir models/Qwen3.6-35B-A3B
hf download ZhuYuhan/VideoJev-4B \
  --revision 1c1a253bf90dfa66176124d25a48ffb791d517d5 --local-dir models/VideoJev-4B
```

然后按 05 文档依次执行：完整清单 → 抽样 → `check_bench_inputs.py --budget` → VideoJev pilot → 正式两模型 → CPU 融合 → 汇总。下载完成后再设置：

```bash
source scripts/env_qwen36.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

容器内用 `/opt/jev/bin/python`，**无需也不要激活上传的 Windows `.venv`**。原 env 脚本只补环境变量与运行时搜索路径。

长任务建议在宿主机 tmux 中执行 `docker exec`。重连不会丢掉 tmux 内的前台任务。也可以在容器内部保存日志；使用管道时打开 `pipefail`，保留真实退出码：

```bash
set -o pipefail
# 把 05 文档的某条完整评测命令放在下面，路径选择新日志文件。
python scripts/eval_bench_videojev.py \
  --manifest runs/screen/videomme/sample.jsonl --out runs/screen/videomme/videojev \
  --checkpoint models/VideoJev-4B --device cuda:0 --seed 42 \
  --fps 2 --max-frames 2048 --context 262144 --visual-token-budget 24576 \
  2>&1 | tee runs/screen/videomme/videojev-console.log
```

重入容器需要重新 source 环境；中断后同配置加 `--resume`，不要删除旧结果。完成后把 `runs/screen/*/summary/results.*` 和总表取回本地即可查看；原视频和权重无需回传。

## 8. 保存可复用环境与验证范围

镜像不含运行时挂载的数据/代码/权重/编译后端。可保存并在另一个兼容服务器加载：

```bash
docker save jev-bench:cu128 -o jev-bench-cu128.tar
# 另一台服务器：docker load -i jev-bench-cu128.tar
```

如以后要发布到自己的 Docker Hub，需要单独给镜像打上你账号下的 tag 并执行 `docker push`；本文没有发布镜像，也没有可用的个人 Hub 地址。专用 `.so` 仍应在相同 Python/Torch/CUDA/SM89 环境重建，或连同已验证的独立后端保存来源证明后迁移。

本次开发机是 Windows CPU 环境，无 Docker daemon、无 CUDA/模型/正式 benchmark 视频。已验证项目代码的 CPU 测试，不代表 Dockerfile 在服务器已成功构建，也不代表正式三 bench 已执行。服务器还需完成：镜像构建、`pip check`、GPU 张量自检、原生扩展加载/数学验证、真实视频预算预检、pilot 和正式推理。这些执行状态不得提前写成通过。
