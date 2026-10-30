# 11 · 新增扩展：Windows 本地分批下载与 SCP 上传

适用于本地网络较快、服务器下载慢、本地磁盘不能同时存下全部模型/视频的情况。新入口 `scripts/stage_local_assets.py` 每次只准备一个模型文件、Video-MME ZIP、MLVU 文件或 LVBench 视频，SCP 到服务器，核验服务器 SHA256 后再按显式参数删除该本地文件。元数据、版本锁定与上传回执留在本地，以便中断重跑。

本地无需 CUDA、GPU 或 Docker；需要 Python >=3.10、支持 `scp -O` 的现代 OpenSSH 客户端和 SSH 密钥/agent 登录。服务器需要 SSH、SCP、GNU coreutils 和可写目录。本文的命令没有替你启动真实下载或上传。

本次检查本机 C 盘约剩 100 GiB、D 盘约剩 149 GiB，不建议先将全部文件下载完再打包。模型版本沿用 06；数据集首次记录 HF commit，重跑使用同一 commit。Video-MME 不在 Windows 解压；视频 ZIP 上传后才在服务器解压，因此本地只需单个文件及下载临时文件的空间。LVBench 失败时停止，保留当前未完成文件，避免积累许多视频的残片。

## 1. 本地 PowerShell：准备独立环境

以下在本地 Windows PowerShell 执行，不是在 SSH 或 Docker 内。当前机器已有 Python 3.12，脚本不安装模型推理依赖：

```powershell
Set-Location 'D:\Jev_as_controller\Jev-as-Controller'
python -m venv .venv-bench-data
# 分批上传模型/Video-MME/MLVU 只需 huggingface_hub；LVBench 再需要 yt-dlp。
# 清掉此前 wheel 离线构建可能遗留的 pip 环境变量。
Remove-Item Env:PIP_NO_INDEX,Env:PIP_FIND_LINKS,Env:PIP_CONSTRAINT -ErrorAction SilentlyContinue
& .\.venv-bench-data\Scripts\python.exe -m pip install -U huggingface_hub 'yt-dlp[default]' `
  --index-url https://pypi.org/simple
$Py = (Resolve-Path .\.venv-bench-data\Scripts\python.exe).Path
$Hf = (Resolve-Path .\.venv-bench-data\Scripts\hf.exe).Path
$Stage = Join-Path (Get-Location).Path 'data\local-stage'
$Server = 'hyj@10.168.100.124' # 改成实际目标服务器
$RemoteRoot = '/public/hyj/jev'
```

这是独立下载环境。pip 源慢时可在安装命令增加 `--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。重新打开 PowerShell 后需要重新设置上述变量。

MLVU 仍需先在 [官方页面](https://huggingface.co/datasets/MLVU/MVLU)取得访问权限，并使用自己的账号登录；CLI 登录不能替代网页授权。不要将 token 上传到服务器的项目数据目录。

```powershell
# 先查看当前登录状态；若显示 token expired，重新登录。
& $Hf auth whoami
& $Hf auth logout
& $Hf auth login
```

登录时粘贴新的 Hugging Face **read token**，不要把 token 写进命令行、脚本、Git 或服务器目录。Qwen 若有 gated access，必须先在对应模型页面完成访问申请。登录成功后验证：

```powershell
& $Hf auth whoami
Remove-Item Env:HF_HUB_OFFLINE,Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue
```

如果 `auth login` 仍报 `SSL: UNEXPECTED_EOF_WHILE_READING`，先检查当前 PowerShell 是否设置了失效代理：

```powershell
Get-ChildItem Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
```

代理不可用时临时清除后重试：

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
& $Hf auth login
```

如果官方站点仍无法连接，可在下载命令中追加 `--endpoint https://hf-mirror.com`；镜像不保证所有 gated 模型和固定 revision 都可用，登录授权仍需在官方 Hugging Face 页面完成。不要把过期 token 传给镜像或脚本。

## 2. SSH 密钥：避免每个文件重复输入密码

如果已有密钥/agent，可直接验证并使用它。没有时可新建专用密钥，下面路径应不存在，避免覆盖现有密钥：

```powershell
$Key = Join-Path $HOME '.ssh\jev_transfer_ed25519'
ssh-keygen -t ed25519 -f $Key
Get-Content "$Key.pub" | ssh $Server 'umask 077; mkdir -p "$HOME/.ssh"; cat >> "$HOME/.ssh/authorized_keys"'
ssh -i $Key -o BatchMode=yes $Server 'echo SSH_OK'
```

最后必须输出 `SSH_OK`；脚本采用 BatchMode，不会逐文件弹密码。密钥设有口令时需要先配置 ssh-agent 并 `ssh-add $Key`，再验证。第一次连接服务器时正常核对并接受主机密钥后，再做 BatchMode 验证。私钥只保留本地。

在服务器确认目标目录可写；此前容器用 root 下载的文件可能导致普通用户无权限。只有这些目录确实是你自己的项目数据目录时，才按需修正所有权（不会删除文件）：

```powershell
ssh -t $Server 'sudo mkdir -p /public/hyj/jev/models /public/hyj/jev/data; sudo chown -R hyj:"$(id -gn)" /public/hyj/jev/models /public/hyj/jev/data'
```

先停止服务器上对相同模型/数据目录的下载任务，避免两边重复写文件。

## 3. 按目标分批下载、上传和清理

参数 `--remove-after-upload` 才会启用清理，且只删除本次 staging 目录中服务器已经核验 SHA256 的文件。SCP 失败、核验失败或远程发布失败时保留本地文件。不要手动删除整个 staging 目录；其中 pins 和 receipts 用于重跑。每条命令返回 0 并输出完成提示后，再继续下一项。

```powershell
# Qwen 固定 995ad96eacd98c81ed38be0c5b274b04031597b0
& $Py scripts/stage_local_assets.py --target qwen --server $Server --remote-root $RemoteRoot --staging-dir $Stage --ssh-key $Key --remove-after-upload

# VideoJev 固定 1c1a253bf90dfa66176124d25a48ffb791d517d5
& $Py scripts/stage_local_assets.py --target videojev --server $Server --remote-root $RemoteRoot --staging-dir $Stage --ssh-key $Key --remove-after-upload

# Video-MME：逐 ZIP 上传，不在本地解压
& $Py scripts/stage_local_assets.py --target videomme --server $Server --remote-root $RemoteRoot --staging-dir $Stage --ssh-key $Key --remove-after-upload

# MLVU：只下载七个 MCQ 任务及对应原视频
& $Py scripts/stage_local_assets.py --target mlvu --server $Server --remote-root $RemoteRoot --staging-dir $Stage --ssh-key $Key --remove-after-upload
```

可在任意一条命令追加 `--endpoint https://hf-mirror.com` 使用第三方镜像。默认使用官方站点，环境变量在 Python 导入 HF 前设置，下载时会解除离线模式。重跑会重新核验服务器回执对应文件，只有远程文件仍存在且 SHA256 一致时才跳过本地下载；回执本身不等于远程数据仍完整。

上传使用远程 `.jev-upload.part` 文件，核验成功后原子替换目标文件。现有目标文件内容完全相同时可直接复用；内容不同时会替换为当前锁定版本。未完成的单文件传输可能需要从头重传，但已核验的其它文件不会重复下载。服务器文件校验和本地哈希会增加磁盘读取时间。

## 4. LVBench：先准备 YouTube 下载条件

HF-Mirror 不镜像 YouTube，必须保证本地能正常访问 YouTube；下架或受限的视频不能保证取全。本地按 [Deno 官方 Windows 安装方式](https://docs.deno.com/runtime/getting_started/installation/)准备 JS runtime：

```powershell
irm https://deno.land/install.ps1 | iex
$env:Path = "$HOME\.deno\bin;$env:Path"
deno --version

& $Py scripts/stage_local_assets.py --target lvbench --server $Server --remote-root $RemoteRoot --staging-dir $Stage --ssh-key $Key --remove-after-upload
```

已有 Node >=22 可追加 `--js-runtime node`。需要自己的 cookies 时追加 `--cookies '本地cookies文件路径'`，脚本不上传 cookies。下载的是完整 MP4 视频流，保持当前无音频/无字幕协议。遇到首个视频失败时停止并保留该视频残片，服务器 `missing-videos.json` 记录本次失败 ID；它不是全库缺失统计，后续未尝试的文件也可能缺失。修复网络/登录条件后重跑，不自动删题。

## 5. 服务器容器：只整理已上传文件，不再联网下载

先将更新后的 `scripts/download_bench_data.py` 同步到服务器仓库。按现有挂载进入目标容器，服务器宿主机 `/public/hyj/jev/models`、`data` 应分别对应容器内项目的 models、data。

```bash
sudo docker exec -it jev-bench-migrated-v2 bash
```

以下在容器内执行。三个 bench 都上传完成时才使用 `--bench all`；也可逐个指定 `--bench videomme` 等单独整理。

```bash
cd /workspace/Jev-as-Controller
python scripts/download_bench_data.py --bench all --prepare-local \
  --data-root data/benchmarks
```

`--prepare-local` 读取上传的固定版本记录和原始文件，不调用 HF，也不调用 yt-dlp。Video-MME 在服务器解压并生成 official.json；MLVU 整理七任务 JSON 和 video-map.json；LVBench 整理标注并检查全部预期视频是否存在。缺文件会报错，不能将未完成上传当作完整数据集。

按 09 第 6 节构建 manifest（尤其使用其中的 MLVU 视频根目录和映射参数），再做解码、预算预检和 pilot。已上传文件的 SHA256 通过仅说明传输一致，不替代实际解码和模型运行验证。

模型上传完成后可直接使用容器本地路径：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 评测入口继续传 models/Qwen3.6-35B-A3B 和 models/VideoJev-4B。
```

源码、Windows 虚拟环境、SSH 私钥和 HF 登录缓存不需要上传。只读接口及本地合成上传失败测试不代表已完成真实网络下载；实际完成状态以每个目标命令和服务器预检为准。
