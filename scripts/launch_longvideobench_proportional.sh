#!/usr/bin/env bash
# 持久运行比例抽样队列；监控完成后验收，收尾日志保存在实验目录之外。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
source scripts/env_qwen36.sh
# 环境脚本会重定向 /dev/null，因此在加载环境之后才启用防覆盖。
set -o noclobber

EXPERIMENT_DIR="${1:-$PROJECT_DIR/runs/lvb100_proportional_seed42_43_44}"
FINALIZATION_DIR="${2:-$PROJECT_DIR/reports/lvb100_proportional_finalization}"
mkdir -p "$FINALIZATION_DIR"

python - "$EXPERIMENT_DIR" "$FINALIZATION_DIR" "$$" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import socket
import sys

master, finalizer = map(lambda x: Path(x).resolve(), sys.argv[1:3])
if master == finalizer or master in finalizer.parents:
    raise ValueError('收尾日志必须保存在实验目录之外')
for name in ('full.log', 'full.exitcode', 'launch_record.json'):
    if (master/name).exists():
        raise FileExistsError(f'启动现场已存在，拒绝覆盖：{master/name}')
for name in ('finalize.log', 'finalize.exitcode', 'launcher.exitcode'):
    if (finalizer/name).exists():
        raise FileExistsError(f'收尾现场已存在，拒绝覆盖：{finalizer/name}')
with (master/'launch_record.json').open('x') as stream:
    json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), wrapper_pid=int(sys.argv[3]),
        python=sys.executable, master=str(master), finalization_directory=str(finalizer),
        entrypoint='scripts/launch_longvideobench_proportional.sh',
        qwen_devices='0,1,2,3,4,6', videojev_devices='7'), stream, indent=2)
    stream.write('\n')
PY

if python -u scripts/run_longvideobench_proportional.py --phase full \
    --out "$EXPERIMENT_DIR" --devices 0,1,2,3,4,6 --videojev-devices 7 \
    > "$EXPERIMENT_DIR/full.log" 2>&1; then
    FULL_CODE=0
else
    FULL_CODE=$?
fi
printf '%s\n' "$FULL_CODE" > "$EXPERIMENT_DIR/full.exitcode"
if (( FULL_CODE != 0 )); then
    printf '%s\n' "$FULL_CODE" > "$FINALIZATION_DIR/launcher.exitcode"
    exit "$FULL_CODE"
fi

# 此处仅等待指定模型监控器的阶段交接；不读取模型日志、资源或准确率。
if python - "$EXPERIMENT_DIR" <<'PY'
import json
from pathlib import Path
import sys
import time

status = Path(sys.argv[1])/'monitor/status.json'
deadline = time.monotonic()+24*3600
while True:
    if status.exists():
        value = json.loads(status.read_text())
        state = value.get('monitor_status', value.get('status'))
        if state == 'completed':
            break
        if state in ('failed', 'blocked'):
            raise RuntimeError(f'监控交接失败：{state}')
    if time.monotonic() >= deadline:
        raise TimeoutError('等待监控完成交接超过24小时，保留实验现场')
    time.sleep(60)
PY
then
    HANDOFF_CODE=0
else
    HANDOFF_CODE=$?
fi
if (( HANDOFF_CODE != 0 )); then
    printf '%s\n' "$HANDOFF_CODE" > "$FINALIZATION_DIR/launcher.exitcode"
    exit "$HANDOFF_CODE"
fi

if python -u scripts/run_longvideobench_proportional.py --phase finalize \
    --out "$EXPERIMENT_DIR" --devices 0,1,2,3,4,6 --videojev-devices 7 \
    > "$FINALIZATION_DIR/finalize.log" 2>&1; then
    FINAL_CODE=0
else
    FINAL_CODE=$?
fi
printf '%s\n' "$FINAL_CODE" > "$FINALIZATION_DIR/finalize.exitcode"
printf '%s\n' "$FINAL_CODE" > "$FINALIZATION_DIR/launcher.exitcode"
exit "$FINAL_CODE"
