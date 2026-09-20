#!/bin/bash
# RL gate 并发矩阵（Part 五+六）：TP2 c2/c4 → TP4 c8 → TP1 c1
set -x
cd /data/wangshenghua/wsh/teacher_data
source /data/wangshenghua/miniconda3/etc/profile.d/conda.sh
conda activate slime
export SLIME_AGENT_CC_EXTRA_ENVS='{"IS_SANDBOX":"1","DISABLE_AUTOUPDATER":"1","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}'
export PYTHONPATH=/data/wangshenghua/wsh/teacher_data:/data/wangshenghua/wsh/slime
LOG=outputs/rl_gate/logs

# 1) TP2 @30101 (GPU0,1)：c=2 (4 tasks) 与 c=4 (8 tasks)
python rl_gate/standalone_concurrency.py 30101 2 4 --tag tp2_c2 2>&1 | grep -E '^\{"task' 
python rl_gate/standalone_concurrency.py 30101 4 8 --tag tp2_c4 2>&1 | grep -E '^\{"task'

# 2) TP4 @30100 (GPU4-7)：c=8 (8 tasks)
python rl_gate/standalone_concurrency.py 30100 8 8 --tag tp4_c8 2>&1 | grep -E '^\{"task'

# 3) TP1 (GPU2, port 30102)：c=1 (2 tasks)
pkill -f 'port 3011[0]' 2>/dev/null || true
nohup bash serve_rl.sh 2 30102 0.92 1 32768 > $LOG/sgl_tp1_rlgate.log 2>&1 &
for i in $(seq 1 40); do sleep 12; curl -s -m 3 http://127.0.0.1:30102/health >/dev/null 2>&1 && break; done
python rl_gate/standalone_concurrency.py 30102 1 2 --tag tp1_c1 2>&1 | grep -E '^\{"task'
python rl_gate/standalone_concurrency.py 30102 2 2 --tag tp1_c2 2>&1 | grep -E '^\{"task'

echo "MATRIX DONE"
