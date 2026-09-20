#!/usr/bin/env bash
# Teacher Production V2 启动器：proxy（0.0.0.0，iptables 已收紧）+ production_v2.py
# 用法（经 login shell 拿 GLM key）：
#   PROD_V2_LIMIT=20  PROD_V2_CONCURRENCY=2 bash -lc 'bash run_production_v2.sh'   # 20-task smoke
#   PROD_V2_CONCURRENCY=6               bash -lc 'bash run_production_v2.sh'       # 正式（resume 安全）
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/wangshenghua/miniconda3/envs/slime/bin/python
export HF_HOME=/data/wangshenghua/.cache/huggingface

# ---- iptables 守卫（幂等；重启后规则会丢，这里重加） ----
sudo iptables -C INPUT -s 127.0.0.1/32 -p tcp --dport 18734 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 -s 127.0.0.1/32 -p tcp --dport 18734 -j ACCEPT
sudo iptables -C INPUT -s 172.16.0.0/12 -p tcp --dport 18734 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 2 -s 172.16.0.0/12 -p tcp --dport 18734 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 18734 -j DROP 2>/dev/null || sudo iptables -A INPUT -p tcp --dport 18734 -j DROP

# ---- proxy：健康则复用，否则启动（0.0.0.0：docker bridge 经 host.docker.internal 回拨） ----
if ! curl -s --max-time 5 http://127.0.0.1:18734/_health >/dev/null 2>&1; then
  [ -n "${GLM_API_KEY:-}" ] || { echo "[ABORT] GLM_API_KEY 未设置"; exit 3; }
  setsid nohup $PY teacher_proxy.py --host 0.0.0.0 --port 18734 > outputs/proxy_v2.log 2>&1 < /dev/null &
  disown
  for i in $(seq 1 30); do curl -s --max-time 3 http://127.0.0.1:18734/_health >/dev/null 2>&1 && break; sleep 1; done
fi
KEYP=$(curl -s --max-time 5 http://127.0.0.1:18734/_control/state | python3 -c 'import json,sys; print(json.load(sys.stdin).get("api_key_present"))')
[ "$KEYP" = "True" ] || { echo "[ABORT] proxy 无 GLM key（api_key_present=$KEYP）"; exit 3; }
echo "[launcher] proxy OK (api_key_present=True)"

# ---- 生产（resume 安全；日志 tee） ----
exec $PY -u production_v2.py 2>&1 | tee -a outputs/production_v2_run.log
