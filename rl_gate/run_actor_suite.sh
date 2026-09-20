#!/bin/bash
# Actor 边界套件：TP4×{4096,3072,2048,1024} + TP8×4096（每配置 2 step）
cd /data/wangshenghua/wsh/teacher_data
for MT in 4096 3072 2048 1024; do
  echo "=== ACTOR TP4 MT=$MT ==="
  bash run_actor_boundary.sh 4 $MT 0,1,2,3 2>&1 | tail -3
done
echo "=== ACTOR TP8 MT=4096 ==="
bash run_actor_boundary.sh 8 4096 0,1,2,3,4,5,6,7 2>&1 | tail -3
echo "ACTOR SUITE DONE"
