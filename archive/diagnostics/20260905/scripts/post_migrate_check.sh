#!/bin/bash
# 用途：历史运维：检查迁移后的训练和 GPU 状态。
echo "=== NUMA placement new ranks ==="
for P in $(pgrep -f "trainer_ostia.py --output-dir" | head -2); do
  echo "--- rank $P ---"
  grep -oE "N[0-9]+=[0-9]+" /proc/$P/numa_maps 2>/dev/null | awk -F'[= ]' '{s[$1]+=$2} END {for (k in s) printf "%s=%.1fGiB ", k, s[k]*4/1048576; print ""}'
done
echo "=== node 0/3 free ==="
for N in /sys/devices/system/node/node0/meminfo /sys/devices/system/node/node3/meminfo; do
  echo "$N: $(grep MemFree $N | awk '{printf "%.1f", $4/1048576}') GiB free"
done
echo "=== watcher count + snapshots ==="
pgrep -af finetune_epoch_watcher.py | grep -v grep | awk '{print $1}' | tr '\n' ' '; echo
ls -lat /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/epoch_snapshots/ | head -4
echo "=== training log tail ==="
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -2
echo CHECK_DONE
