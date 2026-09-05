#!/bin/bash
# 用途：历史运维：读取当时的探测结果。
echo "=== load probe result ==="
cat /tmp/ostia_ft_logs/load_probe.log 2>/dev/null || echo "probe not done"
echo "=== training rate now ==="
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1
echo "=== node 0/3 free ==="
for N in /sys/devices/system/node/node0/meminfo /sys/devices/system/node/node3/meminfo; do
  echo "$N: $(grep MemFree $N | awk '{printf "%.1f", $4/1048576}') GiB free"
done
echo "=== watcher ok? ==="
tail -2 /tmp/ostia_ft_logs/residual_scratch_watcher.log 2>/dev/null
echo CHECK_DONE
