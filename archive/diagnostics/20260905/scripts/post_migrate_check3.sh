#!/bin/bash
# 用途：历史运维：再次核对迁移后的训练进度。
echo "=== rank processes detail ==="
for P in 391836 391837; do
  echo "--- rank $P ---"
  ps -p $P -o pid,state,pcpu,rss,etime,wchan:30 2>/dev/null
  cat /proc/$P/stack 2>/dev/null | head -5
done
echo "=== worker states (D/R/S counts) ==="
ps -u zzx -o pid,state,wchan:20,pcpu | awk '$2=="D" || $2=="R" {print}' | head -10
ps -u zzx -o pid,state | awk '$2=="D" {n++} END {print "D-state workers:", n+0}'
echo "=== node free + pressure ==="
for N in /sys/devices/system/node/node0/meminfo /sys/devices/system/node/node3/meminfo; do
  echo "$N: $(grep MemFree $N | awk '{printf "%.1f", $4/1048576}') GiB free"
done
cat /proc/pressure/memory
cat /proc/pressure/io 2>/dev/null
echo "=== iostat nvme ==="
iostat -x 1 2 2>/dev/null | grep -E "nvme" | tail -2
echo "=== hjc current ==="
ps -u hjc -o pid,rss,pcpu,etime --sort=-rss 2>/dev/null | head -4
echo "=== watcher validation done? ==="
ls -la /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/epoch_snapshots/metrics_epoch_023.json 2>/dev/null || echo "still validating"
echo "=== step count over 60s ==="
S1=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
sleep 60
S2=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
echo "steps $S1 -> $S2 in 60s"
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1
echo "=== gpu board ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo CHECK_DONE
