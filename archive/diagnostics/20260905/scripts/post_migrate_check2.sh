#!/bin/bash
# 用途：历史运维：补查迁移后进程、日志与训练状态。
echo "=== rank/worker processes (ppid chain) ==="
ps -u zzx -o pid,ppid,rss,pcpu,cmd | grep "python -u trainer_ostia" | grep -v grep | awk '{printf "pid=%-8s ppid=%-8s rss=%.1fGB cpu=%s%%\n", $1, $2, $3/1048576, $4}'
echo "=== NUMA of the two ranks (children of torchrun) ==="
TORCH=$(pgrep -f "torch.distributed.run" | head -1)
echo "torchrun=$TORCH"
for P in $(ps -u zzx -o pid,ppid | awk -v t=$TORCH '$2==t {print $1}' | head -2); do
  echo "--- rank $P ---"
  grep -oE "N[0-9]+=[0-9]+" /proc/$P/numa_maps 2>/dev/null | awk -F'[= ]' '{s[$1]+=$2} END {for (k in s) printf "%s=%.1fGiB ", k, s[k]*4/1048576; print ""}'
done
echo "=== step rate over 90s ==="
S1=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
sleep 90
S2=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
echo "steps $S1 -> $S2 in 90s ($(echo "scale=1; 90/($S2-$S1)" | bc 2>/dev/null || python3 -c "print(round(90/($S2-$S1),1))") s/it)"
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1
echo "=== watcher epoch-23 metrics done? ==="
ls -la /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/epoch_snapshots/metrics_epoch_023.json 2>/dev/null || echo "still validating"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'
echo CHECK_DONE
