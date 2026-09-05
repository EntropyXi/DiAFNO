#!/bin/bash
# 用途：历史运维：检查 GPU 拓扑及通信相关环境。
echo "=== GPU topology matrix (P2P/NVLink) ==="
nvidia-smi topo -m 2>/dev/null
echo "=== watcher validation status ==="
ls -la /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/epoch_snapshots/metrics_epoch_023.json 2>/dev/null || echo "still validating"
ps -u zzx -o pid,pcpu,etime,cmd | grep validate_ostia | grep -v grep | head -3
echo "=== rate sample 60s ==="
S1=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
sleep 60
S2=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
echo "steps $S1 -> $S2 in 60s"
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1
echo "=== gpu 0/3 + ecc ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1p;4p'
nvidia-smi --query-gpu=index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader | sed -n '1p;4p'
echo CHECK_DONE
