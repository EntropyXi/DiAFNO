#!/bin/bash
# 用途：历史运维：调整当时验证监测任务的运行位置。
MET=/data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/epoch_snapshots/metrics_epoch_023.json
echo "=== wait for epoch-23 validation to finish (max 6 min) ==="
for i in $(seq 1 90); do
  if [ -f "$MET" ]; then break; fi
  sleep 4
done
if [ -f "$MET" ]; then
  echo "validation done at $(date -u +%H:%M:%S)"
  python3 -c "import json; d=json.load(open('$MET')); o=d['overall']; print('epoch23 val overall_rmse=%.4f day1=%.4f' % (o['rmse'], d['by_lead_day']['1']['rmse']))" 2>/dev/null || head -c 300 "$MET"
else
  echo "TIMEOUT waiting for validation (still running?)"
  ps -u zzx -o pid,etime,cmd | grep validate_ostia | grep -v grep | head -2
fi
echo "=== relocate watcher GPU3 -> GPU1 ==="
kill 103665 103668 2>/dev/null && echo "old watcher killed" || echo "old watcher gone"
tmux kill-session -t ostia_epoch_watcher 2>/dev/null || true
tmux new-session -d -s ostia_epoch_watcher "CUDA_VISIBLE_DEVICES=1 /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u scripts/finetune_epoch_watcher.py --repo /data2/user/zzx/exam_preprocessed/DiAFNO --exp-dir /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 --log-file /tmp/ostia_ft_logs/scratch_h5runs.log --device cuda:0 --sampling-steps 16 --s-churn 0 --ensemble-members 4 --max-samples 200 --poll-seconds 15 --settle-seconds 8 --exit-grace-seconds 600 > /tmp/ostia_ft_logs/residual_scratch_watcher.log 2>&1"
echo "watcher relaunched on GPU1:"
pgrep -af finetune_epoch_watcher.py | grep -v grep | head -2
echo "=== rate sample 120s (training exclusive on 0,3 now) ==="
S1=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
sleep 120
S2=$(grep -oE "step=[0-9]+" /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1 | cut -d= -f2)
echo "steps $S1 -> $S2 in 120s ($(python3 -c "print(round(120/max($S2-$S1,1),1))") s/it)"
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'
echo WATCHER_MOVE_DONE
