#!/bin/bash
# 用途：历史运维：收集训练迁移前的进程与资源状态。
echo "=== last epoch line in h5runs log ==="
tr '\r' '\n' < /tmp/ostia_ft_logs/scratch_h5runs.log 2>/dev/null | grep -E "epoch=[0-9]+ train_loss=" | tail -3
echo "=== checkpoints ==="
ls -lat /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch/*.pth | head -4
echo "=== trainer alive ==="
pgrep -f "trainer_ostia.py" | head -3
echo "=== gpu 0..3 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'
echo "=== watcher alive ==="
pgrep -af finetune_epoch_watcher.py | grep -v grep | awk '{print $1}' | tr '\n' ' '; echo
echo "=== tmux sessions ==="
tmux ls 2>/dev/null || echo "no tmux sessions"
echo PRECHECK_DONE
