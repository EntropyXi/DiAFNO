#!/bin/bash
# 用途：历史运维：等待指定 epoch 后迁移训练到 GPU 0 和 3。
# Auto-migrate: wait for epoch 23 completion, then restart training on GPUs 0/3 with --resume latest.
LOG=/tmp/ostia_ft_logs/scratch_h5runs.log
MIGLOG=/tmp/ostia_ft_logs/migrate_gpu03.log
OUT=/data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch
PATT="trainer_ostia.py --output-dir experiments/ostia_7day_to15day_residual_scratch"

echo "[migrate] $(date -u +%H:%M:%S) waiting for 'epoch=23 train_loss=' in $LOG"
for i in $(seq 1 720); do
  if grep -q "epoch=23 train_loss=" "$LOG" 2>/dev/null; then
    echo "[migrate] $(date -u +%H:%M:%S) epoch=23 line seen (wait $((i*20))s)"
    break
  fi
  sleep 20
done
if ! grep -q "epoch=23 train_loss=" "$LOG" 2>/dev/null; then
  echo "[migrate] TIMEOUT after 4h without epoch=23 line; aborting" >> "$MIGLOG"
  exit 1
fi

# the trainer prints the epoch line BEFORE saving latest.pth; wait for the save to land
T0=$(date +%s)
echo "[migrate] waiting for latest.pth save (mtime >= trigger time)"
for i in $(seq 1 300); do
  MT=$(stat -c %Y "$OUT/latest.pth" 2>/dev/null || echo 0)
  if [ "$MT" -ge "$T0" ]; then
    echo "[migrate] latest.pth saved (waited $((i*2))s)"
    break
  fi
  sleep 2
done
sleep 5
ls -la "$OUT/latest.pth"

echo "[migrate] backup pre-migration checkpoint"
cp -p "$OUT/latest.pth" /tmp/ostia_ft_logs/epoch23_pre_migrate.pth
ls -la /tmp/ostia_ft_logs/epoch23_pre_migrate.pth

echo "[migrate] $(date -u +%H:%M:%S) stopping training tree (SIGTERM)"
pkill -TERM -f "$PATT" 2>/dev/null || echo "no match (already gone?)"
sleep 12
pkill -KILL -f "$PATT" 2>/dev/null || true
sleep 3
if pgrep -f "$PATT" > /dev/null; then
  echo "[migrate] WARN: training tree still alive:" >> "$MIGLOG"
  pgrep -af "$PATT" | head -3 >> "$MIGLOG"
else
  echo "[migrate] training tree fully stopped"
fi

echo "[migrate] kill zombie watcher 2577941 (dead-log watcher only)"
if kill -0 2577941 2>/dev/null; then kill 2577941 2>/dev/null && echo "[migrate] zombie watcher killed" || echo "[migrate] zombie watcher kill failed"; else echo "[migrate] zombie watcher already gone"; fi

echo "[migrate] GPU state after stop:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'

echo "[migrate] $(date -u +%H:%M:%S) relaunching on GPUs 0,3 with --resume latest (tmux ostia_residual_scratch)"
tmux kill-session -t ostia_residual_scratch 2>/dev/null || true
tmux new-session -d -s ostia_residual_scratch "cd /data2/user/zzx/exam_preprocessed/DiAFNO && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py --output-dir experiments/ostia_7day_to15day_residual_scratch --target-mode residual --sigma-data 0.15 --sigma-max 1.0 --sigma-min 0.0005 --p-mean -3.0 --learning-rate 2e-4 --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4 --checkpoint-interval 5 --resume latest 2>&1 | tee -a /tmp/ostia_ft_logs/scratch_h5runs.log"
sleep 90

echo "[migrate] relaunch verification:"
pgrep -af "$PATT" | head -3
echo "[migrate] GPU 0..3:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'
echo "[migrate] log tail:"
tr '\r' '\n' < "$LOG" | tail -6
echo "[migrate] DONE $(date -u +%H:%M:%S)"
