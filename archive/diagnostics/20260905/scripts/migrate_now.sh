#!/bin/bash
# 用途：历史运维：按当时配置迁移并恢复训练，不能直接用于当前实验。
# Immediate migration: epoch 23 checkpoint already on disk. Stop -> backup -> relaunch on GPUs 0,3.
OUT=/data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch
LOG=/tmp/ostia_ft_logs/scratch_h5runs.log
PATT="trainer_ostia.py --output-dir experiments/ostia_7day_to15day_residual_scratch"

echo "=== [1/6] confirm epoch-23 checkpoint fresh ==="
date -u +"%H:%M:%S UTC"
ls -la "$OUT/latest.pth"
MT=$(stat -c %Y "$OUT/latest.pth")
NOW=$(date +%s)
echo "latest.pth age: $((NOW-MT))s"
tr '\r' '\n' < "$LOG" | grep -E "epoch=2[0-9] train_loss=" | tail -3

echo "=== [2/6] backup checkpoint ==="
cp -p "$OUT/latest.pth" /tmp/ostia_ft_logs/epoch23_pre_migrate.pth
ls -la /tmp/ostia_ft_logs/epoch23_pre_migrate.pth

echo "=== [3/6] stop training tree ==="
pkill -TERM -f "$PATT" 2>/dev/null && echo "SIGTERM sent" || echo "no match"
sleep 12
pkill -KILL -f "$PATT" 2>/dev/null || true
sleep 3
if pgrep -f "$PATT" > /dev/null; then echo "WARN: still alive:"; pgrep -af "$PATT" | head -3; else echo "training tree stopped"; fi
kill 2577941 2>/dev/null && echo "zombie watcher killed" || echo "zombie watcher already gone"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'

echo "=== [4/6] relaunch on GPUs 0,3 (--resume latest) ==="
tmux kill-session -t ostia_residual_scratch 2>/dev/null || true
tmux new-session -d -s ostia_residual_scratch "cd /data2/user/zzx/exam_preprocessed/DiAFNO && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py --output-dir experiments/ostia_7day_to15day_residual_scratch --target-mode residual --sigma-data 0.15 --sigma-max 1.0 --sigma-min 0.0005 --p-mean -3.0 --learning-rate 2e-4 --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4 --checkpoint-interval 5 --resume latest 2>&1 | tee -a /tmp/ostia_ft_logs/scratch_h5runs.log"
echo "tmux launched:"
tmux ls | grep ostia_residual_scratch || echo "tmux session missing!"

echo "=== [5/6] wait 90s for setup ==="
sleep 90

echo "=== [6/6] verify ==="
pgrep -af "$PATT" | head -3
echo "--- gpu 0..3 ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '1,4p'
echo "--- log tail ---"
tr '\r' '\n' < "$LOG" | grep -vE "^$" | tail -8
echo MIGRATION_COMPLETE $(date -u +"%H:%M:%S UTC")
