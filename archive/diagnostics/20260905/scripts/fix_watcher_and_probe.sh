#!/bin/bash
# 用途：历史运维：修复当时的验证监测进程并探测状态。
echo "=== [1] relaunch watcher with correct cwd ==="
tmux new-session -d -s ostia_epoch_watcher "cd /data2/user/zzx/exam_preprocessed/DiAFNO && CUDA_VISIBLE_DEVICES=1 /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u scripts/finetune_epoch_watcher.py --repo /data2/user/zzx/exam_preprocessed/DiAFNO --exp-dir /data2/user/zzx/exam_preprocessed/DiAFNO/experiments/ostia_7day_to15day_residual_scratch --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 --log-file /tmp/ostia_ft_logs/scratch_h5runs.log --device cuda:0 --sampling-steps 16 --s-churn 0 --ensemble-members 4 --max-samples 200 --poll-seconds 15 --settle-seconds 8 --exit-grace-seconds 600 > /tmp/ostia_ft_logs/residual_scratch_watcher.log 2>&1"
sleep 5
echo "watcher now:"
pgrep -af finetune_epoch_watcher.py | grep -v grep | head -2
tail -2 /tmp/ostia_ft_logs/residual_scratch_watcher.log 2>/dev/null
echo "=== [2] data-path probe (isolate batch-load cost) ==="
cat > /tmp/ostia_ft_logs/load_probe.py <<'PY'
import time, sys
sys.path.insert(0, "/data2/user/zzx/exam_preprocessed/DiAFNO")
from diafno.data.ostia import OSTIADailyDataset
from diafno.training.data import DistributedSpatialBlockSampler
from torch.utils.data import DataLoader

ds = OSTIADailyDataset(
    h5_path="/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5",
    split="train", input_days=7, output_days=15, condition_mode="sst_mask",
)
print("spd=%d chunk_rows=%d" % (ds.samples_per_day, ds.chunk_rows), flush=True)
sampler = DistributedSpatialBlockSampler(
    dataset=ds, samples_per_epoch=31200, batch_size=16,
    num_replicas=1, rank=0, seed=123,
)
loader = DataLoader(
    ds, batch_size=16, sampler=sampler, shuffle=False,
    num_workers=4, pin_memory=False, persistent_workers=True, prefetch_factor=2,
)
it = iter(loader)
t0 = time.time()
n = 0
for _ in it:
    n += 1
    el = time.time() - t0
    if n >= 15 or el > 240:
        break
el = time.time() - t0
print("probe: %d batches in %.1fs -> %.2f s/batch" % (n, el, el / max(n, 1)), flush=True)
PY
nohup /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u /tmp/ostia_ft_logs/load_probe.py > /tmp/ostia_ft_logs/load_probe.log 2>&1 &
echo "probe launched pid=$!"
echo DONE
