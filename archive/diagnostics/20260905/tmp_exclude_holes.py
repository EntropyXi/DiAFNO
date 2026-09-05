# 用途：历史诊断：排除疑似缺测窗口后重新比较验证误差。
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from diafno.evaluation.config import build_validation_parser, OSTIAValidationConfig
from diafno.evaluation.validator import OSTIAValidator

h5 = "/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5"
ckpt = "/data2/user/zzx/exam_preprocessed/DiAFNO/experiments/det_lead_standardized/epoch_015.pth"
parser = build_validation_parser()
args = parser.parse_args([
    "--checkpoint", ckpt, "--h5-path", h5, "--split", "val",
    "--prediction-mode", "model", "--max-samples", "200", "--seed", "123",
    "--batch-size", "16", "--num-workers", "2", "--device", "cuda:0",
    "--output-path", "/tmp/exclude_out.json",
])
val = OSTIAValidator(OSTIAValidationConfig.from_args(args)).setup()
indices = val._build_indices()
rows = json.load(open("/tmp/per_sample_val200.json"))
bad = {r["index"] for r in rows if r["rmse"] is not None and r["rmse"] > 3}
clean = [i for i in indices if i not in bad]
print("excluding indices:", sorted(bad))
loader = DataLoader(
    Subset(val.dataset, clean), batch_size=16, shuffle=False,
    num_workers=2, pin_memory=val.device.type == "cuda", drop_last=False,
)

overall = 0.0
pixels = 0
with torch.no_grad():
    for batch in loader:
        cond = batch["condition"].to(val.device, non_blocking=True)
        tgt = batch["target"].to(val.device, non_blocking=True)
        msk = batch["target_mask"].to(val.device, non_blocking=True)
        with torch.autocast("cuda", enabled=val.amp_enabled):
            pred = val._predict(cond, 0)[..., 0]
        m = msk[..., 0].bool()
        t = tgt[..., 0]
        p = pred * val.dataset.sst_std + val.dataset.sst_mean
        t = t * val.dataset.sst_std + val.dataset.sst_mean
        e = ((p - t) ** 2)[m]
        overall += float(e.sum().item())
        pixels += int(m.sum().item())
print("clean val (n=%d): overall_rmse=%.5f  (original with 4 hole samples: 1.13703)" % (len(clean), (overall / pixels) ** 0.5))
