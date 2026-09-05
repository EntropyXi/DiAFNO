# 用途：历史诊断：提取固定验证样本的逐样本误差。
import json
import numpy as np
import torch
from diafno.evaluation.config import build_validation_parser, OSTIAValidationConfig
from diafno.evaluation.validator import OSTIAValidator

h5 = "/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5"
ckpt = "/data2/user/zzx/exam_preprocessed/DiAFNO/experiments/det_lead_standardized/epoch_015.pth"
parser = build_validation_parser()
args = parser.parse_args([
    "--checkpoint", ckpt, "--h5-path", h5, "--split", "val",
    "--prediction-mode", "model", "--max-samples", "200", "--seed", "123",
    "--batch-size", "16", "--num-workers", "2", "--device", "cuda:0",
    "--output-path", "/tmp/per_sample_out.json",
])
val = OSTIAValidator(OSTIAValidationConfig.from_args(args)).setup()
indices = val._build_indices()
ds = val.dataset
rows = []
with torch.no_grad():
    for i, di in enumerate(indices):
        s = ds[di]
        cond = s["condition"].to(val.device).unsqueeze(0)
        tgt = s["target"].to(val.device).unsqueeze(0)
        msk = s["target_mask"].to(val.device).unsqueeze(0)
        with torch.autocast("cuda", enabled=val.amp_enabled):
            pred = val._predict(cond, i)[..., 0]
        m = msk[..., 0].bool()
        t = tgt[..., 0]
        p = pred * ds.sst_std + ds.sst_mean
        t = t * ds.sst_std + ds.sst_mean
        errs = (p - t) ** 2
        seq_index = int(di) // ds.samples_per_day
        init_day = int(ds.split_start_day) + seq_index
        lead_rmses = []
        for l in range(errs.shape[1]):
            e = errs[0, l][m[0, l]]
            lead_rmses.append(float(e.mean().sqrt().item()) if e.numel() else None)
        all_e = errs[0][m[0]]
        day5plus = torch.cat(
            [errs[0, l][m[0, l]].reshape(-1) for l in range(4, 15)]
        )
        rows.append({
            "index": int(di), "init_day": int(init_day), "seq_index": int(seq_index),
            "rmse": float(all_e.mean().sqrt().item()) if all_e.numel() else None,
            "rmse_day1": lead_rmses[0],
            "rmse_day5plus": float(day5plus.mean().sqrt().item()) if day5plus.numel() else None,
            "by_lead": lead_rmses,
        })
with open("/tmp/per_sample_val200.json", "w") as f:
    json.dump(rows, f, indent=1)
rows.sort(key=lambda r: -(r["rmse"] or 0))
print("worst 20 by overall rmse:")
for r in rows[:20]:
    print("init_day=%d rmse=%.4f day1=%.4f day5plus=%.4f" % (
        r["init_day"], r["rmse"], r["rmse_day1"], r["rmse_day5plus"]))
print("best 5:")
for r in rows[-5:]:
    print("init_day=%d rmse=%.4f" % (r["init_day"], r["rmse"]))
