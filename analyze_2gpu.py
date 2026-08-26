import numpy as np
import torch

# 1) forward_features 里 if 条件的真实解析（Python 运算符优先级）
ex_layer, nlayer = 4, 2
actual = (ex_layer != 1 & nlayer == 1)
chained = (ex_layer != (1 & nlayer)) and ((1 & nlayer) == 1)
intended = (ex_layer != 1) and (nlayer == 1)
print("parsed  :", actual)
print("chained :", chained)
print("intended:", intended)
print("(1 & nlayer) =", 1 & nlayer)

# 2) checkpoint 的 epoch 级损失
for tag in ("ostia_smoke", "ostia_smoke_2gpu"):
    path = "experiments/" + tag + "/latest.pth"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tl = ck["train_loss"]
    print("==", tag, "== train_loss=%.6f" % tl,
          "epoch=", ck["epoch"], "global_step=", ck["global_step"])
    print("   scaler scale:", ck["scaler"].get("scale", "N/A"))

# 3) 尖峰位置
d = np.load("experiments/ostia_smoke_2gpu/training_curves.npz")
lv, gv = d["loss_values"], d["gradient_norms"]
print()
print("loss peaks (value, step):",
      sorted(zip(lv, range(1, 251)), reverse=True)[:5])
print("grad peaks (value, step):",
      sorted(zip(gv, range(1, 251)), reverse=True)[:5])
print("grad > 1.0 at steps:", [int(i + 1) for i, v in enumerate(gv) if v > 1.0])

# 4) loss 滞后自相关
x = lv - lv.mean()
x = x / x.std()
for lag in (1, 2, 3, 4, 5, 6, 7, 8, 10):
    corr = float((x[lag:] * x[:-lag]).mean())
    print("loss autocorr lag=%d: %+.3f" % (lag, corr))

# 5) 归一化统计对比（数据集一致性）
import json
for tag in ("ostia_smoke", "ostia_smoke_2gpu"):
    with open("experiments/" + tag + "/normalization.json") as f:
        norm = json.load(f)
    print(tag, "sst_mean=%.6f sst_std=%.6f" % (norm["sst_mean"], norm["sst_std"]))
