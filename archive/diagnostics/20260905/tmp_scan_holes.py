# 用途：历史诊断：抽查 HDF5 每日切片中的全 NaN 数据。
import h5py
import numpy as np
import time

h5 = "/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5"
t0 = time.time()
holes = []
with h5py.File(h5, "r") as f:
    for day in range(11261):
        s = np.asarray(f["sst"][day * 100, 0, 0:16, 0:16])
        if np.isnan(s).all():
            holes.append(day)
        if day % 500 == 0:
            print("progress day %d, holes so far %d (%.0fs)" % (
                day, len(holes), time.time() - t0), flush=True)
print("hole days total:", len(holes), flush=True)
runs = []
for d in holes:
    if runs and d == runs[-1][-1] + 1:
        runs[-1].append(d)
    else:
        runs.append([d])
for r in runs:
    tag = "train" if r[0] < 7882 else ("val" if r[0] < 10134 else "test")
    print("day %d-%d (%d days) [%s]" % (r[0], r[-1], len(r), tag), flush=True)
print("done in %.0fs" % (time.time() - t0), flush=True)
