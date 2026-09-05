# archive/diagnostics/20260905

历史排障脚本备份；含旧路径或进程操作，勿直接执行。
本层文件用途见下方；子目录详见各自 README。历史内容仅供追溯，不代表当前执行指令。

## 本层项目树

```text
20260905/
├── scripts/  # 历史排障脚本备份；含旧路径或进程操作，勿直接执行
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── tmp_exclude_holes.py  # 历史诊断：排除疑似缺测窗口后重新比较验证误差
├── tmp_per_sample_analysis.py  # 历史诊断：提取固定验证样本的逐样本误差
└── tmp_scan_holes.py  # 历史诊断：抽查 HDF5 每日切片中的全 NaN 数据
```
