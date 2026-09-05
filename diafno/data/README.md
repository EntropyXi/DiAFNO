# diafno/data

HDF5 样本读取与输入数据契约。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
data/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── condition_schema.py  # 统一 SST、mask、位置和季节条件通道的顺序与版本
├── manifest.py  # 校验真实日期 manifest，约束缺日过滤与数据来源
├── ostia.py  # 从切块 HDF5 构造同区域的 7 日输入与 15 日目标样本
└── README.md  # 说明本目录用途并列出本层文件和子目录
```
