# diafno/training

训练配置、数据加载、训练循环和权重保存。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
training/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── artifacts.py  # 保存与恢复权重、优化器状态及训练曲线等产物
├── config.py  # 解析训练参数并校验配置、统计量和恢复训练约束
├── data.py  # 构造训练采样器与 DataLoader，并组织批量 HDF5 读取
├── main.py  # 连接配置解析与训练器，提供训练模块入口
├── normalization.py  # 管理训练集标准化统计量及张量变换
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── runtime.py  # 初始化设备、随机种子与分布式进程环境
└── trainer.py  # 执行训练循环、AMP、梯度累积及 checkpoint 保存
```
