# diafno/inference

批量推理、权重加载与预测文件保存。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
inference/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── config.py  # 解析批量预测的 checkpoint、数据和保存配置
├── inferencer.py  # 遍历预测样本、执行模型采样并交给输出器保存
├── main.py  # 提供批量 SST 推理模块入口
├── model.py  # 按配置恢复用于推理的模型权重
├── README.md  # 说明本目录用途并列出本层文件和子目录
└── writer.py  # 保存预测张量、目标及索引等推理产物
```
