# diafno/models

IAFNO 骨干、EDM 扩散与模型配置。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
models/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── config.py  # 定义模型配置并按 checkpoint 语义构造对应模型
├── diffusion.py  # 实现 EDM 扩散训练目标与迭代采样过程
├── iafno.py  # 实现带时间条件的隐式自适应 Fourier 神经算子骨干
└── README.md  # 说明本目录用途并列出本层文件和子目录
```
