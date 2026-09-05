# artifacts/smoke

保留的一套代表性烟测曲线和原始数值。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
smoke/
├── gradient_norm_curve.png  # 训练梯度范数随训练进度变化的曲线图
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── training_curves.npz  # 损失、梯度等训练曲线的原始数值
└── training_loss_curve.png  # 训练损失随训练进度变化的曲线图
```

## 原有使用说明（保留）

# Smoke test result

This directory keeps one representative two-GPU smoke-test result:

- `training_loss_curve.png`
- `gradient_norm_curve.png`
- `training_curves.npz`

The curves are retained as a historical sanity check. Old smoke checkpoints were removed because their temporal metadata predates the finalized daily OSTIA task and they are not valid resume points.
