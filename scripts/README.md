# scripts

训练运行、烟测、验证监测、消融和结果整理工具。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
scripts/
├── archive_legacy_ostia_main.sh  # 校验后归档旧主训练目录，默认只预览而不执行
├── finetune_epoch_watcher.py  # 快照并验证已完成 epoch 的权重，维护验证 RMSE 最优模型
├── init_ostia_centered_main.sh  # 初始化 centered 主训练配置产物，默认预览且拒绝覆盖
├── mem_probe.py  # 用合成输入测量候选 batch 的训练峰值显存
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── run_ostia_centered_main.sh  # 检查权威配置和前置产物后启动 centered 主训练
├── smoke_ostia_centered.sh  # 在独立输出目录执行 centered diffusion 的 GPU 结构烟测
└── watch_ostia_centered.sh  # 监测完成的 checkpoint 并以固定验证协议维护最优权重
```
