# scripts

训练运行、烟测、验证监测、消融和结果整理工具。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
scripts/
├── ablation_common.py  # 提供消融任务的配置、命令构造和安全检查公共函数
├── ablation_summary.py  # 汇总短训消融、阶段复评和配对 bootstrap 结果
├── ablation_supervisor.py  # 按声明式任务依赖调度消融并记录状态，不作为训练入口自动运行
├── archive_legacy_ostia_main.sh  # 校验后归档旧主训练目录，默认只预览而不执行
├── audit_ostia_h5.py  # 只读审计 HDF5 的时间、坐标与形状并生成数据 manifest
├── compare_ostia_methods.py  # 在配对样本上评估 DiAFNO、IAFNO 和 persistence 并输出图表
├── finalize_ostia_comparison.py  # 选择一个展示区域，同时保留完整样本总体指标与来源记录
├── finetune_epoch_watcher.py  # 快照并验证已完成 epoch 的权重，维护验证 RMSE 最优模型
├── init_ostia_centered_main.sh  # 初始化 centered 主训练配置产物，默认预览且拒绝覆盖
├── probe_ablation_vram.py  # 用合成输入测量 A0–A5 架构的显存与吞吐
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── run_ablation_stages.py  # 按固定协议执行一个消融配置的烟测、短训或候选复核
├── run_ostia_centered_main.sh  # 检查权威配置和前置产物后启动 centered 主训练
├── smoke_ostia_centered.sh  # 在独立输出目录执行 centered diffusion 的 GPU 结构烟测
└── watch_ostia_centered.sh  # 监测完成的 checkpoint 并以固定验证协议维护最优权重
```
