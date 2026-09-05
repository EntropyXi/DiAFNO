# deterministic_iafno/tests

确定性和 centered 模型、统计量及恢复训练的回归测试。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
tests/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── test_centered_checkpoint_roundtrip.py  # 验证centered 模型保存与恢复的一致性
├── test_centered_config_cli.py  # 验证centered 配置与命令行参数校验
├── test_centered_cpu_smoke.py  # 验证centered 模型在 CPU 上的训练与恢复烟测
├── test_centered_diffusion.py  # 验证冻结均值和中心化扩散的训练采样行为
├── test_centered_stats.py  # 验证创新统计量和来源校验规则
├── test_checkpoint_roundtrip.py  # 验证模型及优化器 checkpoint 的保存恢复
├── test_checkpoint_semantics.py  # 验证checkpoint 不可变字段与兼容性约束
├── test_deterministic_model.py  # 验证确定性残差模型的前向及损失
├── test_deterministic_real_backbone.py  # 验证确定性包装与真实 IAFNO 骨干的集成
├── test_evaluation_contract.py  # 验证有效像素、残差还原及评估协议
├── test_evaluator_centered.py  # 验证centered 模型的在线验证行为
├── test_evaluator_deterministic.py  # 验证确定性模型的在线验证行为
├── test_lead_stats_cli_validation.py  # 验证逐 lead 统计文件与命令行输入校验
├── test_lead_stats.py  # 验证逐 lead 残差统计量计算
├── test_legacy_diffusion_loss.py  # 验证旧扩散损失的兼容性与数值行为
├── test_legacy_metrics_golden.py  # 验证旧版本指标的固定数值回归
├── test_paired_bootstrap.py  # 验证配对时间分块 bootstrap 的统计行为
├── test_resume_restore.py  # 验证续训时配置和训练状态的恢复
└── test_trainer_amp_skip.py  # 验证AMP 溢出跳步时优化器与调度器的一致性
```
