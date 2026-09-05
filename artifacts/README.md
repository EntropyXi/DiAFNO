# artifacts

审计、统计量、调度状态和代表性烟测等辅助产物。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
artifacts/
├── smoke/  # 保留的一套代表性烟测曲线和原始数值
├── A0_stage1_manifest_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A0_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A1_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A1_stage3_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A2_stage1_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A2_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A3_stage1_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A3_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A3_stage3_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A4_stage1_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A4_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A4_stage3_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_finetune_lr1e4_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_finetune_lr5e-5_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_finetune_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_stage2_2gpu_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_stage2_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── A5_stage3_2gpu_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── bootstrap_A4_step1500.json  # 配对时间分块重采样的 skill 与置信区间结果（服务器产物）
├── bootstrap_A5_step1500.json  # 配对时间分块重采样的 skill 与置信区间结果（服务器产物）
├── lead_stats_A0_manifest.json  # 逐 lead 残差或创新的训练集统计量及来源信息（服务器产物）
├── ostia_data_manifest_real.json  # 真实日期、两处缺日和经纬度数据契约（服务器产物）
├── ostia_h5_audit_real.json  # 源 HDF5 形状、布局和坐标时间审计结果（服务器产物）
├── probe_A0_gpu4.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A1_gpu4.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A2_gpu5.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A3_gpu7.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A4_gpu7.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A5_gpu3_blocks1.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── probe_A5_gpu5.json  # 指定架构或 GPU 的显存、耗时与运行探测结果（服务器产物）
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── stage1_master.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_boot_a4.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_boot_a5.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_ft_eval2000.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_ft_eval2500.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_ft_eval2750.log  # 该运行的追加日志或逐次指标记录（服务器产物）
├── supervisor_plan_ft.json  # 机器可读的消融任务依赖配置；属于运行配置而非文字计划书（服务器产物）
├── supervisor_plan.json  # 机器可读的消融任务依赖配置；属于运行配置而非文字计划书（服务器产物）
├── supervisor_state_ft.json  # 消融调度器记录的任务执行状态（服务器产物）
├── supervisor_state.json  # 消融调度器记录的任务执行状态（服务器产物）
├── supervisor_summary.log  # 该运行的追加日志或逐次指标记录（服务器产物）
└── three_method_ep22_test200_e16_20260905.log  # 该运行的追加日志或逐次指标记录（服务器产物）
```
