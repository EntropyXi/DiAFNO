# tests

数据契约、消融执行和三方法图表的回归测试。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
tests/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── ostia_test_h5.py  # 构造满足时间、坐标与 mask 协议的合成 HDF5 测试夹具
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── test_method_comparison.py  # 验证三方法指标、CRPS、选区及图表输出
├── test_ostia_ablation_runner.py  # 验证消融执行器的安全约束与统计量集成
├── test_ostia_condition_channels.py  # 验证位置季节通道、日期边界与读取一致性
├── test_ostia_data_manifest.py  # 验证真实日期、缺日过滤与空间坐标 manifest
├── test_ostia_model_schema.py  # 验证条件通道配置、checkpoint 语义及模型前后向
└── test_ostia_supervisor.py  # 验证消融调度器与汇总器的纯逻辑
```
