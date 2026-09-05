# configs

正式训练与 A0–A5 架构消融的声明式配置。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
configs/
├── ostia_ablation_A0_baseline_p8_b8_i2.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_ablation_A1_geo_p8_b8_i2.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_ablation_A2_geo_p4_b8_i2.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_ablation_A3_geo_p4_b2_i2.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_ablation_A4_geo_p4_b1_i2.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_ablation_A5_geo_p4_best_i4.json  # 该 A0–A5 方案的条件通道、patch、blocks 与 implicit 配置
├── ostia_centered_diffusion_main.json  # centered diffusion 正式主训练的权威配置
└── README.md  # 说明本目录用途并列出本层文件和子目录
```
