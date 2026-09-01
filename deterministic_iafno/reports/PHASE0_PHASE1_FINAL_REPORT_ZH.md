# DiAFNO 日尺度 SST 预测：Phase 0/1 最终报告

生成日期：2026-09-01  
统计实现提交：`a233a49`  
任务：连续 7 日 SST 输入，预测随后 15 日 SST。

## 1. 结论摘要

Phase 0/1 已完成，固定验证 gate 通过。验证集只用于模型选择，测试集只在 gate 通过后冻结评估一次。

最终选中 `det_std_epoch015`：确定性 IAFNO、残差目标、按 lead day 标准化、训练 15 epoch。

| split | 样本数 | 模型 RMSE (K) | persistence RMSE (K) | RMSE 差 (K) | RMSE 差 95% CI (K) | MSE skill | MSE skill 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 200 | 1.1370 | 1.1871 | -0.0500 | [-0.1075, -0.0271] | 8.25% | [2.95%, 27.27%] |
| test | 1000 | 0.6307 | 0.7170 | -0.0863 | [-0.1014, -0.0712] | 22.62% | [19.62%, 25.28%] |

在本报告的配对时间块 bootstrap 口径下，val 与 test 的总体 RMSE 差区间均完整低于 0；两个 split 的 15 个 lead day 也全部如此。因此可以客观地说：**确定性均值预测已经训练成功，并且在冻结协议下稳定优于 persistence。**

这不是“任务已完全解决”。模型预测残差的振幅仍明显偏小；它证明了确定性条件均值可以超过 persistence，但没有证明概率扩散预测已经成功。

## 2. 数据、任务与冻结协议

- 数据文件：`/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5`
- HDF5 主体：`sst=(1126100,1,448,448)`，`mask=(1126100,448,448)`；每个日期有 100 个空间 patch。
- 时间尺度：日尺度，不是月尺度。
- 样本：7 日输入 + 15 日目标，时间步长 1 日。
- 条件：7 日 SST 加最后输入日 mask，共 8 个条件通道。
- 目标：未来绝对 SST 用最后输入日 SST 重新锚定；网络学习 15 日残差。
- 划分：按时间顺序 70% train、20% val、10% test；不随机混合日期。
- 固定选择集：val 200 个样本，seed=123。
- 冻结测试集：test 1000 个样本，seed=123；只在验证 gate 通过后运行一次。
- 指标：只在有效海洋像素上计算，单位 K；skill 定义为 `1 - MSE_model / MSE_persistence`。
- persistence：将第 7 个输入日的 SST 原样复制到未来 15 日。

## 3. Phase 0：旧模型与基线诊断

固定 val-200 上的总体 RMSE：

| 方法 | RMSE (K) | 结论 |
|---|---:|---|
| persistence | 1.1871 | 必须超过的基线 |
| 7 日线性趋势外推 | 1.6691 | 明显差于 persistence |
| 旧 diffusion，16 steps | 1.3702 | 明显差于 persistence |
| 旧 diffusion，100 steps | 1.3640 | 比 16 steps 仅改善约 0.45%，仍明显落后 |

旧 diffusion 的条件消融结果在 1.3702–1.3752 K 之间：`zero_sst`、`anchor_only`、`reverse_history` 和 `shuffle_history` 对结果影响都很小。结合 16→100 sampling steps 的微弱变化，可排除“主要只是采样步数不足”；旧 diffusion 的核心问题是均值/条件利用不足。

## 4. Phase 1：确定性两条训练臂

所有候选只在相同 val-200、相同 seed、相同有效像素协议下比较。下表 skill 均相对同一次评估中的 persistence 计算。

| 训练臂 | epoch | val RMSE (K) | MSE skill |
|---|---:|---:|---:|
| raw residual | 5 | 1.1880 | -0.15% |
| raw residual | 10 | 1.1846 | 0.41% |
| raw residual | 15 | 1.1766 | 1.76% |
| lead-standardized residual | 5 | 1.1852 | 0.31% |
| lead-standardized residual | 10 | 1.1402 | 7.74% |
| lead-standardized residual | 15 | **1.1370** | **8.25%** |

结论：raw residual 只获得小幅提升；按 lead day 标准化显著改善优化平衡，尤其避免较短 lead 被较大后期残差尺度淹没。`det_std_epoch015` 是预先规定候选中的验证最优项，因此被 gate 选中。

最优 checkpoint 训练语义：

- IAFNO：`embed_dim=128`、8 blocks、patch 8×8、15 输出通道；
- 单 GPU，batch-per-GPU=32，gradient accumulation=1，有效全局 batch=32；
- 每 epoch 31,200 个样本、975 optimizer steps，共 15 epoch；
- AdamW 配置中的初始学习率 2e-4、最小学习率 1e-6、weight decay 1e-4；
- lead 均值/标准差只由 train split 的 4096 个样本估计，未使用 val/test；
- checkpoint 带语义 sidecar，resume 时不允许静默漂移。

## 5. 配对 block bootstrap

旧 JSON 只有总体聚合指标，不能从两个 RMSE 反推出配对置信区间。因此对**同一个已选 checkpoint**按原 split、样本数和 seed 复评一次；点估计与旧 JSON 精确复现。复评不是重训，也没有重新选择模型。

统计过程：

1. 对每个样本和每个 lead，在完全相同的 mask/有效像素上同时累计模型与 persistence 的 SSE 和像素数。
2. 以 forecast initialization time 分组；采用 22 日非重叠时间块，对应完整的 7 日输入 + 15 日预测窗口。
3. 同一块内的日期、空间 patch、lead 和有效像素始终一起重采样；模型与 persistence 始终成对。
4. 对时间块有放回抽样 10,000 次，bootstrap seed=20260901，报告 percentile 95% CI。
5. val-200 形成 85 个非空时间块；test-1000 形成 51 个非空时间块。

该设计避免把同一时间窗口内的空间 patch 以及高度重叠的相邻预测窗口当作完全独立观测。CI 表达的是固定样本协议下跨时间块的不确定性，不包含“重新训练模型”“更换数据划分”或“重新抽取 lead 统计量”的不确定性。

## 6. 分 lead 结果

| split | lead | 模型 RMSE (K) | persistence RMSE (K) | RMSE 差 (K) | RMSE 差 95% CI (K) | MSE skill |
|---|---:|---:|---:|---:|---:|---:|
| val | 1 | 0.2408 | 0.2484 | -0.0076 | [-0.0097, -0.0057] | 6.00% |
| val | 5 | 0.9218 | 0.9466 | -0.0247 | [-0.0628, -0.0050] | 5.16% |
| val | 10 | 1.4196 | 1.4719 | -0.0523 | [-0.1216, -0.0295] | 6.98% |
| val | 15 | 1.4969 | 1.5827 | -0.0858 | [-0.1816, -0.0488] | 10.55% |
| test | 1 | 0.2461 | 0.2525 | -0.0064 | [-0.0073, -0.0056] | 5.01% |
| test | 5 | 0.5419 | 0.5909 | -0.0490 | [-0.0554, -0.0427] | 15.90% |
| test | 10 | 0.6948 | 0.7935 | -0.0987 | [-0.1160, -0.0815] | 23.34% |
| test | 15 | 0.8233 | 0.9662 | -0.1429 | [-0.1711, -0.1145] | 27.40% |

test 的绝对 RMSE 低于 val，不应直接解释成“泛化突然变好”，因为 test 时段本身更容易：同一 test 样本上的 persistence 也从 val 的 1.1870 K 降至 0.7170 K。相对 persistence 的 paired skill 才是跨时段更稳妥的比较。

## 7. 仍然存在的局限

1. 残差振幅偏保守。val 的 residual correlation=0.2898、predicted/target residual std ratio=0.2533；test 分别为 0.4771 和 0.4393。模型方向信息有效，但只恢复了真实残差波动的一部分。
2. 固定 val 只有 200 个样本。22 日分块后为 85 个非空块，验证 CI 比 test 更宽；结果支持 gate，但不应将 CI 精度无限外推。
3. test 只用于最终确认，不能根据 test 结果回头选择 checkpoint 或调整超参数。
4. 训练 loss 全部有限；记录的 gradient norm 中 raw 有 3 次、standardized 有 10 次非有限值。它们与 AMP overflow/step skip 的表现相容，但现有日志没有逐次记录是否跳步，因此这里只作数值稳定性披露，不作更强归因。后续训练宜显式记录 scaler scale 与 skipped-step 计数。
5. 本阶段证明的是确定性均值预测，不是概率校准、ensemble spread 或极端事件技巧。

## 8. 最终判定与下一阶段建议

**Phase 1 gate：PASS。** `det_std_epoch015` 在冻结 val/test 协议中均优于 persistence，且总体与全部 15 个 lead 的 paired 95% CI 均支持这一方向。

建议将 `experiments/det_lead_standardized/epoch_015.pth` 冻结为后续 centered diffusion 的 mean model：确定性网络负责条件均值，扩散分支只学习围绕该均值的扰动。下一阶段不能再用当前 test-1000 做模型选择；仍应在 val 上决定方案，最终测试另行冻结。

## 9. 可复现证据

- 单测：服务器 DiAFNO 环境中 51/51 通过。
- gate：`experiments/deterministic_iafno/persistence_gate.json`
- lead stats：`experiments/deterministic_iafno/lead_stats_train_4096.json`
- val CI：`experiments/deterministic_iafno/val_det_std_epoch015_200_ci.json`
- test CI：`experiments/deterministic_iafno/test_det_std_epoch015_1000_ci.json`
- 最优 checkpoint：`experiments/det_lead_standardized/epoch_015.pth`
- 配对 bootstrap 实现：`diafno/evaluation/bootstrap.py`
- checkpoint SHA-256：`cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6`
- val CI JSON SHA-256：`82ca2a7b3e026ed04780951cde64655a4ee18eec8bc4ed07c0da1be1602a8cc1`
- test CI JSON SHA-256：`7220009ab673d4f4a6f713de8bbe5c09d1b0eacbd113fef8cfce40ba8092b920`

报告中的机器可读摘要见 `deterministic_iafno/reports/phase0_phase1_final_summary.json`。
