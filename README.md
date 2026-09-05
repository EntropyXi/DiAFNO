# DiAFNO OSTIA 主训练项目

输入前 7 日 SST，预测后 15 日 SST；本目录组织源码、配置和实验结果。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
./
├── archive/  # 已归档的计划书与历史排障工具，不作为当前运行入口
├── artifacts/  # 审计、统计量、调度状态和代表性烟测等辅助产物
├── configs/  # 正式训练与 A0–A5 架构消融的声明式配置
├── deterministic_iafno/  # 确定性残差基线与冻结均值的 centered diffusion 扩展
├── diafno/  # 日尺度 SST 的数据、模型、训练、验证和推理核心包
├── docs/  # 仍有参考价值的实现说明、分析报告和项目导航
├── experiments/  # 训练权重、烟测、消融及验证测试结果，原路径保留（服务器产物）
├── scripts/  # 训练运行、烟测、验证监测、消融和结果整理工具
├── .gitignore  # 排除缓存、原始数据、权重和运行输出等非源码文件
├── evaluate_ostia.py  # 读取已保存的预测结果并汇总离线评估指标
├── infer_ostia.py  # 启动 SST 批量推理并保存预测结果
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── smoke_ostia.py  # 选择可用 GPU，执行短训练烟测以检查训练管线
├── trainer_ostia.py  # 启动 OSTIA 训练并解析单卡或分布式运行参数
├── validate_ostia.py  # 加载 checkpoint，在验证或测试划分上推理并计算指标
├── validation_absolute_churn0_ens8.json  # validation_absolute_churn0_ens8.json 对应协议或 checkpoint 的验证评分（服务器产物）
├── validation_metrics_epoch21.json  # validation_metrics_epoch21.json 对应协议或 checkpoint 的验证评分（服务器产物）
├── validation_persistence.json  # validation_persistence.json 对应协议或 checkpoint 的验证评分（服务器产物）
├── validation_probe_abs_sigma002.json  # validation_probe_abs_sigma002.json 对应协议或 checkpoint 的验证评分（服务器产物）
├── validation_probe_abs_sigma005.json  # validation_probe_abs_sigma005.json 对应协议或 checkpoint 的验证评分（服务器产物）
├── validation_probe_abs_sigma03.json  # validation_probe_abs_sigma03.json 对应协议或 checkpoint 的验证评分（服务器产物）
└── validation_residual_smoke.json  # validation_residual_smoke.json 对应协议或 checkpoint 的验证评分（服务器产物）
```

详细保留策略与 worktree 区别见 [项目导航](docs/PROJECT_LAYOUT.md)，历史计划统一见 [计划归档](archive/plans/README.md)。

## 原有使用说明（保留）

# DiAFNO for daily OSTIA SST forecasting

This repository adapts DiAFNO to predict 15 consecutive days of OSTIA sea-surface temperature from the previous 7 consecutive days.

## Task

- Condition: 7 normalized daily SST fields and the latest valid-ocean mask.
- Target: the following 15 normalized daily SST fields.
- Tensor layout: `[batch, channel, 448, 448, 1]`.
- Loss: EDM-weighted MSE over valid target ocean pixels only.
- Split: chronological 70% train, 20% validation and 10% test.

Invalid SST values are filled with the training mean before standardization, so they become zero in model space. Latitude and longitude remain in the HDF5 source but are not model inputs.

## Layout

```text
diafno/
  models/       IAFNO backbone and diffusion process
  data/         OSTIA daily HDF5 dataset
  training/     DDP runtime, sampler, checkpoints and plots
  inference/    checkpoint loading, sampling and output writing
  evaluation/   masked SST metrics by forecast day
trainer_ostia.py
infer_ostia.py
evaluate_ostia.py
validate_ostia.py
smoke_ostia.py
artifacts/smoke/ one retained smoke-test result
```

Runtime outputs, checkpoints, local environments and caches are excluded from Git.

## Training

Two GPUs with global batch 32:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4
```

Resume the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py --resume --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4
```

Four GPUs with the same global batch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --standalone --nproc_per_node=4 trainer_ostia.py --resume --batch-per-gpu 8 --gradient-accumulation 1 --num-workers 2
```

The default output directory is `experiments/ostia_7day_to15day`. Checkpoints produced during the short-lived monthly naming phase remain compatible and are migrated to daily metadata when loaded. Older weekly-stride checkpoints are intentionally rejected.

## Smoke test

```bash
python -u smoke_ostia.py
```

The script selects two idle GPUs and writes disposable output to `experiments/ostia_daily_smoke`.

## Inference and evaluation

Estimate validation metrics directly from a checkpoint without saving
intermediate predictions. By default, 200 validation samples are selected
uniformly and reproducibly from the full validation split:

```bash
CUDA_VISIBLE_DEVICES=2 python -u validate_ostia.py --checkpoint experiments/ostia_7day_to15day/latest.pth --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 --output-path validation_metrics.json --device cuda:0
```

Use `--max-samples N` to change the sample count or `--all-samples` to evaluate
the complete validation split. With `CUDA_VISIBLE_DEVICES=2`, the selected
physical GPU is exposed to the process as `cuda:0`.

To save predictions before evaluating them:

```bash
python -u infer_ostia.py --checkpoint experiments/ostia_7day_to15day/latest.pth --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 --output-dir inference_results
```

```bash
python -u evaluate_ostia.py --prediction-dir inference_results --output-path evaluation_metrics.json
```

Evaluation reports MAE, RMSE, bias and correlation overall and separately for forecast Day +1 through Day +15.

## Dependencies

Python 3.10 with PyTorch, h5py, NumPy, einops, timm, tqdm and Matplotlib is required. Multi-GPU training uses NCCL through `torch.distributed.run`.

## Upstream model

The model is adapted from [Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence](https://arxiv.org/abs/2512.12628) by Yuchi Jiang, Yunpeng Wang, Huiyu Yang and Jianchun Wang.
