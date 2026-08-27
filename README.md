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

```bash
python -u infer_ostia.py --checkpoint experiments/ostia_7day_to15day/latest.pth --h5-path /data/exam_preprocessed_data/zzx/ocean_temperature_data_patched.h5 --output-dir inference_results
```

```bash
python -u evaluate_ostia.py --prediction-dir inference_results --output-path evaluation_metrics.json
```

Evaluation reports MAE, RMSE, bias and correlation overall and separately for forecast Day +1 through Day +15.

## Dependencies

Python 3.10 with PyTorch, h5py, NumPy, einops, timm, tqdm and Matplotlib is required. Multi-GPU training uses NCCL through `torch.distributed.run`.

## Upstream model

The model is adapted from [Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence](https://arxiv.org/abs/2512.12628) by Yuchi Jiang, Yunpeng Wang, Huiyu Yang and Jianchun Wang.
