# DiAFNO for daily OSTIA SST forecasting

This repository adapts DiAFNO to predict 15 consecutive days of OSTIA sea-surface temperature from the previous 7 consecutive days.

## Condition modes and channel contract

Two condition modes exist; both are constructed by the single
`OSTIADailyDataset` builder shared by training, validation and inference:

| mode | channels | layout |
|---|---:|---|
| `sst_mask` (legacy default) | 8 | 7 normalized SST days + `valid_mask_t0` |
| `sst_mask_geo_season` | 14 | the 8 channels + `sin_lat`, `cos_lat`, `sin_lon`, `cos_lon`, `sin_doy`, `cos_doy` |

The channel order is fixed and schema-versioned (`condition_schema_version`
1 for the legacy layout, 2 for the geo-season layout) and is persisted in
every checkpoint and `.semantics.json` sidecar together with `calendar_encoding`,
`time_units_reference` and the lat/lon summary.  Model channel counts are
derived from the condition schema, never hand-trusted: an 8-vs-14 channel
mismatch, a stale `cond_chans` or a hand-edited channel list fails before any
model or optimizer state is loaded, and a new-layout checkpoint can never be
resumed or weight-initialized from an old one.  The semantic-manifest schema
version is 5; version-4 sidecars from older code still validate because the
immutable comparison is restricted to the fields the old file actually stores
(resume never silently adopts newer defaults).

Geo-season details (fail closed):

- latitudes/longitudes come from the HDF5 `lat`/`lon` datasets in one
  of two proven layouts: full-grid 1-D `lat[img_h]`/`lon[img_w]`
  (synthetic/tests), or the real server layout of per-row patch grids
  `[num_rows, img_h, img_w]` where each of the `samples_per_day`
  spatial patches carries its own grid (latitudes constant along the
  width, longitudes along the height) and every day repeats the day-0
  patches exactly.  Only the first day's patches are cached, as
  compact per-patch axes; the middle and last days are sampled row by
  row to prove the repeat, and the full 1.1M-row coordinate datasets
  are never loaded.  Non-finite values, violated constancy or
  non-static repeats fail closed.
- latitudes are strictly bounded ([-90, 90] degrees / [-pi/2, pi/2]
  radians).  Longitudes accept both common source conventions --
  [-180, 180] and [0, 360] degrees ([-pi, pi] and [0, 2*pi] radians);
  the real server grid uses 0..360 (first lon patch 80.025..199.975),
  and the detected convention is recorded as
  `geospatial_summary.longitude_convention`.  Values outside both
  conventions (or a mix of them) fail closed, and the sin/cos
  encoding is 2*pi-periodic, so the two conventions are equivalent at
  the dateline and at 0/360.
- the seasonal phase is the day-of-year of the **last input day (t0)**
  using real Gregorian year lengths (365/366).  The real HDF5 carries
  no calendar metadata at all, so geo-season runs need the read-only
  upstream **data manifest** (`scripts/audit_ostia_h5.py
  --source-netcdf <upstream.nc> --manifest-out <file>`, built from
  the NetCDF that proves the 11261 true day offsets including the two
  31-day gaps).  Training/validation/inference/resume all take
  `--data-manifest`; the dataset validates the manifest against the
  HDF5 (day count, samples-per-day, compact time-axis sha256) and
  only exposes 22-day windows whose true day offsets are consecutive,
  so no forecast window ever crosses a gap and the season never
  drifts by the missing days.  Without a manifest or HDF5 metadata
  the geo-season mode refuses to run (integer time is never a Unix
  epoch guess).  Time decoding and the manifest identity are recorded
  in every checkpoint and re-verified on resume/validation/inference.
- every geo-season dataset records a deterministic coordinate
  fingerprint in `geospatial_summary` (lat/lon min/max plus a SHA-256 of
  the canonical little-endian float64 bytes, spec
  `sha256_le_f8_raw_order`, or `sha256_le_f8_per_patch_axes` for the
  per-spatial-patch layout), so a checkpoint trained on one grid can
  never silently validate or resume against another grid of the same
  shape.  Training data setup compares restored provenance against the
  current HDF5 on resume and writes provenance only on fresh runs.

Validation and inference restore the condition contract from the checkpoint
instead of trusting a CLI default; an explicit conflicting `--condition-mode`
is a launch error.

## Spatiotemporal architecture ablation (A0..A5)

Server-side small-scale ablation with fixed seed, chronological split,
effective batch 32, per-mode train-only lead statistics, deterministic
residual IAFNO and a fixed validation protocol that never uses the test
split.  Layout per the plan:

```text
experiments/ostia_spatiotemporal_ablation/
  A0_baseline_p8_b8_i2/
  A1_geo_p8_b8_i2/
  A2_geo_p4_b8_i2/
  A3_geo_p4_b2_i2/
  A4_geo_p4_b1_i2/
  A5_geo_p4_best_i4/
  manifests/
  summary/
```

Before any stage, audit the real HDF5 read-only (server pre-check,
plan section 7), build the geo-season data manifest from the upstream
NetCDF, and probe the real geometry:

```bash
python scripts/audit_ostia_h5.py \
  --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 \
  --out experiments/ostia_spatiotemporal_ablation/manifests/ostia_audit.json \
  --source-netcdf /data/sst_data/sst_missing_value_imputation/copernicus_data/copernicus_sst_monthly_1991_2021.nc \
  --manifest-out experiments/ostia_spatiotemporal_ablation/manifests/ostia_data_manifest.json

python scripts/probe_ablation_vram.py \
  --config configs/ostia_ablation_A2_geo_p4_b8_i2.json \
  --out experiments/ostia_spatiotemporal_ablation/manifests/vram_A2.json \
  --batch-sizes 1,2,4,8
```

Then run the fixed stages (GPU host, repository root).  Stage 1 runs
5 epochs x 10 optimizer steps (checkpoint at epoch 5, global_step 50)
and then resumes with `num_epochs=6` at the same 10 steps/epoch, so
the restored scheduler (last epoch 50, horizon 60) runs exactly one
more epoch and global_step continues 50 -> 60 before the fixed val-16
evaluation of both checkpoints:

```bash
python scripts/run_ablation_stages.py --config-id A0 --stage 1 \
  --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5
# Geo-season configs (A1..A5) additionally require the manifest:
python scripts/run_ablation_stages.py --config-id A1 --stage 1 \
  --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 \
  --data-manifest experiments/ostia_spatiotemporal_ablation/manifests/ostia_data_manifest.json
# stage 2 (300 steps, val-200) and stage 3 (1500 steps, evals at
# 500/1000/1500 on the same val-200 protocol) use --stage 2 / --stage 3.
```

Runners are non-destructive: they refuse to start into an existing
non-empty stage directory, never delete or overwrite previous results and
fail closed without a CUDA GPU.  A5 must never silently run with the
interim `num_blocks=2`: pass `--a5-winner-blocks {1|2}` after the
Stage-2 decision between A3 (`num_blocks=2`) and A4 (`num_blocks=1`),
and re-probe the winner geometry (`scripts/probe_ablation_vram.py
--blocks N`) first.  The old `experiments/ostia_7day_to15day*`
directories are never touched, and a geo-season checkpoint cannot
resume an 8-channel run (the semantic manifest rejects the
`cond_chans`/condition contract mismatch before weights load).
The training CLI also exposes `--patch-size H W Z`, `--num-blocks` and
`--implicit-layer` directly (values on the CLI win over the config
JSON); `cond_chans` is intentionally never a flag because it derives
from the condition mode.

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

## Tests

The existing regression suite plus the new condition/checkpoint/runner
tests are plain `unittest` (no extra runner needed):

```bash
python -m unittest discover -s tests -p "test_*.py" -t .
python -m unittest discover -s deterministic_iafno/tests -p "test_*.py" -t .
```

## Dependencies

Python 3.10 with PyTorch, h5py, NumPy, einops, timm, tqdm and Matplotlib is required. Multi-GPU training uses NCCL through `torch.distributed.run`.

## Upstream model

The model is adapted from [Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence](https://arxiv.org/abs/2512.12628) by Yuchi Jiang, Yunpeng Wang, Huiyu Yang and Jianchun Wang.
