# Phase 2 centered diffusion — implementation notes and reproducible commands

Status: local implementation complete, awaiting Codex review (no commit, no push).

Authoritative task (unchanged): **7 consecutive days of SST input → the
following 15 days of SST.**  Any weekly/monthly material is obsolete context.

## Target algebra (locked)

All tensors are normalized SST `x = (SST - sst_mean) / sst_std` unless
stated otherwise.  For one sample:

- condition `c = [x_(t-6..t), mask_t]`, anchor `a = x_t`
- true normalized residual `r_l = x_(t+l) - a`, `l = 1..15`
- frozen deterministic mean `mu_l = F_mean(c)` (already in normalized
  residual space; its own lead stats are NOT re-applied)
- centered innovation `e_l = r_l - mu_l` (fp32)
- train-only innovation stats `m_l, s_l` → `z_l = (e_l - m_l) / s_l`
- diffusion learns `p(z | c)` with `sigma_data = 1.0`
- sampling: `z_hat → e_hat = z_hat*s + m → r_hat = mu + e_hat`
  (single reconstruction; no anchor, no SST denormalization in the wrapper)
- evaluator adds the anchor exactly once: `x_hat = a + r_hat`

Hard invariants: anchor added once (evaluator only), mean added once
(wrapper only), training target is `target - anchor - frozen_mean`
before innovation standardization, innovation `lead_mean/lead_std` are
the CENTERED stats (never Phase 1 raw residual stats), `sigma_data=1.0`.

## New files

| file | role |
|---|---|
| `deterministic_iafno/centered_diffusion.py` | `FrozenMeanCenteredDiffusion` wrapper (freeze/eval, fp32 z, single reconstruction, sampler two-way delegation) |
| `deterministic_iafno/centered_stats.py` | locked mean SHA, stats payload validator, mean sidecar cross-check, per-rank fresh-input validation |
| `deterministic_iafno/compute_centered_stats.py` | train-only centered innovation stats tool (chunk-aware deterministic indices, float64 accumulators) |
| `configs/ostia_centered_diffusion_main.json` | the single authoritative main training config |
| `scripts/archive_legacy_ostia_main.sh` | legacy archive; **default dry-run**, `--execute` is Codex-only |
| `scripts/run_ostia_centered_main.sh` | preflight + launch from the authoritative JSON (effective batch 32, 1/2 GPU) |
| `scripts/smoke_ostia_centered.sh` | GPU smoke (phase 1 = 20 steps, phase 2 = 1000 steps), independent output dir |
| `scripts/watch_ostia_centered.sh` | read-only val-200 watcher; epoch-5 ablation then human lock; never touches test |

Checkpoint schema was raised to 4 (`mean_lead_mean/mean_lead_std/
mean_checkpoint_sha256/mean_semantics_sha256` added to the immutable
block).  Schema 3 remains read-only/resume compatible.  Centered
checkpoints are self-contained: the frozen mean weights (`mean_model.*`
prefix), innovation stats, mean residual stats and both identity hashes
all live in the checkpoint + sidecar; inference and resume never need
the original mean path.

## Local verification (already run, evidence in this workspace)

```bash
PY=/d/anaconda3/envs/DL/python            # any torch env with einops/timm/h5py
$PY -m unittest discover -s deterministic_iafno/tests          # 110/110 OK
$PY -m compileall -q diafno deterministic_iafno
$PY -c "import diafno.models.config, deterministic_iafno.centered_diffusion, deterministic_iafno.compute_centered_stats; print('imports OK')"
bash -n scripts/archive_legacy_ostia_main.sh scripts/init_ostia_centered_main.sh scripts/run_ostia_centered_main.sh scripts/smoke_ostia_centered.sh scripts/watch_ostia_centered.sh
bash scripts/archive_legacy_ostia_main.sh                      # dry-run, must not modify anything
```

## Server steps (Codex only, after review passes)

1. `git pull --ff-only origin OSTIA_SST`; verify HEAD equals local HEAD;
   re-run the full test suite in the server DiAFNO env.
2. Compute centered stats into a staging directory (train-only,
   deterministic; the old canonical main still exists at this point):
   ```bash
   mkdir -p experiments/ostia_centered_preflight
   $PY -m deterministic_iafno.compute_centered_stats \
     --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 \
     --mean-checkpoint experiments/det_lead_standardized/epoch_015.pth \
     --num-samples 8192 --batch-size 32 \
     --output experiments/ostia_centered_preflight/centered_stats_train.json
   ```
   Run twice; outputs must be byte-identical.  If 8192 samples are too
   slow in the window, 4096 is allowed but the decision and timing must
   be recorded.
3. GPU smoke (independent dir, never canonical/archive):
   ```bash
   scripts/smoke_ostia_centered.sh --gpus <gpu> --phase 1 \
     --mean-checkpoint experiments/det_lead_standardized/epoch_015.pth \
     --centered-stats experiments/ostia_centered_preflight/centered_stats_train.json \
     --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5
   scripts/smoke_ostia_centered.sh --gpus <gpu> --phase 2 \
     --mean-checkpoint experiments/det_lead_standardized/epoch_015.pth \
     --centered-stats experiments/ostia_centered_preflight/centered_stats_train.json \
     --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5
   ```
   Acceptance: all losses finite, `skipped_optimizer_steps == 0`, mean
   grads None, diffusion updated, resume continuous, finite 4-step
   sample.
4. Archive the legacy main (default dry-run first, then Codex
   `--execute --expected-commit <reviewed-commit>`):
   ```bash
   scripts/archive_legacy_ostia_main.sh                        # dry-run
   scripts/archive_legacy_ostia_main.sh --execute --expected-commit <sha>
   ```
5. Initialize the canonical dir config files (default dry-run first):
   ```bash
   scripts/init_ostia_centered_main.sh \
     --mean-checkpoint experiments/det_lead_standardized/epoch_015.pth \
     --centered-stats experiments/ostia_centered_preflight/centered_stats_train.json \
     --expected-commit <sha>
   scripts/init_ostia_centered_main.sh --execute \
     --mean-checkpoint experiments/det_lead_standardized/epoch_015.pth \
     --centered-stats experiments/ostia_centered_preflight/centered_stats_train.json \
     --expected-commit <sha>
   ```
6. Launch in tmux `ostia_centered_main`:
   ```bash
   scripts/run_ostia_centered_main.sh --gpus <a,b>   # or --gpus <a> for the single-GPU fallback
   ```
   Launch manifest (`logs/launch_manifest.json`) must record: commit,
   config JSON hash, mean SHA, stats SHA, GPUs, nproc, effective
   batch 32.
7. Watcher in tmux `ostia_centered_watch`:
   ```bash
   scripts/watch_ostia_centered.sh --output-dir experiments/ostia_7day_to15day_residual_scratch \
     --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 \
     --gpu-id <free-physical-gpu-id>
   ```
   After the epoch-5 ablation it stops and asks Codex to lock the
   profile in `watcher_config.json`.  `best_val_mean_rmse.pth` naming is
   used from the start; the test split is never read by the watcher.

## Main config summary (authoritative JSON)

`model_type=centered_diffusion`, `target_mode=residual`,
`target_scaling=lead_standardized` (centered), `sigma_data=1.0`,
`sigma_min=0.002`, `sigma_max=80`, `P_mean=-1.2`, `P_std=1.2`,
`rho=7`, `sampling_steps=16`, 35 epochs × 31200 samples,
`lr=2e-4`→`1e-6`, `weight_decay=1e-4`, `max_grad_norm=1.0`, AMP on,
checkpoint every epoch, seed 123, 2 workers/rank.  Effective batch 32:
2 GPUs = 8×2×2; 1 GPU = 8×4×1.  OOM fallback keeps batch 32 (4×4×2 /
4×8×1).
