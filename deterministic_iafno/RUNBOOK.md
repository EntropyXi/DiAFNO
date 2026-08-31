# RUNBOOK — deterministic IAFNO Phase 0/1 (strict execution order)

This runbook fixes the order of the Phase 0/1 experiments and their
decision gates.  It is written for the training server (read/write is
allowed there when the user executes it); the local repository changes
are complete and reviewed before any of these steps run.

Conventions:

- H5: `/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5`
- Repo: `/data2/user/zzx/exam_preprocessed/DiAFNO`
- Env: `/data2/user/zzx/ENTER/envs/DiAFNO/bin/python`
- Fixed validation indices: 200 samples, seed 123, split=val.
  The test split (1000, seed 123) is **never** used for selection.

---

## Step 0 — Freeze legacy evaluation & resume semantics

1. Re-run the legacy evaluator on the locked checkpoints with the
   *unchanged* protocol and record the numbers as the golden baseline
   (do this before any evaluator change was merged; the JSONs
   `test_best_epoch004_1000_b16.json` / `test_latest_epoch035_1000_b16.json`
   / `test_persistence_1000_b16.json` from the audit are the reference):

   ```
   cd $REPO
   CUDA_VISIBLE_DEVICES=2 $PY validate_ostia.py \
     --checkpoint experiments/ostia_7day_to15day_residual_scratch/best_model.pth \
     --h5-path $H5 --sampling-steps 16 --s-churn 0 --ensemble-members 4 \
     --max-samples 200 --output-path /tmp/golden_best_200.json
   ```

2. Verify the local unit tests still pin the legacy evaluator contract
   (`tests/test_legacy_metrics_golden.py`) and the legacy diffusion
   masked-loss formula (`tests/test_legacy_diffusion_loss.py`).

3. Resume drift regression (local, no GPU): a resume whose CLI/defaults
   differ from the checkpoint immutable semantics must **fail closed**
   (`tests/test_resume_restore.py`), and a bare resume must restore the
   checkpoint semantics from the sidecar.  On the server, the epoch23
   scenario is the acceptance test: resuming `epoch_023.pth` with a bare
   `--resume` must either restore (sidecar) or error — never silently
   adopt `sigma_max=80`.

   Go/No-Go: all unit tests pass; golden JSON numbers unchanged.

## Step 1 — Baselines & condition ablations (no new model)

Run on the fixed 200 val indices (batch 16, seed 123), same H5:

```
$PY validate_ostia.py --checkpoint <best> --h5-path $H5 \
  --prediction-mode persistence --max-samples 200 --output-path /tmp/b_pers.json
$PY validate_ostia.py --checkpoint <best> --h5-path $H5 \
  --prediction-mode linear_trend --max-samples 200 --output-path /tmp/b_trend.json
# legacy diffusion condition ablations (only for the diffusion checkpoint):
for MODE in none zero_sst anchor_only reverse_history shuffle_history; do
  $PY validate_ostia.py --checkpoint <best> --h5-path $H5 \
    --condition-ablation $MODE --sampling-steps 16 --s-churn 0 \
    --ensemble-members 4 --max-samples 200 \
    --output-path /tmp/b_abl_$MODE.json
done
```

Notes: `shuffle_history` requires `--batch-size >= 2`.  The persistence
baseline is always raw residual = 0 (the re-anchored day 7); it is never
confused with a standardized zero.  Record: per-lead and overall
absolute + residual RMSE/MAE/bias/corr/std-ratio, and MSE-based skill
vs persistence (paired on the same indices).

## Step 2 — Lead stats (train-only)

```
$PY -m deterministic_iafno.compute_lead_stats --h5-path $H5 \
  --num-samples 4096 --batch-size 32 \
  --output /tmp/lead_stats_train.json
```

Verify the JSON declares `split=train`, `target_space=normalized_residual`,
15 leads, positive stds.  The CLI (`--lead-stats`) validates length /
positivity / target space / split / day counts and refuses anything else.

## Step 3 — Deterministic raw arm (A)

```
cd $REPO
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 nohup \
$PY -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py \
  --output-dir experiments/det_raw \
  --model-type deterministic --target-scaling raw \
  --target-mode residual --learning-rate 2e-4 \
  --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4 \
  --num-epochs 15 --checkpoint-interval 5 \
  > /tmp/ostia_ft_logs/det_raw.log 2>&1 &
```

Evaluate the saved epoch-5/10/15 checkpoints on the same fixed 200 val
indices.  The current evaluator reports paired-set aggregate skill but
does not yet emit per-initialization errors for a block bootstrap; do
not claim confidence intervals until that analysis output is added.
Never touch the test split.

## Step 4 — Deterministic lead-standardized arm (B)

Same as Step 3 with `--target-scaling lead_standardized
--lead-stats /tmp/lead_stats_train.json`.

## Step 5 — Fixed validation gate (decision)

Compare the best arm's validation skill vs persistence, both overall
and day 1.  Positive point skill is necessary; final statistical claims
also require a paired block bootstrap by initialization date, which is
explicitly still pending in this Phase 0/1 implementation.

- If the best deterministic arm does **not** beat persistence on
  validation (non-positive point skill, or a later paired interval that
  crosses zero):
  **stop**.  No diffusion tuning, no hybrid.  Report back to the
  data/backbone diagnosis.
- If it beats persistence: freeze the validation protocol, then run the
  test split (1000, seed 123) exactly once for the final report.

## Step 6 — After the gate (Phase 2+, out of current scope)

Frozen-mean centered diffusion, per the reviewed plan, only after the
gate passes.  Not implemented yet.

---

## Hard rules

1. No model selection on the test split; the 200-sample validation set
   is frozen before any test run.
2. No persistence-vs-model comparison across different index sets; all
   paired metrics are computed in the same validation pass.
3. No silent resume: immutable semantics come from the checkpoint
   sidecar or fail closed; sampler profile changes are warnings and the
   resulting curves are not compared with the old protocol.
4. No new lead statistics from validation/test data.
