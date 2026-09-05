# Status

## Phase 0 / Phase 1 implementation status (local, reviewed)

- [x] Phase 0a: legacy evaluator golden compatibility (synthetic golden)
- [x] Phase 0b: checkpoint/resume semantic manifest + sidecar
- [x] Phase 0c: persistence/trend and condition ablations
- [x] Phase 1a: deterministic raw-residual training path
- [x] Phase 1b: train-only per-lead statistics script
- [x] Phase 1c: lead-standardized deterministic training path
- [ ] Phase 1d: fixed-validation comparison against persistence (server)
- [ ] Gate decision: stop or proceed to centered diffusion (server)

## Audit fixes applied (this round)

1. **Legacy diffusion loss semantics pinned** — `diafno/models/diffusion.py`
   reverted to HEAD: the draft's global masked normalization was removed
   because it changed the legacy per-sample mask mean contract.  The
   original formula is now locked by
   `tests/test_legacy_diffusion_loss.py` (reference replication with the
   same RNG stream).  The DDP-correct global normalization lives only in
   the deterministic path (`deterministic_iafno/losses.py`).
2. **Resume semantics restored from the checkpoint, not the CLI** —
   new checkpoints write a per-checkpoint sidecar
   `<checkpoint>.semantics.json`; the trainer restores immutable
   model/data/training-noise semantics from the sidecar before building
   the model, so a bare `--resume latest` continues with exactly the
   checkpoint semantics (epoch23-style drift now impossible without an
   explicit conflict error).  Legacy checkpoints (no sidecar) fall back
   to fail-fast validation of the fields they actually store; no double
   load of the large checkpoint is needed.
3. **Compatible-field grading** — optimizer/schedule/effective-batch,
   epoch budget and exposure changes require
   `--allow-resume-override`.  Reviewed overrides are applied after
   state restoration, so optimizer/scheduler objects and the next
   manifest cannot silently disagree.
4. **Lead-stats CLI validation** — `validate_lead_stats_dict` enforces
   train split, `target_space=normalized_residual`, length ==
   target_chans, positive stds, and day-count consistency; the stats
   script records selection method, sample count, normalization and
   all 15 leads.
5. **Evaluator** — absolute + residual overall/per-lead metrics,
   MSE-based persistence skill, std ratio, `anchor_only` /
   `reverse_history` / `shuffle_history` ablations (all keep the
   original day-7 anchor), linear-trend baseline, near-identity-only
   tiny-sigma probe, deterministic prediction path with
   `ensemble-members 1` enforcement.

## Verification

- 46 unit tests pass in the Conda DL environment
  (`python -m unittest discover -s deterministic_iafno/tests -t .`).
- `python -m compileall diafno deterministic_iafno` clean; imports OK.
- Tiny-size CPU smoke through `OSTIAModelConfig.build_model` for both
  `diffusion` and `deterministic` model types (forward/backward/predict).
- Codex review added `P_std` to immutable training-noise semantics,
  `rho` to the sampler profile, finite lead-stat checks, invalid
  diffusion/lead-standardized CLI rejection, and a real reviewed
  optimizer/scheduler override round-trip test.
- Resume now tracks which model/sampler fields were explicitly passed
  by the CLI.  A value equal to the factory default is therefore no
  longer mistaken for an omitted argument.

## Diffusion sampling-step check (server, Aug 31, gate pre-evidence)

Same frozen protocol (200 val samples, seed 123, absolute-space RMSE),
diffusion `ostia_7day_to15day_residual_scratch/best_model.pth`, only the
sampler step count changed:

| model / config | overall RMSE | day1 RMSE |
|---|---|---|
| diffusion, 16 steps (s_churn=0, ens=4) | 1.3702 | 0.6769 |
| diffusion, 100 steps (s_churn=0, ens=4) | 1.3640 | 0.6660 |
| det_raw epoch 15 (deterministic) | **1.1766** | **0.2471** |
| persistence baseline | 1.1871 | 0.2484 |

Evidence: `experiments/deterministic_iafno/val_diffusion_best_100step_200.json`
(on the training server).

Conclusion: 16 → 100 steps improves the diffusion arm by only -0.45%
(1.3702 → 1.3640).  Undersampling is **not** the cause of the diffusion
arm's deficit; the model itself does not learn an adequate mean
prediction.  This strengthens the case for the frozen-mean centered
diffusion (Phase 2): the diffusion distribution is stable across step
counts, only its center is wrong — the deterministic arm provides the
center, diffusion models the perturbation.

## Not yet done (server, real data)

- Real HDF5 lead stats (`deterministic_iafno/compute_lead_stats.py`).
- Deterministic raw / lead-standardized training runs (5/10/15 epoch
  trend checks) and the fixed-validation gate — see RUNBOOK.md.
- Per-initialization outputs and paired block-bootstrap confidence
  intervals (the aggregate evaluator must not be described as a CI).
- No model has been trained or evaluated on real data in this phase.
