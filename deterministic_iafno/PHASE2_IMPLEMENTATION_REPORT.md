# Phase 2 centered diffusion — local implementation report

Plan: `PHASE2_MAIN_TRAINING_PLAN.md` (final PASS revision), SHA-256
`1d89f4c76fb93a6fc9c8e1e02939b625ff0278249c2368501f0cabc392e75d19`
(verified against the working tree before implementation).

Status: **local implementation complete — stopped for Codex review.**
No commit, no push, no archive execution, no GPU training started.

Authoritative task: 7 consecutive days of SST input → the following
15 days of SST (daily scale; weekly/monthly material is obsolete).

## Files changed (all intentional)

Stage 1 — centered wrapper + config/checkpoint semantics:
- `deterministic_iafno/centered_diffusion.py` (new): `FrozenMeanCenteredDiffusion`
- `diafno/models/config.py`: `centered_diffusion` model type, 4 new
  identity fields, fail-closed build rules (sigma_data=1.0, stats
  required)
- `deterministic_iafno/checkpoint_semantics.py`: schema 3 → 4, 4 new
  immutable fields, list→tuple normalization on restore
- `diafno/training/config.py`: `--mean-checkpoint`, `--centered-stats`,
  `--config` JSON, centered CLI rules, centered target-space branch in
  `validate_lead_stats_dict`
- `diafno/training/main.py`: `--config` merge into args
- `deterministic_iafno/tests/test_centered_diffusion.py` (new, 17 tests)

Stage 2 — train-only centered stats + provenance validator:
- `deterministic_iafno/centered_stats.py` (new): locked mean SHA,
  payload validator, mean sidecar cross-check, per-rank fresh-input
  validation
- `deterministic_iafno/compute_centered_stats.py` (new): chunk-aware
  deterministic indices, float64 accumulators, byte-deterministic JSON
- `deterministic_iafno/tests/test_centered_stats.py` (new, 14 tests;
  synthetic HDF5 + synthetic frozen-mean end-to-end)

Stage 3 — trainer integration:
- `diafno/training/trainer.py`: per-rank fresh validation, frozen mean
  weight load, optimizer over trainable params only, AMP overflow guard
  (optimizer + scheduler skip), mean-grad assertion after first step,
  skip counters in checkpoint/log/history
- `diafno/training/artifacts.py`: skip counters in checkpoint and
  `training_curves.npz`
- `deterministic_iafno/tests/test_trainer_amp_skip.py` (new, 7 tests)

Stage 4 — inference/evaluator self-contained rebuild:
- no evaluator/loader code change required (centered flows through the
  residual diffusion branch: sample → single re-anchor); contracts
  locked by new tests
- `deterministic_iafno/tests/test_evaluator_centered.py` (new, 4 tests)
- `deterministic_iafno/tests/test_centered_checkpoint_roundtrip.py`
  (new, 6 tests)
- `deterministic_iafno/tests/test_centered_config_cli.py` (new, 9 tests)

Stage 5 — tiny CPU smoke:
- `deterministic_iafno/tests/test_centered_cpu_smoke.py` (new, 2 tests;
  real trainer loop + AMP guard + schema-4 checkpoint/sidecar)

Stage 6 — config JSON + ops scripts:
- `configs/ostia_centered_diffusion_main.json` (new; single
  authoritative main config)
- `scripts/archive_legacy_ostia_main.sh` (new; **default dry-run**,
  `--execute` is Codex-only; fail-closed checks per plan 5.2/5.3,
  rollback per plan 5.5)
- `scripts/init_ostia_centered_main.sh` (new; **default dry-run**;
  verified staging directory followed by atomic canonical config move)
- `scripts/run_ostia_centered_main.sh` (new; preflight + hash printing
  + launch manifest; 1/2-GPU effective batch 32)
- `scripts/smoke_ostia_centered.sh` (new; independent per-run directory;
  phase 1 = 20 steps + resume to step 40 + finite 4-step sampling,
  phase 2 = 1000 steps; post-run skip==0 verification)
- `scripts/watch_ostia_centered.sh` (new; read-only val-200 protocol,
  epoch-5 ablation then human lock, never test)

Stage 7 — docs:
- `deterministic_iafno/PHASE2_CENTERED_RUNBOOK.md` (new)
- `deterministic_iafno/README.md` (updated)

Untouched: pre-existing dirty `deterministic_iafno/STATUS.md`,
untracked `tmp_*.py`, `docs/scratch_*`, migration/diagnostic scripts.

## Verification battery (all run in this workspace, Git Bash, DL env)

| check | result |
|---|---|
| `unittest discover -s deterministic_iafno/tests` | **110/110 OK** (51 legacy + 59 new) |
| `compileall -q diafno deterministic_iafno` | OK |
| imports (models/training/inference/evaluation/data/centered) | OK |
| config JSON schema + sigma_data=1.0 + effective batch 32 (1/2 GPU) | OK |
| tiny CPU smoke (real trainer loop, schema-4 sidecar) | 2/2 OK |
| archive dry-run | fail-closed (exit 1, zero changes) |
| launcher preflight | fail-closed at missing frozen mean, no launch |
| `git diff --check` | clean |
| `bash -n` × 5 scripts | OK |

## Requirement-to-evidence matrix

| Requirement (plan §14 / gate) | Evidence |
|---|---|
| 7→15 day task not drifted | dataset untouched; stats payload validation locks `input_days=7/output_days=15`; trainer `_training_target` slices `input_days`; config JSON has no time-scale fields |
| frozen mean identity fixed | `LOCKED_MEAN_CHECKPOINT_SHA256` in `centered_stats.py`; file SHA checked per rank in `validate_centered_fresh_inputs`; launcher re-checks before launch; `test_mean_sha_mismatch_fails_closed` |
| centered target algebra | `test_forward_computes_standardized_innovation` (exact `z=(e-m)/s`); fp32 under AMP (`test_standardization_stays_fp32_under_amp`); end-to-end numpy recomputation (`test_end_to_end_stats_payload_and_determinism`) |
| anchor/mean added exactly once | wrapper returns `mu+m+s*z_hat` without anchor (`test_sample_returns_normalized_residual_no_anchor`); evaluator adds anchor once (`test_centered_reanchored_exactly_once`, ensemble variant); zero innovation degenerates to mean (`test_zero_innovation_degenerates_to_deterministic_mean`) |
| mean truly frozen | `CenteredFreezeTests`; `test_train_mode_keeps_mean_eval`; optimizer param exclusion (`test_optimizer_params_exclude_mean`); trainer `_assert_mean_frozen_grads` after first step; CPU smoke asserts mean grads None after 4 real steps |
| stats train-only, no leakage | validator rejects `split!=train`, wrong target space, wrong lead counts, non-finite/non-positive stds, val/test keys, missing provenance hashes (`CenteredStatsValidatorTests`); tool declares split/target_space/selection/indices_sha256 and self-validates; deterministic chunk-aware indices locked by test |
| checkpoint self-contained | `test_inference_rebuilds_without_mean_path` (no mean file exists anywhere); state dict contains `mean_model.*` (`test_state_dict_contains_frozen_mean`); schema-4 sidecar carries mean/innovation stats + both hashes (`test_schema4_sidecar_carries_centered_semantics`) |
| legacy paths unbroken | all 51 legacy tests (diffusion loss golden, metrics golden, deterministic backbone, resume restore, checkpoint semantics/roundtrip, paired bootstrap, evaluator contracts) pass unchanged |
| resume semantics safe | bare resume restores all centered semantics (`test_bare_resume_restores_centered_semantics`); explicit conflict fails closed (`test_explicit_centered_conflict_fails_closed`); schema-3 manifest still validates (`test_schema3_legacy_manifest_still_validates`); skip counters restored in `_resume_training` |
| fresh-run per-rank fail-closed | `validate_centered_fresh_inputs` tests: positive path, missing sidecar, architecture mismatch, SHA mismatch; trainer runs it on every rank before any data read |
| sampler attribute two-way delegation | `test_sampler_attributes_two_way_delegation` (S_churn/sigma_min/sigma_max/rho/num_sample_steps); evaluator `--s-churn` reaches the wrapper property |
| AMP overflow skip semantics | `test_overflow_skips_optimizer_and_scheduler`, `test_clean_step_advances_optimizer_and_scheduler`, `test_global_step_tracks_updates_even_when_skipped`; skip counters in checkpoint + history (`test_checkpoint_contains_mean_weights_and_skip_counters`); smoke script post-check enforces `skipped_optimizer_steps==0` |
| sigma_data=1.0 locked | build_model raises on ≠1.0; CLI rejects the 0.15 factory default (`test_factory_sigma_data_rejected_for_centered`, `test_nonunit_sigma_data_rejected_for_centered`); config JSON asserts it; launcher preflights it |
| archive/init scripts safe | archive default dry-run; execute requires locked commit; fail-closed: source exists/not-symlink, target absent, realpath/symlink-component containment, active writers, free space; post-move SHA verification + rollback. Init uses a verified staging directory and atomic move; never overwrites canonical config |
| smoke uses independent dir | each `scripts/smoke_ostia_centered.sh` invocation writes a unique phase/run-id directory under `experiments/ostia_centered_smoke_scratch`; canonical/archive never referenced for output |
| launcher reads only the frozen JSON | prints config JSON SHA, mean SHA, stats SHA; structural stats validation; launch manifest; no invented values |
| 2-GPU / 1-GPU effective batch 32 | launcher arithmetic 8×2×2=32 and 8×4×1=32 with hard check; config schema test asserts both |
| watcher never selects on test | `--split val --max-samples 200 --seed 123` only; epoch-5 ablation then human lock of `watcher_config.json`; no test path anywhere in the script |
| server-only steps (archive execute, GPU smoke, main launch) | intentionally NOT run locally; delegated to Codex with exact commands in `PHASE2_CENTERED_RUNBOOK.md` |

## Known risks (for Codex review)

1. The frozen mean sidecar on the server is expected to carry the
   architecture fields in its immutable block (Phase-1 checkpoint saved
   by the current schema code).  If it does not, the per-rank
   validation fails closed — correct behavior, but worth confirming
   during server pull before the stats run.
2. Overflow detection uses post-`unscale_` gradient finiteness, which
   is equivalent to the scaler's internal inf check and works on CPU
   and CUDA; the plan's wording ("compare scale/inf states around
   scaler.step") is satisfied semantically, not via the private API.
3. `merge_config_json` treats config-file values as explicit for
   resume-conflict purposes; a bare `--resume --config <different>`
   fails closed instead of silently overriding.  Launchers only use
   `--config` for fresh starts.
4. The DDP path (`find_unused_parameters=False`, frozen mean untracked)
   is correct by construction but not exercised locally (no local GPU);
   the server 20-step smoke is its gate, per the plan.
