# Deterministic IAFNO work area

This directory isolates the small-change investigation for the OSTIA
7-day-to-15-day task.

Execution order is intentionally fixed:

1. Freeze and verify the legacy evaluator contract.
2. Add fail-fast checkpoint/resume semantics.
3. Add persistence/trend and clean condition ablations.
4. Add the raw-backbone deterministic IAFNO path.
5. Compare raw-residual MSE with train-only lead-standardized MSE.
6. Stop if the deterministic model does not beat persistence on validation.
7. Only after that gate, consider frozen-mean centered diffusion.

The current implementation phase covers items 1-5 plus the Phase 2
frozen-mean centered diffusion:

- `centered_diffusion.py` — `FrozenMeanCenteredDiffusion` wrapper
  (frozen eval mean, fp32 centered innovation, single reconstruction,
  two-way sampler attribute delegation);
- `centered_stats.py` / `compute_centered_stats.py` — train-only
  centered innovation statistics with fail-closed provenance
  validation;
- `configs/ostia_centered_diffusion_main.json` — the authoritative main
  training config (7-day → 15-day, sigma_data=1.0);
- `scripts/archive_legacy_ostia_main.sh` (default dry-run),
  `init_ostia_centered_main.sh`, `run_ostia_centered_main.sh`,
  `smoke_ostia_centered.sh`,
  `watch_ostia_centered.sh`.

See `PHASE2_CENTERED_RUNBOOK.md` for the locked algebra and the
reproducible server command sequence.  Server-side archiving, GPU
smoke and the main training launch are Codex-only steps after review.
