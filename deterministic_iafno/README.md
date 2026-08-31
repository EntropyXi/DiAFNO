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

The current implementation phase covers items 1-5 only. Centered diffusion,
new condition encoders, cross-attention, joint training, calendar features,
and broad hyperparameter searches are out of scope until the validation gate
is passed.
