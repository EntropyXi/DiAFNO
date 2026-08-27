# Smoke test result

This directory keeps one representative two-GPU smoke-test result:

- `training_loss_curve.png`
- `gradient_norm_curve.png`
- `training_curves.npz`

The curves are retained as a historical sanity check. Old smoke checkpoints were removed because their temporal metadata predates the finalized daily OSTIA task and they are not valid resume points.
