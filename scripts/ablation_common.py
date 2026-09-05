# 用途：提供消融任务的配置、命令构造和安全检查公共函数。
"""Shared helpers for the OSTIA spatiotemporal ablation runner.

Non-destructive by construction:

- every run output lives under its own directory and a stage refuses
  to start when its target directory already exists and is non-empty;
- nothing in this module ever deletes, overwrites or cleans up prior
  results (an operator must move failed runs aside and start a fresh
  directory instead);
- the ablation directory layout follows the plan:

    experiments/ostia_spatiotemporal_ablation/
      A0_baseline_p8_b8_i2/
      A1_geo_p8_b8_i2/
      A2_geo_p4_b8_i2/
      A3_geo_p4_b2_i2/
      A4_geo_p4_b1_i2/
      A5_geo_p4_best_i4/
"""

import os
import subprocess
import sys

ABLATION_ROOT = os.path.join(
    "experiments",
    "ostia_spatiotemporal_ablation",
)

# Canonical A0..A5 identity.  ``blocks`` is the AFNO frequency-block
# count (``num_blocks``), ``implicit`` the implicit-layer iterations.
# A5's ``blocks`` starts at the interim value 2 and must be updated to
# the Stage-2 winner between A3 and A4 before A5 is launched.
ABLATION_CONFIGS = {
    "A0": {
        "config": "configs/ostia_ablation_A0_baseline_p8_b8_i2.json",
        "dir": "A0_baseline_p8_b8_i2",
        "mode": "sst_mask",
        "patch_size": (8, 8, 1),
        "blocks": 8,
        "implicit": 2,
    },
    "A1": {
        "config": "configs/ostia_ablation_A1_geo_p8_b8_i2.json",
        "dir": "A1_geo_p8_b8_i2",
        "mode": "sst_mask_geo_season",
        "patch_size": (8, 8, 1),
        "blocks": 8,
        "implicit": 2,
    },
    "A2": {
        "config": "configs/ostia_ablation_A2_geo_p4_b8_i2.json",
        "dir": "A2_geo_p4_b8_i2",
        "mode": "sst_mask_geo_season",
        "patch_size": (4, 4, 1),
        "blocks": 8,
        "implicit": 2,
    },
    "A3": {
        "config": "configs/ostia_ablation_A3_geo_p4_b2_i2.json",
        "dir": "A3_geo_p4_b2_i2",
        "mode": "sst_mask_geo_season",
        "patch_size": (4, 4, 1),
        "blocks": 2,
        "implicit": 2,
    },
    "A4": {
        "config": "configs/ostia_ablation_A4_geo_p4_b1_i2.json",
        "dir": "A4_geo_p4_b1_i2",
        "mode": "sst_mask_geo_season",
        "patch_size": (4, 4, 1),
        "blocks": 1,
        "implicit": 2,
    },
    "A5": {
        "config": "configs/ostia_ablation_A5_geo_p4_best_i4.json",
        "dir": "A5_geo_p4_best_i4",
        "mode": "sst_mask_geo_season",
        "patch_size": (4, 4, 1),
        # Interim value; replace with the A3/A4 Stage-2 winner before
        # launching A5 and re-run probe/memory stage.
        "blocks": 2,
        "implicit": 4,
    },
}

# Fixed stage protocol (plan section 10).  Validation always uses the
# same split='val' subset selection (validator seed 123) and never
# touches the test split.  Every stage runs whole epochs of exactly
# ``steps_per_epoch`` optimizer steps so checkpoints land on epoch
# boundaries and a resume horizon can be expressed in epochs.
STAGE_PROTOCOL = {
    "stage1": {
        # Smoke: 5 epochs x 10 optimizer steps = 50 steps
        # (checkpoint at epoch 5), then a resume run with num_epochs=6
        # and the same 10 steps/epoch runs exactly one more epoch
        # (epochs 5->6, global_step 50 -> 60) before the fixed val-16
        # evaluation of both checkpoints.
        "steps": 50,
        "steps_per_epoch": 10,
        "resume_steps": 10,
        "val_samples": 16,
        "checkpoint_interval": 5,
    },
    "stage2": {
        "steps": 300,
        "steps_per_epoch": 300,
        "resume_steps": 0,
        "val_samples": 200,
        "checkpoint_interval": 1,
    },
    "stage3": {
        # 6 epochs x 250 optimizer steps = 1500 steps; checkpoints at
        # epochs 2/4/6 give the 500/1000/1500-step re-evaluations of
        # the same fixed val-200 protocol.
        "steps": 1500,
        "steps_per_epoch": 250,
        "resume_steps": 0,
        "val_samples": 200,
        "checkpoint_interval": 2,
        "eval_steps": (500, 1000, 1500),
    },
}


def assert_directory_available(path):
    """Refuse to start into an existing non-empty directory.

    The runner must never delete, overwrite or reuse previous results;
    a refusal tells the operator to point at a fresh directory.
    """
    if not os.path.exists(path):
        return
    if not os.path.isdir(path):
        raise RuntimeError(
            f"output path exists and is not a directory: {path}"
        )
    if os.listdir(path):
        raise RuntimeError(
            "refusing to run into an existing non-empty directory: "
            f"{path} (nothing is ever deleted or overwritten; move "
            "the previous run aside and use a fresh directory)"
        )


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the ablation runner and the VRAM probe require a CUDA "
            "GPU; they are server-only tools by design (no local "
            "long training is started)"
        )


def samples_per_epoch_for_steps(
        steps,
        gpus,
        batch_per_gpu,
        gradient_accumulation,
    ):
    """Samples per epoch that yields exactly ``steps`` optimizer
    updates: loader_batches / gradient_accumulation with
    loader_batches = samples / (gpus * batch_per_gpu)."""
    return (
        int(steps)
        * int(gpus)
        * int(batch_per_gpu)
        * int(gradient_accumulation)
    )


def git_revision(repo_root=None):
    """Best-effort (branch, commit-sha) of the checked-out code."""
    root = repo_root or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"branch": branch, "commit": sha}
    except Exception as error:  # pragma: no cover - best effort
        return {"branch": None, "commit": None, "error": str(error)}


def run_command(command, description, cwd=None, env=None):
    print(f"[ablation] {description}")
    print("[ablation] " + " ".join(str(part) for part in command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code "
            f"{completed.returncode}"
        )
    return completed


def check_json_result_finite(path):
    """Hard gate: parsed validation JSON must be finite and complete."""
    import json
    import math
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    sample_count = payload.get("num_samples")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise RuntimeError(
            f"validation result {path} has no valid num_samples"
        )
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        raise RuntimeError(
            f"validation result {path} has no overall metrics"
        )
    for key in ("rmse", "mae", "bias", "correlation", "mse"):
        value = overall.get(key)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(
                f"validation result {path} has non-finite "
                f"overall.{key}={value!r}"
            )
    return payload
