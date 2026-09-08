#!/usr/bin/env python
# 用途：A5 长训逐 epoch 验证队列：对源权重与每个已完成 epoch 跑冻结 val-200，按 overall RMSE 选 best。
"""A5 long-run per-epoch validation queue and best selection (plan 4.2 / 5).

Every candidate (the source 2750-step checkpoint plus each completed
stage epoch) is validated on the SAME frozen val-200 sample manifest,
single GPU, deterministic single output, then the lowest unrounded
overall RMSE(K) wins with ties resolved to the earlier cumulative
training amount.  Results land under ``--validation-dir/<label>/`` as
``protocol.json`` / ``checkpoint.json`` / ``validation.json`` and the
winner is written to ``--best-output`` (``best_selection.json``) with a
byte copy to ``--best-checkpoint`` (``best_val_rmse.pth``).

The driver is restart-safe (a candidate whose ``validation.json``
already exists is never re-run) and reads only immutable epoch
snapshots: an epoch is considered complete only once BOTH
``epoch_NNN.pth`` and its ``.semantics.json`` sidecar exist (the
sidecar is written after the checkpoint, so its presence proves the
snapshot finished writing).  ``latest.pth`` is never validated here.

Usage (validation queue tmux, one free GPU):

    CUDA_VISIBLE_DEVICES=<free-gpu-id> python -u scripts/validate_a5_epochs.py \
        --h5-path /data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5 \
        --data-manifest artifacts/ostia_data_manifest_real.json \
        --sample-manifest experiments/a5_longtrain_v1_20260908/protocol/val200_manifest.json \
        --source-checkpoint experiments/ostia_spatiotemporal_ablation/A5_geo_p4_best_i4/finetune_lr1e4/epoch_011.pth \
        --train-dir experiments/a5_longtrain_v1_20260908/train \
        --validation-dir experiments/a5_longtrain_v1_20260908/validation \
        --best-output experiments/a5_longtrain_v1_20260908/validation/best_selection.json \
        --best-checkpoint experiments/a5_longtrain_v1_20260908/validation/best_val_rmse.pth \
        --device cuda:0 --num-epochs 30 --watch
"""

import argparse
import json
import os
import shutil
import sys
import time

from diafno.evaluation.candidate_eval import (
    checkpoint_summary,
    select_best_val_rmse,
    write_candidate_artifacts,
)
from diafno.evaluation.config import OSTIAValidationConfig
from diafno.evaluation.validator import OSTIAValidator

# Source A5 checkpoint already accounts for 2750 optimizer attempts;
# every long-run stage checkpoint adds its own global_step on top.
SOURCE_CUMULATIVE_STEPS = 2750


# 用途：构造固定协议下的单候选验证配置（冻结 val-200、确定性单输出、AMP）。
# 参数：输入 checkpoint/h5/data_manifest/sample_manifest/output_path/device；输出 OSTIAValidationConfig。
def build_validation_config(
        checkpoint,
        h5_path,
        data_manifest,
        sample_manifest,
        output_path,
        device,
        num_workers=2,
        batch_size=1,
        use_amp=True,
    ):
    return OSTIAValidationConfig(
        checkpoint=checkpoint,
        h5_path=h5_path,
        output_path=output_path,
        split="val",
        condition_mode=None,
        data_manifest=data_manifest,
        sample_manifest=sample_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_steps=None,
        s_churn=None,
        ensemble_members=1,
        prediction_mode="model",
        probe_sigma=None,
        condition_ablation="none",
        seed=123,
        device=device,
        max_samples=200,
        use_amp=use_amp,
        paired_bootstrap_replicates=0,
        bootstrap_block_days=22,
        bootstrap_confidence=0.95,
        bootstrap_seed=123,
    )


# 用途：列出候选（源 checkpoint + 已完成且带 sidecar 的 epoch 快照）。
# 参数：输入 source_checkpoint、train_dir、num_epochs；输出 [(label, checkpoint_path)] 列表。
def list_candidates(source_checkpoint, train_dir, num_epochs):
    """Source first, then completed (sidecar-proven) epoch snapshots."""
    candidates = [("source_epoch011", os.path.abspath(source_checkpoint))]
    if not os.path.isdir(train_dir):
        return candidates
    for epoch in range(1, num_epochs + 1):
        checkpoint = os.path.join(
            train_dir, f"epoch_{epoch:03d}.pth"
        )
        sidecar = checkpoint + ".semantics.json"
        if os.path.isfile(checkpoint) and os.path.isfile(sidecar):
            candidates.append((f"epoch_{epoch:03d}", checkpoint))
    return candidates


# 用途：计算候选的累计训练量（源=2750；stage epoch=2750+该快照 global_step）。
# 参数：输入 label、summary（checkpoint_summary 结果）；输出 int。
def cumulative_training_steps(label, summary):
    if label == "source_epoch011":
        return SOURCE_CUMULATIVE_STEPS
    return SOURCE_CUMULATIVE_STEPS + int(summary.get("global_step", 0))


# 用途：验证单个候选并落盘协议/checkpoint/validation 三件套。
# 参数：输入 args、label、checkpoint_path、protocol；输出 候选 dict（含 overall_rmse 与累计步数）。
def validate_candidate(args, label, checkpoint_path, protocol):
    candidate_dir = os.path.join(args.validation_dir, label)
    output_path = os.path.join(candidate_dir, "validation.json")
    if os.path.isfile(output_path):
        # Restart-safe: re-read the already-written result instead of
        # re-running inference (validations are expensive).
        with open(output_path, "r", encoding="utf-8") as file:
            result = json.load(file)
    else:
        config = build_validation_config(
            checkpoint=checkpoint_path,
            h5_path=args.h5_path,
            data_manifest=args.data_manifest,
            sample_manifest=args.sample_manifest,
            output_path=output_path,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
        )
        result = OSTIAValidator(config).run()
    summary = write_candidate_artifacts(
        candidate_dir,
        protocol,
        result,
        checkpoint_path,
    )
    return {
        "label": label,
        "source": checkpoint_path,
        "overall_rmse": result["overall"]["rmse"],
        "epoch": summary["epoch"],
        "global_step": summary["global_step"],
        "cumulative_training_steps": cumulative_training_steps(
            label, summary
        ),
        "summary": summary,
    }


# 用途：构建写入 protocol.json 的验证协议摘要。
# 参数：输入 args；输出 dict。
def build_protocol(args):
    return {
        "split": "val",
        "sample_manifest": os.path.abspath(args.sample_manifest),
        "data_manifest": os.path.abspath(args.data_manifest),
        "selection_rule": (
            "lowest unrounded overall RMSE(K); ties -> earlier "
            "cumulative training steps"
        ),
        "ensemble_members": 1,
        "prediction_mode": "model",
        "condition_ablation": "none",
        "seed": 123,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_amp": not args.no_amp,
    }


# 用途：从已落盘候选重建用于选 best 的候选列表（含累计步数）。
# 参数：输入 validation_dir；输出 候选 dict 列表。
def load_validated_candidates(validation_dir):
    candidates = []
    for label in sorted(os.listdir(validation_dir)):
        candidate_dir = os.path.join(validation_dir, label)
        checkpoint_path = os.path.join(candidate_dir, "checkpoint.json")
        validation_path = os.path.join(candidate_dir, "validation.json")
        if not (
                os.path.isfile(checkpoint_path)
                and os.path.isfile(validation_path)
            ):
            continue
        with open(checkpoint_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        with open(validation_path, "r", encoding="utf-8") as file:
            result = json.load(file)
        candidates.append({
            "label": label,
            "source": summary["path"],
            "overall_rmse": result["overall"]["rmse"],
            "epoch": summary["epoch"],
            "global_step": summary["global_step"],
            "cumulative_training_steps": cumulative_training_steps(
                label, summary
            ),
            "summary": summary,
        })
    return candidates


# 用途：选出 best 并写 best_selection.json（含协议摘要）与 best_val_rmse.pth（原子替换）。
# 参数：输入 candidates、best_output、best_checkpoint、protocol；输出 best dict。
def finalize_best(candidates, best_output, best_checkpoint, protocol):
    best = select_best_val_rmse(candidates)
    selection = dict(best)
    selection["protocol"] = protocol
    output_dir = os.path.dirname(best_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(best_output, "w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)
        file.write("\n")
    source = best["source"]
    temporary = best_checkpoint + ".tmp"
    shutil.copyfile(source, temporary)
    os.replace(temporary, best_checkpoint)
    return best


# 用途：判断训练进程是否仍在运行。
# 参数：无输入；输出 布尔值。
def training_running():
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", "trainer_ostia.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# 用途：入口：解析参数，按 watch/once 驱动验证与选 best。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description=(
            "A5 long-run per-epoch val-200 validation queue and best "
            "selection (plan 4.2 / 5)"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--best-output", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--watch", action="store_true",
                        help="loop until all candidates are validated "
                             "and training has finished")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--exit-grace-seconds", type=float, default=900.0)
    args = parser.parse_args()

    os.makedirs(args.validation_dir, exist_ok=True)
    protocol = build_protocol(args)

    def validate_available():
        validated = []
        for label, checkpoint_path in list_candidates(
                args.source_checkpoint, args.train_dir, args.num_epochs
            ):
            candidate = validate_candidate(
                args, label, checkpoint_path, protocol
            )
            print(
                f"[val] {label}: overall_rmse={candidate['overall_rmse']} "
                f"cum_steps={candidate['cumulative_training_steps']}",
                flush=True,
            )
            validated.append(candidate)
        return validated

    validated = validate_available()
    total_needed = args.num_epochs + 1  # source + every stage epoch
    if len(validated) >= total_needed:
        candidates = load_validated_candidates(args.validation_dir)
        best = finalize_best(
            candidates, args.best_output, args.best_checkpoint, protocol
        )
        print(
            f"[val] all candidates done; best={best['label']} "
            f"overall_rmse={best['overall_rmse']}",
            flush=True,
        )
        return 0

    if not args.watch:
        print(
            f"[val] only {len(validated)}/{total_needed} candidates "
            "available; use --watch to keep polling",
            flush=True,
        )
        return 1

    print(
        f"[val] {len(validated)}/{total_needed} candidates validated; "
        "watching for more epochs",
        flush=True,
    )
    idle_since = None
    while True:
        for label, checkpoint_path in list_candidates(
                args.source_checkpoint, args.train_dir, args.num_epochs
            ):
            if os.path.isfile(
                    os.path.join(
                        args.validation_dir, label, "validation.json"
                    )
                ):
                continue
            candidate = validate_candidate(
                args, label, checkpoint_path, protocol
            )
            print(
                f"[val] {label}: overall_rmse={candidate['overall_rmse']} "
                f"cum_steps={candidate['cumulative_training_steps']}",
                flush=True,
            )
        validated = load_validated_candidates(args.validation_dir)
        if len(validated) >= total_needed:
            best = finalize_best(
                validated, args.best_output, args.best_checkpoint, protocol
            )
            print(
                f"[val] all candidates done; best={best['label']} "
                f"overall_rmse={best['overall_rmse']}",
                flush=True,
            )
            return 0
        if training_running():
            idle_since = None
        elif idle_since is None:
            idle_since = time.time()
        elif time.time() - idle_since >= args.exit_grace_seconds:
            print(
                "[val] training stopped but not all epochs present; "
                "exiting for manual review",
                flush=True,
            )
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
