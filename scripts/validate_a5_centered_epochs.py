#!/usr/bin/env python
# 用途：A5-centered 逐 epoch 验证队列：16 成员 val-200 双指标（mean RMSE / CRPS）选模。
"""Per-epoch val-200 validation queue for the A5-centered run.

Every completed (sidecar-proven) epoch checkpoint of the centered run is
scored on the SAME frozen val-200 sample manifest with 16 members x 16
sampling steps, S_churn=0, member seeds fixed to the physical sample
key (``123 + legacy_dataset_index*1000 + member``, the same convention
as the four-method test-200).  Two best checkpoints are produced after
all candidates are validated, by unrounded metrics with ties resolved
to the earlier cumulative training steps (plan 7.2):

- ``best_val_mean_rmse`` -- lowest ensemble-mean overall RMSE (K);
- ``best_val_crps``      -- lowest empirical ensemble CRPS (K).

The frozen A5 mean is a baseline, never a diffusion candidate.
``latest.pth`` is never scored here.  Restart-safe (a candidate whose
validation JSON exists is skipped) and watches for new epochs.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from diafno.evaluation.method_comparison import empirical_crps
from diafno.evaluation.sample_manifest import (
    ensure_manifest_universe,
    load_sample_manifest,
    manifest_dataset_indices,
)
from scripts.compare_ostia_protocol import ProtocolValidator

HORIZON = 15
MEMBER_COUNT = 16


# 用途：单候选评分：逐样本 16 成员采样，累计 mean-RMSE 与成员 CRPS（物理 K 空间）。
# 参数：输入 validator（ProtocolValidator）、manifest 载荷、label（日志标签）、progress_every（进度打印间隔）；输出 结果 dict。
def score_candidate(validator, manifest_payload, label="", progress_every=10):
    """Pooled per-lead point metrics of the ensemble mean and the
    per-lead empirical CRPS of the 16 members, in physical Kelvin.

    One candidate costs ~2-3 h of GPU time, so the sweep prints a
    progress line every ``progress_every`` samples (elapsed seconds and
    the running pooled RMSE in K); ``progress_every=0`` is silent.
    """
    indices = manifest_dataset_indices(manifest_payload)
    entries = {
        int(entry["dataset_index"]): entry
        for entry in manifest_payload["entries"]
    }
    mean, std = validator.dataset.sst_mean, validator.dataset.sst_std

    def to_kelvin(value):
        return value * std + mean

    # Accumulators per lead (Kelvin space).
    counts = np.zeros(HORIZON, dtype=np.int64)
    sse = np.zeros(HORIZON, dtype=np.float64)
    abs_sum = np.zeros(HORIZON, dtype=np.float64)
    err_sum = np.zeros(HORIZON, dtype=np.float64)
    crps_sum = np.zeros(HORIZON, dtype=np.float64)
    per_lead = []
    started = time.time()
    for position, sample_index in enumerate(indices):
        entry = entries[sample_index]
        legacy_index = int(entry["legacy_dataset_index"])
        seed_base = 123 + legacy_index * 1000
        members = []
        for member in range(MEMBER_COUNT):
            (prediction, _, _, _, _) = validator.sample_at(
                validator.sample_index(entry),
                seed_base=seed_base + member,
            )
            members.append(to_kelvin(prediction))
        members = np.asarray(members, dtype=np.float64)
        mean_prediction = members.mean(axis=0)
        sample = validator.decoded_sample(validator.sample_index(entry))
        target = to_kelvin(
            sample["target"].numpy()[..., 0]
        ).astype(np.float64)
        mask = sample["target_mask"].numpy()[..., 0] > 0
        for lead in range(HORIZON):
            valid = mask[lead]
            t = target[lead][valid]
            p = mean_prediction[lead][valid]
            error = p - t
            counts[lead] += int(valid.sum())
            sse[lead] += float(np.square(error).sum())
            abs_sum[lead] += float(np.abs(error).sum())
            err_sum[lead] += float(error.sum())
            member_values = members[:, lead][:, valid]
            crps_sum[lead] += float(
                empirical_crps(member_values, t).sum()
            )
        if progress_every and (
                (position + 1) % int(progress_every) == 0
                or position + 1 == len(indices)
            ):
            done = int(counts.sum())
            print(
                f"[val] {label} {position + 1}/{len(indices)} samples "
                f"elapsed={time.time() - started:.0f}s "
                f"running_rmse="
                f"{float(np.sqrt(sse.sum() / max(done, 1))):.4f}K",
                flush=True,
            )
    overall_count = int(counts.sum())
    result = {
        "num_samples": len(indices),
        "overall": {
            "rmse": float(np.sqrt(sse.sum() / overall_count)),
            "mae": float(abs_sum.sum() / overall_count),
            "bias": float(err_sum.sum() / overall_count),
            "crps": float(crps_sum.sum() / overall_count),
        },
        "by_lead_day": {},
    }
    for lead in range(HORIZON):
        n = int(counts[lead])
        result["by_lead_day"][str(lead + 1)] = {
            "rmse": float(np.sqrt(sse[lead] / n)),
            "mae": float(abs_sum[lead] / n),
            "crps": float(crps_sum[lead] / n),
            "valid_pixels": n,
        }
    result["overall"]["valid_pixels"] = overall_count
    return result


# 用途：双指标选 best（未舍入最小；并列取较早累计步数）。
# 参数：输入 candidates（含 metric 键与 cumulative_training_steps）、metric_key、output_path；输出 best dict。
def select_best(candidates, metric_key, output_path=None):
    if not candidates:
        raise ValueError("select_best needs at least one candidate")
    best = min(
        candidates,
        key=lambda item: (
            item[metric_key],
            int(item.get("cumulative_training_steps", 0)),
        ),
    )
    if output_path is not None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(best, file, ensure_ascii=False, indent=2)
            file.write("\n")
    return best


# 用途：把候选 checkpoint 字节复制为 best 权重文件（原子替换）。
# 参数：输入 best（候选 dict）、best_checkpoint_path；输出 无。
def copy_best_checkpoint(best, best_checkpoint_path):
    temporary = best_checkpoint_path + ".tmp"
    shutil.copyfile(best["source"], temporary)
    os.replace(temporary, best_checkpoint_path)


# 用途：构建固定协议描述。
# 参数：输入 args；输出 dict。
def build_protocol(args):
    return {
        "split": "val",
        "sample_manifest": os.path.abspath(args.sample_manifest),
        "data_manifest": os.path.abspath(args.data_manifest),
        "members": MEMBER_COUNT,
        "sampling_steps": args.sampling_steps,
        "s_churn": args.s_churn,
        "member_seed_rule": "123 + legacy_dataset_index*1000 + member",
        "selection": (
            "unrounded ensemble-mean RMSE / empirical CRPS minima; "
            "ties -> earlier cumulative training steps"
        ),
        "device": args.device,
        "use_amp": not args.no_amp,
    }


# 用途：本次扫描需要验证的 epoch 列表（显式清单优先，否则按间隔规则）。
# 参数：输入 num_epochs、every（间隔）、only_epochs（显式 epoch 列表或 None）；输出 升序 epoch 列表。
def candidate_epochs(num_epochs, every=1, only_epochs=None):
    """Epoch numbers this sweep validates.

    Default cadence (plan 7.2): every ``every``-th epoch plus the final
    epoch.  An explicit ``only_epochs`` list overrides the cadence so
    independent processes can shard one frozen sweep across GPUs while
    sharing a single validation directory.
    """
    if only_epochs:
        return sorted({int(epoch) for epoch in only_epochs})
    return [
        epoch
        for epoch in range(1, int(num_epochs) + 1)
        if epoch % int(every) == 0 or epoch == int(num_epochs)
    ]


# 用途：候选列表：已完成（sidecar 在场）的 epoch 快照，按轮次升序。
# 参数：输入 train_dir、num_epochs、every（验证间隔，末轮恒在）、only_epochs（显式 epoch 列表）；输出 [(label, path)]。
def list_candidates(train_dir, num_epochs, every=1, only_epochs=None):
    candidates = []
    if not os.path.isdir(train_dir):
        return candidates
    for epoch in candidate_epochs(num_epochs, every, only_epochs):
        checkpoint = os.path.join(
            train_dir, f"epoch_{epoch:03d}.pth"
        )
        if (
                os.path.isfile(checkpoint)
                and os.path.isfile(checkpoint + ".semantics.json")
            ):
            candidates.append((f"epoch_{epoch:03d}", checkpoint))
    return candidates


# 用途：按 label 统计累计训练步数（stage global_step；无源候选）。
# 参数：输入 label、summary；输出 int。
def cumulative_training_steps(label, summary):
    return int(summary.get("global_step", 0))


# 用途：验证单个候选并落盘。
# 参数：输入 args、label、checkpoint_path、protocol、validator（惰性构建回调）；输出 候选 dict。
def validate_candidate(args, label, checkpoint_path, protocol):
    from diafno.evaluation.candidate_eval import (
        checkpoint_summary,
        write_candidate_artifacts,
    )
    candidate_dir = os.path.join(args.validation_dir, label)
    output_path = os.path.join(candidate_dir, "validation.json")
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            result = json.load(file)
    else:
        import torch
        validator = ProtocolValidator(
            checkpoint_path,
            args.h5_path,
            args.data_manifest,
            torch.device(args.device),
            ensemble_members=1,
            sampling_steps=args.sampling_steps,
            s_churn=args.s_churn,
            use_amp=not args.no_amp,
            split=protocol["split"],
        )
        # The frozen val-200 manifest addresses physical samples by
        # ``dataset_index`` inside the val universe it was frozen from;
        # scoring it against another split (e.g. the test loader, length
        # 110600) is exactly the historical IndexError(110894) bug.
        ensure_manifest_universe(
            args.manifest_payload,
            len(validator.dataset),
            split=protocol["split"],
            label=f"{label} val sample manifest",
        )
        result = score_candidate(
            validator, args.manifest_payload, label=label
        )
    summary = write_candidate_artifacts(
        candidate_dir, protocol, result, checkpoint_path
    )
    return {
        "label": label,
        "source": checkpoint_path,
        "overall_rmse": result["overall"]["rmse"],
        "overall_crps": result["overall"]["crps"],
        "epoch": summary["epoch"],
        "global_step": summary["global_step"],
        "cumulative_training_steps": cumulative_training_steps(
            label, summary
        ),
    }


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description=(
            "A5-centered per-epoch 16-member val-200 queue with dual "
            "best selection (plan 7.2)"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help=(
            "validate every K-th epoch (final epoch always included); "
            "pre-recorded cadence revision for the expensive 16-member "
            "sweep (plan 7.2)"
        ),
    )
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--only-epochs",
        default=None,
        help=(
            "explicit comma-separated epoch list overriding --every; "
            "lets several processes shard one frozen sweep across GPUs "
            "while sharing a single validation directory (pair with "
            "--no-select and finish with one full-cadence pass)"
        ),
    )
    parser.add_argument(
        "--no-select",
        action="store_true",
        help=(
            "score candidates but never write the frozen best_* "
            "artifacts (shard workers must not select from a partial "
            "candidate set)"
        ),
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--exit-grace-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.only_epochs is not None:
        args.only_epochs = [
            int(part)
            for part in str(args.only_epochs).replace(
                " ", ""
            ).split(",")
            if part
        ]
        if not args.only_epochs:
            raise ValueError("--only-epochs needs at least one epoch")
        if any(epoch < 1 for epoch in args.only_epochs):
            raise ValueError("--only-epochs entries must be >= 1")

    os.makedirs(args.validation_dir, exist_ok=True)
    protocol = build_protocol(args)
    args.manifest_payload = load_sample_manifest(
        args.sample_manifest, split=protocol["split"]
    )

    def run_sweep():
        validated = []
        for label, path in list_candidates(
                args.train_dir, args.num_epochs, args.every,
                args.only_epochs,
            ):
            try:
                candidate = validate_candidate(
                    args, label, path, protocol
                )
            except Exception as error:  # keep the queue alive
                print(
                    f"[val] {label}: FAILED {error!r}",
                    flush=True,
                )
                continue
            print(
                f"[val] {label}: mean_rmse={candidate['overall_rmse']} "
                f"crps={candidate['overall_crps']} "
                f"cum_steps={candidate['cumulative_training_steps']}",
                flush=True,
            )
            validated.append(candidate)
        return validated

    validated = run_sweep()
    total_needed = len(candidate_epochs(
        args.num_epochs, args.every, args.only_epochs
    ))
    if len(validated) >= total_needed and not args.no_select:
        best_rmse = select_best(
            validated,
            "overall_rmse",
            os.path.join(args.validation_dir, "best_val_mean_rmse.json"),
        )
        best_crps = select_best(
            validated,
            "overall_crps",
            os.path.join(args.validation_dir, "best_val_crps.json"),
        )
        copy_best_checkpoint(
            best_rmse,
            os.path.join(args.validation_dir, "best_val_mean_rmse.pth"),
        )
        copy_best_checkpoint(
            best_crps,
            os.path.join(args.validation_dir, "best_val_crps.pth"),
        )
        print(
            f"[val] all done; best_val_mean_rmse={best_rmse['label']} "
            f"({best_rmse['overall_rmse']}), "
            f"best_val_crps={best_crps['label']} "
            f"({best_crps['overall_crps']})",
            flush=True,
        )
        return 0
    if not args.watch:
        if len(validated) >= total_needed:
            print(
                f"[val] shard complete: {len(validated)}/"
                f"{total_needed} candidates validated; --no-select "
                "leaves best selection to the full-cadence pass",
                flush=True,
            )
            return 0
        print(
            f"[val] {len(validated)}/{total_needed} validated; use "
            "--watch to keep polling",
            flush=True,
        )
        return 1
    idle_since = None
    while True:
        for label, path in list_candidates(
                args.train_dir, args.num_epochs, args.every,
                args.only_epochs,
            ):
            if os.path.isfile(
                    os.path.join(args.validation_dir, label, "validation.json")
                ):
                continue
            try:
                candidate = validate_candidate(
                    args, label, path, protocol
                )
            except Exception as error:
                print(
                    f"[val] {label}: FAILED {error!r}",
                    flush=True,
                )
                continue
            print(
                f"[val] {label}: mean_rmse={candidate['overall_rmse']} "
                f"crps={candidate['overall_crps']} "
                f"cum_steps={candidate['cumulative_training_steps']}",
                flush=True,
            )
        validated = []
        for label, _ in list_candidates(
                args.train_dir, args.num_epochs, args.every,
                args.only_epochs,
            ):
            candidate_dir = os.path.join(args.validation_dir, label)
            path = os.path.join(candidate_dir, "validation.json")
            checkpoint_json = os.path.join(
                candidate_dir, "checkpoint.json"
            )
            if os.path.isfile(path) and os.path.isfile(checkpoint_json):
                with open(path, "r", encoding="utf-8") as file:
                    result = json.load(file)
                with open(checkpoint_json, "r", encoding="utf-8") as file:
                    summary = json.load(file)
                validated.append({
                    "label": label,
                    "source": summary["path"],
                    "overall_rmse": result["overall"]["rmse"],
                    "overall_crps": result["overall"]["crps"],
                    "epoch": summary["epoch"],
                    "global_step": summary["global_step"],
                    "cumulative_training_steps": int(
                        summary.get("global_step", 0)
                    ),
                })
        if len(validated) >= total_needed:
            if args.no_select:
                print(
                    f"[val] shard complete: {len(validated)}/"
                    f"{total_needed} candidates validated; --no-select "
                    "leaves best selection to the full-cadence pass",
                    flush=True,
                )
                return 0
            best_rmse = select_best(
                validated,
                "overall_rmse",
                os.path.join(
                    args.validation_dir, "best_val_mean_rmse.json"
                ),
            )
            best_crps = select_best(
                validated,
                "overall_crps",
                os.path.join(args.validation_dir, "best_val_crps.json"),
            )
            copy_best_checkpoint(
                best_rmse,
                os.path.join(
                    args.validation_dir, "best_val_mean_rmse.pth"
                ),
            )
            copy_best_checkpoint(
                best_crps,
                os.path.join(args.validation_dir, "best_val_crps.pth"),
            )
            print(
                f"[val] all done; best_val_mean_rmse="
                f"{best_rmse['label']} ({best_rmse['overall_rmse']}), "
                f"best_val_crps={best_crps['label']} "
                f"({best_crps['overall_crps']})",
                flush=True,
            )
            return 0
        running = subprocess.run(
            ["pgrep", "-f", "trainer_ostia.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if running:
            idle_since = None
        elif idle_since is None:
            idle_since = time.time()
        elif time.time() - idle_since >= args.exit_grace_seconds:
            print(
                "[val] training stopped but not all epochs validated; "
                "exiting for manual review",
                flush=True,
            )
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
