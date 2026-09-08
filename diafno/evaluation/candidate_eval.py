# 用途：候选验证产物（协议摘要、checkpoint 摘要）与 best_val_rmse 选择。
"""Candidate validation artifacts and best-val selection (A5 plan 4.2/5).

Every validated candidate stores a small self-describing artifact set:

- ``protocol.json``  -- resolved validation protocol (sampler profile,
  split/sample rule, manifest identity, seed, device);
- ``checkpoint.json`` -- the validated checkpoint path, SHA-256, stage /
  epoch / global step and scheduler counts read from the checkpoint;
- ``validation.json`` -- a copy of the validator output.

Selection uses the same frozen val-200 overall RMSE (K) across the
source checkpoint replay and all stage epochs; ties resolve to the
earlier cumulative training amount.  The result is written to
``best_selection.json`` with the unrounded RMSE, provenance and the
protocol summary.  latest.pth is never used for selection.
"""

import hashlib
import json
import os

import torch


# 用途：checkpoint 文件的规范 SHA256。
# 参数：输入 path；输出 十六进制摘要。
def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 用途：从 checkpoint 读取阶段/步数摘要（epoch/global_step/scheduler/skips）。
# 参数：输入 path；输出 dict。
def checkpoint_summary(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    scheduler = checkpoint.get("scheduler", {}) or {}
    return {
        "path": os.path.abspath(path),
        "sha256": checkpoint_sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "successful_updates": int(scheduler.get("last_epoch", -1)),
        "skipped_optimizer_steps": int(
            checkpoint.get("skipped_optimizer_steps", 0)
        ),
    }


# 用途：写出单个候选的产物目录（protocol/checkpoint/validation）。
# 参数：输入 candidate_dir、protocol、validation_result、checkpoint_path；输出 checkpoint.json dict。
def write_candidate_artifacts(
        candidate_dir,
        protocol,
        validation_result,
        checkpoint_path,
    ):
    os.makedirs(candidate_dir, exist_ok=True)
    summary = checkpoint_summary(checkpoint_path)
    with open(
            os.path.join(candidate_dir, "checkpoint.json"),
            "w",
            encoding="utf-8",
        ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(
            os.path.join(candidate_dir, "protocol.json"),
            "w",
            encoding="utf-8",
        ) as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(
            os.path.join(candidate_dir, "validation.json"),
            "w",
            encoding="utf-8",
        ) as file:
        json.dump(validation_result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


# 用途：按冻结 val-200 overall RMSE 选择 best（未舍入；并列取较早累计训练量）。
# 参数：输入 candidates（含 rmse、epoch/global_step、summary 等 dict 列表）、output_path；输出 best dict。
def select_best_val_rmse(candidates, output_path=None):
    """Select the lowest unrounded overall RMSE candidate.

    ``candidates`` entries need ``overall_rmse``, ``epoch``,
    ``global_step`` and a ``source`` path.  Ties resolve to the earlier
    cumulative training amount (epoch, then global step, then the
    source 2750-step stage before any long-run epoch).
    """
    if not candidates:
        raise ValueError("select_best_val_rmse needs at least one candidate")
    best = min(
        candidates,
        key=lambda item: (
            item["overall_rmse"],
            int(item.get("cumulative_training_steps", 0)),
        ),
    )
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(best, file, ensure_ascii=False, indent=2)
            file.write("\n")
    return best
