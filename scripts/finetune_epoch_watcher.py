# 用途：快照并验证已完成 epoch 的权重，维护验证 RMSE 最优模型。
"""Offline epoch watcher for the residual fine-tune run (plan v2.1, sec 1.6).

Snapshots latest.pth after each completed epoch, validates the snapshot on a
side GPU with the fixed 200-sample validation split, appends per-epoch metrics
to epoch_metrics.jsonl, and maintains best_model.pth by overall RMSE (day1 RMSE
as tie-break). The training process itself is never touched; validations share
GPU 2/3 with batch 1 (~1 GB).

Launch on the server next to the fine-tune job, e.g.:

    CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/finetune_epoch_watcher.py \
        --repo /data2/user/zzx/exam_preprocessed/DiAFNO \
        --exp-dir experiments/ostia_7day_to15day_residual_ft \
        --h5-path /data/exam_preprocessed_data/zzx/ocean_temperature_data_patched.h5 \
        --log-file /tmp/ostia_ft_logs/ft.log \
        > /tmp/ostia_ft_logs/watcher.log 2>&1 &

Notes:
- Restart-safe: epochs whose metrics json already exists are not re-validated;
  records already in epoch_metrics.jsonl are not duplicated.
- The watcher exits `--exit-grace-seconds` after the training process
  disappears (checked via pgrep on trainer_ostia.py).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

EPOCH_LINE = re.compile(r"epoch=(\d+) train_loss=([\d.eE+-]+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--ensemble-members", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--exit-grace-seconds", type=float, default=600.0)
    return parser.parse_args()


def read_last_epoch(log_path):
    """Return (max_epoch, train_loss_of_last_line) from the training log."""
    epoch = 0
    train_loss = None
    if not os.path.isfile(log_path):
        return epoch, train_loss
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EPOCH_LINE.search(line)
            if match:
                epoch = max(epoch, int(match.group(1)))
                train_loss = float(match.group(2))
    return epoch, train_loss


def training_running():
    result = subprocess.run(
        ["pgrep", "-f", "trainer_ostia.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def run_validation(args, snapshot_path, output_path):
    command = [
        sys.executable,
        "-u",
        "validate_ostia.py",
        "--checkpoint", snapshot_path,
        "--h5-path", args.h5_path,
        "--sampling-steps", str(args.sampling_steps),
        "--s-churn", str(args.s_churn),
        "--ensemble-members", str(args.ensemble_members),
        "--max-samples", str(args.max_samples),
        "--device", args.device,
        "--output-path", output_path,
    ]
    subprocess.run(command, cwd=args.repo, check=False)
    if not os.path.isfile(output_path):
        return None
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_recorded_epochs(metrics_path):
    epochs = set()
    if not os.path.isfile(metrics_path):
        return epochs
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                epochs.add(int(json.loads(line)["epoch"]))
            except (ValueError, KeyError):
                continue
    return epochs


def main():
    args = parse_args()
    args.exp_dir = os.path.abspath(args.exp_dir)
    snapshot_dir = os.path.join(args.exp_dir, "epoch_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    metrics_path = os.path.join(args.exp_dir, "epoch_metrics.jsonl")
    best_path = os.path.join(args.exp_dir, "best_model.pth")
    latest_path = os.path.join(args.exp_dir, "latest.pth")

    validated_epochs = load_recorded_epochs(metrics_path)
    best = None  # (overall_rmse, day1_rmse)
    for metrics_file in sorted(os.listdir(snapshot_dir)):
        if metrics_file.startswith("metrics_epoch_"):
            # already validated in a previous watcher run: track best only
            with open(
                    os.path.join(snapshot_dir, metrics_file),
                    "r",
                    encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            overall = payload.get("overall", {})
            day1 = payload.get("by_lead_day", {}).get("1", {})
            rmse, day1_rmse = overall.get("rmse"), day1.get("rmse")
            if rmse is None:
                continue
            candidate = (rmse, day1_rmse if day1_rmse is not None else float("inf"))
            if best is None or candidate < best:
                best = candidate

    idle_since = None
    print("[watcher] started", flush=True)
    while True:
        epoch, train_loss = read_last_epoch(args.log_file)
        for completed in range(1, epoch + 1):
            if completed in validated_epochs:
                continue
            time.sleep(args.settle_seconds)
            if not os.path.isfile(latest_path):
                print(
                    f"[watcher] epoch {completed}: latest.pth missing, skip",
                    flush=True,
                )
                validated_epochs.add(completed)
                continue
            snapshot_path = os.path.join(
                snapshot_dir, f"epoch_{completed:03d}.pth"
            )
            metrics_path_out = os.path.join(
                snapshot_dir, f"metrics_epoch_{completed:03d}.json"
            )
            shutil.copyfile(latest_path, snapshot_path)
            print(
                f"[watcher] validating epoch {completed} "
                f"(train_loss={train_loss}) ...",
                flush=True,
            )
            payload = run_validation(
                args, snapshot_path, metrics_path_out
            )
            if payload is None:
                print(
                    f"[watcher] epoch {completed}: validation failed, skip",
                    flush=True,
                )
                validated_epochs.add(completed)
                continue
            overall = payload.get("overall", {})
            day1 = payload.get("by_lead_day", {}).get("1", {})
            rmse = overall.get("rmse")
            day1_rmse = day1.get("rmse")
            record = {
                "epoch": completed,
                "train_loss": train_loss,
                "overall_rmse": rmse,
                "overall_mae": overall.get("mae"),
                "day1_rmse": day1_rmse,
                "bias": overall.get("bias"),
                "correlation": overall.get("correlation"),
                "valid_pixels": overall.get("valid_pixels"),
            }
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[watcher] epoch {completed}: {json.dumps(record)}",
                flush=True,
            )
            if rmse is not None:
                candidate = (
                    rmse,
                    day1_rmse if day1_rmse is not None else float("inf"),
                )
                if best is None or candidate < best:
                    best = candidate
                    tmp_path = best_path + ".tmp"
                    shutil.copyfile(snapshot_path, tmp_path)
                    os.replace(tmp_path, best_path)
                    print(
                        f"[watcher] new best -> best_model.pth "
                        f"(overall_rmse={rmse}, day1_rmse={day1_rmse})",
                        flush=True,
                    )
            validated_epochs.add(completed)

        if training_running():
            idle_since = None
        elif idle_since is None:
            idle_since = time.time()
        elif time.time() - idle_since >= args.exit_grace_seconds:
            print(
                f"[watcher] training finished; best={best}; exiting",
                flush=True,
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
