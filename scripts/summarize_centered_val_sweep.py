#!/usr/bin/env python
# 用途：汇总 A5-centered 逐 epoch val-200 扫描结果为 Markdown 表（含冻结 A5 基线对照）。
"""Summarize the per-epoch val-200 sweep of the A5-centered run.

Reads every ``<validation-dir>/<label>/validation.json`` (plus the
matching ``checkpoint.json``) and writes a Markdown table with the
frozen protocol identity, the per-epoch optional overall RMSE / CRPS /
MAE / bias, and the two frozen best selections when they exist.  The
frozen A5 baseline (deterministic, same val-200) is quoted as a
reference row, never as a candidate.

Usage::

    python scripts/summarize_centered_val_sweep.py \
        --validation-dir experiments/.../validation \
        --output experiments/.../validation/VAL_SWEEP.md
"""

import argparse
import json
import os
import sys


# 用途：读取一个候选的 validation.json + checkpoint.json。
# 参数：输入 candidate_dir；输出 (validation dict, checkpoint dict) 或 None。
def read_candidate(candidate_dir):
    validation_path = os.path.join(candidate_dir, "validation.json")
    checkpoint_path = os.path.join(candidate_dir, "checkpoint.json")
    if not (
            os.path.isfile(validation_path)
            and os.path.isfile(checkpoint_path)
        ):
        return None
    with open(validation_path, "r", encoding="utf-8") as file:
        validation = json.load(file)
    with open(checkpoint_path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    return validation, checkpoint


# 用途：收集按 epoch 升序的候选记录。
# 参数：输入 validation_dir；输出 (候选列表, 协议 dict 或 None)。
def collect_candidates(validation_dir):
    candidates = []
    protocol = None
    for name in sorted(os.listdir(validation_dir)):
        candidate_dir = os.path.join(validation_dir, name)
        if not os.path.isdir(candidate_dir):
            continue
        payload = read_candidate(candidate_dir)
        if payload is None:
            continue
        validation, checkpoint = payload
        if protocol is None:
            protocol_path = os.path.join(candidate_dir, "protocol.json")
            if os.path.isfile(protocol_path):
                with open(protocol_path, "r", encoding="utf-8") as file:
                    protocol = json.load(file)
        overall = validation["overall"]
        candidates.append({
            "label": name,
            "epoch": int(checkpoint.get("epoch", -1)),
            "global_step": int(checkpoint.get("global_step", -1)),
            "successful_updates": int(
                checkpoint.get("successful_updates", -1)
            ),
            "sha256": checkpoint.get("sha256", ""),
            "rmse": float(overall["rmse"]),
            "crps": float(overall["crps"]),
            "mae": float(overall.get("mae", float("nan"))),
            "bias": float(overall.get("bias", float("nan"))),
            "pixels": int(overall.get("valid_pixels", 0)),
        })
    candidates.sort(key=lambda entry: (entry["epoch"], entry["label"]))
    return candidates, protocol


# 用途：写 Markdown 汇总（含两种 best 与基线对照）。
# 参数：输入 validation_dir、output_path、baseline（dict 或 None）；输出 汇总 dict。
def write_summary(validation_dir, output_path, baseline=None):
    candidates, protocol = collect_candidates(validation_dir)
    if not candidates:
        raise ValueError(
            f"no validated candidates under {validation_dir}"
        )
    best_rmse = min(
        candidates,
        key=lambda entry: (entry["rmse"], entry["global_step"]),
    )
    best_crps = min(
        candidates,
        key=lambda entry: (entry["crps"], entry["global_step"]),
    )
    lines = [
        "# A5-centered DiAFNO val-200 sweep summary",
        "",
        "Frozen protocol: 16 members, 16 sampling steps, S_churn=0, "
        "member seed `123 + legacy_dataset_index*1000 + member`, pooled "
        "overall metrics over all 15 leads and ocean pixels, one frozen "
        "val-200 physical sample manifest.",
        "",
    ]
    if protocol:
        lines += [
            "```json",
            json.dumps(protocol, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "| label | epoch | global step | successful updates | "
        "overall RMSE (K) | overall CRPS (K) | MAE (K) | bias (K) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in candidates:
        lines.append(
            f"| {entry['label']} | {entry['epoch']} | "
            f"{entry['global_step']} | {entry['successful_updates']} | "
            f"{entry['rmse']:.4f} | {entry['crps']:.4f} | "
            f"{entry['mae']:.4f} | {entry['bias']:+.4f} |"
        )
    if baseline:
        lines.append(
            f"| {baseline['label']} | {baseline.get('epoch', '-')} | "
            f"{baseline.get('global_step', '-')} | "
            f"{baseline.get('successful_updates', '-')} | "
            f"{baseline['rmse']:.4f} | {baseline['crps']:.4f} | "
            f"{baseline.get('mae', float('nan')):.4f} | "
            f"{baseline.get('bias', float('nan')):+.4f} |"
        )
    lines += [
        "",
        f"Frozen best by RMSE: **{best_rmse['label']}** "
        f"({best_rmse['rmse']:.4f} K) -> `best_val_mean_rmse.pth`.",
        "",
        f"Frozen best by CRPS: **{best_crps['label']}** "
        f"({best_crps['crps']:.4f} K) -> `best_val_crps.pth`.",
        "",
        "Ties resolve to the earlier cumulative training amount; "
        "`latest.pth` is never a selection candidate.",
        "",
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return {
        "candidates": candidates,
        "best_rmse": best_rmse,
        "best_crps": best_crps,
        "protocol": protocol,
        "output": os.path.abspath(output_path),
    }


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description="Markdown summary of the A5-centered val-200 sweep"
    )
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "optional JSON with a frozen baseline row "
            "({'label','rmse','crps','mae','bias'})"
        ),
    )
    args = parser.parse_args()
    baseline = None
    if args.baseline_json:
        with open(args.baseline_json, "r", encoding="utf-8") as file:
            baseline = json.load(file)
    summary = write_summary(
        args.validation_dir, args.output, baseline=baseline
    )
    print(json.dumps(
        {
            "num_candidates": len(summary["candidates"]),
            "best_rmse": summary["best_rmse"],
            "best_crps": summary["best_crps"],
            "output": summary["output"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
