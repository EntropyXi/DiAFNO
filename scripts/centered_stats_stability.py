#!/usr/bin/env python
# 用途：训练集 innovation 统计的嵌套稳定性检查（8192→16384→32768→65536）与固化。
"""Train-only innovation statistics with nested stability gating.

Runs ``compute_centered_stats`` at increasing nested sample counts
(plan 6.1).  The stability decision is a pure function of two payloads:
every lead must satisfy ``|std_new - std_prev| / std_new <= 0.02`` and
``|mean_new - mean_prev| / std_new <= 0.02``.  The first stable count
is frozen to ``--final-output``; running to 65536 without stability
fails with the per-lead report instead of silently relaxing the
threshold.

Every payload written is re-validated by the shared centered-stats
validator before it is kept.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# 用途：纯函数：比较相邻两轮统计的逐 lead 稳定性。
# 参数：输入 prev_payload、cur_payload、std_rtol/mean_scale（阈值）；输出 (stable, 明细 dict)。
def stability_summary(
        prev_payload,
        cur_payload,
        std_rtol=0.02,
        mean_scale=0.02,
    ):
    """Per-lead relative std change and mean shift vs the new std.

    ``std_rtol`` bounds ``|s_cur - s_prev| / s_cur``; ``mean_scale``
    bounds ``|m_cur - m_prev| / s_cur``.  Both must hold for every
    lead for the pair to be stable.
    """
    prev_std = np.asarray(prev_payload["lead_std"], dtype=np.float64)
    cur_std = np.asarray(cur_payload["lead_std"], dtype=np.float64)
    prev_mean = np.asarray(prev_payload["lead_mean"], dtype=np.float64)
    cur_mean = np.asarray(cur_payload["lead_mean"], dtype=np.float64)
    if cur_std.shape != prev_std.shape:
        raise ValueError("consecutive stats payloads disagree in length")
    rel_std = np.abs(cur_std - prev_std) / cur_std
    rel_mean = np.abs(cur_mean - prev_mean) / cur_std
    stable = bool(
        np.all(rel_std <= std_rtol)
        and np.all(rel_mean <= mean_scale)
    )
    details = {
        "prev_num_samples": int(prev_payload["num_samples"]),
        "cur_num_samples": int(cur_payload["num_samples"]),
        "std_rtol": float(std_rtol),
        "mean_scale": float(mean_scale),
        "rel_std_change": rel_std.tolist(),
        "rel_mean_shift": rel_mean.tolist(),
        "stable": stable,
        "valid_pixels": cur_payload["valid_pixels"],
    }
    return stable, details


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Nested train-only innovation statistics with stability "
            "gating (A5_CENTERED_DIAFNO_MAINTRAIN plan 6.1)"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--mean-checkpoint", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-output", required=True)
    parser.add_argument("--sizes", default="8192,16384,32768,65536")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-days", type=int, default=7)
    parser.add_argument("--output-days", type=int, default=15)
    parser.add_argument("--master-size", type=int, default=65536)
    args = parser.parse_args()

    from deterministic_iafno.compute_centered_stats import (
        compute_centered_stats,
    )
    from deterministic_iafno.centered_stats import (
        validate_centered_stats_payload,
    )

    sizes = [int(item) for item in args.sizes.split(",")]
    if len(sizes) < 2:
        raise ValueError("--sizes must list at least two counts")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch_device(args.device)

    report = {"sizes_attempted": sizes, "stages": []}
    previous = None
    chosen = None
    for size in sizes:
        payload, elapsed = compute_centered_stats(
            h5_path=args.h5_path,
            mean_checkpoint_path=args.mean_checkpoint,
            num_samples=size,
            batch_size=args.batch_size,
            input_days=args.input_days,
            output_days=args.output_days,
            device=device,
            use_amp=False,
            data_manifest=args.data_manifest,
            master_indices_size=args.master_size,
        )
        validate_centered_stats_payload(
            payload,
            target_chans=args.output_days,
            input_days=args.input_days,
            output_days=args.output_days,
        )
        size_path = os.path.join(
            args.output_dir, f"stats_size_{size}.json"
        )
        with open(size_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        stage = {
            "num_samples": int(size),
            "elapsed_seconds": float(elapsed),
            "payload": size_path,
            "indices_sha256": payload["indices_sha256"],
        }
        print(
            f"computed stats at {size} samples "
            f"({elapsed:.1f}s, sha {payload['indices_sha256'][:12]}...)",
            flush=True,
        )
        if previous is not None:
            stable, details = stability_summary(previous, payload)
            stage["stability"] = details
            print(
                f"nested {previous['num_samples']} -> {size}: "
                f"stable={stable}",
                flush=True,
            )
            if stable:
                chosen = payload
                report["frozen_num_samples"] = int(
                    payload["num_samples"]
                )
                report["stable_after"] = int(size)
                break
        previous = payload
        report["stages"].append(stage)
    if chosen is None:
        report["decision"] = "unstable_at_all_sizes"
        report_path = os.path.join(
            args.output_dir, "stability_report.json"
        )
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(
            "ERROR: innovation stats did not stabilise up to "
            f"{sizes[-1]} samples; per-lead report at {report_path}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    report["decision"] = "frozen"
    report_path = os.path.join(
        args.output_dir, "stability_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(args.final_output, "w", encoding="utf-8") as file:
        json.dump(chosen, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"FROZEN stats at {chosen['num_samples']} samples -> "
        f"{os.path.abspath(args.final_output)}",
        flush=True,
    )
    return 0


# 用途：解析 torch 设备（缺省按 CUDA 可用性）。
# 参数：输入 device（字符串或 None）；输出 torch.device。
def torch_device(device):
    import torch
    if device is not None:
        return torch.device(device)
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    sys.exit(main())
