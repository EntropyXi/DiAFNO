#!/usr/bin/env python
# 用途：用统一协议导出的 figure_fields.npz 绘制含真值的共享色标预报图。
"""Forecast figures for the unified OSTIA protocol (plan section 9).

Reads ``figure_fields.npz`` (written by ``compare_ostia_protocol.py``)
which holds, for a fixed small set of physical samples, the ground
truth, the target mask and every method's ensemble mean in Kelvin.  Each
figure fixes one sample and one relative crop and shows the ground truth
plus every method over Day 1 / 5 / 10 / 15 with one shared colour scale
and grey invalid pixels -- no per-method autoscaling, no clipping.

Usage::

    python scripts/plot_ostia_protocol_figures.py \
        --fields experiments/.../test200/figure_fields.npz \
        --output-dir experiments/.../test200/figures
"""

import argparse
import json
import os
import sys

import numpy as np

from diafno.evaluation.method_comparison import draw_forecasts

LEADS = (1, 5, 10, 15)

METHOD_LABELS = {
    "target": "Ground Truth",
    "a5_centered": "A5-centered DiAFNO (ensemble mean)",
    "a5_centered_crps": "A5-centered DiAFNO CRPS-best",
    "a5": "A5 frozen mean (deterministic)",
    "old_iafno": "old IAFNO (deterministic)",
    "old_diafno": "old centered DiAFNO (ensemble mean)",
    "persistence": "Persistence",
}


# 用途：解析 "r0:r1,c0:c1" 裁剪窗口。
# 参数：输入 text；输出 (行切片, 列切片)。
def parse_crop(text):
    try:
        rows, columns = str(text).split(",")
        r0, r1 = (int(part) for part in rows.split(":"))
        c0, c1 = (int(part) for part in columns.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"--crop must be r0:r1,c0:c1, got {text!r}"
        ) from error
    if not (0 <= r0 < r1 and 0 <= c0 < c1):
        raise ValueError(f"empty or negative crop window {text!r}")
    return (slice(r0, r1), slice(c0, c1))


# 用途：把导出的场切成一个绘图 case（固定样本 + 固定相对窗口）。
# 参数：输入 fields（npz 载荷）、position、crop（(行,列) 切片）、method_slugs（方法列表）；输出 case dict。
def build_case(fields, position, crop, method_slugs):
    row_slice, column_slice = crop
    target = np.asarray(
        fields["target_kelvin"][position], dtype=np.float32
    )
    mask = np.asarray(fields["target_mask"][position], dtype=np.uint8)
    if target.ndim != 3 or mask.shape != target.shape:
        raise ValueError(
            "figure fields must have [lead,H,W] target/mask shapes"
        )
    case = {
        "target": target[:, row_slice, column_slice],
        "target_mask": mask[:, row_slice, column_slice],
        "metadata": {
            "spatial_index": int(fields["spatial_index"][position]),
            "input_start_time": str(
                fields["input_date_last"][position]
            ),
        },
    }
    if case["target_mask"].max() == 0:
        raise ValueError(
            f"crop window {crop} contains no valid ocean pixel for "
            f"figure sample {position}"
        )
    for slug in method_slugs:
        key = f"prediction_{slug}"
        if key not in fields:
            raise ValueError(f"figure fields are missing {key}")
        case[slug] = np.asarray(
            fields[key][position], dtype=np.float32
        )[:, row_slice, column_slice]
    return case


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ground-truth + all-method forecast figures from the unified "
            "protocol field dump (shared colour scale, Day 1/5/10/15)"
        )
    )
    parser.add_argument("--fields", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--crop",
        default="112:336,112:336",
        help="fixed relative window r0:r1,c0:c1 (default centre half)",
    )
    parser.add_argument(
        "--methods",
        default="a5_centered,a5,old_diafno,old_iafno,persistence",
        help="comma-separated method slugs, in panel order",
    )
    parser.add_argument("--unit", default="K")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    crop = parse_crop(args.crop)
    slugs = [
        part.strip() for part in str(args.methods).split(",") if part.strip()
    ]
    if not slugs:
        raise ValueError("--methods needs at least one method")
    with np.load(args.fields) as payload:
        fields = {key: payload[key] for key in payload.files}
    samples = int(fields["target_kelvin"].shape[0])
    if args.max_samples:
        samples = min(samples, int(args.max_samples))
    cases = [
        build_case(fields, position, crop, slugs)
        for position in range(samples)
    ]
    panel_keys = ("target", *slugs)
    panel_labels = tuple(
        METHOD_LABELS.get(key, key) for key in panel_keys
    )
    os.makedirs(args.output_dir, exist_ok=True)
    images = draw_forecasts(
        cases,
        args.output_dir,
        unit=args.unit,
        dpi=args.dpi,
        panel_keys=panel_keys,
        panel_labels=panel_labels,
        leads=LEADS,
    )
    summary = {
        "fields": os.path.abspath(args.fields),
        "samples": samples,
        "crop": args.crop,
        "methods": slugs,
        "leads": list(LEADS),
        "images": images,
        "colour_scale": (
            "one shared linear turbo scale over every panel of every "
            "figure; grey = invalid target pixel; ensemble methods show "
            "the ensemble mean"
        ),
    }
    with open(
            os.path.join(args.output_dir, "figures.json"),
            "w", encoding="utf-8",
        ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
