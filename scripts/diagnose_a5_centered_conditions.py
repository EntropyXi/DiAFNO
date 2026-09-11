#!/usr/bin/env python
# 用途：§7.3 条件诊断：固定 mu/anchor/z/噪声，只改扩散条件，测分档去噪误差与置乱差异。
"""Condition diagnostics for the A5-centered DiAFNO checkpoints.

Plan section 7.3: at epochs 1 / 10 / 30, on a fixed small sample set,
measure the denoiser error at ``r_sigma`` in {0.1, 0.3, 1, 2, 5} while
changing *only* the diffusion condition.  The frozen mean, the anchor,
the standardized innovation ``z`` and the noise draw are held fixed, so
any difference is attributable to the condition and never to a changed
mean.

Condition variants (plan 7.3):

- ``correct``                -- the sample's own history, t0, geo, season;
- ``history_shuffled``       -- first six history days from a donor
                                (t0, geo and season stay the sample's own);
- ``same_region_other_date`` -- same spatial patch, different window
                                (the whole condition, t0 included);
- ``season_shuffled``        -- only sin_doy/cos_doy from a donor date;
- ``geo_shuffled``           -- only the four static geo channels from a
                                donor patch;
- ``season_geo_shuffled``    -- both of the above.

A variant whose donor does not exist (for example no other window on the
same patch) is reported as skipped instead of being paired with itself.

Usage::

    python scripts/diagnose_a5_centered_conditions.py \
        --checkpoint epoch_001=train/epoch_001.pth \
        --checkpoint epoch_030=train/epoch_030.pth \
        --h5-path ... --data-manifest ... --sample-manifest ... \
        --output-dir experiments/a5_centered_diafno_v1_20260909/diagnostics
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from diafno.evaluation.sample_manifest import (
    ensure_manifest_universe,
    load_sample_manifest,
)
from scripts.compare_ostia_protocol import ProtocolValidator

# Fixed condition layout (14 channels, condition schema v2).
HISTORY_CHANNELS = slice(0, 6)
T0_CHANNEL = 6
GEO_CHANNELS = slice(8, 12)
SEASON_CHANNELS = slice(12, 14)

VARIANTS = (
    "correct",
    "history_shuffled",
    "same_region_other_date",
    "season_shuffled",
    "geo_shuffled",
    "season_geo_shuffled",
)


# 用途：按物理约束为每个样本挑选 donor（禁止自身配对）。
# 参数：输入 entries（清单条目，按 (compact_start, spatial_index) 排序）；输出 {dataset_index: donor dict}。
def select_donors(entries):
    """Deterministic donor assignment for every condition variant.

    Each sample needs at most three donors: a different date (history and
    season), a different spatial patch (geo) and another window on the
    *same* patch (same-region control).  Missing partners are reported
    as ``None`` rather than silently pairing a sample with itself.
    """
    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry["compact_start"]), int(entry["spatial_index"])
        ),
    )
    donors = {}
    for position, entry in enumerate(ordered):
        index = int(entry["dataset_index"])
        start = int(entry["compact_start"])
        patch = int(entry["spatial_index"])
        record = {
            "history": None,
            "season": None,
            "geo": None,
            "same_region": None,
        }
        for offset in range(1, len(ordered)):
            other = ordered[(position + offset) % len(ordered)]
            other_start = int(other["compact_start"])
            other_patch = int(other["spatial_index"])
            if record["history"] is None and other_start != start:
                record["history"] = other
                record["season"] = other
            if record["geo"] is None and other_patch != patch:
                record["geo"] = other
            if (
                    record["same_region"] is None
                    and other_patch == patch
                    and other_start != start
                ):
                record["same_region"] = other
            if all(value is not None for value in record.values()):
                break
        donors[index] = record
    return donors


# 用途：按变体组装被破坏的扩散条件（只改条件，不动 mu/anchor/z）。
# 参数：输入 condition（[14,H,W,1] 真条件）、donor_condition（[14,H,W,1] 或 None）、variant；输出 (条件, 是否可用)。
def build_variant_condition(condition, donor_condition, variant):
    """Condition array of one variant (a copy; the input is never mutated)."""
    if variant == "correct":
        return condition.copy(), True
    if donor_condition is None:
        return None, False
    broken = condition.copy()
    if variant == "history_shuffled":
        broken[HISTORY_CHANNELS] = donor_condition[HISTORY_CHANNELS]
    elif variant == "same_region_other_date":
        broken = donor_condition.copy()
    elif variant == "season_shuffled":
        broken[SEASON_CHANNELS] = donor_condition[SEASON_CHANNELS]
    elif variant == "geo_shuffled":
        broken[GEO_CHANNELS] = donor_condition[GEO_CHANNELS]
    elif variant == "season_geo_shuffled":
        broken[SEASON_CHANNELS] = donor_condition[SEASON_CHANNELS]
        broken[GEO_CHANNELS] = donor_condition[GEO_CHANNELS]
    else:
        raise ValueError(f"unknown condition variant {variant!r}")
    return broken, True


# 用途：单 checkpoint 的分档去噪误差扫描（固定 z 与噪声，只改条件）。
# 参数：输入 validator、entry 列表、donors、sigmas、num_samples、device；输出 结果 dict。
def run_diagnostics(validator, entries, donors, sigmas, num_samples=None,
                    progress_every=4):
    """Masked denoising error per r_sigma and per condition variant."""
    wrapper = validator.model
    if not (
            hasattr(wrapper, "diffusion")
            and hasattr(wrapper, "transform_innovation")
            and hasattr(wrapper, "inverse_innovation")
        ):
        raise ValueError(
            "condition diagnostics require a frozen-mean centered "
            "diffusion checkpoint (mean_model + diffusion + innovation "
            "statistics)"
        )
    selected = entries[:num_samples] if num_samples else list(entries)
    accumulated = {
        str(sigma): {
            variant: {
                "sq_error_z": 0.0,
                "sq_error_residual": 0.0,
                "bias_z": 0.0,
                "abs_z": 0.0,
                "count": 0,
                "samples": 0,
            }
            for variant in VARIANTS
        }
        for sigma in sigmas
    }
    signal = {"sq_z": 0.0, "count": 0}
    skipped = {variant: 0 for variant in VARIANTS}
    started = time.time()

    def to_device(array):
        return torch.as_tensor(array)[None].to(validator.device).float()

    for position, entry in enumerate(selected):
        index = int(entry["dataset_index"])
        sample = validator.decoded_sample(index)
        condition = sample["condition"].numpy()
        target = sample["target"].numpy()
        mask = sample["target_mask"].numpy() > 0
        anchor = condition[T0_CHANNEL: T0_CHANNEL + 1]
        residual = target - anchor
        donor_record = donors[index]
        donors_conditions = {}
        for role in ("history", "season", "geo", "same_region"):
            donor = donor_record.get(role)
            donors_conditions[role] = (
                validator.decoded_sample(
                    int(donor["dataset_index"])
                )["condition"].numpy()
                if donor is not None else None
            )
        with torch.no_grad():
            condition_tensor = to_device(condition)
            residual_tensor = to_device(residual)
            mask_tensor = to_device(mask.astype(np.float32))
            mu = wrapper._frozen_mean_prediction(condition_tensor)
            innovation = residual_tensor - mu
            standardized = wrapper.transform_innovation(innovation)
            generator = torch.Generator(device="cpu").manual_seed(
                20260909 + index
            )
            for sigma in sigmas:
                noise = torch.randn(
                    standardized.shape,
                    generator=generator,
                    dtype=torch.float32,
                ).to(validator.device)
                noised = standardized + float(sigma) * noise
                signal["sq_z"] += float(
                    (standardized * mask_tensor).square().sum()
                )
                signal["count"] += int(mask_tensor.sum())
                for variant in VARIANTS:
                    role = {
                        "history_shuffled": "history",
                        "same_region_other_date": "same_region",
                        "season_shuffled": "season",
                        "geo_shuffled": "geo",
                        "season_geo_shuffled": "season",
                    }.get(variant)
                    damaged, usable = build_variant_condition(
                        condition,
                        donors_conditions.get(role) if role else None,
                        variant,
                    )
                    if not usable:
                        skipped[variant] += 1
                        continue
                    denoised = wrapper.diffusion.preconditioned_network_forward(
                        noised,
                        float(sigma),
                        to_device(damaged),
                    )
                    error_z = (denoised - standardized) * mask_tensor
                    error_residual = (
                        wrapper.inverse_innovation(denoised)
                        - wrapper.inverse_innovation(standardized)
                    ) * mask_tensor
                    bucket = accumulated[str(sigma)][variant]
                    bucket["sq_error_z"] += float(error_z.square().sum())
                    bucket["sq_error_residual"] += float(
                        error_residual.square().sum()
                    )
                    bucket["bias_z"] += float(error_z.sum())
                    bucket["abs_z"] += float(
                        (denoised * mask_tensor).abs().sum()
                    )
                    bucket["count"] += int(mask_tensor.sum())
                    bucket["samples"] += 1
        if progress_every and (
                (position + 1) % int(progress_every) == 0
                or position + 1 == len(selected)
            ):
            print(
                f"[diagnostics] {position + 1}/{len(selected)} samples "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    results = {"sigmas": [float(sigma) for sigma in sigmas], "by_sigma": {}}
    for sigma in sigmas:
        rows = {}
        for variant in VARIANTS:
            bucket = accumulated[str(sigma)][variant]
            count = max(bucket["count"], 1)
            rows[variant] = {
                "mse_z": bucket["sq_error_z"] / count,
                "mse_residual": bucket["sq_error_residual"] / count,
                "bias_z": bucket["bias_z"] / count,
                "mean_abs_denoised_z": bucket["abs_z"] / count,
                "pixels": bucket["count"],
                "samples": bucket["samples"],
            }
        reference = rows["correct"]["mse_z"]
        for variant, row in rows.items():
            row["mse_z_ratio_vs_correct"] = (
                row["mse_z"] / reference if reference > 0 else None
            )
        rows["_zero_predictor"] = {
            "mse_z": signal["sq_z"] / max(signal["count"], 1),
            "note": (
                "masked MSE of the trivial predictor z_hat=0 (the plan's "
                "high-noise near-zero-output check)"
            ),
        }
        results["by_sigma"][str(float(sigma))] = rows
    results["skipped_variant_samples"] = skipped
    results["variants"] = list(VARIANTS)
    return results


# 用途：写 Markdown 摘要（每个 checkpoint 一张变体 x sigma 的 mse_z 表）。
# 参数：输入 results_by_label、sigmas、output_path；输出 无。
def write_markdown(results_by_label, sigmas, output_path):
    lines = [
        "# A5-centered DiAFNO condition diagnostics (plan 7.3)",
        "",
        "Masked MSE of the standardized innovation ``z`` between the "
        "denoiser output and the true ``z``; the frozen mean, anchor, "
        "``z`` and the noise draw are identical across variants, so only "
        "the diffusion condition differs.  15 forecast leads, ocean "
        "pixels only.",
        "",
    ]
    for label, results in results_by_label.items():
        lines.append(f"## {label}")
        lines.append("")
        header = "| variant | " + " | ".join(
            f"r_sigma={float(sigma):g}" for sigma in sigmas
        ) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(sigmas) + 1))
        for variant in VARIANTS:
            cells = []
            for sigma in sigmas:
                row = results["by_sigma"][str(float(sigma))][variant]
                cells.append(f"{row['mse_z']:.4f}")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        zero_cells = [
            f"{results['by_sigma'][str(float(sigma))]['_zero_predictor']['mse_z']:.4f}"
            for sigma in sigmas
        ]
        lines.append("| z=0 predictor | " + " | ".join(zero_cells) + " |")
        lines.append("")
        lines.append("Ratio vs ``correct`` (mse_z):")
        lines.append("")
        lines.append(header)
        lines.append("|" + "---|" * (len(sigmas) + 1))
        for variant in VARIANTS:
            cells = []
            for sigma in sigmas:
                row = results["by_sigma"][str(float(sigma))][variant]
                ratio = row["mse_z_ratio_vs_correct"]
                cells.append("n/a" if ratio is None else f"{ratio:.3f}")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        lines.append("")
        skipped = results["skipped_variant_samples"]
        if any(value for value in skipped.values()):
            lines.append(
                "Skipped samples without a usable partner: "
                + ", ".join(
                    f"{name}={value}" for name, value in skipped.items()
                    if value
                )
            )
            lines.append("")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# 用途：解析 --checkpoint LABEL=PATH。
# 参数：输入 items（字符串列表）；输出 [(label, path)]。
def parse_checkpoints(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--checkpoint must be LABEL=PATH, got {item!r}"
            )
        label, path = item.split("=", 1)
        if not label or not path:
            raise ValueError(
                f"--checkpoint must be LABEL=PATH, got {item!r}"
            )
        parsed.append((label, path))
    return parsed


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-mean centered diffusion condition diagnostics "
            "(plan 7.3): per-r_sigma denoising error under history / "
            "date / geo condition variants"
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="LABEL=PATH (repeatable)",
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sigmas",
        default="0.1,0.3,1,2,5",
        help="comma-separated r_sigma buckets",
    )
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    checkpoints = parse_checkpoints(args.checkpoint)
    sigmas = [
        float(part) for part in str(args.sigmas).replace(" ", "").split(",")
        if part
    ]
    if not sigmas or any(sigma <= 0 for sigma in sigmas):
        raise ValueError("--sigmas must be positive")

    manifest_payload = load_sample_manifest(args.sample_manifest)
    entries = sorted(
        manifest_payload["entries"],
        key=lambda entry: (
            int(entry["compact_start"]), int(entry["spatial_index"])
        ),
    )
    donors = select_donors(entries)
    os.makedirs(args.output_dir, exist_ok=True)
    results_by_label = {}
    provenance = {
        "sample_manifest": os.path.abspath(args.sample_manifest),
        "sample_manifest_sha256": manifest_payload["manifest_sha256"],
        "data_manifest": os.path.abspath(args.data_manifest),
        "num_samples": int(args.num_samples),
        "sigmas": sigmas,
        "variants": list(VARIANTS),
        "checkpoints": {},
        "seed_rule": "torch.randn(generator=manual_seed(20260909 + index))",
        "fixed_quantities": (
            "frozen mean mu, anchor, standardized innovation z and the "
            "noise draw are identical across variants; only the "
            "diffusion condition changes"
        ),
    }
    for label, path in checkpoints:
        validator = ProtocolValidator(
            path, args.h5_path, args.data_manifest,
            torch.device(args.device), ensemble_members=1,
            sampling_steps=args.sampling_steps, s_churn=0.0,
            use_amp=not args.no_amp, split=manifest_payload["split"],
        )
        ensure_manifest_universe(
            manifest_payload, len(validator.dataset),
            split=manifest_payload["split"],
            label=f"{label} sample manifest",
        )
        results = run_diagnostics(
            validator, entries, donors, sigmas,
            num_samples=args.num_samples,
        )
        results_by_label[label] = results
        provenance["checkpoints"][label] = {
            "path": os.path.abspath(path),
            "dataset_len": len(validator.dataset),
            "split": validator.split,
        }
        with open(
                os.path.join(args.output_dir, f"diagnostics_{label}.json"),
                "w", encoding="utf-8",
            ) as file:
            json.dump(
                {"provenance": provenance, "results": results},
                file, ensure_ascii=False, indent=2,
            )
            file.write("\n")
        del validator
        torch.cuda.empty_cache()
    with open(
            os.path.join(args.output_dir, "diagnostics.json"),
            "w", encoding="utf-8",
        ) as file:
        json.dump(
            {"provenance": provenance, "results": results_by_label},
            file, ensure_ascii=False, indent=2,
        )
        file.write("\n")
    write_markdown(
        results_by_label, sigmas,
        os.path.join(args.output_dir, "DIAGNOSTICS.md"),
    )
    print(
        json.dumps(
            {
                label: {
                    "correct_mse_z": {
                        sigma: results["by_sigma"][sigma]["correct"]["mse_z"]
                        for sigma in results["by_sigma"]
                    }
                }
                for label, results in results_by_label.items()
            },
            ensure_ascii=False, indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
