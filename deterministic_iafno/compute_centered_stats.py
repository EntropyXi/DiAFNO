# 用途：只用训练集计算相对冻结均值的逐 lead 创新统计量。
"""Compute train-only centered innovation statistics for the frozen mean.

Target space: normalized residual ``r = x_target - anchor`` (both
normalized SST), frozen deterministic mean ``mu(c)`` (already in
normalized residual space), centered innovation ``e = r - mu(c)``, and
per-lead standardization ``z = (e - m) / s``.

Only the train split is read.  The index selection is deterministic and
chunk-aware (same construction as ``compute_lead_stats``); the frozen
mean checkpoint identity, its sidecar semantics and its own lead stats
are validated fail-closed before any accumulation starts.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from deterministic_iafno.centered_stats import (
    CENTERED_TARGET_SPACE,
    LOCKED_MEAN_CHECKPOINT_SHA256,
    cross_check_mean_sidecar,
    indices_sha256,
    sha256_hex_file,
    sha256_of_normalized,
    validate_centered_stats_payload,
)
from deterministic_iafno.checkpoint_semantics import (
    load_semantic_sidecar,
)
from deterministic_iafno.compute_lead_stats import (
    LeadStatsAccumulator,
    build_chunk_aware_indices,
)
from diafno.data.ostia import OSTIADailyDataset
from diafno.models.config import OSTIAModelConfig


def load_frozen_mean(mean_checkpoint_path, device):
    """Load and validate the frozen deterministic mean checkpoint.

    Fails closed unless: the sidecar exists, its immutable semantics
    match the locked deterministic contract (cross-checked against the
    file SHA), the checkpoint's own config agrees with the sidecar on
    the architecture fields, and the weights load strictly.
    """
    mean_checkpoint_path = os.path.abspath(mean_checkpoint_path)
    if not os.path.isfile(mean_checkpoint_path):
        raise FileNotFoundError(mean_checkpoint_path)
    file_sha = sha256_hex_file(mean_checkpoint_path)
    if file_sha != LOCKED_MEAN_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen mean checkpoint SHA-256 mismatch: file "
            f"{file_sha} vs locked {LOCKED_MEAN_CHECKPOINT_SHA256}"
        )
    sidecar = load_semantic_sidecar(mean_checkpoint_path)
    if sidecar is None:
        raise ValueError(
            "frozen mean checkpoint has no semantic sidecar; "
            "centered stats fail closed without it"
        )
    manifest = sidecar.get("semantic_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "frozen mean sidecar has no semantic_manifest"
        )
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise ValueError(
            "frozen mean sidecar manifest has no immutable block"
        )
    stats_draft = {
        "mean_lead_mean": immutable["lead_mean"],
        "mean_lead_std": immutable["lead_std"],
        "mean_semantics_sha256": sha256_of_normalized(immutable),
    }
    cross_check_mean_sidecar(stats_draft, mean_checkpoint_path)
    sidecar_config = sidecar.get("config")
    if not isinstance(sidecar_config, dict):
        raise ValueError("frozen mean sidecar has no model config")
    model_config = OSTIAModelConfig.from_checkpoint(sidecar_config)
    if (
            model_config.model_type != "deterministic"
            or model_config.target_mode != "residual"
            or model_config.target_scaling != "lead_standardized"
        ):
        raise ValueError(
            "frozen mean is not a lead-standardized deterministic "
            "residual model: "
            f"model_type={model_config.model_type}, "
            f"target_mode={model_config.target_mode}, "
            f"target_scaling={model_config.target_scaling}"
        )
    checkpoint = torch.load(
        mean_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = OSTIAModelConfig.from_checkpoint(
        checkpoint.get("config", {})
    )
    arch_fields = (
        "input_days",
        "output_days",
        "cond_chans",
        "target_chans",
        "image_size",
        "patch_size",
        "embed_dim",
        "num_blocks",
        "explicit_layer",
        "implicit_layer",
        "hidden_size_factor",
    )
    for field in arch_fields:
        if getattr(checkpoint_config, field) != getattr(
                model_config, field
            ):
            raise ValueError(
                "frozen mean checkpoint config disagrees with its "
                f"sidecar on {field}"
            )
    model = model_config.build_model(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, immutable


def compute_centered_stats(
        h5_path,
        mean_checkpoint_path,
        num_samples,
        batch_size,
        input_days,
        output_days,
        device,
        use_amp,
    ):
    started = time.time()
    dataset = OSTIADailyDataset(
        h5_path=h5_path,
        split="train",
        input_days=input_days,
        output_days=output_days,
        condition_mode="sst_mask",
    )
    model, mean_immutable = load_frozen_mean(
        mean_checkpoint_path,
        device,
    )
    if (
            not hasattr(model, "lead_mean")
            or not hasattr(model, "lead_std")
        ):
        raise ValueError(
            "frozen mean model was not built with its own lead stats"
        )
    indices = build_chunk_aware_indices(dataset, num_samples)
    accumulator = LeadStatsAccumulator(output_days)
    model_device = next(model.parameters()).device

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        samples = dataset.__getitems__(batch_indices.tolist())
        condition = torch.stack([
            sample["condition"] for sample in samples
        ]).to(model_device)
        target = torch.stack([
            sample["target"] for sample in samples
        ]).to(model_device)
        target_mask = torch.stack([
            sample["target_mask"] for sample in samples
        ]).numpy()
        anchor = condition[
            :,
            input_days - 1:input_days,
        ]
        residual = target - anchor
        with torch.no_grad(), torch.autocast(
                device_type=model_device.type,
                enabled=use_amp,
            ):
            mu = model.predict(condition)
        innovation = residual.float() - mu.float()
        accumulator.update(
            innovation.float().cpu().numpy(),
            target_mask,
        )

    stats = accumulator.compute()
    pooled_variance = (
        np.sum(
            np.asarray(stats["lead_std"]) ** 2
            * np.asarray(stats["valid_pixels"])
        )
        / max(np.sum(stats["valid_pixels"]), 1)
    )
    payload = {
        "schema_version": 1,
        "split": "train",
        "target_space": CENTERED_TARGET_SPACE,
        "input_days": input_days,
        "output_days": output_days,
        "condition_mode": "sst_mask",
        "num_samples": int(len(indices)),
        "dataset_size": int(len(dataset)),
        "selection": (
            "evenly_spaced_sequence_spatial_chunk_blocks"
        ),
        "selection_description": (
            "deterministic chunk-aware selection: evenly spaced "
            "initialization dates across the train split, each reading "
            "one contiguous spatial block of chunk_rows samples; "
            "spatial start rotates between dates; no random seed"
        ),
        "indices_sha256": indices_sha256(indices),
        "mean_checkpoint": os.path.abspath(mean_checkpoint_path),
        "mean_checkpoint_sha256": LOCKED_MEAN_CHECKPOINT_SHA256,
        "mean_semantics_sha256": sha256_of_normalized(mean_immutable),
        "mean_lead_mean": list(mean_immutable["lead_mean"]),
        "mean_lead_std": list(mean_immutable["lead_std"]),
        "sst_mean": dataset.sst_mean,
        "sst_std": dataset.sst_std,
        "lead_mean": stats["lead_mean"],
        "lead_std": stats["lead_std"],
        "overall_innovation_std": float(
            np.sqrt(max(pooled_variance, 0.0))
        ),
        "valid_pixels": stats["valid_pixels"],
        "h5_path": os.path.abspath(h5_path),
    }
    # Self-check the payload with the shared validator before writing.
    validate_centered_stats_payload(
        payload,
        target_chans=output_days,
        input_days=input_days,
        output_days=output_days,
    )
    elapsed_seconds = time.time() - started
    return payload, elapsed_seconds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-only centered innovation statistics for "
            "the frozen deterministic mean"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--mean-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-days", type=int, default=7)
    parser.add_argument("--output-days", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to cuda when available",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="inference AMP for the frozen mean (accumulators stay float64)",
    )
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=None)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = (
        args.use_amp
        if args.use_amp is not None
        else (device.type == "cuda")
    )
    payload, elapsed_seconds = compute_centered_stats(
        h5_path=args.h5_path,
        mean_checkpoint_path=args.mean_checkpoint,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        input_days=args.input_days,
        output_days=args.output_days,
        device=device,
        use_amp=use_amp,
    )
    # Timing is reported on stderr only: the JSON must stay
    # byte-identical across identical invocations.
    print(
        f"centered stats elapsed: {elapsed_seconds:.1f}s "
        f"({args.num_samples} samples)",
        file=__import__("sys").stderr,
    )
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
