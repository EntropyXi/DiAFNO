#!/usr/bin/env python
# 用途：四方法（A5/旧IAFNO/旧centered DiAFNO/persistence）统一配对 test-200 评估。
"""Unified four-method paired test evaluation (A5 plan section 6).

Methods keep their own training/condition contracts:
  A5                -- sst_mask_geo_season deterministic, single output
  old IAFNO         -- legacy sst_mask deterministic, single output
  old centered DiAFNO -- legacy centered diffusion, 16 members, 16
                        steps, S_churn=0 (AMP)
  persistence       -- no checkpoint; day-7 SST repeated

All four methods are evaluated on the SAME physical samples taken from
a frozen sample manifest (physical pairing keys: real calendar dates +
spatial index).  Real-time blocks use manifest day offsets with the
fixed rule of A5 plan 6.2; every output needed to recompute the tables
is saved (paired_contributions.npz plus the manifest).

This script intentionally does NOT reuse the three-method entry point:
the four method slots, per-method source contracts and the frozen
sample manifest are a separate protocol.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from diafno.data.condition_schema import resolve_condition_mode
from diafno.data.ostia import OSTIADailyDataset, verify_checkpoint_data_contract
from diafno.evaluation.metrics import RunningSSTMetrics
from diafno.evaluation.method_comparison import empirical_crps
from diafno.evaluation.sample_manifest import (
    load_sample_manifest,
    manifest_dataset_indices,
)
from diafno.inference.model import InferenceModelLoader

# Fixed legacy identities (A5 plan 6.1); verified before any sampling.
OLD_IAFNO_SHA256 = "cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6"
OLD_DIAFNO_SHA256 = "4d62f6c250aa4ebb19f68920398660b824bfe8963d3f39c969104a026be30291"
LEADS = (1, 5, 10, 15)
HORIZON = 15


def _file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256_must(path, expected, label):
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: file {path} has {actual}, "
            f"expected {expected}"
        )


class ProtocolValidator:
    """Minimal per-model sampling adapter over one physical sample.

    Mirrors OSTIAValidator's prediction math (residual re-anchor once,
    deterministic predict or member sampling) but is driven by explicit
    dataset indices from the frozen sample manifest.
    """

    def __init__(self, checkpoint_path, h5_path, data_manifest,
                 device, ensemble_members=1, sampling_steps=16,
                 s_churn=None, use_amp=True):
        (
            self.model,
            self.model_config,
            self.sampling_steps,
            self.normalization,
        ) = InferenceModelLoader.load(
            checkpoint_path,
            device,
            sampling_steps=sampling_steps,
        )
        if s_churn is not None:
            if not hasattr(self.model, "S_churn"):
                raise ValueError(
                    "--s-churn only applies to diffusion checkpoints"
                )
            self.model.S_churn = s_churn
        self.model.eval()
        self.ensemble_members = int(ensemble_members)
        self.use_amp = bool(use_amp)
        condition_mode = resolve_condition_mode(
            None,
            self.model_config.condition_mode,
            "four-method evaluation",
        )
        # Each method keeps its own source contract (plan 6.1): a
        # checkpoint bound to an upstream data manifest uses the shared
        # gap-filtered universe, while a legacy manifest-less checkpoint
        # keeps the raw compact-time universe (no gap filtering).  The
        # manifest is therefore only bound when the checkpoint itself
        # declares one.
        checkpoint_manifest = getattr(
            self.model_config, "data_manifest_sha256", None
        )
        self.uses_manifest = checkpoint_manifest is not None
        self.dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="test",
            input_days=self.model_config.input_days,
            output_days=self.model_config.output_days,
            condition_mode=condition_mode,
            data_manifest=data_manifest if self.uses_manifest else None,
        )
        verify_checkpoint_data_contract(
            self.dataset,
            self.model_config,
        )
        self.device = device

    # 用途：返回该模型在自己的数据宇宙中定位同一条物理样本的索引。
    # 参数：输入 entry（冻结清单条目）；输出 int 索引。
    def sample_index(self, entry):
        """Address one frozen physical sample in this method's universe.

        Manifest-bound methods use the gap-filtered ``dataset_index``;
        legacy (manifest-less) methods use ``legacy_dataset_index`` so
        the same compact window + spatial patch maps to the same sample
        without changing the gap-filtered universe (plan 6.1).
        """
        if self.uses_manifest:
            return int(entry["dataset_index"])
        return int(entry["legacy_dataset_index"])

    def sample_at(self, dataset_index, seed_base):
        sample = self.dataset[int(dataset_index)]
        condition = sample["condition"][None].to(self.device).float()
        with torch.no_grad(), autocast(
                "cuda",
                enabled=self.use_amp and self.device.type == "cuda",
            ):
            if self.model_config.model_type == "deterministic":
                if self.ensemble_members != 1:
                    raise ValueError(
                        "deterministic methods require one member"
                    )
                prediction = self.model.predict(condition)
            else:
                members = []
                for member in range(self.ensemble_members):
                    seed = (
                        seed_base + member
                        if seed_base is not None
                        else None
                    )
                    members.append(
                        self.model.sample(
                            condition=condition,
                            num_sample_steps=self.sampling_steps,
                            seed=seed,
                        )
                    )
                prediction = torch.stack(
                    members,
                    dim=0,
                ).mean(dim=0)
        if self.model_config.target_mode == "residual":
            anchor = condition[
                :,
                self.model_config.input_days - 1:
                self.model_config.input_days
            ]
            prediction = prediction + anchor
        prediction = prediction[0]
        target = sample["target"]
        mask = sample["target_mask"]
        return (
            prediction.float().cpu().numpy()[..., 0],
            target.numpy()[..., 0],
            mask.numpy()[..., 0],
            self.dataset.sst_mean,
            self.dataset.sst_std,
        )


# 用途：对任意方法对的样本级累计量做真实日块 bootstrap（ΔRMSE/ΔCRPS CI）。
# 参数：输入 各方法逐样本逐 lead 累计与时间；输出 dict。
def delta_block_bootstrap(
        sse_a, sse_b, crps_a, crps_b, counts,
        times, origin, block_days=22, replicates=2000, seed=123,
    ):
    """Block bootstrap of pooled ΔRMSE and ΔCRPS between two methods."""
    sse_a = np.asarray(sse_a, dtype=np.float64)
    sse_b = np.asarray(sse_b, dtype=np.float64)
    crps_a = np.asarray(crps_a, dtype=np.float64)
    crps_b = np.asarray(crps_b, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    times = np.asarray(times, dtype=np.int64)
    blocks, inverse = np.unique(
        (times - origin) // block_days,
        return_inverse=True,
    )
    num_blocks = len(blocks)
    result = {"num_blocks": num_blocks}
    if num_blocks < 2 or replicates == 0:
        result["reason"] = "disabled or fewer than two temporal blocks"
        result["interval"] = None
        return result
    block_sse_a = np.zeros((num_blocks, sse_a.shape[1]))
    block_sse_b = np.zeros((num_blocks, sse_b.shape[1]))
    block_crps_a = np.zeros((num_blocks, crps_a.shape[1]))
    block_crps_b = np.zeros((num_blocks, crps_b.shape[1]))
    block_counts = np.zeros((num_blocks, counts.shape[1]))
    np.add.at(block_sse_a, inverse, sse_a)
    np.add.at(block_sse_b, inverse, sse_b)
    np.add.at(block_crps_a, inverse, crps_a)
    np.add.at(block_crps_b, inverse, crps_b)
    np.add.at(block_counts, inverse, counts)
    overall_sse_a = block_sse_a.sum(axis=1)
    overall_sse_b = block_sse_b.sum(axis=1)
    overall_crps_a = block_crps_a.sum(axis=1)
    overall_crps_b = block_crps_b.sum(axis=1)
    overall_counts = block_counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    rmse_diffs = []
    crps_diffs = []
    for _ in range(replicates):
        picked = rng.integers(0, num_blocks, size=num_blocks)
        n = overall_counts[picked].sum()
        if n <= 0:
            continue
        mse_a = overall_sse_a[picked].sum() / n
        mse_b = overall_sse_b[picked].sum() / n
        rmse_diffs.append(np.sqrt(mse_a) - np.sqrt(mse_b))
        crps_diffs.append(
            overall_crps_a[picked].sum() / n
            - overall_crps_b[picked].sum() / n
        )
    result["rmse_difference"] = float(
        np.sqrt(overall_sse_a.sum() / overall_counts.sum())
        - np.sqrt(overall_sse_b.sum() / overall_counts.sum())
    )
    result["rmse_difference_ci"] = np.quantile(
        rmse_diffs, [0.025, 0.975]
    ).tolist()
    result["crps_difference"] = float(
        overall_crps_a.sum() / overall_counts.sum()
        - overall_crps_b.sum() / overall_counts.sum()
    )
    result["crps_difference_ci"] = np.quantile(
        crps_diffs, [0.025, 0.975]
    ).tolist()
    return result


# 用途：校验四方法主要身份与参数。
# 参数：输入 args；输出 无。
def validate_protocol_args(args):
    _checkpoint_sha256_must(
        args.old_iafno_checkpoint,
        OLD_IAFNO_SHA256,
        "old IAFNO checkpoint",
    )
    _checkpoint_sha256_must(
        args.old_diafno_checkpoint,
        OLD_DIAFNO_SHA256,
        "old centered DiAFNO checkpoint",
    )
    if args.ensemble_members < 2 or args.ensemble_members > 1000:
        raise ValueError("--ensemble-members must be within [2, 1000]")
    if args.sampling_steps < 1:
        raise ValueError("--sampling-steps must be positive")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Four-method paired test evaluation (A5 plan section 6): "
            "A5 / old IAFNO / old centered DiAFNO / persistence on a "
            "frozen physical sample manifest."
        )
    )
    parser.add_argument("--a5-checkpoint", required=True)
    parser.add_argument("--old-iafno-checkpoint", required=True)
    parser.add_argument("--old-diafno-checkpoint", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ensemble-members", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=22)
    parser.add_argument("--sst-unit", default="K")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    validate_protocol_args(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "refusing to write into a non-empty output dir; the four-"
            "method run must start into an empty directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    manifest_payload = load_sample_manifest(args.sample_manifest)
    indices = manifest_dataset_indices(manifest_payload)
    entries_by_index = {
        int(entry["dataset_index"]): entry
        for entry in manifest_payload["entries"]
    }

    a5 = ProtocolValidator(
        args.a5_checkpoint, args.h5_path, args.data_manifest,
        device, ensemble_members=1,
        use_amp=not args.no_amp,
    )
    old_iafno = ProtocolValidator(
        args.old_iafno_checkpoint, args.h5_path, args.data_manifest,
        device, ensemble_members=1,
        use_amp=not args.no_amp,
    )
    old_diafno = ProtocolValidator(
        args.old_diafno_checkpoint, args.h5_path, args.data_manifest,
        device, ensemble_members=args.ensemble_members,
        sampling_steps=args.sampling_steps,
        s_churn=args.s_churn,
        use_amp=not args.no_amp,
    )
    for validator in (a5, old_iafno, old_diafno):
        if len(validator.dataset) != len(a5.dataset):
            raise ValueError(
                "method datasets disagree in length; the frozen sample "
                "manifest cannot be shared across these checkpoints"
            )

    if manifest_payload["split"] != "test":
        raise ValueError(
            "the four-method protocol runs on the frozen test sample "
            "manifest (split='test')"
        )

    def to_kelvin(value, mean, std):
        return value * std + mean

    # Per-sample accumulators (overall + per lead) for the report and
    # for the recomputable contribution file.
    contributions = {
        "times_real_t0": [],
        "dataset_index": [],
        "legacy_dataset_index": [],
        "block_id": [],
    }
    methods = ("A5", "old_IAFNO", "old_DiAFNO", "persistence")
    horizon = HORIZON
    sums = {
        method: {
            "counts": np.zeros(horizon, dtype=np.int64),
            "sse": np.zeros(horizon, dtype=np.float64),
            "abs": np.zeros(horizon, dtype=np.float64),
            "err": np.zeros(horizon, dtype=np.float64),
            "pred": np.zeros(horizon, dtype=np.float64),
            "pred2": np.zeros(horizon, dtype=np.float64),
            "prod": np.zeros(horizon, dtype=np.float64),
            "tgt": np.zeros(horizon, dtype=np.float64),
            "tgt2": np.zeros(horizon, dtype=np.float64),
            "crps": np.zeros(horizon, dtype=np.float64),
        }
        for method in methods
    }
    per_sample = {key: [] for key in ("counts",)}
    per_sample["sse"] = {m: [] for m in methods}
    per_sample["crps"] = {m: [] for m in methods}

    # Bootstrap pairing times use the real-day rule of plan 6.2:
    # real_t0 = day_offsets[compact window start + 6].
    real_offsets = a5.dataset.real_day_offsets
    if real_offsets is None:
        raise ValueError("four-method protocol requires a data manifest")
    origin = int(real_offsets[
        a5.dataset.split_start_day + 6
    ])

    for position, dataset_index in enumerate(indices):
        entry = entries_by_index[dataset_index]
        legacy_index = int(entry["legacy_dataset_index"])
        # Same physical sample across the three loaded models; each
        # method addresses it in its own universe (plan 6.1): the
        # manifest-bound A5 uses the gap-filtered dataset_index, the
        # legacy methods use the unfiltered legacy_dataset_index.
        (pred_a5, target_a5, mask_a5, mean_a5, std_a5) = a5.sample_at(
            a5.sample_index(entry), seed_base=None
        )
        (pred_iafno, target_iafno, mask_iafno, mean_iafno,
         std_iafno) = old_iafno.sample_at(
            old_iafno.sample_index(entry), seed_base=None
        )
        # DiAFNO member seeding follows the legacy rule
        # 123 + legacy_dataset_index*1000 + member (plan 6.1).
        di_members = []
        for member in range(args.ensemble_members):
            seed = (
                args.seed
                + legacy_index * 1000
                + member
            )
            (pred_member, _, _, _, _) = old_diafno.sample_at(
                old_diafno.sample_index(entry), seed_base=seed
            )
            di_members.append(pred_member)
        pred_diafno = np.mean(di_members, axis=0)

        if (
                not np.allclose(target_a5, target_iafno, atol=1e-5)
                or not np.array_equal(mask_a5, mask_iafno)
            ):
            raise ValueError(
                "physical target/mask mismatch between methods at "
                f"dataset_index {dataset_index}"
            )
        target = target_a5
        mask = mask_a5 > 0
        anchor = a5.dataset[int(dataset_index)]["condition"][
            6, :, :, 0
        ].numpy()
        anchor_kelvin = anchor * std_a5 + mean_a5
        pers_kelvin = np.repeat(
            anchor_kelvin[None, :, :], horizon, axis=0
        )
        mean_std = (mean_a5, std_a5)
        preds = {
            "A5": to_kelvin(pred_a5, *mean_std),
            "old_IAFNO": to_kelvin(pred_iafno, *mean_std),
            "old_DiAFNO": to_kelvin(pred_diafno, *mean_std),
            "persistence": pers_kelvin,
        }
        if not np.isfinite(target).all() or not np.isfinite(mask).all():
            raise ValueError("non-finite target or mask")
        for method in methods:
            values = preds[method]
            if values.shape != target.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid prediction: {method}")
        # Real t0 = day_offsets[compact window start + 6] (plan 6.2).
        real_t0 = int(
            real_offsets[int(entry["compact_start"]) + 6]
        )
        block_id = int((real_t0 - origin) // args.block_days)
        contributions["times_real_t0"].append(real_t0)
        contributions["dataset_index"].append(dataset_index)
        contributions["legacy_dataset_index"].append(legacy_index)
        contributions["block_id"].append(block_id)

        sample_sse = {method: [] for method in methods}
        sample_crps = {method: [] for method in methods}
        sample_counts = []
        # Deterministic member stack: a single member equals the point
        # forecast; CRPS then collapses to MAE by construction.
        di_members_kelvin = np.asarray([
            to_kelvin(member, *mean_std)
            for member in di_members
        ])
        for lead in range(horizon):
            valid = mask[lead]
            if not valid.any():
                raise ValueError("empty valid pixels per sample")
            t = (
                target[lead][valid] * std_a5 + mean_a5
            ).astype(np.float64)
            for method in methods:
                p = preds[method][lead][valid].astype(np.float64)
                e = p - t
                sums[method]["counts"][lead] += int(valid.sum())
                sums[method]["sse"][lead] += float(np.square(e).sum())
                sums[method]["abs"][lead] += float(np.abs(e).sum())
                sums[method]["err"][lead] += float(e.sum())
                sums[method]["pred"][lead] += float(p.sum())
                sums[method]["pred2"][lead] += float(np.square(p).sum())
                sums[method]["prod"][lead] += float((p * t).sum())
                sums[method]["tgt"][lead] += float(t.sum())
                sums[method]["tgt2"][lead] += float(np.square(t).sum())
                sample_sse[method].append(float(np.square(e).sum()))
            sample_counts.append(int(valid.sum()))
            for method in methods:
                if method == "old_DiAFNO":
                    members = di_members_kelvin[:, lead][:, valid]
                else:
                    members = preds[method][lead][valid][None]
                crps_lead = empirical_crps(
                    np.asarray(members, dtype=np.float64),
                    t,
                )
                value = float(crps_lead.sum())
                sums[method]["crps"][lead] += value
                sample_crps[method].append(value)
        per_sample["counts"].append(sample_counts)
        for method in methods:
            per_sample["sse"][method].append(sample_sse[method])
            per_sample["crps"][method].append(sample_crps[method])

    np.savez(
        output_dir / "paired_contributions.npz",
        **contributions,
        per_sample_sse_a5=np.asarray(per_sample["sse"]["A5"]),
        per_sample_sse_old_iafno=np.asarray(per_sample["sse"]["old_IAFNO"]),
        per_sample_sse_old_diafno=np.asarray(per_sample["sse"]["old_DiAFNO"]),
        per_sample_sse_persistence=np.asarray(per_sample["sse"]["persistence"]),
        per_sample_crps_a5=np.asarray(per_sample["crps"]["A5"]),
        per_sample_crps_old_iafno=np.asarray(per_sample["crps"]["old_IAFNO"]),
        per_sample_crps_old_diafno=np.asarray(per_sample["crps"]["old_DiAFNO"]),
        per_sample_crps_persistence=np.asarray(per_sample["crps"]["persistence"]),
        per_sample_counts=np.asarray(per_sample["counts"]),
    )

    report = {
        "provenance": {
            "split": "test",
            "sample_manifest": args.sample_manifest,
            "sample_manifest_sha256": manifest_payload["manifest_sha256"],
            "a5_checkpoint": args.a5_checkpoint,
            "old_iafno_checkpoint": args.old_iafno_checkpoint,
            "old_diafno_checkpoint": args.old_diafno_checkpoint,
            "ensemble_members": args.ensemble_members,
            "sampling_steps": args.sampling_steps,
            "s_churn": args.s_churn,
            "seed": args.seed,
            "block_days": args.block_days,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "num_samples": len(indices),
    }
    report["methods"] = {}
    for method in methods:
        s = sums[method]
        count = s["counts"].sum()
        report["methods"][method] = {
            "overall": {
                "rmse": float(np.sqrt(s["sse"].sum() / count)),
                "mse": float(s["sse"].sum() / count),
                "mae": float(s["abs"].sum() / count),
                "bias": float(s["err"].sum() / count),
                "correlation": float((
                    s["prod"].sum()
                    - s["pred"].sum() * s["tgt"].sum() / count
                ) / np.sqrt(
                    (s["pred2"].sum() - s["pred"].sum() ** 2 / count)
                    * (s["tgt2"].sum() - s["tgt"].sum() ** 2 / count)
                )),
                "crps": float(s["crps"].sum() / count),
            },
            "by_lead_day": {},
        }
        for lead in range(horizon):
            n = s["counts"][lead]
            mse = s["sse"][lead] / n
            denom = np.sqrt(
                (s["pred2"][lead] - s["pred"][lead] ** 2 / n)
                * (s["tgt2"][lead] - s["tgt"][lead] ** 2 / n)
            )
            report["methods"][method]["by_lead_day"][str(lead + 1)] = {
                "rmse": float(np.sqrt(mse)),
                "mse": float(mse),
                "mae": float(s["abs"][lead] / n),
                "bias": float(s["err"][lead] / n),
                "correlation": float(
                    (s["prod"][lead] - s["pred"][lead] * s["tgt"][lead] / n)
                    / denom
                ) if denom > 0 else None,
                "crps": float(s["crps"][lead] / n),
            }
    # Direct paired ΔCI: A5 minus each other method (overall pooled).
    for other in ("old_IAFNO", "old_DiAFNO", "persistence"):
        report[f"delta_A5_minus_{other}"] = delta_block_bootstrap(
            np.asarray(per_sample["sse"]["A5"]),
            np.asarray(per_sample["sse"][other]),
            np.asarray(per_sample["crps"]["A5"]),
            np.asarray(per_sample["crps"][other]),
            np.asarray(per_sample["counts"]),
            np.asarray(contributions["times_real_t0"]),
            origin=origin,
            block_days=args.block_days,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(output_dir / "evaluation_manifest.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "protocol": report["provenance"],
                "num_samples": report["num_samples"],
                "outputs": [
                    "metrics.json",
                    "paired_contributions.npz",
                ],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
