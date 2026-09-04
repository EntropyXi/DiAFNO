import argparse
import json
import os

import numpy as np
import torch

from diafno.data.ostia import OSTIADailyDataset


class LeadStatsAccumulator:
    def __init__(self, leads):
        self.count = np.zeros(leads, dtype=np.float64)
        self.total = np.zeros(leads, dtype=np.float64)
        self.total_squared = np.zeros(leads, dtype=np.float64)

    def update(self, residual, mask):
        residual = np.asarray(residual, dtype=np.float64)
        mask = np.asarray(mask) > 0
        if residual.shape != mask.shape:
            raise ValueError("residual and mask shapes must match")
        if residual.ndim < 2:
            raise ValueError("expected [batch,lead,...] residuals")
        for lead in range(residual.shape[1]):
            values = residual[:, lead][mask[:, lead]]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            self.count[lead] += values.size
            self.total[lead] += values.sum()
            self.total_squared[lead] += np.square(values).sum()

    def compute(self):
        if np.any(self.count < 2):
            raise ValueError("each lead needs at least two valid values")
        mean = self.total / self.count
        variance = np.maximum(
            self.total_squared / self.count - np.square(mean),
            1e-12,
        )
        return {
            "lead_mean": mean.tolist(),
            "lead_std": np.sqrt(variance).tolist(),
            "valid_pixels": self.count.astype(np.int64).tolist(),
        }


def build_indices(dataset_size, num_samples):
    if num_samples is None or num_samples >= dataset_size:
        return np.arange(dataset_size, dtype=np.int64)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    return np.unique(np.linspace(
        0,
        dataset_size - 1,
        num_samples,
        dtype=np.int64,
    ))


def build_chunk_aware_indices(dataset, num_samples):
    """Select train samples in contiguous spatial blocks.

    The OSTIA HDF5 datasets are chunked along the flattened row axis.
    Sampling isolated ``sequence * samples_per_day + spatial`` rows
    causes a whole chunk to be read for nearly every single sample.
    Instead, distribute initialization dates across the train split and
    read one chunk-sized contiguous spatial block at each date.  The
    spatial start rotates between dates so all patch positions remain
    represented.
    """
    dataset_size = len(dataset)
    if num_samples is None or num_samples >= dataset_size:
        return np.arange(dataset_size, dtype=np.int64)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    spatial_block_size = min(
        int(getattr(dataset, "chunk_rows", 1)),
        int(dataset.samples_per_day),
        int(num_samples),
    )
    spatial_block_size = max(spatial_block_size, 1)
    block_count = int(np.ceil(num_samples / spatial_block_size))
    sequence_indices = np.linspace(
        0,
        dataset.sequences_per_window - 1,
        block_count,
        dtype=np.int64,
    )
    selected = []
    for block_index, sequence_index in enumerate(sequence_indices):
        remaining = num_samples - len(selected)
        if remaining <= 0:
            break
        current_size = min(spatial_block_size, remaining)
        spatial_start = (
            block_index * spatial_block_size
            % dataset.samples_per_day
        )
        spatial_indices = (
            spatial_start
            + np.arange(current_size, dtype=np.int64)
        ) % dataset.samples_per_day
        selected.extend(
            (
                int(sequence_index) * dataset.samples_per_day
                + spatial_indices
            ).tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def compute_lead_stats_file(
        h5_path,
        output,
        input_days=7,
        output_days=15,
        num_samples=4096,
        batch_size=32,
        condition_mode="sst_mask",
        data_manifest=None,
    ):
    """Compute train-only per-lead residual stats and write them.

    Shared by the CLI and the ablation runner so the deterministic
    lead-statistics protocol is identical everywhere.  The condition
    mode (and the real-day data manifest identity, when one is used)
    is recorded for provenance (geo-season runs must recompute their
    own stats under the same mode/manifest).
    """
    dataset = OSTIADailyDataset(
        h5_path=h5_path,
        split="train",
        input_days=input_days,
        output_days=output_days,
        condition_mode=condition_mode,
        data_manifest=data_manifest,
    )
    indices = build_chunk_aware_indices(dataset, num_samples)
    accumulator = LeadStatsAccumulator(output_days)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        samples = dataset.__getitems__(batch_indices.tolist())
        condition = torch.stack([
            sample["condition"] for sample in samples
        ]).numpy()
        target = torch.stack([
            sample["target"] for sample in samples
        ]).numpy()
        target_mask = torch.stack([
            sample["target_mask"] for sample in samples
        ]).numpy()
        anchor = condition[
            :,
            input_days - 1:input_days,
        ]
        residual = target - anchor
        accumulator.update(residual, target_mask)

    result = accumulator.compute()
    result.update(
        {
            "schema_version": 1,
            "target_space": "normalized_residual",
            "split": "train",
            "selection": (
                "evenly_spaced_sequence_spatial_chunk_blocks"
            ),
            "spatial_block_size": min(
                int(dataset.chunk_rows),
                int(dataset.samples_per_day),
                int(len(indices)),
            ),
            "num_samples": int(len(indices)),
            "dataset_size": int(len(dataset)),
            "input_days": input_days,
            "output_days": output_days,
            "condition_mode": condition_mode,
            "data_manifest_sha256": dataset.data_manifest_sha256,
            "day_offset_sha256": (
                (dataset.time_axis_summary or {}).get(
                    "day_offset_sha256"
                )
            ),
            "sst_mean": dataset.sst_mean,
            "sst_std": dataset.sst_std,
            "h5_path": os.path.abspath(h5_path),
        }
    )
    output_path = os.path.abspath(output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic train-only per-lead residual "
            "statistics for OSTIA"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-days", type=int, default=7)
    parser.add_argument("--output-days", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--condition-mode",
        default="sst_mask",
        help=(
            "condition contract the stats were computed under; "
            "recorded in the payload for provenance"
        ),
    )
    parser.add_argument(
        "--data-manifest",
        default=None,
        help=(
            "upstream data manifest (geo-season runs whose HDF5 "
            "lacks calendar metadata)"
        ),
    )
    args = parser.parse_args()
    result = compute_lead_stats_file(
        h5_path=args.h5_path,
        output=args.output,
        input_days=args.input_days,
        output_days=args.output_days,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        condition_mode=args.condition_mode,
        data_manifest=args.data_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
