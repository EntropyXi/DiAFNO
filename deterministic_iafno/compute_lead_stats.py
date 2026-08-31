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
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dataset = OSTIADailyDataset(
        h5_path=args.h5_path,
        split="train",
        input_days=args.input_days,
        output_days=args.output_days,
        condition_mode="sst_mask",
    )
    indices = build_indices(len(dataset), args.num_samples)
    accumulator = LeadStatsAccumulator(args.output_days)

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start + args.batch_size]
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
            args.input_days - 1:args.input_days,
        ]
        residual = target - anchor
        accumulator.update(residual, target_mask)

    result = accumulator.compute()
    result.update(
        {
            "schema_version": 1,
            "target_space": "normalized_residual",
            "split": "train",
            "selection": "evenly_spaced_dataset_indices",
            "num_samples": int(len(indices)),
            "dataset_size": int(len(dataset)),
            "input_days": args.input_days,
            "output_days": args.output_days,
            "sst_mean": dataset.sst_mean,
            "sst_std": dataset.sst_std,
            "h5_path": os.path.abspath(args.h5_path),
        }
    )
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
