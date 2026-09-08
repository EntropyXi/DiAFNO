#!/usr/bin/env python
# 用途：按冻结规则抽取并保存某 split 的物理样本清单（服务器只读数据上运行）。
"""Freeze the val-200 / test-200 physical sample manifests.

Selection rule (plans/A5_LONGTRAIN_TEST200_20260908.md):

``np.sort(np.random.default_rng(123).choice(dataset_size, 200,
replace=False))`` over the manifest-filtered dataset of the requested
split, then record the physical identity of every chosen dataset index.
The output file is a validated ``sample_manifest`` payload carrying its
own ``manifest_sha256``.

The selection never filters by error, ocean fraction or any method's
behaviour, and the frozen files are read-only inputs for later runs.
"""

import argparse
import json
import os
import sys

import numpy as np

from diafno.data.ostia import OSTIADailyDataset
from diafno.evaluation.sample_manifest import (
    build_sample_manifest_payload,
    load_sample_manifest,
)


def freeze_manifest(
        h5_path,
        data_manifest,
        split,
        count,
        seed,
        output_path,
    ):
    dataset = OSTIADailyDataset(
        h5_path=h5_path,
        split=split,
        input_days=7,
        output_days=15,
        condition_mode="sst_mask_geo_season",
        data_manifest=data_manifest,
    )
    dataset_size = len(dataset)
    if count > dataset_size:
        raise ValueError(
            f"cannot freeze {count} samples from a dataset of size "
            f"{dataset_size}"
        )
    generator = np.random.default_rng(seed)
    indices = np.sort(
        generator.choice(
            dataset_size,
            size=count,
            replace=False,
        )
    ).tolist()
    spatial_entries = {}
    for dataset_index in indices:
        sequence_index = dataset_index // dataset.samples_per_day
        spatial_index = dataset_index % dataset.samples_per_day
        spatial_entries[dataset_index] = {
            "compact_start": int(
                dataset.valid_start_days[int(sequence_index)]
            ),
            "spatial_index": int(spatial_index),
        }
    payload = build_sample_manifest_payload(
        dataset,
        indices,
        split=split,
        spatial_entries=spatial_entries,
        dataset_size_geo=dataset_size,
        legacy_split_start=dataset.split_start_day,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a physical val/test sample manifest for the A5 "
            "long-run (read-only over the server HDF5 + data manifest)"
        )
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    payload = freeze_manifest(
        args.h5_path,
        args.data_manifest,
        args.split,
        args.count,
        args.seed,
        args.output,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    # Re-read from disk and validate before declaring success.
    load_sample_manifest(output_path)
    print(
        json.dumps(
            {
                "split": payload["split"],
                "count": payload["count"],
                "manifest_sha256": payload["manifest_sha256"],
                "output": output_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
