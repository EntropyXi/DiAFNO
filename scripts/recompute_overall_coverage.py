#!/usr/bin/env python
# 用途：修正 metrics.json 的 overall coverage 聚合（除以样本数而非像素数）。
"""Recompute the pooled coverage rows of a protocol metrics.json.

The probability auxiliary coverage accumulates a *per-sample* mean per
lead, so the pooled value must divide by the number of (sample, lead)
contributions.  A buggy build divided by the pixel count, which turned
~0.82 into ~3e-06 while every other number stayed correct.

Every lead of a protocol run scores the same sample set, so the pooled
value is exactly the unweighted mean of the stored per-lead values:
``sum_lead cov[lead] / (num_leads * samples_per_lead)``.  This script
therefore repairs the affected fields without resampling anything, keeps
the original file as a backup and records what changed.

Usage::

    python scripts/recompute_overall_coverage.py \\
        --metrics experiments/.../test200/metrics.json
"""

import argparse
import json
import os
import shutil
import sys

COVERAGE_FIELDS = (
    "coverage_50",
    "coverage_90",
    "coverage_50_8members",
    "coverage_90_8members",
)


# 用途：用逐 lead coverage 重算 overall coverage（等样本数下与正确聚合完全等价）。
# 参数：输入 metrics（载荷）；输出 (修正后的载荷, 修改清单)。
def recompute_overall_coverage(metrics):
    changes = []
    for method, entry in metrics["methods"].items():
        auxiliary = entry.get("probability_auxiliary")
        if auxiliary is None:
            continue
        per_lead = entry.get("by_lead_day") or {}
        if not per_lead:
            raise ValueError(
                f"{method} has no by_lead_day rows to aggregate"
            )
        for field in COVERAGE_FIELDS:
            values = []
            for lead, row in per_lead.items():
                if field not in row:
                    raise ValueError(
                        f"{method} lead {lead} is missing {field}"
                    )
                values.append(float(row[field]))
            pooled = sum(values) / len(values)
            previous = auxiliary.get(field)
            if previous is None or abs(previous - pooled) > 1e-9:
                changes.append({
                    "method": method,
                    "field": field,
                    "before": previous,
                    "after": pooled,
                })
            auxiliary[field] = pooled
        auxiliary["coverage_units"] = (
            "mean per-sample pixel coverage, pooled over (sample, lead) "
            "pairs; recomputed from the per-lead rows (all leads score "
            "the same sample set, so the pooled value is their "
            "unweighted mean)"
        )
    provenance = metrics.setdefault("provenance", {})
    if changes:
        provenance["coverage_aggregation_correction"] = (
            "overall coverage fields were recomputed from the per-lead "
            "rows because the original build divided the per-sample "
            "coverage sum by the pixel count; no sampling was repeated "
            "and every other metric is untouched"
        )
    return metrics, changes


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description="Repair the pooled coverage fields of metrics.json"
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument(
        "--backup-suffix",
        default=".coverage_bug_backup",
        help="suffix for the untouched copy of the input file",
    )
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    corrected, changes = recompute_overall_coverage(metrics)
    if not changes:
        print("nothing to correct")
        return 0
    backup = args.metrics + args.backup_suffix
    if not os.path.isfile(backup):
        shutil.copyfile(args.metrics, backup)
    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(corrected, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(
        {"backup": backup, "changes": changes},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
