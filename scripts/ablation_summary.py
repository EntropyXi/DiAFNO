"""Final ablation comparison summary generator (read-only).

Aggregates the Stage-2 quick screen (val-200), the Stage-3
500/1000/1500-step re-evaluations and the paired temporal-block
bootstrap of the final candidates into one JSON + Markdown table.

Usage (server):

    python scripts/ablation_summary.py \
        --root experiments/ostia_spatiotemporal_ablation \
        --out experiments/ostia_spatiotemporal_ablation/summary/final_summary.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Stage-2 configuration identities in display order.
CONFIG_IDS = (
    "A0_baseline_p8_b8_i2",
    "A1_geo_p8_b8_i2",
    "A2_geo_p4_b8_i2",
    "A3_geo_p4_b2_i2",
    "A4_geo_p4_b1_i2",
    "A5_geo_p4_best_i4",
)

STAGE3_TAGS = {
    "A4_geo_p4_b1_i2": "stage3",
    "A1_geo_p8_b8_i2": "stage3",
    "A3_geo_p4_b2_i2": "stage3",
    "A5_geo_p4_best_i4": "stage3_2gpu",
}


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _metric_row(payload):
    if payload is None:
        return None
    overall = payload.get("overall") or {}
    by_lead = payload.get("by_lead_day") or {}
    skill = payload.get("persistence_skill") or {}
    bootstrap = payload.get("paired_block_bootstrap")
    row = {
        "rmse": overall.get("rmse"),
        "mae": overall.get("mae"),
        "bias": overall.get("bias"),
        "correlation": overall.get("correlation"),
        "day1_rmse": (by_lead.get("1") or {}).get("rmse"),
        "day7_rmse": (by_lead.get("7") or {}).get("rmse"),
        "day15_rmse": (by_lead.get("15") or {}).get("rmse"),
        "skill_vs_persistence": skill.get("overall"),
        "num_samples": payload.get("num_samples"),
        "seed": payload.get("seed"),
    }
    if bootstrap is not None:
        overall = bootstrap.get("overall") or {}
        skill_ci = overall.get("mse_skill_ci")
        rmse_diff_ci = overall.get("rmse_difference_ci")
        row["bootstrap_skill_95ci"] = {
            "mean": overall.get("mse_skill"),
            "ci_low": skill_ci[0] if skill_ci else None,
            "ci_high": skill_ci[1] if skill_ci else None,
            "rmse_difference": overall.get("rmse_difference"),
            "rmse_difference_ci": rmse_diff_ci,
            "fraction_model_better": overall.get(
                "bootstrap_fraction_model_better"
            ),
        }
    return row


def build_summary(root, bootstrap_paths=None):
    root = os.path.abspath(root)
    bootstrap_paths = bootstrap_paths or {}
    stage2 = {}
    for config_id in CONFIG_IDS:
        path = os.path.join(
            root,
            config_id,
            "stage2",
            "val_200_run.json",
        )
        if config_id == "A5_geo_p4_best_i4":
            path = os.path.join(
                root,
                config_id,
                "stage2_2gpu",
                "val_200_run.json",
            )
        stage2[config_id] = _metric_row(_load(path))
    stage3 = {}
    for config_id, tag in STAGE3_TAGS.items():
        entries = {}
        for step in (500, 1000, 1500):
            path = os.path.join(
                root,
                config_id,
                tag,
                f"val_200_step{step}.json",
            )
            entries[str(step)] = _metric_row(_load(path))
        stage3[config_id] = entries
    bootstrap = {}
    for config_id, path in (bootstrap_paths or {}).items():
        payload = _load(path)
        row = _metric_row(payload)
        bootstrap[config_id] = row
    now_utc = datetime.now(timezone.utc)
    summary = {
        "generated_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_beijing": (
            now_utc + timedelta(hours=8)
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "stage2_val200": stage2,
        "stage3_reevals": stage3,
        "bootstrap": bootstrap,
    }
    return summary


def _fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_markdown(summary):
    lines = []
    lines.append("# OSTIA spatiotemporal ablation summary")
    lines.append("")
    lines.append(
        f"Generated: {summary['generated_utc']} UTC / "
        f"{summary['generated_beijing']} 北京时间"
    )
    lines.append("")
    lines.append("## Stage 2 quick screen (val-200, seed=123, shared "
                "manifest)")
    lines.append("")
    lines.append(
        "| 配置 | RMSE | MAE | bias | corr | Day1 | Day7 | Day15 | "
        "skill |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for config_id in CONFIG_IDS:
        row = summary["stage2_val200"].get(config_id)
        if row is None:
            continue
        lines.append(
            f"| {config_id} | {_fmt(row['rmse'])} | "
            f"{_fmt(row['mae'])} | {_fmt(row['bias'])} | "
            f"{_fmt(row['correlation'])} | "
            f"{_fmt(row['day1_rmse'])} | {_fmt(row['day7_rmse'])} | "
            f"{_fmt(row['day15_rmse'])} | "
            f"{_fmt(row['skill_vs_persistence'])} |"
        )
    lines.append("")
    lines.append("## Stage 3 re-evaluations (same val-200 at "
                "500/1000/1500 optimizer steps)")
    lines.append("")
    for config_id, tag in STAGE3_TAGS.items():
        lines.append(f"### {config_id} ({tag})")
        lines.append("")
        lines.append("| step | RMSE | MAE | bias | corr | Day15 | skill |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        entries = summary["stage3_reevals"].get(config_id) or {}
        for step in ("500", "1000", "1500"):
            row = entries.get(step)
            if row is None:
                continue
            lines.append(
                f"| {step} | {_fmt(row['rmse'])} | {_fmt(row['mae'])} "
                f"| {_fmt(row['bias'])} | {_fmt(row['correlation'])} "
                f"| {_fmt(row['day15_rmse'])} | "
                f"{_fmt(row['skill_vs_persistence'])} |"
            )
        lines.append("")
    if summary["bootstrap"]:
        lines.append("## Paired temporal block bootstrap (95% CI)")
        lines.append("")
        for config_id, row in summary["bootstrap"].items():
            ci = row.get("bootstrap_skill_95ci")
            if isinstance(ci, dict):
                diff_ci = ci.get("rmse_difference_ci") or [None, None]
                lines.append(
                    f"- **{config_id}**: skill "
                    f"{_fmt(ci.get('mean'))}, 95% CI "
                    f"[{_fmt(ci.get('ci_low'))}, "
                    f"{_fmt(ci.get('ci_high'))}]; RMSE diff vs "
                    f"persistence {_fmt(ci.get('rmse_difference'))} "
                    f"[{_fmt(diff_ci[0])}, {_fmt(diff_ci[1])}], "
                    f"fraction better "
                    f"{_fmt(ci.get('fraction_model_better'))}"
                )
            else:
                lines.append(f"- **{config_id}**: {ci}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--bootstrap",
        action="append",
        default=[],
        help="CONFIG_ID=PATH pairs of bootstrap result JSONs",
    )
    args = parser.parse_args()
    bootstrap_paths = {}
    for item in args.bootstrap:
        config_id, path = item.split("=", 1)
        bootstrap_paths[config_id] = path
    summary = build_summary(args.root, bootstrap_paths)
    if os.path.exists(args.out):
        raise RuntimeError(
            f"refusing to overwrite existing summary: {args.out}"
        )
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    markdown = render_markdown(summary)
    print(markdown)
    markdown_path = args.out.rsplit(".", 1)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(markdown)
    print(f"summary written to {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
