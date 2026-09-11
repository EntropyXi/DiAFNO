#!/usr/bin/env python
# 用途：把统一协议产物（metrics/bootstrap/figures）渲染为 Markdown 报告（计划 §9）。
"""Render the unified-protocol artifacts into REPORT.md (plan section 9).

Inputs are the frozen artifacts of one protocol run: ``metrics.json``
(per-method overall and per-lead metrics plus the probability auxiliary
rows), ``bootstrap.json`` (paired real-day block bootstrap deltas) and,
optionally, ``figures/figures.json``.  The report states skill against
persistence, the paired deltas with their 95% intervals, the
probabilistic auxiliary rows, the protocol provenance and the honest
limitations; nothing is recomputed from a different sample set.

Usage::

    python scripts/report_ostia_protocol.py \
        --metrics experiments/.../test200/metrics.json \
        --bootstrap experiments/.../test200/bootstrap.json \
        --figures experiments/.../test200/figures/figures.json \
        --output experiments/.../test200/REPORT.md
"""

import argparse
import json
import os
import sys

LEAD_ROWS = ("1", "5", "10", "15")


# 用途：技能分（1 - metric/metric_persistence）。
# 参数：输入 value、reference；输出 float 或 None。
def skill(value, reference):
    if reference is None or reference == 0:
        return None
    return 1.0 - float(value) / float(reference)


# 用途：格式化数值 / 百分比 / 区间。
# 参数：输入 value；输出 str。
def fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def fmt_percent(value):
    return "—" if value is None else f"{float(value):+.2%}"


def fmt_interval(entry):
    interval = (entry or {}).get("interval")
    if not interval:
        return "—"
    return f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"


# 用途：渲染报告正文。
# 参数：输入 metrics、bootstrap（可空）、figures（可空）、title；输出 Markdown 字符串。
def build_report(metrics, bootstrap=None, figures=None,
                 title="A5-centered DiAFNO 统一 test-200 协议结果"):
    provenance = metrics["provenance"]
    methods = list(metrics["methods"])
    persistence = metrics["methods"].get("persistence", {})
    lines = [
        f"# {title}",
        "",
        f"配对物理样本数：{metrics['num_samples']}；方法行："
        + "、".join(methods)
        + "。同一样本、同一真值、同一有效 mask，统一在 K 空间评分。",
        "",
        "## 1. 总体指标（全部 15 个 lead 的有效像素，按像素数加权）",
        "",
        "| 方法 | RMSE (K) | MSE (K²) | MAE (K) | bias (K) | corr | "
        "CRPS (K) | MSE skill | CRPS skill |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    persistence_overall = persistence.get("overall", {})
    for method in methods:
        overall = metrics["methods"][method]["overall"]
        mse_skill = skill(
            overall["mse"], persistence_overall.get("mse")
        )
        crps_skill = skill(
            overall["crps"], persistence_overall.get("crps")
        )
        lines.append(
            f"| {method} | {fmt(overall['rmse'])} | {fmt(overall['mse'])} "
            f"| {fmt(overall['mae'])} | {fmt(overall['bias'])} "
            f"| {fmt(overall['correlation'])} | {fmt(overall['crps'])} "
            f"| {fmt_percent(mse_skill)} | {fmt_percent(crps_skill)} |"
        )
    lines += ["", "## 2. 代表 lead（Day 1/5/10/15）", ""]
    for lead in LEAD_ROWS:
        lines += [
            f"**Day {lead}**",
            "",
            "| 方法 | RMSE (K) | MAE (K) | bias (K) | corr | CRPS (K) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        reference = persistence.get("by_lead_day", {}).get(lead, {})
        for method in methods:
            row = metrics["methods"][method]["by_lead_day"][lead]
            lines.append(
                f"| {method} | {fmt(row['rmse'])} | {fmt(row['mae'])} "
                f"| {fmt(row['bias'])} | {fmt(row['correlation'])} "
                f"| {fmt(row['crps'])} |"
            )
        lines.append("")
    lines += [
        "## 3. 配对真实日块 bootstrap（ΔRMSE / ΔCRPS，95% CI）",
        "",
    ]
    if bootstrap:
        lines += [
            f"块规则：{bootstrap.get('block_rule', '—')}；"
            f"block_days={bootstrap.get('block_days')}，"
            f"replicates={bootstrap.get('replicates')}，"
            f"seed={bootstrap.get('seed')}。",
            "",
            "| 对照 | ΔRMSE (K) | ΔRMSE 95% CI | ΔCRPS (K) | "
            "ΔCRPS 95% CI | 时间块数 |",
            "|---|---:|---|---:|---|---:|",
        ]
        for key, entry in bootstrap.get("deltas", {}).items():
            comparison = key.replace("delta_", "").replace("_minus_", " − ")
            lines.append(
                f"| {comparison} | {fmt(entry.get('rmse_difference'))} "
                f"| {fmt_interval({'interval': entry.get('rmse_difference_ci')})} "
                f"| {fmt(entry.get('crps_difference'))} "
                f"| {fmt_interval({'interval': entry.get('crps_difference_ci')})} "
                f"| {entry.get('num_blocks', '—')} |"
            )
        if any(
                entry.get("interval") is None
                for entry in bootstrap.get("deltas", {}).values()
            ):
            lines += [
                "",
                "注：区间为 — 的对照其时间块少于两个，bootstrap 不适用，"
                "只报告点估计。",
            ]
    else:
        lines.append("未提供 bootstrap.json，本节留空。")
    lines += ["", "## 4. 概率辅助项（16 成员；8 成员为同一批抽样的嵌套子集）", ""]
    auxiliary_rows = [
        method for method in methods
        if "probability_auxiliary" in metrics["methods"][method]
    ]
    if auxiliary_rows:
        lines += [
            "| 方法 | spread (K) | skill (K) | spread/skill | "
            "coverage 50% | coverage 90% | coverage 50%（8 成员） | "
            "coverage 90%（8 成员） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in auxiliary_rows:
            auxiliary = metrics["methods"][method]["probability_auxiliary"]
            lines.append(
                f"| {method} | {fmt(auxiliary['spread'])} "
                f"| {fmt(auxiliary['skill'])} "
                f"| {fmt(auxiliary['spread_skill_ratio'], 3)} "
                f"| {fmt(auxiliary['coverage_50'], 4)} "
                f"| {fmt(auxiliary['coverage_90'], 4)} "
                f"| {fmt(auxiliary['coverage_50_8members'], 4)} "
                f"| {fmt(auxiliary['coverage_90_8members'], 4)} |"
            )
        lines += [
            "",
            "区间为有限成员经验分位数（np.quantile，线性插值），16 成员下"
            "50%/90% 区间很粗，仅作辅助诊断，不据此宣称校准。"
            "确定性方法的 CRPS 等于 MAE。",
        ]
    else:
        lines.append("本运行未记录集合行，概率辅助项不适用。")
    lines += ["", "## 5. 预报图", ""]
    if figures:
        lines += [
            f"固定样本 {figures.get('samples')} 例、固定相对窗口 "
            f"`{figures.get('crop')}`，四列为 Day 1/5/10/15，"
            "每图共用一条线性 turbo 色标，灰色为无效目标像素；"
            "集合方法展示集合均值。",
            "",
        ]
        for image in figures.get("images", []):
            lines += [f"![{image}](figures/{image})", ""]
    else:
        lines += ["未提供 figure_fields.npz/figures.json，本节留空。", ""]
    lines += [
        "## 6. 来源与复现",
        "",
        "```json",
        json.dumps(provenance, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 7. 限制与不宣称",
        "",
        "- 既有 test-200 结果此前已被查看，本轮是在已知基准上的统一比较，"
        "不是从未接触的确认性 holdout。",
        "- persistence 的 CRPS 等于其 MAE；扩散方法使用 16 成员经验 CRPS，"
        "不与 fair CRPS 混用，也不据低 CRPS 单独宣称校准。",
        "- 集合行展示集合均值；概率辅助项受有限成员数限制。",
        "- 若均值指标退化而概率指标改善，按权衡如实记录，不总结为全面更优。",
        "",
    ]
    return "\n".join(lines)


# 用途：入口。
# 参数：无输入（读命令行）；输出 exit code。
def main():
    parser = argparse.ArgumentParser(
        description="Render REPORT.md from the unified-protocol artifacts"
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--bootstrap", default=None)
    parser.add_argument("--figures", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    bootstrap = None
    if args.bootstrap and os.path.isfile(args.bootstrap):
        with open(args.bootstrap, "r", encoding="utf-8") as file:
            bootstrap = json.load(file)
    figures = None
    if args.figures and os.path.isfile(args.figures):
        with open(args.figures, "r", encoding="utf-8") as file:
            figures = json.load(file)
    title = args.title or "A5-centered DiAFNO 统一 test-200 协议结果"
    text = build_report(
        metrics, bootstrap=bootstrap, figures=figures, title=title
    )
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"wrote {os.path.abspath(args.output)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
