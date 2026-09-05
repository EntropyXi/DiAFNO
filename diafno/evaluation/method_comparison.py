"""Paired three-method forecast scores, Markdown tables and 3 x 4 SST maps."""
import json
from pathlib import Path

import numpy as np

from .metrics import RunningSSTMetrics

METHODS = ("DiAFNO", "IAFNO", "persistence")
LABELS = ("DiAFNO ensemble mean", "IAFNO deterministic", "Persistence")
LEADS = (1, 5, 10, 15)


def empirical_crps(members, target):
    """Empirical ensemble CRPS, not fair CRPS. Shape [member, valid pixel].

    E|X-y| - E|X-X'|/2, with both draws from the empirical distribution.
    Sorting avoids allocating a quadratic member-pair matrix.
    A one-member (deterministic) distribution gives absolute error.
    """
    members = np.asarray(members, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if members.ndim != 2 or members.shape[0] < 1 or members.shape[1:] != target.shape:
        raise ValueError("CRPS needs [members,pixels] and [pixels]")
    if not np.isfinite(members).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite CRPS inputs")
    count = members.shape[0]
    weights = (2 * np.arange(1, count + 1) - count - 1)[:, None]
    score = np.abs(members - target).mean(axis=0) - (
        np.sort(members, axis=0) * weights
    ).sum(axis=0) / count ** 2
    return np.maximum(score, 0.0)


def skill(value, reference):
    return None if reference is None or reference <= 0 else 1.0 - value / reference


def paired_skill_ci(scores, reference, times, *, origin, block_days=22, replicates=2000, seed=123):
    """Resample initialization-day blocks, retaining paired pixels/methods.

    Inputs contain sample-level score SUMS, not averages; common valid
    counts cancel in the ratio. Spatial samples of one time stay together.
    """
    if block_days < 1 or replicates < 0:
        raise ValueError("Invalid bootstrap settings")
    scores, reference = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    times = np.asarray(times, dtype=np.int64)
    if scores.shape != reference.shape or scores.ndim != 1 or times.shape != scores.shape:
        raise ValueError("Paired bootstrap needs aligned one-dimensional inputs")
    if not np.isfinite(scores).all() or not np.isfinite(reference).all() or (scores < 0).any() or (reference < 0).any():
        raise ValueError("Invalid score sums")
    blocks, inverse = np.unique((times - origin) // block_days, return_inverse=True)
    if len(blocks) < 2 or replicates == 0:
        return {"interval": None, "num_blocks": len(blocks), "reason": "disabled or fewer than two temporal blocks"}
    sums = np.bincount(inverse, weights=scores)
    refs = np.bincount(inverse, weights=reference)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        denominator = refs[selected].sum()
        if denominator > 0:
            values.append(1 - sums[selected].sum() / denominator)
    interval = np.quantile(values, [0.025, 0.975]).tolist() if len(values) == replicates else None
    return {"interval": interval, "num_blocks": len(blocks), "valid_replicates": len(values), "requested_replicates": replicates}


class ComparisonScores:
    def __init__(self, horizon=15):
        self.horizon = horizon
        self.metrics = {m: [RunningSSTMetrics() for _ in range(horizon)] for m in METHODS}
        self.overall = {m: RunningSSTMetrics() for m in METHODS}
        self.sse = {m: [] for m in METHODS}
        self.crps = {m: [] for m in METHODS}
        self.counts = []
        self.times = []

    def update(self, forecasts, target, mask, initialization_day):
        """One sample, each forecast [member,lead,H,W], target [lead,H,W]."""
        target, mask = np.asarray(target), np.asarray(mask)
        if target.ndim != 3 or target.shape != mask.shape or target.shape[0] != self.horizon:
            raise ValueError("Target/mask must have matching [lead,H,W] shapes")
        if not np.isfinite(mask).all():
            raise ValueError("Nonfinite target mask")
        valid = mask > 0
        if not valid.any() or not np.isfinite(target[valid]).all():
            raise ValueError("No valid target pixels or nonfinite target")
        # Fail before accumulating anything: do not silently give methods
        # different evaluation masks when a forecast has NaN/Inf.
        for method in METHODS:
            members = np.asarray(forecasts[method])
            if members.ndim != 4 or members.shape[0] < 1 or members.shape[1:] != target.shape:
                raise ValueError(f"Invalid forecast shape: {method}")
            if not np.isfinite(members[:, valid]).all():
                raise ValueError(f"Nonfinite valid-pixel forecast: {method}")
        self.counts.append(valid.sum(axis=(1, 2)))
        self.times.append(int(initialization_day))
        for method in METHODS:
            members = np.asarray(forecasts[method])
            prediction = members.mean(axis=0, dtype=np.float64)
            self.overall[method].update(prediction, target, mask)
            sse, crps = [], []
            for lead in range(self.horizon):
                ocean = valid[lead]
                truth = target[lead][ocean].astype(np.float64)
                pred = prediction[lead][ocean]
                self.metrics[method][lead].update(prediction[lead], target[lead], mask[lead])
                sse.append(float(np.square(pred - truth).sum()))
                crps.append(float(empirical_crps(members[:, lead, ocean], truth).sum()))
            self.sse[method].append(sse)
            self.crps[method].append(crps)

    def compute(self, *, origin, block_days=22, replicates=2000, seed=123):
        if not self.times:
            raise ValueError("No evaluated samples")
        counts = np.asarray(self.counts)
        result = {}
        for scope in [*map(str, LEADS), "overall"]:
            index = None if scope == "overall" else int(scope) - 1
            if index is not None and index >= self.horizon:
                continue
            count = counts.sum() if index is None else counts[:, index].sum()
            if count == 0:
                raise ValueError(f"No valid pixels for {scope}")
            def reduce(values):
                values = np.asarray(values)
                return values.sum(axis=1) if index is None else values[:, index]
            result[scope] = {}
            for method in METHODS:
                metrics = (self.overall[method] if index is None else self.metrics[method][index]).compute()
                metrics.pop("acc", None)  # Raw correlation is not climatological ACC.
                sums = reduce(self.crps[method])
                metrics["crps"] = float(sums.sum() / count)
                for metric, source in (("mse", self.sse), ("crps", self.crps)):
                    score, ref = reduce(source[method]), reduce(source["persistence"])
                    metrics[f"{metric}_skill"] = skill(float(score.sum()), float(ref.sum()))
                    metrics[f"{metric}_skill_95ci"] = paired_skill_ci(score, ref, self.times, origin=origin, block_days=block_days, replicates=replicates, seed=seed)
                result[scope][method] = metrics
        return result


def write_markdown(report, path):
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    def percent(value):
        return "—" if value is None else f"{value:+.1%}"
    def interval(value):
        bounds = value["interval"]
        return "—" if bounds is None else f"[{bounds[0]:+.1%}, {bounds[1]:+.1%}]"
    provenance = report["provenance"]
    unit = provenance["sst_unit"]
    lines = ["# 三种方法的 SST 预测对比", "", f"数据划分：{provenance['split']}；配对样本数：{report['num_samples']}；DiAFNO 集合成员数：{provenance['ensemble_members']}。", "",
             "DiAFNO 的 RMSE/MSE/MAE/bias/corr 使用集合均值；CRPS 使用完整经验集合分布。IAFNO 和 persistence 是点分布，因此 CRPS = MAE。",
             "", "MSE skill = 1 − MSE / MSE(persistence)；CRPS skill = 1 − CRPS / CRPS(persistence)。正值表示优于 persistence，persistence 自身为 0（基线分母为 0 时未定义）。",
             "", "overall 汇总全部 15 个预测日的有效像素，按像素数加权，不是只平均图中的四天，也不是对各天 RMSE 取平均。corr 为原始 SST 的 Pearson 相关系数，不是去气候态 ACC。",
             "", f"95% CI：按初始化时间作 {provenance['block_days']} 日配对分块 bootstrap，{provenance['bootstrap_replicates']} 次重采样。空间样本与方法保持配对；少于两个时间块或关闭 bootstrap 时记为 —。少样本 CI 仅作探索性参考。", ""]
    for scope, methods in report["metrics"].items():
        lines += [f"## {'Overall（Day 1–15）' if scope == 'overall' else 'Day +' + scope}", "",
                  f"| forecast | RMSE ({unit}) | MSE ({unit}²) | MAE ({unit}) | bias ({unit}) | corr | CRPS ({unit}) | MSE skill | MSE skill 95% CI | CRPS skill | CRPS skill 95% CI |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
        for method in METHODS:
            m = methods[method]
            values = [method, *[number(m[k]) for k in ("rmse", "mse", "mae", "bias", "correlation", "crps")], percent(m["mse_skill"]), interval(m["mse_skill_95ci"]), percent(m["crps_skill"]), interval(m["crps_skill_95ci"])]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## 预测图", "", "每张图固定一个区域与初始化时间，三行依次为 DiAFNO、IAFNO、persistence，四列为 Day 1/5/10/15。图中 DiAFNO 显示集合均值。所有图共享 SST 色标；灰色为对应预测日的无效目标像素。", ""]
    for image in report.get("figures", []):
        caption = report.get("figure_caption", image)
        lines += [f"![{caption}]({image})", "", caption, ""]
    lines += ["## 来源与复现", "", "```json", json.dumps(provenance, ensure_ascii=False, indent=2), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def draw_forecasts(cases, output_dir, unit="source units", dpi=250, title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    # All methods, days and regions use the same scale. No resampling,
    # interpolation, percentile clipping, or per-method autoscaling.
    lo, hi = np.inf, -np.inf
    for case in cases:
        for lead in LEADS:
            valid = case["target_mask"][lead - 1] > 0
            for method in METHODS:
                values = case[method][lead - 1][valid]
                if values.size:
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite plotted ocean prediction")
                    lo, hi = min(lo, values.min()), max(hi, values.max())
    if not np.isfinite([lo, hi]).all():
        raise ValueError("No valid pixels to plot")
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    images = []
    for index, case in enumerate(cases):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8.6), layout="constrained")
        for row, (method, label) in enumerate(zip(METHODS, LABELS)):
            for col, lead in enumerate(LEADS):
                ax = axes[row, col]
                values = np.ma.array(case[method][lead - 1], mask=case["target_mask"][lead - 1] <= 0)
                im = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
                if row == 0:
                    ax.set_title(f"Day {lead}", fontsize=12, fontweight="bold")
                ax.set_ylabel(label + "\nY (pixel)" if col == 0 else "")
                if row == 2:
                    ax.set_xlabel("X (pixel)")
                ax.tick_params(labelsize=7)
        metadata = case["metadata"]
        fig.suptitle(f"{title_prefix}SST forecast | spatial index {metadata['spatial_index']} | input start {metadata['input_start_time']}", fontsize=12)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.023, pad=0.02, label=f"SST ({unit})")
        name = f"forecast_region_{index:03d}"
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", dpi=dpi)
        plt.close(fig)
        images.append(name + ".png")
    return images
