"""Select one illustrative region while retaining ALL-sample metric tables.

Requires a completed paired evaluation. Selection is explicitly recorded in
case_selection.json; a neutral figure caption makes no random-sampling claim.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diafno.evaluation.method_comparison import write_markdown
from scripts.compare_ostia_methods import sha256


def choose_region(indices, counts, main_sse, persistence_sse, pixels, ocean_min=0.4, ocean_max=0.9):
    counts, main_sse, persistence_sse = map(np.asarray, (counts, main_sse, persistence_sse))
    if counts.shape != main_sse.shape or counts.shape != persistence_sse.shape or counts.ndim != 2 or counts.shape[0] != len(indices):
        raise ValueError("Inconsistent paired score arrays")
    if pixels <= 0 or not 0 < ocean_min < ocean_max <= 1:
        raise ValueError("Invalid region selection settings")
    records = []
    for pos, index in enumerate(indices):
        count, error, baseline = counts[pos].sum(), main_sse[pos].sum(), persistence_sse[pos].sum()
        fraction = float(count / (counts.shape[1] * pixels))
        rmse = float(np.sqrt(error / count)) if count else None
        skill = float(1 - error / baseline) if baseline > 0 else None
        eligible = ocean_min <= fraction <= ocean_max and rmse is not None and np.isfinite(rmse)
        records.append(dict(position=pos, dataset_index=int(index), valid_ocean_fraction=fraction, diafno_rmse=rmse, mse_skill_vs_persistence=skill, eligible=bool(eligible)))
    candidates = [r for r in records if r["eligible"]]
    if not candidates:
        raise ValueError("No candidate has the specified valid ocean fraction")
    improved = [r for r in candidates if r["mse_skill_vs_persistence"] is not None and r["mse_skill_vs_persistence"] > 0]
    selected = min(improved or candidates, key=lambda r: (r["diafno_rmse"], r["dataset_index"]))
    return {"selected": selected, "candidates": records, "criterion": {"valid_ocean_fraction_range": [ocean_min, ocean_max], "prefer_positive_mse_skill": True, "ranking": "lowest DiAFNO overall RMSE; dataset index tie-break", "fallback": "if no positive-skill eligible sample, lowest RMSE among eligible samples"}, "purpose": "illustrative case selected after evaluation; aggregate tables unchanged; not random case sampling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ocean-min", type=float, default=0.4)
    parser.add_argument("--ocean-max", type=float, default=0.9)
    args = parser.parse_args()
    source, out = Path(args.evaluation_dir), Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Final output directory must be empty")
    report = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    provenance = report["provenance"]
    with np.load(next(source.glob("case_*.npz")), allow_pickle=False) as case:
        pixels = int(np.prod(case["target_mask"].shape[-2:]))
    with np.load(source / "paired_score_sums.npz", allow_pickle=False) as data:
        selection = choose_region(provenance["sample_indices"], data["counts"], data["DiAFNO_sse"], data["persistence_sse"], pixels, args.ocean_min, args.ocean_max)
    for checkpoint in provenance["checkpoints"].values():
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed since aggregate evaluation")
    command = [sys.executable, "-u", str(Path(__file__).with_name("compare_ostia_methods.py")), "--diafno-checkpoint", provenance["checkpoints"]["DiAFNO"]["path"], "--iafno-checkpoint", provenance["checkpoints"]["IAFNO"]["path"], "--h5-path", provenance["h5_path"], "--output-dir", str(out), "--split", provenance["split"], "--max-samples", "1", "--plot-samples", "1", "--sample-index", str(selection["selected"]["dataset_index"]), "--ensemble-members", str(provenance["ensemble_members"]), "--sampling-steps", str(provenance["sampling_steps"]), "--seed", str(provenance["seed"]), "--sst-unit", provenance["sst_unit"], "--sst-offset", str(provenance["sst_offset"]), "--bootstrap-replicates", "0", "--num-workers", "0", "--device", args.device]
    if provenance.get("data_manifest"):
        command.extend(["--data-manifest", provenance["data_manifest"]])
    if provenance.get("s_churn") is not None:
        command.extend(["--s-churn", str(provenance["s_churn"])])
    if not provenance["use_amp"]:
        command.append("--no-amp")
    print(json.dumps(selection["selected"]), flush=True)
    subprocess.run(command, check=True)
    case_report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    # Preserve selected-sample statistics separately before replacing the
    # presentation report with the unfiltered, full evaluation tables.
    (out / "selected_case_metrics.json").write_text(json.dumps(case_report, indent=2), encoding="utf-8")
    actual = case_report["metrics"]["overall"]["DiAFNO"]["rmse"]
    expected = selection["selected"]["diafno_rmse"]
    if not np.isclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise ValueError(f"Selected rerun changed RMSE: {expected} -> {actual}")
    final = copy.deepcopy(report)
    final["figures"] = case_report["figures"]
    final["figure_caption"] = "测试集某区域的 Day 1/5/10/15 SST 预测对比"
    final["provenance"]["case_selection_record"] = "case_selection.json"
    final["provenance"]["aggregate_evaluation_source"] = str(source.resolve())
    (out / "case_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    # Selected case files keep their names; score sums and run_manifest in
    # this directory describe that single rerun. Full tables explicitly
    # refer to aggregate_evaluation_source and its original run_manifest.
    (out / "comparison.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(final, out / "REPORT.md")
    print(f"Final one-region figure and {report['num_samples']}-sample tables: {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
