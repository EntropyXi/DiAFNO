"""Evaluate DiAFNO / deterministic IAFNO / persistence on paired samples.

Outputs 3-row x 4-column Day 1/5/10/15 SST panels, PNG/PDF, a Markdown
report and machine-readable scores. Uses existing checkpoint/data contracts.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from diafno.evaluation.method_comparison import (
    METHODS, ComparisonScores, draw_forecasts, write_markdown,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_paired(left, right):
    import torch
    for key in ("input_start_time", "target_start_time", "target_end_time", "spatial_index"):
        if not torch.equal(left["metadata"][key], right["metadata"][key]):
            raise ValueError(f"Unpaired test samples: {key}; align date manifests first")
    if not torch.equal(left["target_mask"], right["target_mask"]):
        raise ValueError("Unpaired target masks")


def predict_physical_members(validator, condition, sample_id, count):
    """Call the scored predictor for each member: residual anchoring once.

    Each sample is a one-item inference batch, so seeds do not change with
    unrelated loader batching. Returned members are absolute SST in source
    physical units, including centered diffusion's restored mean.
    """
    import torch
    from torch.amp import autocast
    original_seed, original_count = validator.config.seed, validator.config.ensemble_members
    members = []
    try:
        validator.config.ensemble_members = 1
        for member in range(count):
            validator.config.seed = original_seed + int(sample_id) * 1000 + member
            with torch.no_grad(), autocast("cuda", enabled=validator.amp_enabled):
                result = validator._predict(condition, 0)
            result = validator._inverse_transform(result.float())
            members.append(result.detach().cpu().numpy()[0, ..., 0])
    finally:
        validator.config.seed, validator.config.ensemble_members = original_seed, original_count
    return np.stack(members)


def render_existing(directory, dpi):
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    cases = []
    for path in sorted(directory.glob("case_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            case = {key: data[key].copy() for key in (*METHODS, "target_mask")}
            case["metadata"] = json.loads(str(data["metadata_json"].item()))
            cases.append(case)
    if not cases:
        raise ValueError("No saved case maps found")
    # Use a new output subdirectory for repeat rendering.
    out = directory / ("render_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    report["figures"] = draw_forecasts(cases, out, report["provenance"]["sst_unit"], dpi)
    write_markdown(report, out / "REPORT.md")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diafno-checkpoint")
    parser.add_argument("--iafno-checkpoint")
    parser.add_argument("--h5-path")
    parser.add_argument("--data-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-only", action="store_true", help="Render saved cases/scores without loading checkpoints or GPU")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--sample-index", type=int, help="Evaluate this exact split-relative dataset index, with the same member seeds")
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--ensemble-members", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--s-churn", type=float)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sst-unit", default="source units")
    parser.add_argument("--sst-offset", type=float, default=0.0, help="Explicit additive SST conversion, e.g. -273.15 for K to Celsius")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=22)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.render_only:
        render_existing(out, args.dpi)
        return
    if not all((args.diafno_checkpoint, args.iafno_checkpoint, args.h5_path)):
        parser.error("Both checkpoints and --h5-path are required")
    if min(args.max_samples, args.plot_samples, args.sampling_steps, args.block_days) < 1 or args.ensemble_members < 2 or args.ensemble_members > 1000:
        parser.error("Positive sample/step/block counts and 2..1000 ensemble members required")
    if args.bootstrap_replicates < 0 or args.num_workers < 0 or args.dpi < 72 or not np.isfinite(args.sst_offset):
        parser.error("Invalid bootstrap, worker, dpi or temperature-offset value")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {out}")
    import torch
    from tqdm import tqdm
    from diafno.evaluation.config import OSTIAValidationConfig
    from diafno.evaluation.validator import OSTIAValidator
    from diafno.inference.writer import InferenceSampleWriter
    class SelectedValidator(OSTIAValidator):
        def _build_indices(self):
            if args.sample_index is None:
                return super()._build_indices()
            if not 0 <= args.sample_index < len(self.dataset):
                raise ValueError("sample-index outside dataset")
            return [args.sample_index]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    validators = []
    checkpoints = [args.diafno_checkpoint, args.iafno_checkpoint]
    checkpoint_hashes = [sha256(path) for path in checkpoints]
    for index, checkpoint in enumerate(checkpoints):
        config = OSTIAValidationConfig(
            checkpoint=checkpoint, h5_path=args.h5_path, data_manifest=args.data_manifest,
            split=args.split, max_samples=args.max_samples, seed=args.seed, batch_size=1,
            num_workers=args.num_workers, device=args.device,
            sampling_steps=args.sampling_steps, s_churn=args.s_churn if index == 0 else None,
            ensemble_members=1, use_amp=not args.no_amp,
        )
        validators.append(SelectedValidator(config).setup())
    main_model, deterministic = validators
    if main_model.model_config.model_type not in ("diffusion", "centered_diffusion") or deterministic.model_config.model_type != "deterministic":
        raise ValueError("Expected diffusion/centered DiAFNO and deterministic IAFNO checkpoints")
    for validator in validators:
        if validator.model_config.input_days != 7 or validator.model_config.output_days != 15:
            raise ValueError("This comparison requires 7-day input and 15-day forecast")
    indices = main_model._build_indices()
    if len(main_model.dataset) != len(deterministic.dataset) or indices != deterministic._build_indices():
        raise ValueError("Different test sample universes; align both date manifests")
    out.mkdir(parents=True, exist_ok=True)
    count = len(indices)
    plot_positions = set(np.linspace(0, count - 1, min(args.plot_samples, count), dtype=int).tolist())
    provenance = {
        "created_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "split": args.split, "sst_unit": args.sst_unit, "sst_offset": args.sst_offset,
        "ensemble_members": args.ensemble_members, "sampling_steps": args.sampling_steps,
        "seed": args.seed, "sample_indices": indices, "plot_positions": sorted(plot_positions),
        "sampling": "explicit dataset index" if args.sample_index is not None else "fixed seed random subset; sorted indices; figures evenly spaced positions chosen before forecasts",
        "member_seed": "seed + dataset_index * 1000 + member_index; one sample per inference batch",
        "h5_path": str(Path(args.h5_path).resolve()), "h5_size": Path(args.h5_path).stat().st_size,
        "data_manifest": args.data_manifest,
        "data_manifest_sha256": sha256(args.data_manifest) if args.data_manifest else None,
        "checkpoints": {m: {"path": str(Path(p).resolve()), "sha256": digest} for m, p, digest in zip(METHODS, checkpoints, checkpoint_hashes)},
        "block_days": args.block_days, "bootstrap_replicates": args.bootstrap_replicates,
        "ci_confidence": 0.95, "crps_definition": "empirical ensemble CRPS (not fair CRPS); deterministic CRPS=MAE",
        "overall_definition": "all valid pixels over all 15 lead days",
        "s_churn": getattr(main_model.model, "S_churn", None), "use_amp": main_model.amp_enabled,
    }
    (out / "run_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    scores, cases = ComparisonScores(), []
    start = time.perf_counter()
    for position, (left, right) in enumerate(tqdm(zip(main_model.loader, deterministic.loader), total=count, desc="Paired three-method evaluation")):
        assert_paired(left, right)
        physical_targets = [v._inverse_transform(batch["target"]).float().numpy()[0, ..., 0] for v, batch in zip(validators, (left, right))]
        mask = left["target_mask"].numpy()[0, ..., 0]
        if not np.allclose(physical_targets[0][mask > 0], physical_targets[1][mask > 0], atol=1e-5, rtol=0):
            raise ValueError("Different physical SST references")
        target = physical_targets[0] + args.sst_offset
        # Check the physical persistence anchor agrees too, even if the
        # models have different normalized input channel contracts.
        anchors = [v._inverse_transform(b["condition"][:, 6:7]).float().numpy()[0, ..., 0] for v, b in zip(validators, (left, right))]
        if not np.allclose(anchors[0], anchors[1], atol=1e-5, rtol=0):
            raise ValueError("Different physical persistence anchors")
        forecasts = {
            "DiAFNO": predict_physical_members(main_model, left["condition"].to(main_model.device), indices[position], args.ensemble_members) + args.sst_offset,
            "IAFNO": predict_physical_members(deterministic, right["condition"].to(deterministic.device), indices[position], 1) + args.sst_offset,
            "persistence": np.repeat(anchors[0], 15, axis=0)[None] + args.sst_offset,
        }
        metadata = InferenceSampleWriter.metadata_item(left["metadata"], 0)
        scores.update(forecasts, target, mask, metadata["input_start_time"])
        if position in plot_positions:
            case = {method: np.where(mask > 0, forecasts[method].mean(axis=0, dtype=np.float64), np.nan).astype(np.float32) for method in METHODS}
            case.update(target_mask=mask, metadata=metadata)
            cases.append(case)
            np.savez_compressed(out / f"case_{position:06d}.npz", **{k: v for k, v in case.items() if k != "metadata"}, target=target, metadata_json=np.asarray(json.dumps(metadata)))
        del forecasts
    if len(scores.times) != count:
        raise RuntimeError("Incomplete paired data iteration")
    provenance["elapsed_seconds_inference_and_scores"] = time.perf_counter() - start
    provenance["seconds_per_sample_three_methods"] = provenance["elapsed_seconds_inference_and_scores"] / count
    # Compact per-sample sufficient statistics permit CI auditing without
    # storing all full-resolution ensemble members.
    np.savez_compressed(out / "paired_score_sums.npz", times=scores.times, counts=scores.counts, **{f"{m}_sse": scores.sse[m] for m in METHODS}, **{f"{m}_crps": scores.crps[m] for m in METHODS})
    report = {"num_samples": count, "provenance": provenance, "metrics": scores.compute(origin=int(main_model.dataset.first_time + main_model.dataset.split_start_day), block_days=args.block_days, replicates=args.bootstrap_replicates, seed=args.seed)}
    report["figures"] = draw_forecasts(cases, out, args.sst_unit, args.dpi)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, out / "REPORT.md")
    print(f"Finished {count} paired samples. Figures and Markdown: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
