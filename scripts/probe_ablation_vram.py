"""Non-destructive per-architecture VRAM / throughput probe.

For every ablation configuration (A0..A5) and each micro-batch in
{1,2,4,8} this script runs warm-up iterations followed by several
measured forward/backward iterations and records allocated/reserved/
peak memory, seconds per iteration, throughput and any OOM/NaN or
gradient anomaly.  It only instantiates the real architecture (448x448
grid, fixed 7->15 day contract, condition schema of the config) on
random tensors; it writes one JSON report to --out and refuses to
overwrite an existing file.  Nothing is trained, saved or deleted.

Server-only: requires a CUDA GPU; the micro-batch chosen here is then
paired with gradient accumulation so the global effective batch stays
32.
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_probe_config(config_path, num_blocks=None,
                       implicit_layer=None):
    """Model config mirroring the ablation JSON exactly (same schema,
    same architecture), with the deterministic raw-scaling wrapper so
    no lead-stats file is needed for the probe.  ``num_blocks`` /
    ``implicit_layer`` optionally override the JSON (A5 winner
    re-probe)."""
    import json as jsonlib
    with open(config_path, "r", encoding="utf-8") as file:
        payload = jsonlib.load(file)
    from diafno.models.config import OSTIAModelConfig
    config = OSTIAModelConfig(
        target_mode="residual",
        model_type="deterministic",
        target_scaling="raw",
        sigma_data=0.15,
    )
    config.adopt_condition_mode(payload["condition_mode"])
    config.patch_size = tuple(payload["patch_size"])
    config.num_blocks = (
        int(payload["num_blocks"])
        if num_blocks is None
        else int(num_blocks)
    )
    config.implicit_layer = (
        int(payload["implicit_layer"])
        if implicit_layer is None
        else int(implicit_layer)
    )
    return config


def measure(config, batch_size, device, warmup=2, iterations=3,
            use_amp=True):
    import torch
    from torch.amp import autocast
    model = config.build_model(device)
    model.train()
    condition = torch.randn(
        batch_size,
        config.cond_chans,
        *config.image_size,
        device=device,
    )
    target = torch.randn(
        batch_size,
        config.target_chans,
        *config.image_size,
        device=device,
    )
    target_mask = torch.ones_like(target)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=1e-4,
    )
    timings = []
    peak = 0.0
    anomaly = None
    for index in range(warmup + iterations):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        try:
            with autocast(
                    "cuda",
                    enabled=use_amp,
                ):
                loss = model(target, condition, target_mask)
            loss.backward()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {
                "batch_size": batch_size,
                "oom": True,
                "peak_allocated_gib": None,
                "sec_per_iter": None,
            }
        elapsed = time.perf_counter() - start
        if not torch.isfinite(loss):
            anomaly = "non-finite loss"
            break
        gradients_finite = all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if not gradients_finite:
            anomaly = "non-finite gradient"
            break
        optimizer.step()
        if index >= warmup:
            timings.append(elapsed)
        peak = max(
            peak,
            torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        )
    if anomaly is not None or not timings:
        return {
            "batch_size": batch_size,
            "oom": False,
            "anomaly": anomaly or "no measured iterations",
            "peak_allocated_gib": round(peak, 4),
            "sec_per_iter": None,
        }
    allocated = (
        torch.cuda.memory_allocated(device) / 1024 ** 3
    )
    reserved = (
        torch.cuda.memory_reserved(device) / 1024 ** 3
    )
    return {
        "batch_size": batch_size,
        "oom": False,
        "sec_per_iter": round(
            sum(timings) / len(timings), 4
        ),
        "samples_per_sec": round(
            batch_size * len(timings) / sum(timings), 2
        ),
        "peak_allocated_gib": round(peak, 4),
        "allocated_gib": round(allocated, 4),
        "reserved_gib": round(reserved, 4),
        "anomaly": anomaly,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="ablation config JSON to probe")
    parser.add_argument("--out", required=True,
                        help="JSON report path (must not exist)")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help=(
            "override num_blocks from the JSON (used to re-probe A5 "
            "with the Stage-2 winner blocks)"
        ),
    )
    parser.add_argument(
        "--implicit-layer",
        type=int,
        default=None,
        help="override implicit_layer from the JSON",
    )
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the VRAM probe requires a CUDA GPU (server-only tool); "
            "no local long training is started"
        )
    if os.path.exists(args.out):
        raise RuntimeError(
            f"refusing to overwrite existing probe report: {args.out}"
        )
    device = torch.device("cuda:0")
    # Force the CUDA context up front (laptop/hybrid-GPU drivers can
    # otherwise reject memory-stat calls before any allocation).
    torch.zeros(1, device=device)
    torch.cuda.synchronize()
    batch_sizes = [
        int(value.strip()) for value in args.batch_sizes.split(",")
        if value.strip()
    ]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError(
            f"invalid --batch-sizes {args.batch_sizes!r}"
        )
    config = build_probe_config(
        args.config,
        num_blocks=args.blocks,
        implicit_layer=args.implicit_layer,
    )
    rows = []
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats(device)
        row = measure(
            config,
            batch_size,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
            use_amp=not args.no_amp,
        )
        rows.append(row)
        print(f"batch {batch_size}: {row}")
    report = {
        "config": os.path.abspath(args.config),
        "condition_mode": config.condition_mode,
        "cond_chans": config.cond_chans,
        "patch_size": list(config.patch_size),
        "num_blocks": config.num_blocks,
        "implicit_layer": config.implicit_layer,
        "image_size": list(config.image_size),
        "target_chans": config.target_chans,
        "use_amp": not args.no_amp,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"probe report written to {args.out}")


if __name__ == "__main__":
    main()
