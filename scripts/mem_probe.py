# 用途：用合成输入测量候选 batch 的训练峰值显存。
"""Measure per-GPU peak memory for candidate batch sizes.

Builds the real training model and feeds synthetic batches with the exact
tensor shapes of the OSTIA dataset, running forward+backward through the same
autocast/GradScaler path as OSTIATrainer._train_epoch. No HDF5 access and no
DDP, so it works on any single GPU of the training machine.

Peak activation memory scales almost linearly with batch size (model +
optimizer state is ~1.5M params, i.e. tens of MB), so the numbers reported
here transfer directly to real training.

Usage on the training machine:

    python -u scripts/mem_probe.py                          # default sweep
    python -u scripts/mem_probe.py --batches 32 48 64 80    # custom sweep
    python -u scripts/mem_probe.py --batches 16 --no-amp    # fp32 ceiling
"""

import argparse
import gc

import torch
from torch.amp import GradScaler, autocast

from diafno.models.config import OSTIAModelConfig

COND_CHANS = 8
TARGET_CHANS = 15
H, W, Z = 448, 448, 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 80, 96],
        help="batch sizes to probe (default: %(default)s)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="disable autocast/GradScaler (fp32 memory ceiling)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="training steps per batch size after warmup",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup steps before resetting peak stats",
    )
    return parser.parse_args()


def make_batch(batch, device):
    condition = torch.randn(
        batch, COND_CHANS, H, W, Z, device=device
    )
    target = torch.randn(
        batch, TARGET_CHANS, H, W, Z, device=device
    )
    target_mask = (
        torch.rand(
            batch, TARGET_CHANS, H, W, Z, device=device
        ) > 0.5
    ).float()
    return condition, target, target_mask


def probe(batch, model, scaler, amp_enabled, warmup, steps):
    device = next(model.parameters()).device

    def step(condition, target, target_mask):
        with autocast("cuda", enabled=amp_enabled):
            loss = model(target, condition, target_mask)
        scaler.scale(loss).backward()

    condition, target, target_mask = make_batch(batch, device)
    for _ in range(warmup):
        step(condition, target, target_mask)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        step(condition, target, target_mask)
    allocated = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
    )
    reserved = (
        torch.cuda.max_memory_reserved(device) / 1024 ** 3
    )
    del condition, target, target_mask
    return allocated, reserved


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / 1024 ** 3
    print(
        f"GPU: {props.name}, total memory {total_gib:.1f} GiB, "
        f"amp={'on' if not args.no_amp else 'off'}"
    )
    model = OSTIAModelConfig().build_model(device)
    amp_enabled = not args.no_amp
    scaler = GradScaler("cuda", enabled=amp_enabled)
    rows = []
    for batch in args.batches:
        try:
            allocated, reserved = probe(
                batch,
                model,
                scaler,
                amp_enabled,
                args.warmup,
                args.steps,
            )
            rows.append((batch, allocated, reserved))
            print(
                f"batch={batch:>4}: "
                f"peak_allocated={allocated:6.2f} GiB "
                f"peak_reserved={reserved:6.2f} GiB"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"batch={batch:>4}: OOM")
            torch.cuda.empty_cache()
            gc.collect()
    if len(rows) < 2:
        return
    biggest = rows[-1]
    per_sample = biggest[1] / biggest[0]
    print(
        "\npeak memory scales ~linearly with batch size "
        f"({per_sample * 1024:.0f} MiB/sample at batch "
        f"{biggest[0]}, fixed overhead included)"
    )
    target_gib = 20.0
    suggested = int(target_gib / per_sample)
    print(
        f"for ~{target_gib:.0f} GiB on this GPU try "
        f"--batch-per-gpu {suggested} "
        "(leave 2-3 GiB headroom below the card's total; "
        "verify with one short epoch and the trainer's "
        "peak_memory print)"
    )


if __name__ == "__main__":
    main()
