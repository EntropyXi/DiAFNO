# 用途：解析在线验证的样本、采样、设备及输出配置。
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSTIAValidationConfig:
    checkpoint: str
    h5_path: str
    output_path: str = "./validation_metrics.json"
    split: str = "val"
    # None = restore the condition contract from the checkpoint model
    # config (legacy checkpoints resolve to 'sst_mask').
    condition_mode: Optional[str] = None
    # Upstream data manifest required by any checkpoint bound to the
    # shared real-day, gap-filtered sample universe.
    data_manifest: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 2
    sampling_steps: Optional[int] = None
    s_churn: Optional[float] = None
    ensemble_members: int = 1
    prediction_mode: str = "model"
    probe_sigma: Optional[float] = None
    condition_ablation: str = "none"
    seed: int = 123
    device: str = "cuda:0"
    max_samples: Optional[int] = 200
    use_amp: bool = True
    paired_bootstrap_replicates: int = 0
    bootstrap_block_days: int = 22
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 123

    @classmethod
    def from_args(cls, args):
        prediction_mode = (
            "probe"
            if getattr(args, "probe_sigma", None) is not None
            else args.prediction_mode
        )
        return cls(
            checkpoint=args.checkpoint,
            h5_path=args.h5_path,
            output_path=args.output_path,
            split=args.split,
            condition_mode=args.condition_mode,
            data_manifest=args.data_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling_steps=args.sampling_steps,
            s_churn=args.s_churn,
            ensemble_members=args.ensemble_members,
            prediction_mode=prediction_mode,
            probe_sigma=args.probe_sigma,
            condition_ablation=args.condition_ablation,
            seed=args.seed,
            device=args.device,
            max_samples=(
                None
                if args.all_samples
                else args.max_samples
            ),
            use_amp=not args.no_amp,
            paired_bootstrap_replicates=(
                args.paired_bootstrap_replicates
            ),
            bootstrap_block_days=args.bootstrap_block_days,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
        )


def build_validation_parser():
    parser = argparse.ArgumentParser(
        description="Validate DiAFNO on the OSTIA validation split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument(
        "--output-path",
        default="./validation_metrics.json"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--condition-mode",
        default=None,
        help=(
            "condition contract override; defaults to the mode "
            "restored from the checkpoint"
        )
    )
    parser.add_argument(
        "--data-manifest",
        default=None,
        help=(
            "upstream data manifest (required when the checkpoint "
            "was trained with one)"
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--s-churn", type=float)
    parser.add_argument(
        "--ensemble-members",
        type=int,
        default=1
    )
    parser.add_argument(
        "--prediction-mode",
        choices=(
            "model",
            "persistence",
            "linear_trend",
            "probe"
        ),
        default="model"
    )
    parser.add_argument("--probe-sigma", type=float)
    parser.add_argument(
        "--condition-ablation",
        choices=(
            "none",
            "anchor_only",
            "reverse_history",
            "shuffle_history",
            "zero_sst",
            "reverse_sst"
        ),
        default="none"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200
    )
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--paired-bootstrap-replicates",
        type=int,
        default=0,
        help=(
            "paired temporal-block bootstrap replicates; "
            "0 disables confidence intervals"
        )
    )
    parser.add_argument(
        "--bootstrap-block-days",
        type=int,
        default=22,
        help="days per temporal block (default: 7 input + 15 forecast)"
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95
    )
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    return parser
