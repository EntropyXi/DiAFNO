import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSTIAValidationConfig:
    checkpoint: str
    h5_path: str
    output_path: str = "./validation_metrics.json"
    split: str = "val"
    condition_mode: str = "sst_mask"
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
            use_amp=not args.no_amp
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
        default="sst_mask"
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
        choices=("model", "persistence", "probe"),
        default="model"
    )
    parser.add_argument("--probe-sigma", type=float)
    parser.add_argument(
        "--condition-ablation",
        choices=("none", "zero_sst", "reverse_sst"),
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
    return parser
