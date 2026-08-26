import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSTIAInferenceConfig:
    checkpoint: str
    h5_path: str
    output_dir: str = "./inference_results"
    split: str = "test"
    condition_mode: str = "sst_mask"
    batch_size: int = 1
    num_workers: int = 2
    sampling_steps: Optional[int] = None
    ensemble_members: int = 1
    seed: int = 123
    device: str = "cuda:0"
    max_samples: Optional[int] = None
    save_members: bool = False
    compress: bool = False
    use_amp: bool = True

    @classmethod
    def from_args(cls, args):
        return cls(
            checkpoint=args.checkpoint,
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            split=args.split,
            condition_mode=args.condition_mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling_steps=args.sampling_steps,
            ensemble_members=args.ensemble_members,
            seed=args.seed,
            device=args.device,
            max_samples=args.max_samples,
            save_members=args.save_members,
            compress=args.compress,
            use_amp=not args.no_amp
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="DiAFNO OSTIA inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./inference_results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--condition-mode",
        type=str,
        default="sst_mask"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--ensemble-members",
        type=int,
        default=1
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )
    parser.add_argument(
        "--save-members",
        action="store_true"
    )
    parser.add_argument(
        "--compress",
        action="store_true"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true"
    )
    return parser
