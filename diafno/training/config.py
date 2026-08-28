import argparse
from dataclasses import dataclass, field
from typing import Optional

from ..models.config import OSTIAModelConfig


@dataclass
class OSTIATrainingConfig:
    seed: int = 123
    train_h5_path: str = "/data/exam_preprocessed_data/zzx/ocean_temperature_data_patched.h5"
    output_dir: str = "./experiments/ostia_7day_to15day_residual"
    resume_path: Optional[str] = None
    model: OSTIAModelConfig = field(
        default_factory=lambda: OSTIAModelConfig(
            sigma_data=0.15,
            target_mode="residual"
        )
    )
    num_epochs: int = 35
    samples_per_epoch: int = 31200
    # batch_per_gpu=16 x gradient_accumulation=2
    # -> effective batch 32 on one GPU, about 55 training hours
    batch_per_gpu: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4
    prefetch_factor: int = 2
    checkpoint_interval: int = 5
    use_amp: bool = True
    split: str = "train"
    condition_mode: str = "sst_mask"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train DiAFNO for OSTIA SST forecasting"
    )
    parser.add_argument("--train-h5-path")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--resume",
        dest="resume_path",
        nargs="?",
        const="latest"
    )
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--batch-per-gpu", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "residual")
    )
    parser.add_argument("--sigma-data", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split")
    parser.add_argument("--condition-mode")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true"
    )
    amp_group.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false"
    )
    parser.set_defaults(use_amp=None)
    return parser


def training_config_from_args(args):
    config = OSTIATrainingConfig()
    field_names = (
        "train_h5_path",
        "output_dir",
        "resume_path",
        "num_epochs",
        "samples_per_epoch",
        "batch_per_gpu",
        "gradient_accumulation",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "max_grad_norm",
        "num_workers",
        "prefetch_factor",
        "checkpoint_interval",
        "seed",
        "split",
        "condition_mode",
        "use_amp"
    )
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)
    if args.sampling_steps is not None:
        config.model.sampling_steps = args.sampling_steps
    if args.target_mode is not None:
        config.model.target_mode = args.target_mode
    if args.sigma_data is not None:
        config.model.sigma_data = args.sigma_data
    return config
