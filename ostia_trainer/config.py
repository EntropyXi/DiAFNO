from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import torch

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff


@dataclass
class OSTIAModelConfig:
    input_weeks: int = 7
    output_weeks: int = 15
    cond_chans: int = 8
    target_chans: int = 15
    image_size: Tuple[int, int, int] = (448, 448, 1)
    patch_size: Tuple[int, int, int] = (8, 8, 1)
    embed_dim: int = 128
    num_blocks: int = 8
    explicit_layer: int = 4
    implicit_layer: int = 2
    hidden_size_factor: int = 4
    sampling_steps: int = 16
    sigma_data: float = 1.0

    def to_checkpoint(self):
        return asdict(self)

    def build_model(self, device):
        backbone = IAFNODiff(
            dim=self.image_size,
            dim_f=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            cond_chans=self.cond_chans,
            target_chans=self.target_chans,
            ex_layer=self.explicit_layer,
            nlayer=self.implicit_layer,
            hidden_size_factor=self.hidden_size_factor,
            drop_rate=0.,        
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        )
        model = ElucidatedDiffusion(
            backbone,
            channels=self.target_chans,
            num_sample_steps=self.sampling_steps,
            image_size_h=self.image_size[0],
            image_size_w=self.image_size[1],
            image_size_z=self.image_size[2],
            sigma_data=self.sigma_data
        )
        return model.to(
            device=device,
            dtype=torch.float32
        )


@dataclass
class OSTIATrainingConfig:
    seed: int = 123
    train_h5_path: str = "/data/exam_preprocessed_data/zzx/ocean_temperature_data_patched.h5"
    output_dir: str = "./experiments/ostia_7to15"
    resume_path: Optional[str] = None
    model: OSTIAModelConfig = field(default_factory=OSTIAModelConfig)
    num_epochs: int = 35
    samples_per_epoch: int = 7040
    # batch_per_gpu=16 x 2 GPUs -> global batch 32
    # 35 epochs x 7040 samples -> about 50 training hours
    batch_per_gpu: int = 16
    gradient_accumulation: int = 1
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


import argparse


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
    return config
