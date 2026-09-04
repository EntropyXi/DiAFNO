import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..models.config import OSTIAModelConfig
from deterministic_iafno.centered_stats import (
    validate_centered_stats_payload,
)


def default_training_model():
    """Factory default model config for OSTIA training.

    Kept as a named helper so checkpoint-resume semantics can compare
    the current values against exactly the same defaults the dataclass
    factory uses (a bare ``--resume`` must not silently adopt newer
    defaults over the checkpoint's recorded semantics).
    """
    return OSTIAModelConfig(
        sigma_data=0.15,
        target_mode="residual"
    )


_CONFIG_JSON_FIELDS = (
    "train_h5_path",
    "output_dir",
    "mean_checkpoint_path",
    "centered_stats_path",
    "lead_stats_path",
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
    "sampling_steps",
    "target_mode",
    "model_type",
    "target_scaling",
    "sigma_data",
    "sigma_max",
    "sigma_min",
    "p_mean",
    "p_std",
    "rho",
    "seed",
    "split",
    "condition_mode",
    "data_manifest_path",
    "use_amp",
    # Explicit small-architecture entries for the controlled
    # spatiotemporal ablation (A0..A5).  cond_chans is deliberately
    # NOT configurable: it is derived from condition_mode.
    "patch_size",
    "num_blocks",
    "implicit_layer",
)


def merge_config_json(args, config_path):
    """Fill CLI args from an authoritative training config JSON.

    A JSON value only fills an argument the CLI did not explicitly set;
    explicit CLI arguments win and are reported so the launcher can
    verify the JSON-vs-CLI expansion.  Returns a list of human-readable
    override notes.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"training config JSON not found: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(
            "training config JSON must contain an object"
        )
    unknown = sorted(
        key
        for key in payload
        if key not in _CONFIG_JSON_FIELDS
    )
    if unknown:
        raise ValueError(
            "training config JSON contains unknown keys: "
            f"{unknown}"
        )
    overrides = []
    for key, value in payload.items():
        current = getattr(args, key, None)
        if current is not None:
            overrides.append(
                f"{key}: CLI value {current!r} overrides config "
                f"file value {value!r}"
            )
            continue
        setattr(args, key, value)
    return overrides


@dataclass
class OSTIATrainingConfig:
    seed: int = 123
    train_h5_path: str = "/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5"
    output_dir: str = "./experiments/ostia_7day_to15day_residual"
    resume_path: Optional[str] = None
    init_from: Optional[str] = None
    lead_stats_path: Optional[str] = None
    mean_checkpoint_path: Optional[str] = None
    centered_stats_path: Optional[str] = None
    model: OSTIAModelConfig = field(
        default_factory=default_training_model
    )
    num_epochs: int = 35
    samples_per_epoch: int = 31200
    optimizer_steps_per_epoch: Optional[int] = None
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
    allow_resume_override: bool = False
    explicit_resume_fields: Optional[Tuple[str, ...]] = field(
        default=None,
        repr=False,
    )
    split: str = "train"
    condition_mode: str = "sst_mask"
    # Optional upstream-proven real-day data manifest.  The ablation
    # runner requires it for A0..A5 so every configuration shares the
    # same gap-filtered sample universe.
    data_manifest_path: Optional[str] = None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train DiAFNO for OSTIA SST forecasting"
    )
    parser.add_argument(
        "--config",
        help=(
            "authoritative training config JSON (values fill any "
            "CLI argument not explicitly given)"
        )
    )
    parser.add_argument("--train-h5-path")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--resume",
        dest="resume_path",
        nargs="?",
        const="latest"
    )
    parser.add_argument("--init-from")
    parser.add_argument(
        "--lead-stats",
        dest="lead_stats_path"
    )
    parser.add_argument(
        "--mean-checkpoint",
        dest="mean_checkpoint_path",
        help=(
            "frozen deterministic mean checkpoint "
            "(centered_diffusion fresh runs only)"
        )
    )
    parser.add_argument(
        "--centered-stats",
        dest="centered_stats_path",
        help=(
            "train-only centered innovation stats JSON "
            "(centered_diffusion fresh runs only)"
        )
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
    parser.add_argument(
        "--allow-resume-override",
        action="store_true",
        help=(
            "accept reviewed optimizer/schedule/effective-batch "
            "resume mismatches; immutable model/data semantics "
            "still cannot be overridden"
        )
    )
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "residual")
    )
    parser.add_argument(
        "--model-type",
        choices=("diffusion", "deterministic", "centered_diffusion")
    )
    parser.add_argument(
        "--target-scaling",
        choices=("raw", "lead_standardized")
    )
    parser.add_argument("--sigma-data", type=float)
    parser.add_argument("--sigma-max", type=float)
    parser.add_argument("--sigma-min", type=float)
    parser.add_argument("--p-mean", type=float)
    parser.add_argument("--p-std", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split")
    parser.add_argument("--condition-mode")
    parser.add_argument(
        "--data-manifest",
        dest="data_manifest_path",
        help=(
            "read-only upstream data manifest mapping every compact "
            "HDF5 day to its true daily offset; ablation runs bind "
            "all configurations to it for identical gap filtering"
        )
    )
    # Explicit small-architecture switches for the A0..A5 ablation.
    # Values given on the CLI win over the config JSON; cond_chans is
    # deliberately never a CLI flag (it derives from condition_mode).
    parser.add_argument(
        "--patch-size",
        nargs=3,
        type=int,
        metavar=("H", "W", "Z"),
        help="3D patch size [H W Z] for the IAFNO backbone"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="AFNO frequency-block count for the IAFNO backbone"
    )
    parser.add_argument(
        "--implicit-layer",
        type=int,
        help="implicit (weight-tied) iteration count for the backbone"
    )
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
    explicit_resume_fields = []
    field_names = (
        "train_h5_path",
        "output_dir",
        "resume_path",
        "init_from",
        "lead_stats_path",
        "mean_checkpoint_path",
        "centered_stats_path",
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
        "allow_resume_override",
        "seed",
        "split",
        "condition_mode",
        "data_manifest_path",
        "use_amp"
    )
    for field_name in field_names:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)
    # The condition schema is authoritative: the model-level
    # condition_mode mirrors the training-level switch and the channel
    # count/names/version are derived from the canonical tables, never
    # hand-filled.
    if config.condition_mode != config.model.condition_mode:
        config.model.adopt_condition_mode(config.condition_mode)
    patch_size = getattr(args, "patch_size", None)
    if patch_size is not None:
        if not isinstance(patch_size, (list, tuple)) or (
                len(patch_size) != 3
            ):
            raise ValueError(
                "patch_size must be a 3-element array of positive "
                f"integers, but got {patch_size!r}"
            )
        resolved_patch = tuple(
            int(value) for value in patch_size
        )
        if any(value < 1 for value in resolved_patch):
            raise ValueError(
                f"patch_size entries must be positive: {resolved_patch}"
            )
        config.model.patch_size = resolved_patch
        explicit_resume_fields.append("patch_size")
    num_blocks = getattr(args, "num_blocks", None)
    if num_blocks is not None:
        num_blocks = int(num_blocks)
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        config.model.num_blocks = num_blocks
        explicit_resume_fields.append("num_blocks")
    implicit_layer = getattr(args, "implicit_layer", None)
    if implicit_layer is not None:
        implicit_layer = int(implicit_layer)
        if implicit_layer < 1:
            raise ValueError("implicit_layer must be positive")
        config.model.implicit_layer = implicit_layer
        explicit_resume_fields.append("implicit_layer")
    if args.sampling_steps is not None:
        config.model.sampling_steps = args.sampling_steps
        explicit_resume_fields.append("sampling_steps")
    if args.target_mode is not None:
        config.model.target_mode = args.target_mode
        explicit_resume_fields.append("target_mode")
    if args.model_type is not None:
        config.model.model_type = args.model_type
        explicit_resume_fields.append("model_type")
    if args.target_scaling is not None:
        config.model.target_scaling = args.target_scaling
        explicit_resume_fields.append("target_scaling")
    if args.sigma_data is not None:
        config.model.sigma_data = args.sigma_data
        explicit_resume_fields.append("sigma_data")
    if args.sigma_max is not None:
        config.model.sigma_max = args.sigma_max
        explicit_resume_fields.append("sigma_max")
    if args.sigma_min is not None:
        config.model.sigma_min = args.sigma_min
        explicit_resume_fields.append("sigma_min")
    if args.p_mean is not None:
        config.model.p_mean = args.p_mean
        explicit_resume_fields.append("p_mean")
    if args.p_std is not None:
        config.model.p_std = args.p_std
        explicit_resume_fields.append("p_std")
    if args.rho is not None:
        config.model.rho = args.rho
        explicit_resume_fields.append("rho")
    # Centered path rejections happen before any legacy stats file is
    # opened: a centered run must never silently consume raw residual
    # stats or reuse another checkpoint's weights.
    if config.model.model_type == "centered_diffusion":
        if config.init_from is not None:
            raise ValueError(
                "--init-from is not supported for centered_diffusion"
            )
        if config.lead_stats_path is not None:
            raise ValueError(
                "--lead-stats carries raw residual statistics and is "
                "not valid for centered_diffusion; use --centered-stats"
            )
    if config.lead_stats_path is not None:
        stats_path = os.path.abspath(config.lead_stats_path)
        with open(stats_path, "r", encoding="utf-8") as file:
            stats = json.load(file)
        lead_mean, lead_std = validate_lead_stats_dict(
            stats,
            target_chans=config.model.target_chans,
            input_days=config.model.input_days,
            output_days=config.model.output_days,
        )
        config.model.lead_mean = lead_mean
        config.model.lead_std = lead_std
        explicit_resume_fields.extend(("lead_mean", "lead_std"))
    if (
            config.lead_stats_path is not None
            and config.model.target_scaling != "lead_standardized"
        ):
        raise ValueError(
            "--lead-stats is only valid with "
            "--target-scaling lead_standardized"
        )
    if (
            config.model.target_scaling == "lead_standardized"
            and config.lead_stats_path is None
            and config.resume_path is None
            and config.model.model_type != "centered_diffusion"
        ):
        raise ValueError(
            "--target-scaling lead_standardized requires "
            "--lead-stats"
        )
    if config.model.model_type == "centered_diffusion":
        _apply_centered_config_rules(config, explicit_resume_fields)
    for field_name in ("split", "condition_mode"):
        if getattr(args, field_name, None) is not None:
            explicit_resume_fields.append(field_name)
    config.explicit_resume_fields = tuple(
        sorted(set(explicit_resume_fields))
    )
    return config


def _apply_centered_config_rules(config, explicit_resume_fields):
    """Fail-closed CLI rules for centered_diffusion runs.

    Fresh runs must carry the frozen mean checkpoint and the train-only
    centered stats JSON; resume restores the same identity from the
    checkpoint sidecar instead (no mean path required).  A non-unit
    sigma_data is rejected outright (the init-from / lead-stats
    rejections already ran before any file IO in the caller).
    """
    sigma_data_explicit = "sigma_data" in set(
        explicit_resume_fields
    )
    if (
            config.resume_path is None
            or sigma_data_explicit
        ) and float(config.model.sigma_data) != 1.0:
        raise ValueError(
            "centered_diffusion requires sigma_data=1.0 "
            f"(got {config.model.sigma_data}); the factory default "
            "0.15 is a launch error for centered runs"
        )
    if config.centered_stats_path is None:
        if config.resume_path is None:
            raise ValueError(
                "fresh centered run requires --centered-stats"
            )
    else:
        stats_path = os.path.abspath(config.centered_stats_path)
        with open(stats_path, "r", encoding="utf-8") as file:
            stats = json.load(file)
        validated = validate_centered_stats_payload(
            stats,
            target_chans=config.model.target_chans,
            input_days=config.model.input_days,
            output_days=config.model.output_days,
        )
        config.model.lead_mean = validated["lead_mean"]
        config.model.lead_std = validated["lead_std"]
        config.model.mean_lead_mean = validated["mean_lead_mean"]
        config.model.mean_lead_std = validated["mean_lead_std"]
        config.model.mean_checkpoint_sha256 = validated[
            "mean_checkpoint_sha256"
        ]
        config.model.mean_semantics_sha256 = validated[
            "mean_semantics_sha256"
        ]
        explicit_resume_fields.extend(
            (
                "lead_mean",
                "lead_std",
                "mean_lead_mean",
                "mean_lead_std",
                "mean_checkpoint_sha256",
                "mean_semantics_sha256",
            )
        )
    if (
            config.resume_path is None
            and config.mean_checkpoint_path is None
        ):
        raise ValueError(
            "fresh centered run requires --mean-checkpoint"
        )


def validate_lead_stats_dict(
        stats,
        target_chans,
        input_days,
        output_days,
    ):
    """Validate a lead-stats JSON payload and return
    (lead_mean_tuple, lead_std_tuple).

    Enforces the train-only provenance contract: the payload must
    declare the normalized residual target space, the train split, the
    selection method, and consistent lead dimensions with positive stds.
    """
    if not isinstance(stats, dict):
        raise ValueError(
            "lead stats payload must be a JSON object"
        )
    lead_mean = stats.get(
        "lead_mean",
        stats.get("residual_lead_mean")
    )
    lead_std = stats.get(
        "lead_std",
        stats.get("residual_lead_std")
    )
    if lead_mean is None or lead_std is None:
        raise ValueError(
            "lead stats JSON must contain lead_mean/lead_std"
        )
    if not isinstance(lead_mean, (list, tuple)) or not isinstance(
            lead_std, (list, tuple)
        ):
        raise ValueError(
            "lead_mean/lead_std must be arrays of floats"
        )
    if len(lead_mean) != target_chans or len(lead_std) != target_chans:
        raise ValueError(
            f"lead stats length {len(lead_mean)}/{len(lead_std)} "
            f"does not match target_chans={target_chans}"
        )
    try:
        lead_mean = tuple(float(value) for value in lead_mean)
        lead_std = tuple(float(value) for value in lead_std)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "lead stats must contain only numeric values"
        ) from error
    if any(not math.isfinite(value) for value in lead_mean):
        raise ValueError("all lead_mean values must be finite")
    if any(
            not math.isfinite(value) or value <= 0.0
            for value in lead_std
        ):
        raise ValueError(
            "all lead_std values must be finite and positive"
        )
    target_space = stats.get("target_space")
    if target_space not in (
            "normalized_residual",
            "residual",
            "normalized_centered_residual",
        ):
        raise ValueError(
            "lead stats must declare target_space="
            "'normalized_residual' or "
            "'normalized_centered_residual' (got "
            f"{target_space!r})"
        )
    split = stats.get("split")
    if split != "train":
        raise ValueError(
            f"lead stats must come from the train split (got {split!r}); "
            "validation/test data must never enter training statistics"
        )
    if stats.get("input_days") not in (None, input_days):
        raise ValueError(
            f"lead stats input_days={stats.get('input_days')} does not "
            f"match current input_days={input_days}"
        )
    if stats.get("output_days") not in (None, output_days):
        raise ValueError(
            f"lead stats output_days={stats.get('output_days')} does "
            f"not match current output_days={output_days}"
        )
    return lead_mean, lead_std
