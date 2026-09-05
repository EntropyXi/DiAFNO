# 用途：定义模型配置并按 checkpoint 语义构造对应模型。
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import torch

from .diffusion import ElucidatedDiffusion
from .iafno import IAFNODiff
from deterministic_iafno.centered_diffusion import (
    FrozenMeanCenteredDiffusion,
)
from deterministic_iafno.model import DeterministicIAFNO


@dataclass
class OSTIAModelConfig:
    input_days: int = 7
    output_days: int = 15
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
    # NOTE: keep these defaults at the legacy absolute-training values so that
    # old checkpoints (which lack the fields) reproduce their original
    # sampling schedule when validated with new code.
    sigma_max: float = 80.0
    sigma_min: float = 0.002
    p_mean: float = -1.2
    p_std: float = 1.2
    rho: float = 7.0
    target_mode: str = "absolute"
    model_type: str = "diffusion"
    target_scaling: str = "raw"
    # Phase 1 raw-residual (or Phase 2 centered innovation) lead stats.
    lead_mean: Optional[Tuple[float, ...]] = None
    lead_std: Optional[Tuple[float, ...]] = None
    # Frozen deterministic mean identity + its own residual lead stats.
    # Only meaningful for model_type='centered_diffusion'; None keeps
    # legacy checkpoints byte-compatible.
    mean_lead_mean: Optional[Tuple[float, ...]] = None
    mean_lead_std: Optional[Tuple[float, ...]] = None
    mean_checkpoint_sha256: Optional[str] = None
    mean_semantics_sha256: Optional[str] = None

    def to_checkpoint(self):
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, config):
        config = dict(config)
        if not all(
                key in config
                for key in ("input_days", "output_days")
            ):
            if all(
                    key in config
                    for key in ("input_months", "output_months")
                ):
                config["input_days"] = config["input_months"]
                config["output_days"] = config["output_months"]
            else:
                raise ValueError(
                    "checkpoint uses the old weekly time indexing and "
                    "cannot be used for daily OSTIA inference"
                )
        field_names = {item.name for item in fields(cls)}
        values = {
            key: value
            for key, value in config.items()
            if key in field_names
        }
        if "image_size" in values:
            values["image_size"] = tuple(values["image_size"])
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        if values.get("lead_mean") is not None:
            values["lead_mean"] = tuple(values["lead_mean"])
        if values.get("lead_std") is not None:
            values["lead_std"] = tuple(values["lead_std"])
        if values.get("mean_lead_mean") is not None:
            values["mean_lead_mean"] = tuple(values["mean_lead_mean"])
        if values.get("mean_lead_std") is not None:
            values["mean_lead_std"] = tuple(values["mean_lead_std"])
        return cls(**values)

    def build_model(self, device, sampling_steps=None):
        if self.target_mode not in ("absolute", "residual"):
            raise ValueError(
                "target_mode must be 'absolute' or 'residual'"
            )
        if self.model_type not in (
                "diffusion",
                "deterministic",
                "centered_diffusion",
            ):
            raise ValueError(
                "model_type must be 'diffusion', 'deterministic' "
                "or 'centered_diffusion'"
            )
        if (
                self.model_type == "deterministic"
                and self.target_mode != "residual"
            ):
            raise ValueError(
                "deterministic OSTIA training requires "
                "target_mode='residual'"
            )
        if (
                self.model_type == "diffusion"
                and self.target_scaling != "raw"
            ):
            raise ValueError(
                "lead-standardized targets are only supported by "
                "model_type='deterministic'"
            )
        if self.model_type == "centered_diffusion":
            if self.target_mode != "residual":
                raise ValueError(
                    "centered_diffusion requires "
                    "target_mode='residual'"
                )
            if self.target_scaling != "lead_standardized":
                raise ValueError(
                    "centered_diffusion requires "
                    "target_scaling='lead_standardized' "
                    "(centered innovation semantics)"
                )
            # Centered targets are standardized innovations; any
            # sigma_data != 1.0 is a launch error, not a warning.
            if float(self.sigma_data) != 1.0:
                raise ValueError(
                    "centered_diffusion requires sigma_data=1.0 "
                    f"(got {self.sigma_data})"
                )
            if self.lead_mean is None or self.lead_std is None:
                raise ValueError(
                    "centered_diffusion requires innovation "
                    "lead_mean/lead_std"
                )
            if (
                    self.mean_lead_mean is None
                    or self.mean_lead_std is None
                ):
                raise ValueError(
                    "centered_diffusion requires frozen-mean "
                    "mean_lead_mean/mean_lead_std"
                )
        if self.model_type in ("diffusion", "centered_diffusion"):
            if self.sigma_data <= 0.0:
                raise ValueError("sigma_data must be positive")
            if self.p_std <= 0.0:
                raise ValueError("p_std must be positive")
            if self.rho <= 0.0:
                raise ValueError("rho must be positive")
            if self.sigma_min <= 0.0:
                raise ValueError("sigma_min must be positive")
            if self.sigma_max <= self.sigma_min:
                raise ValueError(
                    "sigma_max must be greater than sigma_min"
                )
        resolved_steps = (
            self.sampling_steps
            if sampling_steps is None
            else sampling_steps
        )

        def _build_backbone():
            return IAFNODiff(
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

        if self.model_type == "deterministic":
            model = DeterministicIAFNO(
                _build_backbone(),
                target_chans=self.target_chans,
                target_scaling=self.target_scaling,
                lead_mean=self.lead_mean,
                lead_std=self.lead_std,
            )
        elif self.model_type == "centered_diffusion":
            # The frozen mean is rebuilt from its own residual lead
            # stats; its weights are loaded from the checkpoint state
            # dict (mean_model.* prefix), so no external mean file is
            # needed for inference or resume.
            mean_model = DeterministicIAFNO(
                _build_backbone(),
                target_chans=self.target_chans,
                target_scaling="lead_standardized",
                lead_mean=self.mean_lead_mean,
                lead_std=self.mean_lead_std,
            )
            diffusion = ElucidatedDiffusion(
                _build_backbone(),
                channels=self.target_chans,
                num_sample_steps=resolved_steps,
                image_size_h=self.image_size[0],
                image_size_w=self.image_size[1],
                image_size_z=self.image_size[2],
                sigma_data=self.sigma_data,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                rho=self.rho,
                P_mean=self.p_mean,
                P_std=self.p_std,
            )
            model = FrozenMeanCenteredDiffusion(
                mean_model,
                diffusion,
                lead_mean=self.lead_mean,
                lead_std=self.lead_std,
            )
        else:
            model = ElucidatedDiffusion(
                _build_backbone(),
                channels=self.target_chans,
                num_sample_steps=resolved_steps,
                image_size_h=self.image_size[0],
                image_size_w=self.image_size[1],
                image_size_z=self.image_size[2],
                sigma_data=self.sigma_data,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                rho=self.rho,
                P_mean=self.p_mean,
                P_std=self.p_std,
            )
        return model.to(
            device=device,
            dtype=torch.float32
        )
