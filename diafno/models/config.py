from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import torch

from ..data.condition_schema import (
    CONDITION_MODES,
    condition_channel_names,
    condition_schema_version_for,
)
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
    # Condition-schema contract.  Persisted with every checkpoint and
    # semantic sidecar so training, resume, validation and inference
    # all restore the same data contract.  Legacy checkpoints lack the
    # fields and are interpreted as 'sst_mask'/version 1 with their
    # stored cond_chans; channel counts and ordering are validated
    # against the canonical schema tables at build time instead of
    # trusting hand-written cond_chans values.
    condition_mode: str = "sst_mask"
    condition_schema_version: int = 1
    condition_channel_names: Optional[Tuple[str, ...]] = None
    # Decoded HDF5 date semantics (geo-season mode only): the resolved
    # calendar and the proven time reference, e.g.
    # 'days since 2020-01-01'.  None for legacy modes.
    calendar_encoding: Optional[str] = None
    time_units_reference: Optional[str] = None
    # Provenance of the static lat/lon grids (geo-season mode only).
    geospatial_summary: Optional[dict] = None
    # Provenance of the real-day time axis (geo-season mode only):
    # per-day offsets sha256 / gaps / calendar, plus the identity of
    # the upstream data manifest when one was used.  Two files with
    # the same shape but a different time mapping can never pass the
    # checkpoint contract.
    time_axis_summary: Optional[dict] = None
    data_manifest_sha256: Optional[str] = None

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
        if values.get("condition_channel_names") is not None:
            values["condition_channel_names"] = tuple(
                values["condition_channel_names"]
            )
        return cls(**values)

    # -- condition-schema helpers ------------------------------------

    def canonical_condition_channel_names(self):
        """Fixed channel order required by the declared mode."""
        return condition_channel_names(
            self.condition_mode,
            self.input_days,
        )

    def adopt_condition_mode(self, condition_mode):
        """Make the whole condition schema canonical for one mode.

        The channel table is authoritative: ``cond_chans`` and the
        channel-name tuple are derived, never hand-maintained.  Call
        this from the training config / data setup only; validation
        and inference restore the same fields from the checkpoint.
        """
        if condition_mode not in CONDITION_MODES:
            raise ValueError(
                f"condition_mode must be one of {CONDITION_MODES}, "
                f"but got {condition_mode!r}"
            )
        self.condition_mode = condition_mode
        self.condition_schema_version = (
            condition_schema_version_for(condition_mode)
        )
        self.condition_channel_names = (
            self.canonical_condition_channel_names()
        )
        self.cond_chans = len(self.condition_channel_names)
        return self

    def validate_condition_schema(self):
        """Fail-closed channel/schema checks before any model build.

        Runs before any parameter tensors exist, so an 8-versus-14
        channel mistake or a hand-edited channel list can never reach
        a state-dict load.  The SST history always occupies channels
        ``0 .. input_days-1`` (channel ``input_days-1`` is the t0
        anchor), followed by the t0 mask and then the static channels.

        Schema-managed configs (condition mode adopted by the trainer,
        geo-season mode, or any stored schema-version/channel-name
        metadata) are validated strictly.  The exact legacy-unmanaged
        shape -- default ``sst_mask`` mode, schema version 1 and no
        stored channel names -- keeps the historical freedom of tiny
        hand-built configs (regression tests, toy networks) whose
        cond_chans is an arbitrary tensor-channel count.
        """
        if self.condition_mode not in CONDITION_MODES:
            raise ValueError(
                f"condition_mode must be one of {CONDITION_MODES}, "
                f"but got {self.condition_mode!r}"
            )
        legacy_unmanaged = (
            self.condition_mode == "sst_mask"
            and int(self.condition_schema_version) == 1
            and self.condition_channel_names is None
        )
        if legacy_unmanaged:
            return
        canonical = self.canonical_condition_channel_names()
        if self.condition_channel_names is not None:
            if tuple(self.condition_channel_names) != canonical:
                raise ValueError(
                    "condition_channel_names do not match the fixed "
                    f"schema for condition_mode={self.condition_mode!r} "
                    "with input_days="
                    f"{self.input_days}: stored "
                    f"{tuple(self.condition_channel_names)} versus "
                    f"canonical {canonical}; the channel layout is "
                    "immutable and must not be hand-edited"
                )
        if self.cond_chans != len(canonical):
            raise ValueError(
                "condition channel count mismatch: the condition "
                f"schema for condition_mode={self.condition_mode!r} "
                f"with input_days={self.input_days} requires "
                f"cond_chans={len(canonical)}, but the model config "
                f"declares cond_chans={self.cond_chans}.  cond_chans "
                "must come from the condition schema, not from a "
                "hand-written value"
            )
        expected_version = condition_schema_version_for(
            self.condition_mode
        )
        if self.condition_schema_version != expected_version:
            raise ValueError(
                "condition_schema_version mismatch: "
                f"condition_mode={self.condition_mode!r} requires "
                f"version {expected_version}, but the config stores "
                f"{self.condition_schema_version}"
            )

    def build_model(self, device, sampling_steps=None):
        self.validate_condition_schema()
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
