from dataclasses import dataclass, fields
from typing import Tuple

import torch

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff


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
            values["image_size"] = tuple(
                values["image_size"]
            )
        if "patch_size" in values:
            values["patch_size"] = tuple(
                values["patch_size"]
            )
        return cls(**values)

    def build_model(self, device, sampling_steps):
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
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        )
        model = ElucidatedDiffusion(
            backbone,
            channels=self.target_chans,
            num_sample_steps=sampling_steps,
            image_size_h=self.image_size[0],
            image_size_w=self.image_size[1],
            image_size_z=self.image_size[2],
            sigma_data=self.sigma_data
        )
        return model.to(
            device=device,
            dtype=torch.float32
        )


class InferenceModelLoader:
    @staticmethod
    def load(
            checkpoint_path,
            device,
            sampling_steps=None,
        ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )
        model_config = OSTIAModelConfig.from_checkpoint(
            checkpoint["config"]
        )
        resolved_steps = (
            model_config.sampling_steps
            if sampling_steps is None
            else sampling_steps
        )
        model = model_config.build_model(
            device,
            resolved_steps
        )
        state_dict = checkpoint.get(
            "model",
            checkpoint
        )
        if any(
                key.startswith("module.")
                for key in state_dict
            ):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        model.load_state_dict(
            state_dict,
            strict=True
        )
        model.eval()
        normalization = checkpoint.get("normalization")
        return model, model_config, resolved_steps, normalization
