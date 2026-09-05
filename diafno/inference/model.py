# 用途：按配置恢复用于推理的模型权重。
import torch

from ..models.config import OSTIAModelConfig


class InferenceModelLoader:
    @staticmethod
    def load(
            checkpoint_path,
            device,
            sampling_steps=None,
        ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
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
