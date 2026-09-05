# 用途：按配置恢复用于推理的模型权重。
import torch

from ..models.config import OSTIAModelConfig


class InferenceModelLoader:
    @staticmethod
    # 用途：按 checkpoint 语义加载推理模型（读 sidecar 重建配置，可覆盖采样步数）。
    # 参数：输入 checkpoint_path（checkpoint 路径）、device（设备）、sampling_steps（覆盖采样步数，可选）；输出 组装好的推理模型。
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
