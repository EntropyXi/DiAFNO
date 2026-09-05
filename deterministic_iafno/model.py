# 用途：封装确定性 IAFNO，预测按 lead 标准化的 SST 残差。
import math

import torch
from torch import nn

from .losses import globally_normalized_masked_mse


class DeterministicIAFNO(nn.Module):
    """Regression adapter around the raw IAFNODiff backbone.

    This class deliberately bypasses all EDM preconditioning.  The fixed
    time value is a raw-network embedding input; it is not sigma=0.
    """

    # 用途：初始化确定性回归适配器：登记目标语义与 lead 标准化统计量，旁路一切 EDM 预条件。
    # 参数：输入 net（IAFNODiff 主干）、target_chans（目标通道 15）、target_scaling（raw 或 lead_standardized）、lead_mean/lead_std（逐 lead 残差统计量）、fixed_time_value（固定时间嵌入值）；输出 无（构造模块状态）。
    def __init__(
            self,
            net,
            target_chans,
            target_scaling="raw",
            lead_mean=None,
            lead_std=None,
            fixed_time_value=0.0,
        ):
        super().__init__()
        if target_scaling not in ("raw", "lead_standardized"):
            raise ValueError(
                "target_scaling must be 'raw' or "
                "'lead_standardized'"
            )
        self.net = net
        self.target_chans = int(target_chans)
        self.target_scaling = target_scaling
        self.fixed_time_value = float(fixed_time_value)

        if target_scaling == "raw":
            lead_mean = [0.0] * self.target_chans
            lead_std = [1.0] * self.target_chans
        if lead_mean is None or lead_std is None:
            raise ValueError(
                "lead_mean and lead_std are required for "
                "lead_standardized targets"
            )
        if (
                len(lead_mean) != self.target_chans
                or len(lead_std) != self.target_chans
            ):
            raise ValueError(
                "lead statistics must match target_chans"
            )
        if any(not math.isfinite(float(value)) for value in lead_mean):
            raise ValueError("all lead_mean values must be finite")
        if any(
                not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in lead_std
            ):
            raise ValueError(
                "all lead_std values must be finite and positive"
            )
        stats_shape = (1, self.target_chans, 1, 1, 1)
        self.register_buffer(
            "lead_mean",
            torch.tensor(lead_mean, dtype=torch.float32).view(
                stats_shape
            ),
            persistent=False,
        )
        self.register_buffer(
            "lead_std",
            torch.tensor(lead_std, dtype=torch.float32).view(
                stats_shape
            ),
            persistent=False,
        )

    # 用途：以全零目标与固定时间值跑主干，得到未经反标准化的网络预测。
    # 参数：输入 condition（条件场 [B,C,H,W,Z]）；输出 预测 [B,target_chans,H,W,Z]（标准化空间）。
    def _network_prediction(self, condition):
        batch, _, height, width, depth = condition.shape
        zeros = torch.zeros(
            (
                batch,
                self.target_chans,
                height,
                width,
                depth,
            ),
            device=condition.device,
            dtype=condition.dtype,
        )
        fixed_time = torch.full(
            (batch,),
            self.fixed_time_value,
            device=condition.device,
            dtype=condition.dtype,
        )
        return self.net(zeros, fixed_time, condition)

    # 用途：按 lead 统计量标准化目标（raw 模式下为恒等）。
    # 参数：输入 target（残差目标场）；输出 (target-lead_mean)/lead_std。
    def transform_target(self, target):
        return (target - self.lead_mean) / self.lead_std

    # 用途：按 lead 统计量反标准化（raw 模式下为恒等）。
    # 参数：输入 target（标准化空间值）；输出 target*lead_std+lead_mean。
    def inverse_target(self, target):
        return target * self.lead_std + self.lead_mean

    # 用途：推理接口：网络前向后反标准化为 normalized residual 预测。
    # 参数：输入 condition（条件场）；输出 预测的残差场（normalized residual 空间）。
    def predict(self, condition):
        return self.inverse_target(
            self._network_prediction(condition)
        )

    # 用途：训练前向：预测与标准化目标的逐元素平方误差，按 mask 做全局归一化均值。
    # 参数：输入 target（残差目标）、condition（条件场）、target_mask（有效像素 mask，可选）；输出 标量损失。
    def forward(self, target, condition, target_mask=None):
        prediction = self._network_prediction(condition)
        transformed_target = self.transform_target(target)
        losses = (prediction - transformed_target).square()
        if target_mask is None:
            return losses.mean()
        if target_mask.shape != losses.shape:
            raise ValueError(
                f"target_mask shape {target_mask.shape} does not "
                f"match target shape {losses.shape}"
            )
        return globally_normalized_masked_mse(
            losses,
            target_mask,
        )
