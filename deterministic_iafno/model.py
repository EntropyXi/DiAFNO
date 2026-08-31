import math

import torch
from torch import nn

from .losses import globally_normalized_masked_mse


class DeterministicIAFNO(nn.Module):
    """Regression adapter around the raw IAFNODiff backbone.

    This class deliberately bypasses all EDM preconditioning.  The fixed
    time value is a raw-network embedding input; it is not sigma=0.
    """

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

    def transform_target(self, target):
        return (target - self.lead_mean) / self.lead_std

    def inverse_target(self, target):
        return target * self.lead_std + self.lead_mean

    def predict(self, condition):
        return self.inverse_target(
            self._network_prediction(condition)
        )

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
