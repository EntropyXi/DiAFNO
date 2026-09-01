import math

import torch
from torch import nn


class FrozenMeanCenteredDiffusion(nn.Module):
    """Frozen-mean centered diffusion wrapper.

    Contract (locked by ``deterministic_iafno/tests/test_centered_diffusion.py``):

    - registers ``mean_model`` and ``diffusion`` submodules;
    - freezes ``mean_model`` at construction and forces it back to
      ``eval()`` after every ``train(mode)`` call (still holds after DDP
      wrapping because DDP forwards ``train(mode)`` to the wrapped module);
    - ``forward(residual_target, condition, target_mask)``:
      ``mu = mean_model.predict(condition)`` is evaluated in fp32 under
      ``no_grad`` (its output is ALREADY in normalized residual space —
      no mean lead stats are ever re-applied), then
      ``e = residual_target - mu`` and ``z = (e - m) / s`` are computed
      in fp32, and only ``z`` is passed to the EDM diffusion loss;
    - ``sample(condition, num_sample_steps=None, seed=None)`` returns the
      normalized residual forecast ``r_hat = mu + m + s * z_hat`` (no
      anchor added, no SST denormalization — the evaluator re-anchors
      exactly once);
    - sampler attributes (``S_churn``, ``sigma_min``, ``sigma_max``,
      ``rho``, ``num_sample_steps``) are two-way delegated to the inner
      diffusion;
    - intentionally does NOT expose ``preconditioned_network_forward``.
    """

    def __init__(
            self,
            mean_model,
            diffusion,
            lead_mean,
            lead_std,
        ):
        super().__init__()
        if mean_model is None:
            raise ValueError("mean_model is required")
        if diffusion is None:
            raise ValueError("diffusion is required")
        channels = int(diffusion.channels)
        if len(lead_mean) != channels or len(lead_std) != channels:
            raise ValueError(
                "innovation lead statistics must match "
                f"diffusion channels={channels}"
            )
        if any(
                not math.isfinite(float(value))
                for value in lead_mean
            ):
            raise ValueError(
                "all innovation lead_mean values must be finite"
            )
        if any(
                not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in lead_std
            ):
            raise ValueError(
                "all innovation lead_std values must be finite "
                "and positive"
            )
        self.mean_model = mean_model
        self.diffusion = diffusion
        # Freeze before any optimizer can see the parameters.
        self.mean_model.requires_grad_(False)
        self.mean_model.eval()
        stats_shape = (1, channels, 1, 1, 1)
        self.register_buffer(
            "innovation_mean",
            torch.tensor(lead_mean, dtype=torch.float32).view(
                stats_shape
            ),
            persistent=False,
        )
        self.register_buffer(
            "innovation_std",
            torch.tensor(lead_std, dtype=torch.float32).view(
                stats_shape
            ),
            persistent=False,
        )

    def train(self, mode=True):
        super().train(mode)
        # The frozen deterministic mean must never leave eval mode,
        # even after wrapper.train() / DDP .train().
        self.mean_model.eval()
        return self

    # ---- two-way sampler attribute delegation ---------------------

    @property
    def S_churn(self):
        return self.diffusion.S_churn

    @S_churn.setter
    def S_churn(self, value):
        self.diffusion.S_churn = value

    @property
    def sigma_min(self):
        return self.diffusion.sigma_min

    @sigma_min.setter
    def sigma_min(self, value):
        self.diffusion.sigma_min = value

    @property
    def sigma_max(self):
        return self.diffusion.sigma_max

    @sigma_max.setter
    def sigma_max(self, value):
        self.diffusion.sigma_max = value

    @property
    def rho(self):
        return self.diffusion.rho

    @rho.setter
    def rho(self, value):
        self.diffusion.rho = value

    @property
    def num_sample_steps(self):
        return self.diffusion.num_sample_steps

    @num_sample_steps.setter
    def num_sample_steps(self, value):
        self.diffusion.num_sample_steps = value

    # ---- innovation standardization helpers -----------------------

    def transform_innovation(self, innovation):
        """``z = (e - m) / s`` in fp32 (per-lead innovation stats)."""
        return (
            (innovation.float() - self.innovation_mean)
            / self.innovation_std
        )

    def inverse_innovation(self, standardized):
        """``e = z * s + m`` in fp32 (per-lead innovation stats)."""
        return (
            standardized.float() * self.innovation_std
            + self.innovation_mean
        )

    # ---- forward / sampling ---------------------------------------

    def _check_shapes(self, residual_target, condition):
        if residual_target.ndim != 5:
            raise ValueError(
                "residual_target must have shape [B,C,H,W,Z], "
                f"but got {residual_target.shape}"
            )
        if condition.ndim != 5:
            raise ValueError(
                "condition must have shape [B,C,H,W,Z], "
                f"but got {condition.shape}"
            )
        if residual_target.shape[0] != condition.shape[0]:
            raise ValueError(
                "target and condition batch sizes do not match"
            )
        if residual_target.shape[2:] != condition.shape[2:]:
            raise ValueError(
                "target and condition spatial shapes do not match"
            )

    def _frozen_mean_prediction(self, condition):
        """fp32, no-grad mean prediction in normalized residual space.

        The autocast-disable scope guarantees the mean backbone runs in
        fp32 even when the caller wraps this forward pass in AMP.
        """
        with torch.no_grad(), torch.autocast(
                device_type=condition.device.type,
                enabled=False,
            ):
            return self.mean_model.predict(
                condition.float()
            ).float()

    def forward(self, residual_target, condition, target_mask=None):
        self._check_shapes(residual_target, condition)
        mu = self._frozen_mean_prediction(condition)
        # Both subtractions/divisions are element-wise fp32 operations:
        # the centered innovation is never computed in half precision.
        innovation = residual_target.float() - mu
        standardized = self.transform_innovation(innovation)
        return self.diffusion(
            standardized,
            condition,
            target_mask=target_mask,
        )

    @torch.no_grad()
    def sample(self, condition, num_sample_steps=None, seed=None):
        z_hat = self.diffusion.sample(
            condition,
            num_sample_steps=num_sample_steps,
            seed=seed,
        )
        mu = self._frozen_mean_prediction(condition)
        innovation = self.inverse_innovation(z_hat)
        # Single reconstruction: r_hat = mu + m + s*z_hat.  No anchor,
        # no SST denormalization here.
        return (mu + innovation).float()
