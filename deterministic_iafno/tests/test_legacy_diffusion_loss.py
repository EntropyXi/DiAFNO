# 用途：验证旧扩散损失的兼容性与数值行为。
import unittest

import torch
from torch import nn

from diafno.models.diffusion import ElucidatedDiffusion


class ZeroNet(nn.Module):
    """Dummy network whose output is zero: D = c_skip * y."""

    def __init__(self):
        super().__init__()
        # ElucidatedDiffusion.device derives from the net's parameters,
        # so the dummy net needs at least one parameter.
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, time, condition):
        return torch.zeros_like(x)


class LegacyMaskedLossSemanticsTests(unittest.TestCase):
    """Guard the legacy per-sample masked mean loss semantics.

    ElucidatedDiffusion.forward must keep the ORIGINAL contract:
    per-sample masked mean of (D-target)^2, multiplied by the sigma
    weight, then averaged over the batch.  This test pins the exact
    formula so a future refactor (e.g. global masked normalization)
    cannot silently change legacy diffusion training behavior.
    """

    def _model(self):
        net = ZeroNet()
        return ElucidatedDiffusion(
            net,
            channels=2,
            image_size_h=4,
            image_size_w=4,
            image_size_z=1,
            num_sample_steps=16,
            sigma_min=0.0005,
            sigma_max=1.0,
            sigma_data=0.15,
            P_mean=-3.0,
        )

    def test_masked_loss_matches_reference_formula(self):
        model = self._model()
        target = torch.randn(2, 2, 4, 4, 1)
        condition = torch.randn(2, 8, 4, 4, 1)
        mask = (torch.rand(2, 2, 4, 4, 1) > 0.3).float()

        # Pin the RNG stream immediately before the model call so the
        # internally drawn sigmas/noise are known.
        torch.manual_seed(7)
        loss = model(target, condition, target_mask=mask)

        # Reference replication of the ORIGINAL semantics with the same
        # RNG stream (sigmas, then noise).
        torch.manual_seed(7)
        sigmas = model.noise_distribution(2)
        noise = torch.randn_like(target)
        noised = target + sigmas.view(2, 1, 1, 1, 1) * noise
        denoised = model.preconditioned_network_forward(
            noised,
            sigmas,
            condition,
        )
        losses = (denoised - target) ** 2
        reduce_dims = tuple(range(1, losses.ndim))
        valid_count = mask.sum(dim=reduce_dims).clamp_min(1.0)
        per_sample = (
            losses * mask
        ).sum(dim=reduce_dims) / valid_count
        per_sample = per_sample * model.loss_weight(sigmas)
        reference = per_sample.mean()

        self.assertTrue(torch.allclose(loss, reference, atol=1e-6))

    def test_unmasked_loss_still_per_sample_mean(self):
        model = self._model()
        target = torch.randn(3, 2, 4, 4, 1)
        condition = torch.randn(3, 8, 4, 4, 1)

        torch.manual_seed(11)
        loss = model(target, condition)

        torch.manual_seed(11)
        sigmas = model.noise_distribution(3)
        noise = torch.randn_like(target)
        noised = target + sigmas.view(3, 1, 1, 1, 1) * noise
        denoised = model.preconditioned_network_forward(
            noised,
            sigmas,
            condition,
        )
        losses = (denoised - target) ** 2
        reference = (
            losses.mean(dim=tuple(range(1, losses.ndim)))
            * model.loss_weight(sigmas)
        ).mean()
        self.assertTrue(torch.allclose(loss, reference, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
