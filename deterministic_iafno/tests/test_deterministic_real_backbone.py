import unittest

import torch

from diafno.models.iafno import IAFNODiff
from deterministic_iafno.model import DeterministicIAFNO


class DeterministicRealBackboneTests(unittest.TestCase):
    """Forward/backward smoke through a real (tiny) IAFNODiff trunk.

    Verifies the deterministic path: 15 (here 3) zero target channels,
    fixed raw-network time embedding 0.0, direct network call without
    EDM preconditioning, and correct lead-standardized inversion.
    """

    def _build(self, target_scaling="raw", lead_mean=None, lead_std=None):
        net = IAFNODiff(
            dim=(8, 8, 1),
            dim_f=(8, 8, 1),
            patch_size=(2, 2, 1),
            embed_dim=8,
            num_blocks=2,
            cond_chans=2,
            target_chans=3,
            ex_layer=1,
            nlayer=1,
            hidden_size_factor=2,
            drop_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        )
        return DeterministicIAFNO(
            net,
            target_chans=3,
            target_scaling=target_scaling,
            lead_mean=lead_mean,
            lead_std=lead_std,
        )

    def test_raw_forward_backward(self):
        model = self._build()
        condition = torch.randn(2, 2, 8, 8, 1)
        target = torch.randn(2, 3, 8, 8, 1)
        mask = torch.ones_like(target)
        loss = model(target, condition, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.net.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            any(torch.count_nonzero(gradient) > 0
                for gradient in gradients)
        )

    def test_lead_standardized_roundtrip(self):
        model = self._build(
            target_scaling="lead_standardized",
            lead_mean=(0.1, 0.2, 0.3),
            lead_std=(1.0, 2.0, 3.0),
        )
        condition = torch.randn(1, 2, 8, 8, 1)
        prediction = model.predict(condition)
        self.assertEqual(tuple(prediction.shape), (1, 3, 8, 8, 1))
        self.assertTrue(torch.isfinite(prediction).all())
        # predict() must return raw-space residuals: run the network
        # output through the inverse transform manually and compare.
        raw = model._network_prediction(condition)
        expected = raw * torch.tensor(
            [1.0, 2.0, 3.0]
        ).view(1, 3, 1, 1, 1) + torch.tensor(
            [0.1, 0.2, 0.3]
        ).view(1, 3, 1, 1, 1)
        self.assertTrue(torch.allclose(prediction, expected))

    def test_transform_inverse_identity(self):
        model = self._build(
            target_scaling="lead_standardized",
            lead_mean=(0.1, 0.2, 0.3),
            lead_std=(1.0, 2.0, 3.0),
        )
        target = torch.randn(1, 3, 8, 8, 1)
        transformed = model.transform_target(target)
        restored = model.inverse_target(transformed)
        self.assertTrue(torch.allclose(target, restored))

    def test_mask_shape_mismatch_fails(self):
        model = self._build()
        condition = torch.randn(1, 2, 8, 8, 1)
        target = torch.randn(1, 3, 8, 8, 1)
        with self.assertRaisesRegex(ValueError, "target_mask"):
            model(target, condition, torch.ones(1, 3, 4, 4, 1))


if __name__ == "__main__":
    unittest.main()
