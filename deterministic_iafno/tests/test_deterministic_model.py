# 用途：验证确定性残差模型的前向及损失。
import unittest

import torch
from torch import nn

from deterministic_iafno.model import DeterministicIAFNO


class RecordingNet(nn.Module):
    def __init__(self, target_chans):
        super().__init__()
        self.target_chans = target_chans
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_x = None
        self.last_time = None

    def forward(self, x, time, condition):
        self.last_x = x.detach().clone()
        self.last_time = time.detach().clone()
        return (
            condition[:, :1].repeat(
                1,
                self.target_chans,
                1,
                1,
                1,
            )
            * self.scale
        )


class DeterministicModelTests(unittest.TestCase):
    def test_raw_path_uses_zero_target_and_fixed_time(self):
        net = RecordingNet(2)
        model = DeterministicIAFNO(net, 2)
        condition = torch.ones(3, 8, 2, 2, 1)
        prediction = model.predict(condition)
        self.assertEqual(tuple(prediction.shape), (3, 2, 2, 2, 1))
        self.assertTrue(torch.count_nonzero(net.last_x) == 0)
        self.assertTrue(torch.count_nonzero(net.last_time) == 0)

    def test_lead_standardized_prediction_is_inverted(self):
        net = RecordingNet(2)
        model = DeterministicIAFNO(
            net,
            2,
            target_scaling="lead_standardized",
            lead_mean=(0.5, 1.0),
            lead_std=(2.0, 4.0),
        )
        condition = torch.ones(1, 8, 1, 1, 1)
        prediction = model.predict(condition)
        self.assertTrue(torch.allclose(
            prediction.flatten(),
            torch.tensor([2.5, 5.0]),
        ))

    def test_masked_mse_backpropagates_to_backbone(self):
        net = RecordingNet(2)
        model = DeterministicIAFNO(net, 2)
        condition = torch.ones(1, 8, 1, 1, 1)
        target = torch.zeros(1, 2, 1, 1, 1)
        mask = torch.ones_like(target)
        loss = model(target, condition, mask)
        loss.backward()
        self.assertIsNotNone(net.scale.grad)
        self.assertGreater(abs(net.scale.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
