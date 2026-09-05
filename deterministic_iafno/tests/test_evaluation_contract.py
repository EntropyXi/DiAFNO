# 用途：验证有效像素、残差还原及评估协议。
import unittest
from types import SimpleNamespace

import torch

from diafno.evaluation.metrics import persistence_skill
from diafno.evaluation.validator import OSTIAValidator


class EvaluationContractTests(unittest.TestCase):
    def validator(self, mode="none", prediction_mode="model"):
        validator = OSTIAValidator.__new__(OSTIAValidator)
        validator.config = SimpleNamespace(
            condition_ablation=mode,
            prediction_mode=prediction_mode,
        )
        validator.model_config = SimpleNamespace(
            input_days=7,
            output_days=15,
        )
        return validator

    @staticmethod
    def condition():
        condition = torch.zeros(2, 8, 1, 1, 1)
        condition[0, :7, 0, 0, 0] = torch.arange(7)
        condition[1, :7, 0, 0, 0] = torch.arange(10, 17)
        condition[:, 7, 0, 0, 0] = 1
        return condition

    def test_anchor_only_keeps_anchor_and_mask(self):
        result = self.validator("anchor_only")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.full((7,), 6.0),
        ))
        self.assertEqual(result[0, 7, 0, 0, 0].item(), 1.0)

    def test_reverse_history_does_not_move_anchor(self):
        result = self.validator("reverse_history")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.tensor([5, 4, 3, 2, 1, 0, 6]),
        ))

    def test_shuffle_history_does_not_move_anchor(self):
        result = self.validator("shuffle_history")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.tensor([10, 11, 12, 13, 14, 15, 6]),
        ))

    def test_linear_trend_extrapolates_seven_day_line(self):
        validator = self.validator(
            prediction_mode="linear_trend"
        )
        prediction = validator._predict(self.condition(), 0)
        self.assertTrue(torch.allclose(
            prediction[0, :, 0, 0, 0],
            torch.arange(7, 22, dtype=torch.float32),
            atol=1e-5,
        ))

    def test_persistence_skill_uses_mse(self):
        skill = persistence_skill(
            {"mse": 0.5},
            {"mse": 2.0},
        )
        self.assertAlmostEqual(skill, 0.75)


if __name__ == "__main__":
    unittest.main()
