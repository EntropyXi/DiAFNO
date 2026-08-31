import unittest
from types import SimpleNamespace

import torch

from diafno.evaluation.validator import OSTIAValidator


class StubDeterministicModel:
    def __init__(self):
        self.calls = 0

    def predict(self, condition):
        self.calls += 1
        return torch.zeros(
            condition.shape[0],
            2,
            condition.shape[2],
            condition.shape[3],
            condition.shape[4],
        )


class EvaluatorDeterministicTests(unittest.TestCase):
    def validator(self, ensemble_members=1):
        validator = OSTIAValidator.__new__(OSTIAValidator)
        validator.config = SimpleNamespace(
            prediction_mode="model",
            condition_ablation="none",
            ensemble_members=ensemble_members,
        )
        validator.model_config = SimpleNamespace(
            model_type="deterministic",
            target_mode="residual",
            input_days=7,
            output_days=2,
        )
        validator.model = StubDeterministicModel()
        return validator

    def condition(self):
        condition = torch.zeros(2, 8, 1, 1, 1)
        condition[0, 6, 0, 0, 0] = 3.0
        condition[1, 6, 0, 0, 0] = 9.0
        condition[:, 7, 0, 0, 0] = 1.0
        return condition

    def test_deterministic_prediction_reanchors_anchor(self):
        validator = self.validator()
        prediction = validator._predict(self.condition(), 0)
        self.assertEqual(
            tuple(prediction.shape),
            (2, 2, 1, 1, 1),
        )
        # zero residual prediction + original day-7 anchor
        self.assertTrue(torch.equal(
            prediction[0, :, 0, 0, 0],
            torch.full((2,), 3.0),
        ))
        self.assertTrue(torch.equal(
            prediction[1, :, 0, 0, 0],
            torch.full((2,), 9.0),
        ))

    def test_deterministic_requires_single_member(self):
        validator = self.validator(ensemble_members=4)
        with self.assertRaisesRegex(
                ValueError,
                "ensemble-members",
            ):
            validator._predict(self.condition(), 0)

    def test_ablation_applied_before_deterministic_predict(self):
        validator = self.validator()
        validator.config.condition_ablation = "zero_sst"
        validator.model_config = SimpleNamespace(
            model_type="deterministic",
            target_mode="residual",
            input_days=7,
            output_days=2,
        )
        condition = self.condition()
        prediction = validator._predict(condition, 0)
        # re-anchor must use the ORIGINAL condition day-7, not the
        # ablated (zeroed) one
        self.assertTrue(torch.equal(
            prediction[0, :, 0, 0, 0],
            torch.full((2,), 3.0),
        ))


if __name__ == "__main__":
    unittest.main()
