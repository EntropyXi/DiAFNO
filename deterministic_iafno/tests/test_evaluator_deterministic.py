# 用途：验证确定性模型的在线验证行为。
import unittest
from types import SimpleNamespace

import torch

from diafno.evaluation.validator import OSTIAValidator


class StubDeterministicModel:
    # 用途：测试桩的初始化。
    # 参数：见签名；输出 无。
    def __init__(self):
        self.calls = 0

    # 用途：执行一次预测的测试辅助。
    # 参数：见签名；输出 预测结果。
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
    # 用途：构造测试用验证器实例的辅助函数。
    # 参数：见签名；输出 OSTIAValidator 实例。
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

    # 用途：构造测试用条件张量的辅助函数。
    # 参数：见签名；输出 条件张量。
    def condition(self):
        condition = torch.zeros(2, 8, 1, 1, 1)
        condition[0, 6, 0, 0, 0] = 3.0
        condition[1, 6, 0, 0, 0] = 9.0
        condition[:, 7, 0, 0, 0] = 1.0
        return condition

    # 用途：验证确定性预测重建 anchor 恰好一次。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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

    # 用途：验证确定性路径强制 ensemble=1。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_deterministic_requires_single_member(self):
        validator = self.validator(ensemble_members=4)
        with self.assertRaisesRegex(
                ValueError,
                "ensemble-members",
            ):
            validator._predict(self.condition(), 0)

    # 用途：验证条件消融在确定性预测路径之前生效。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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
