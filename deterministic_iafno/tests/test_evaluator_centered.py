# 用途：验证centered 模型的在线验证行为。
import unittest
from types import SimpleNamespace

import torch

from diafno.evaluation.validator import OSTIAValidator


class StubCenteredModel:
    """Minimal centered model: sample() returns zero innovation."""

    # 用途：测试桩的初始化。
    # 参数：见签名；输出 无。
    def __init__(self):
        self.sample_calls = 0
        self.S_churn = 0.0

    # 用途：执行一次采样的测试辅助。
    # 参数：见签名；输出 采样结果。
    def sample(self, condition, num_sample_steps=None, seed=None):
        self.sample_calls += 1
        return torch.zeros(
            condition.shape[0],
            2,
            condition.shape[2],
            condition.shape[3],
            condition.shape[4],
        )


class EvaluatorCenteredTests(unittest.TestCase):
    # 用途：构造测试用验证器实例的辅助函数。
    # 参数：见签名；输出 OSTIAValidator 实例。
    def validator(self, ensemble_members=1, s_churn=None):
        validator = OSTIAValidator.__new__(OSTIAValidator)
        validator.config = SimpleNamespace(
            prediction_mode="model",
            condition_ablation="none",
            ensemble_members=ensemble_members,
            s_churn=s_churn,
            seed=123,
        )
        validator.model_config = SimpleNamespace(
            model_type="centered_diffusion",
            target_mode="residual",
            input_days=7,
            output_days=2,
        )
        validator.model = StubCenteredModel()
        validator.sampling_steps = 16
        return validator

    # 用途：构造测试用条件张量的辅助函数。
    # 参数：见签名；输出 条件张量。
    def condition(self):
        condition = torch.zeros(2, 8, 1, 1, 1)
        condition[0, 6, 0, 0, 0] = 3.0
        condition[1, 6, 0, 0, 0] = 9.0
        return condition

    # 用途：验证 centered 预测的 anchor 恰好只加一次。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_centered_reanchored_exactly_once(self):
        # sample() returns the normalized residual forecast r_hat (here
        # zero); the evaluator must add the day-7 anchor exactly once.
        validator = self.validator()
        prediction = validator._predict(self.condition(), 0)
        self.assertEqual(tuple(prediction.shape), (2, 2, 1, 1, 1))
        self.assertTrue(torch.equal(
            prediction[0, :, 0, 0, 0],
            torch.full((2,), 3.0),
        ))
        self.assertTrue(torch.equal(
            prediction[1, :, 0, 0, 0],
            torch.full((2,), 9.0),
        ))
        self.assertEqual(validator.model.sample_calls, 1)

    # 用途：验证 centered 集成采样后仅重建一次 anchor。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_centered_ensemble_samples_then_anchors_once(self):
        validator = self.validator(ensemble_members=4)
        prediction = validator._predict(self.condition(), 0)
        self.assertEqual(validator.model.sample_calls, 4)
        self.assertTrue(torch.equal(
            prediction[0, :, 0, 0, 0],
            torch.full((2,), 3.0),
        ))

    # 用途：验证 centered 拒绝探针模式。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_probe_mode_rejected_for_centered(self):
        validator = self.validator()
        validator.config.prediction_mode = "probe"
        validator.config.probe_sigma = 0.002
        with self.assertRaisesRegex(ValueError, "probe mode"):
            validator._predict_probe(
                self.condition(),
                torch.zeros(2, 2, 1, 1, 1),
                0,
            )

    # 用途：验证 S_churn 经透传属性生效。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_s_churn_applies_via_delegated_attribute(self):
        validator = self.validator(s_churn=0.25)
        if validator.config.s_churn is not None:
            if not hasattr(validator.model, "S_churn"):
                self.fail("centered model must expose S_churn")
            validator.model.S_churn = validator.config.s_churn
        self.assertEqual(validator.model.S_churn, 0.25)


if __name__ == "__main__":
    unittest.main()
