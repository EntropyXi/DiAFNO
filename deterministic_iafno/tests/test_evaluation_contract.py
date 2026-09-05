# 用途：验证有效像素、残差还原及评估协议。
import unittest
from types import SimpleNamespace

import torch

from diafno.evaluation.metrics import persistence_skill
from diafno.evaluation.validator import OSTIAValidator


class EvaluationContractTests(unittest.TestCase):
    # 用途：构造测试用验证器实例的辅助函数。
    # 参数：见签名；输出 OSTIAValidator 实例。
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
    # 用途：构造测试用条件张量的辅助函数。
    # 参数：见签名；输出 条件张量。
    def condition():
        condition = torch.zeros(2, 8, 1, 1, 1)
        condition[0, :7, 0, 0, 0] = torch.arange(7)
        condition[1, :7, 0, 0, 0] = torch.arange(10, 17)
        condition[:, 7, 0, 0, 0] = 1
        return condition

    # 用途：验证 anchor-only 消融保留 day-7 锚点与 mask。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_anchor_only_keeps_anchor_and_mask(self):
        result = self.validator("anchor_only")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.full((7,), 6.0),
        ))
        self.assertEqual(result[0, 7, 0, 0, 0].item(), 1.0)

    # 用途：验证逆序历史消融不移动 anchor。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_reverse_history_does_not_move_anchor(self):
        result = self.validator("reverse_history")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.tensor([5, 4, 3, 2, 1, 0, 6]),
        ))

    # 用途：验证打乱历史消融不移动 anchor。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_shuffle_history_does_not_move_anchor(self):
        result = self.validator("shuffle_history")._ablate_condition(
            self.condition()
        )
        self.assertTrue(torch.equal(
            result[0, :7, 0, 0, 0],
            torch.tensor([10, 11, 12, 13, 14, 15, 6]),
        ))

    # 用途：验证线性趋势基线按 7 日直线外推。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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

    # 用途：验证 persistence skill 基于 MSE。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_persistence_skill_uses_mse(self):
        skill = persistence_skill(
            {"mse": 0.5},
            {"mse": 2.0},
        )
        self.assertAlmostEqual(skill, 0.75)


if __name__ == "__main__":
    unittest.main()
