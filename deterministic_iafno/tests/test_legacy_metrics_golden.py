# 用途：验证旧版本指标的固定数值回归。
import math
import unittest

import numpy as np

from diafno.evaluation.metrics import RunningSSTMetrics


class LegacyMetricsGoldenTests(unittest.TestCase):
    # 用途：验证运行指标的口径契约。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_running_metrics_contract(self):
        metrics = RunningSSTMetrics()
        prediction = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 3.0, 5.0])
        mask = np.ones(3, dtype=np.float32)
        metrics.update(prediction, target, mask)
        result = metrics.compute()

        self.assertEqual(result["valid_pixels"], 3)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(result["bias"], -1.0)
        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertAlmostEqual(
            result["prediction_std"],
            math.sqrt(2.0 / 3.0),
        )
        self.assertAlmostEqual(
            result["target_std"],
            math.sqrt(8.0 / 3.0),
        )


if __name__ == "__main__":
    unittest.main()
