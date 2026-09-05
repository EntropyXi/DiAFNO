# 用途：验证逐 lead 统计文件与命令行输入校验。
import unittest

from diafno.training.config import (
    validate_lead_stats_dict,
)


class LeadStatsValidationTests(unittest.TestCase):
    # 用途：构造合法统计载荷的夹具。
    # 参数：见签名（可覆盖字段）；输出 载荷 dict。
    def valid_payload(self):
        return {
            "schema_version": 1,
            "target_space": "normalized_residual",
            "split": "train",
            "selection": "evenly_spaced_dataset_indices",
            "num_samples": 4096,
            "dataset_size": 786100,
            "input_days": 7,
            "output_days": 15,
            "sst_mean": 290.7488927184541,
            "sst_std": 9.57073350168232,
            "lead_mean": [float(value) for value in range(15)],
            "lead_std": [1.0 + value for value in range(15)],
        }

    # 用途：验证合法载荷通过校验。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_valid_payload_passes(self):
        mean, std = validate_lead_stats_dict(
            self.valid_payload(),
            target_chans=15,
            input_days=7,
            output_days=15,
        )
        self.assertEqual(len(mean), 15)
        self.assertEqual(len(std), 15)
        self.assertGreater(min(std), 0.0)

    # 用途：验证缺失键被拒绝。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_missing_keys_fail(self):
        payload = self.valid_payload()
        del payload["lead_mean"]
        with self.assertRaisesRegex(ValueError, "lead_mean/lead_std"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )

    # 用途：验证长度不符时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_wrong_length_fails(self):
        payload = self.valid_payload()
        payload["lead_std"] = [1.0] * 14
        with self.assertRaisesRegex(ValueError, "target_chans"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )

    # 用途：验证非正 std 被拒绝。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_nonpositive_std_fails(self):
        payload = self.valid_payload()
        payload["lead_std"][3] = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )

    # 用途：验证非有限统计量被拒绝。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_nonfinite_stats_fail(self):
        payload = self.valid_payload()
        payload["lead_mean"][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_lead_stats_dict(payload, 15, 7, 15)

        payload = self.valid_payload()
        payload["lead_std"][2] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_lead_stats_dict(payload, 15, 7, 15)

    # 用途：验证 target_space 不符时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_wrong_target_space_fails(self):
        payload = self.valid_payload()
        payload["target_space"] = "absolute_sst"
        with self.assertRaisesRegex(ValueError, "target_space"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )

    # 用途：验证 split 不符时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_wrong_split_fails(self):
        payload = self.valid_payload()
        payload["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train split"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )

    # 用途：验证日数不符时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_wrong_days_fail(self):
        payload = self.valid_payload()
        payload["input_days"] = 14
        with self.assertRaisesRegex(ValueError, "input_days"):
            validate_lead_stats_dict(
                payload, 15, 7, 15
            )


if __name__ == "__main__":
    unittest.main()
