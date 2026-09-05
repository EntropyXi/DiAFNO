# 用途：验证逐 lead 残差统计量计算。
import unittest

import numpy as np

from deterministic_iafno.compute_lead_stats import (
    LeadStatsAccumulator,
    build_chunk_aware_indices,
    build_indices,
)


class LeadStatsTests(unittest.TestCase):
    # 用途：验证残差累计器只统计 mask 有效像素。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_accumulator_respects_mask(self):
        accumulator = LeadStatsAccumulator(2)
        residual = np.array([
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
        ])
        mask = np.array([
            [[1, 0], [1, 0]],
            [[1, 0], [1, 0]],
        ])
        accumulator.update(residual, mask)
        result = accumulator.compute()
        self.assertEqual(result["lead_mean"], [2.0, 20.0])
        self.assertEqual(result["lead_std"], [1.0, 10.0])

    # 用途：验证均匀抽样索引覆盖首尾两端。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_even_indices_cover_both_ends(self):
        indices = build_indices(100, 5)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertEqual(len(indices), 5)

    # 用途：验证 chunk 对齐索引按连续空间行分组。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_chunk_aware_indices_group_contiguous_spatial_rows(self):
        class DatasetStub:
            samples_per_day = 100
            sequences_per_window = 1000
            chunk_rows = 32

            # 用途：测试桩的长度接口。
            # 参数：无输入；输出 int。
            def __len__(self):
                return (
                    self.samples_per_day
                    * self.sequences_per_window
                )

        dataset = DatasetStub()
        indices = build_chunk_aware_indices(dataset, 4096)
        self.assertEqual(len(indices), 4096)
        first_sequence = indices[:32] // dataset.samples_per_day
        self.assertTrue(np.all(first_sequence == first_sequence[0]))
        self.assertTrue(np.array_equal(
            indices[:32] % dataset.samples_per_day,
            np.arange(32),
        ))
        self.assertGreater(
            len(np.unique(indices // dataset.samples_per_day)),
            100,
        )


if __name__ == "__main__":
    unittest.main()
