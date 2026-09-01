import unittest

import numpy as np

from diafno.evaluation.bootstrap import (
    paired_temporal_block_bootstrap,
)


class PairedTemporalBlockBootstrapTests(unittest.TestCase):
    def test_identical_predictions_have_zero_difference_and_skill(self):
        sse = np.array(
            [[1.0, 4.0], [2.0, 8.0], [3.0, 12.0], [4.0, 16.0]]
        )
        counts = np.ones_like(sse)
        result = paired_temporal_block_bootstrap(
            sse,
            sse,
            counts,
            np.array([0, 1, 22, 23]),
            block_days=22,
            replicates=200,
            seed=7,
        )
        self.assertEqual(result["num_blocks"], 2)
        self.assertAlmostEqual(result["overall"]["rmse_difference"], 0.0)
        self.assertAlmostEqual(result["overall"]["mse_skill"], 0.0)
        self.assertEqual(result["overall"]["rmse_difference_ci"], [0.0, 0.0])

    def test_uniformly_better_model_has_strictly_better_interval(self):
        persistence_sse = np.array(
            [[4.0, 9.0], [8.0, 18.0], [12.0, 27.0], [16.0, 36.0]]
        )
        model_sse = persistence_sse * 0.25
        counts = np.ones_like(model_sse)
        result = paired_temporal_block_bootstrap(
            model_sse,
            persistence_sse,
            counts,
            np.array([100, 101, 122, 123]),
            block_days=22,
            replicates=500,
            seed=11,
            block_origin_time=100,
        )
        self.assertLess(result["overall"]["rmse_difference_ci"][1], 0.0)
        self.assertGreater(result["overall"]["mse_skill_ci"][0], 0.0)
        self.assertEqual(
            result["overall"]["bootstrap_fraction_model_better"],
            1.0,
        )

    def test_pairing_aggregates_all_samples_from_same_time_block(self):
        model_sse = np.ones((5, 1))
        persistence_sse = np.full((5, 1), 2.0)
        counts = np.ones((5, 1))
        result = paired_temporal_block_bootstrap(
            model_sse,
            persistence_sse,
            counts,
            np.array([0, 3, 21, 22, 44]),
            block_days=22,
            replicates=100,
        )
        self.assertEqual(result["num_blocks"], 3)
        self.assertEqual(result["samples_per_block_min"], 1)
        self.assertEqual(result["samples_per_block_max"], 3)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            paired_temporal_block_bootstrap(
                np.ones((2, 2)),
                np.ones((2, 1)),
                np.ones((2, 2)),
                np.array([0, 1]),
            )


if __name__ == "__main__":
    unittest.main()
