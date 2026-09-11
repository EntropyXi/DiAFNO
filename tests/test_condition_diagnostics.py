# 用途：§7.3 条件诊断（donor 选择、条件组装、固定噪声去噪扫描）单测。
"""Unit tests for the condition diagnostics (plan section 7.3).

The diagnostics core is driven by a fake centered wrapper and a fake
dataset, so the tests pin the two properties that make the comparison
valid: the frozen mean / anchor / z / noise draw never change across
variants, and each variant only rewrites the condition channels it is
supposed to rewrite.
"""

import unittest

import numpy as np
import torch

from scripts.diagnose_a5_centered_conditions import (
    GEO_CHANNELS,
    HISTORY_CHANNELS,
    SEASON_CHANNELS,
    T0_CHANNEL,
    VARIANTS,
    build_variant_condition,
    run_diagnostics,
    select_donors,
)


def make_entry(dataset_index, compact_start, spatial_index):
    return {
        "dataset_index": dataset_index,
        "compact_start": compact_start,
        "spatial_index": spatial_index,
    }


class DonorSelectionTests(unittest.TestCase):
    def setUp(self):
        # Two patches, three windows.
        self.entries = [
            make_entry(0, 100, 0),
            make_entry(1, 100, 1),
            make_entry(2, 101, 0),
            make_entry(3, 102, 1),
        ]

    def test_donors_respect_their_physical_constraint(self):
        donors = select_donors(self.entries)
        for entry in self.entries:
            index = entry["dataset_index"]
            record = donors[index]
            self.assertNotEqual(
                int(record["history"]["compact_start"]),
                int(entry["compact_start"]),
            )
            self.assertNotEqual(
                int(record["geo"]["spatial_index"]),
                int(entry["spatial_index"]),
            )
            same_region = record["same_region"]
            self.assertIsNotNone(same_region)
            self.assertEqual(
                int(same_region["spatial_index"]),
                int(entry["spatial_index"]),
            )
            self.assertNotEqual(
                int(same_region["compact_start"]),
                int(entry["compact_start"]),
            )

    def test_missing_partner_is_reported_as_none(self):
        # One window per patch -> no same-region control exists.
        entries = [make_entry(0, 100, 0), make_entry(1, 101, 1)]
        donors = select_donors(entries)
        self.assertIsNone(donors[0]["same_region"])
        self.assertIsNone(donors[1]["same_region"])
        self.assertIsNotNone(donors[0]["geo"])

    def test_single_sample_has_no_donors_at_all(self):
        donors = select_donors([make_entry(0, 100, 0)])
        self.assertEqual(
            donors[0],
            {
                "history": None,
                "season": None,
                "geo": None,
                "same_region": None,
            },
        )


class VariantConditionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.condition = rng.normal(size=(14, 4, 5, 1)).astype(np.float32)
        self.donor = rng.normal(size=(14, 4, 5, 1)).astype(np.float32)

    def test_channels_are_rewritten_exactly(self):
        cases = {
            "history_shuffled": (HISTORY_CHANNELS,),
            "season_shuffled": (SEASON_CHANNELS,),
            "geo_shuffled": (GEO_CHANNELS,),
            "season_geo_shuffled": (SEASON_CHANNELS, GEO_CHANNELS),
        }
        for variant, replaced in cases.items():
            broken, usable = build_variant_condition(
                self.condition, self.donor, variant
            )
            self.assertTrue(usable)
            for channel in range(14):
                in_replaced = any(
                    channel in range(
                        selector.start if selector.start is not None else 0,
                        selector.stop,
                    )
                    for selector in replaced
                )
                source = self.donor if in_replaced else self.condition
                self.assertTrue(
                    np.array_equal(broken[channel], source[channel]),
                    f"{variant} channel {channel}",
                )

    def test_correct_variant_keeps_everything_and_copies(self):
        broken, usable = build_variant_condition(
            self.condition, self.donor, "correct"
        )
        self.assertTrue(usable)
        self.assertTrue(np.array_equal(broken, self.condition))
        self.assertFalse(np.shares_memory(broken, self.condition))

    def test_t0_is_preserved_by_history_shuffle(self):
        broken, _ = build_variant_condition(
            self.condition, self.donor, "history_shuffled"
        )
        self.assertTrue(
            np.array_equal(broken[T0_CHANNEL], self.condition[T0_CHANNEL])
        )

    def test_same_region_uses_the_donor_condition_wholesale(self):
        broken, _ = build_variant_condition(
            self.condition, self.donor, "same_region_other_date"
        )
        self.assertTrue(np.array_equal(broken, self.donor))

    def test_missing_donor_is_unusable(self):
        for variant in VARIANTS:
            broken, usable = build_variant_condition(
                self.condition, None, variant
            )
            if variant == "correct":
                self.assertTrue(usable)
            else:
                self.assertFalse(usable)
                self.assertIsNone(broken)

    def test_unknown_variant_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            build_variant_condition(self.condition, self.donor, "nope")


class FakeDiffusion:
    def __init__(self):
        self.calls = []

    def preconditioned_network_forward(self, noised_target, sigma,
                                       condition):
        self.calls.append({
            "noised": noised_target.detach().clone(),
            "sigma": float(sigma),
            "condition": condition.detach().clone(),
            "noised_id": id(noised_target),
        })
        return torch.zeros_like(noised_target)


class FakeCenteredWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion = FakeDiffusion()

    def _frozen_mean_prediction(self, condition):
        return torch.zeros(
            (condition.shape[0], 15) + tuple(condition.shape[2:]),
            dtype=condition.dtype,
        )

    def transform_innovation(self, innovation):
        return innovation

    def inverse_innovation(self, standardized):
        return standardized


class FakeValidator:
    def __init__(self, samples, wrapper):
        self.model = wrapper
        self.samples = samples
        self.split = "val"
        self.device = torch.device("cpu")

    def decoded_sample(self, index):
        return self.samples[int(index)]


class RunDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.samples = {}
        for index in range(2):
            condition = torch.zeros(14, 3, 3, 1)
            condition[0] = 1.0 + index
            condition[T0_CHANNEL] = 5.0 + index
            condition[8] = 0.25 * (index + 1)
            condition[12] = 0.5 * (index + 1)
            target = torch.full((15, 3, 3, 1), 6.0 + index)
            mask = torch.ones(15, 3, 3, 1)
            self.samples[index] = {
                "condition": condition,
                "target": target,
                "target_mask": mask,
            }
        self.entries = [make_entry(0, 100, 0), make_entry(1, 101, 1)]
        self.wrapper = FakeCenteredWrapper()
        self.validator = FakeValidator(self.samples, self.wrapper)

    def test_every_variant_is_scored_once_per_sigma(self):
        donors = select_donors(self.entries)
        results = run_diagnostics(
            self.validator, self.entries, donors, [0.1, 1.0],
            num_samples=2, progress_every=0,
        )
        # Two patches with one window each: no same-region partner, so
        # five variants per sample per sigma.
        calls_per_sigma = {}
        for call in self.wrapper.diffusion.calls:
            calls_per_sigma.setdefault(call["sigma"], []).append(call)
        for sigma in (0.1, 1.0):
            self.assertEqual(len(calls_per_sigma[sigma]), 10)
        self.assertEqual(len(self.wrapper.diffusion.calls), 20)
        # The noise draw is fixed per (sample, sigma): every variant of a
        # sigma sees the identical noised input, and different sigmas (or
        # different samples) differ.  Calls arrive grouped as
        # [sample0 sigma0.1 x5, sample0 sigma1.0 x5, ...].
        for sigma, calls in calls_per_sigma.items():
            for start in range(0, len(calls), 5):
                group = [call["noised"] for call in calls[start:start + 5]]
                self.assertEqual(len(group), 5)
                for other in group[1:]:
                    self.assertTrue(torch.equal(group[0], other))
        self.assertFalse(torch.equal(
            calls_per_sigma[0.1][0]["noised"],
            calls_per_sigma[1.0][0]["noised"],
        ))
        self.assertFalse(torch.equal(
            calls_per_sigma[0.1][0]["noised"],
            calls_per_sigma[0.1][5]["noised"],
        ))
        self.assertEqual(
            set(results["by_sigma"]), {"0.1", "1.0"}
        )
        # Skipped counts are per (sample, sigma): two samples x two
        # sigmas, no same-region partner in this fixture.
        self.assertEqual(
            results["skipped_variant_samples"]["same_region_other_date"], 4
        )

    def test_zero_denoiser_error_equals_the_signal(self):
        donors = select_donors(self.entries)
        results = run_diagnostics(
            self.validator, self.entries, donors, [1.0],
            num_samples=2, progress_every=0,
        )
        row = results["by_sigma"]["1.0"]["correct"]
        # mu = 0, so z = target - anchor(5 or 6) -> 1.0 for both samples.
        self.assertAlmostEqual(row["mse_z"], 1.0, places=5)
        self.assertAlmostEqual(
            row["mse_z"], row["mse_z_ratio_vs_correct"], places=5
        )
        self.assertAlmostEqual(
            results["by_sigma"]["1.0"]["_zero_predictor"]["mse_z"],
            1.0, places=5,
        )
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["pixels"], 2 * 15 * 3 * 3)

    def test_only_the_condition_channels_of_the_variant_change(self):
        donors = select_donors(self.entries)
        run_diagnostics(
            self.validator, self.entries, donors, [1.0],
            num_samples=1, progress_every=0,
        )
        conditions = [
            call["condition"] for call in self.wrapper.diffusion.calls
        ]
        # One sigma, one scored sample -> five variants (no same-region
        # partner), each with its own damaged condition.
        self.assertEqual(len(conditions), 5)
        self.assertEqual(
            len({tuple(condition.flatten().tolist())
                 for condition in conditions}),
            5,
        )
        # Every call shares the sample's own t0 anchor and target mask
        # path: only the condition varies, never the mean or the noise.
        for condition in conditions:
            self.assertTrue(torch.equal(
                condition[0][T0_CHANNEL],
                torch.full_like(condition[0][T0_CHANNEL], 5.0),
            ))

    def test_missing_donors_are_counted_as_skipped(self):
        donors = select_donors(self.entries)
        results = run_diagnostics(
            self.validator, self.entries[:1], donors, [1.0],
            num_samples=1, progress_every=0,
        )
        skipped = results["skipped_variant_samples"]
        # The fixture has one window per patch, so only the same-region
        # control is missing; every other variant has a donor.
        self.assertEqual(skipped["same_region_other_date"], 1)
        self.assertEqual(skipped["history_shuffled"], 0)
        self.assertEqual(skipped["geo_shuffled"], 0)
        self.assertEqual(skipped["season_shuffled"], 0)
        self.assertEqual(skipped["correct"], 0)

    def test_requires_a_centered_wrapper(self):
        class Plain(torch.nn.Module):
            pass
        validator = FakeValidator(self.samples, Plain())
        with self.assertRaisesRegex(ValueError, "centered diffusion"):
            run_diagnostics(
                validator, self.entries, select_donors(self.entries),
                [1.0], num_samples=1, progress_every=0,
            )


if __name__ == "__main__":
    unittest.main()
