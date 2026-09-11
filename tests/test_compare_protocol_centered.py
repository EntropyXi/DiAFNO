# 用途：统一协议评分核心（含新 A5-centered 主实验行）的注入式单测。
"""Unit tests for the unified protocol scoring core.

The five-method main-experiment protocol (plan section 8) is exercised
end to end with injected samplers instead of real checkpoints: the
scoring core only needs ``sample_index(entry)`` and
``sample_at(index, seed_base=...)``, so a tiny synthetic dataset can
drive every accumulator, the probability auxiliary rows, the paired
block bootstrap and the artifact writers.
"""

import argparse
import json
import os
import tempfile
import unittest

import numpy as np

from tests.ostia_test_h5 import OSTIATestCase
from tests.test_a5_longtrain_local import (
    make_dataset_and_manifest,
    make_sample_payload,
)

from diafno.data.ostia import OSTIADailyDataset
from scripts.compare_ostia_protocol import (
    DecodedSampleCache,
    ensemble_probabilistic_stats,
    evaluate_protocol,
    method_npz_slug,
    protocol_method_names,
)


# 用途：只回显索引并记录解码次数的数据集替身（解码缓存单测用）。
class CountingDataset:
    """Dataset stand-in that counts decodes."""

    def __init__(self):
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return index

    def __len__(self):
        return 100


class DecodedSampleCacheTests(unittest.TestCase):
    """One decode per physical sample across all ensemble members."""

    def setUp(self):
        self.dataset = CountingDataset()

    def test_same_index_is_decoded_once(self):
        cache = DecodedSampleCache(self.dataset, capacity=4)
        for _ in range(16):
            self.assertEqual(cache(3), 3)
        self.assertEqual(self.dataset.reads, 1)
        self.assertEqual(cache.decodes, 1)
        self.assertEqual(cache.hits, 15)
        # Distinct indices still decode once each.
        cache(4)
        cache(4)
        self.assertEqual(self.dataset.reads, 2)

    def test_capacity_evicts_the_least_recently_used(self):
        cache = DecodedSampleCache(self.dataset, capacity=2)
        cache(1)
        cache(2)
        cache(1)   # refresh 1
        cache(3)   # evicts 2
        self.assertEqual(self.dataset.reads, 3)
        cache(2)   # 2 was evicted -> decode again, evicts 1
        self.assertEqual(self.dataset.reads, 4)
        cache(1)   # 1 was evicted by the previous insert
        self.assertEqual(self.dataset.reads, 5)


# 用途：鸭子类型采样器（替代 ProtocolValidator，供评分核心注入）。
class FakeSampler:
    """Sampler stand-in with the ProtocolValidator sampling contract."""

    def __init__(self, dataset, checkpoint_path, noise=0.0,
                 seed_offset=0, split="test"):
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.noise = noise
        self.seed_offset = seed_offset
        self.split = split

    # 用途：清单条目 -> 本方法宇宙中的索引。
    # 参数：输入 entry；输出 int。
    def sample_index(self, entry):
        return int(entry["dataset_index"])

    # 用途：解码（或复用）一条物理样本。
    # 参数：输入 dataset_index；输出 样本 dict。
    def decoded_sample(self, dataset_index):
        return self.dataset[int(dataset_index)]

    # 用途：采样单条物理样本（确定性方法 seed_base=None）。
    # 参数：输入 dataset_index、seed_base；输出 5 元组（预测、真值、mask、均值、标准差）。
    def sample_at(self, dataset_index, seed_base=None):
        sample = self.dataset[int(dataset_index)]
        target = sample["target"].numpy()[..., 0].astype(np.float64)
        mask = sample["target_mask"].numpy()[..., 0]
        if seed_base is None:
            prediction = target.copy()
        else:
            rng = np.random.default_rng(int(seed_base) + self.seed_offset)
            prediction = target + rng.normal(
                scale=self.noise, size=target.shape
            )
        return (
            prediction.astype(np.float32),
            target,
            mask,
            self.dataset.sst_mean,
            self.dataset.sst_std,
        )


class ProbabilisticStatsTests(unittest.TestCase):
    """Spread / skill / coverage of the ensemble auxiliary rows."""

    def setUp(self):
        self.target = np.zeros(64, dtype=np.float64)

    def test_spread_skill_and_coverage_are_exact(self):
        members = np.stack(
            [self.target + delta for delta in (-1.0, 0.0, 1.0)]
        )
        stats = ensemble_probabilistic_stats(members, self.target)
        self.assertEqual(stats["members"], 3)
        self.assertAlmostEqual(stats["skill"], 0.0)
        self.assertAlmostEqual(stats["spread"], np.sqrt(2.0 / 3.0))
        # The 3-member empirical 50%/90% intervals still bracket the
        # target for every pixel.
        self.assertEqual(stats["coverage_50"], 1.0)
        self.assertEqual(stats["coverage_90"], 1.0)
        self.assertIn("np.quantile", stats["interval"])

    def test_nested_subset_uses_the_same_draws(self):
        members = np.stack(
            [self.target + delta for delta in (-1.0, 0.0, 1.0)]
        )
        stats = ensemble_probabilistic_stats(
            members, self.target, subset=2
        )
        self.assertEqual(stats["members"], 2)
        # First two members {-1, 0}: mean -0.5, deviations +/-0.5.
        self.assertAlmostEqual(stats["spread"], 0.5)
        self.assertAlmostEqual(stats["skill"], 0.5)
        self.assertAlmostEqual(stats["spread_skill_ratio"], 1.0)

    def test_missed_interval_reduces_coverage(self):
        members = np.stack([
            self.target + 10.0,
            self.target + 11.0,
            self.target + 12.0,
            self.target + 13.0,
        ])
        stats = ensemble_probabilistic_stats(members, self.target)
        self.assertEqual(stats["coverage_50"], 0.0)
        self.assertEqual(stats["coverage_90"], 0.0)


class ProtocolMethodRowTests(unittest.TestCase):
    def test_method_order_is_frozen_without_centered_row(self):
        self.assertEqual(
            protocol_method_names(),
            ("A5", "old_IAFNO", "old_DiAFNO", "persistence"),
        )
        self.assertEqual(
            protocol_method_names(has_centered=True),
            (
                "A5_centered_DiAFNO", "A5", "old_IAFNO",
                "old_DiAFNO", "persistence",
            ),
        )
        self.assertEqual(
            protocol_method_names(has_centered=True, has_centered_crps=True),
            (
                "A5_centered_DiAFNO", "A5_centered_DiAFNO_CRPS", "A5",
                "old_IAFNO", "old_DiAFNO", "persistence",
            ),
        )

    def test_npz_slugs_are_unique_and_stable(self):
        names = protocol_method_names(True, True)
        slugs = [method_npz_slug(name) for name in names]
        self.assertEqual(len(set(slugs)), len(names))
        self.assertIn("a5_centered", slugs)


class EvaluateProtocolTests(OSTIATestCase):
    """End-to-end scoring core over injected samplers."""

    def setUp(self):
        super().setUp()
        self.h5_path, self.data_manifest, self.dataset = (
            make_dataset_and_manifest(self._tmp)
        )
        indices = np.sort(np.random.default_rng(7).choice(
            len(self.dataset), size=6, replace=False
        )).tolist()
        self.payload = make_sample_payload(
            self.dataset, indices, split="test"
        )
        self.payload_path = os.path.join(self._tmp, "test_samples.json")
        with open(self.payload_path, "w", encoding="utf-8") as file:
            json.dump(self.payload, file)
        self.output_dir = os.path.join(self._tmp, "test200")
        self.args = argparse.Namespace(
            sample_manifest=self.payload_path,
            a5_checkpoint=self._checkpoint("a5.pth"),
            old_iafno_checkpoint=self._checkpoint("iafno.pth"),
            old_diafno_checkpoint=self._checkpoint("diafno.pth"),
            centered_checkpoint=self._checkpoint("centered.pth"),
            centered_crps_checkpoint=None,
            ensemble_members=16,
            sampling_steps=16,
            s_churn=0.0,
            seed=123,
            block_days=22,
            bootstrap_replicates=200,
        )

    def _checkpoint(self, name):
        path = os.path.join(self._tmp, name)
        with open(path, "wb") as file:
            file.write(name.encode("utf-8"))
        return path

    def _samplers(self):
        a5 = FakeSampler(
            self.dataset, self.args.a5_checkpoint, split="test"
        )
        old_iafno = FakeSampler(
            self.dataset, self.args.old_iafno_checkpoint, split="test"
        )
        ensemble = {
            "old_DiAFNO": FakeSampler(
                self.dataset, self.args.old_diafno_checkpoint,
                noise=0.35, seed_offset=0,
            ),
            "A5_centered_DiAFNO": FakeSampler(
                self.dataset, self.args.centered_checkpoint,
                noise=0.12, seed_offset=0,
            ),
        }
        return a5, old_iafno, ensemble

    def test_five_method_report_and_artifacts(self):
        a5, old_iafno, ensemble = self._samplers()
        report = evaluate_protocol(
            self.args, self.payload, a5, old_iafno, ensemble,
            self.output_dir,
        )
        self.assertEqual(
            tuple(report["methods"]),
            (
                "A5_centered_DiAFNO", "A5", "old_IAFNO",
                "old_DiAFNO", "persistence",
            ),
        )
        centered = report["methods"]["A5_centered_DiAFNO"]
        self.assertGreater(centered["overall"]["rmse"], 0.0)
        self.assertGreater(centered["overall"]["crps"], 0.0)
        auxiliary = centered["probability_auxiliary"]
        self.assertEqual(auxiliary["members"], 16)
        self.assertGreater(auxiliary["spread"], 0.0)
        self.assertTrue(0.0 <= auxiliary["coverage_90"] <= 1.0)
        self.assertLessEqual(
            auxiliary["coverage_50"], auxiliary["coverage_90"]
        )
        self.assertIn("limitation", auxiliary)
        # Deterministic rows carry no probabilistic auxiliary block.
        self.assertNotIn("probability_auxiliary", report["methods"]["A5"])
        # Per-lead rows of an ensemble method carry spread/coverage.
        lead_one = centered["by_lead_day"]["1"]
        for key in ("spread", "spread_skill_ratio", "coverage_50",
                    "coverage_90", "spread_8members"):
            self.assertIn(key, lead_one)
        # Both bootstrap reference blocks are present.  A synthetic
        # 28-day split holds a single 22-day block, so the bootstrap
        # reports its explicit disabled branch here; the multi-block
        # interval path is covered by the delta_block_bootstrap tests.
        self.assertIn(
            "delta_A5_centered_DiAFNO_minus_persistence", report
        )
        self.assertIn("delta_A5_minus_persistence", report)
        delta = report["delta_A5_centered_DiAFNO_minus_A5"]
        if delta["num_blocks"] < 2:
            self.assertIn("reason", delta)
            self.assertIsNone(delta["interval"])
        else:
            self.assertIn("rmse_difference", delta)
        # Provenance pins every checkpoint with a SHA-256.
        checkpoints = report["provenance"]["checkpoints"]
        self.assertEqual(len(checkpoints["A5"]["sha256"]), 64)
        self.assertEqual(
            checkpoints["A5_centered_DiAFNO"]["members"], 16
        )
        # Artifacts are recomputable and carry the new rows.
        with np.load(
                os.path.join(self.output_dir, "paired_contributions.npz")
            ) as contributions:
            for key in (
                    "per_sample_sse_a5",
                    "per_sample_crps_a5",
                    "per_sample_sse_a5_centered",
                    "per_sample_crps_a5_centered",
                    "per_sample_counts"):
                self.assertIn(key, contributions.files)
            self.assertEqual(
                contributions["per_sample_sse_a5_centered"].shape,
                (len(self.payload["entries"]), 15),
            )
        with open(
                os.path.join(self.output_dir, "metrics.json"),
                "r", encoding="utf-8",
            ) as file:
            self.assertEqual(
                tuple(json.load(file)["methods"]),
                tuple(report["methods"]),
            )
        with open(
                os.path.join(self.output_dir, "evaluation_manifest.json"),
                "r", encoding="utf-8",
            ) as file:
            manifest = json.load(file)
        self.assertEqual(manifest["protocol"]["split"], "test")

    def test_figure_field_dump_feeds_the_plotter(self):
        from scripts.plot_ostia_protocol_figures import (
            build_case,
            parse_crop,
        )
        self.args.figure_samples = 2
        a5, old_iafno, ensemble = self._samplers()
        evaluate_protocol(
            self.args, self.payload, a5, old_iafno, ensemble,
            self.output_dir,
        )
        fields_path = os.path.join(self.output_dir, "figure_fields.npz")
        self.assertTrue(os.path.isfile(fields_path))
        with np.load(fields_path) as payload:
            fields = {key: payload[key] for key in payload.files}
        self.assertEqual(fields["target_kelvin"].shape[0], 2)
        self.assertEqual(fields["target_kelvin"].dtype, np.float16)
        self.assertEqual(
            fields["prediction_a5_centered"].shape,
            fields["target_kelvin"].shape,
        )
        self.assertEqual(
            fields["target_mask"].shape, fields["target_kelvin"].shape
        )
        for slug in ("a5", "old_iafno", "old_diafno", "persistence"):
            self.assertIn(f"prediction_{slug}", fields)
        # The dumped fields are directly plottable through the shared
        # crop helper, on the same physical sample order.
        case = build_case(
            fields, 0, parse_crop("0:16,0:16"),
            ("a5_centered", "a5", "persistence"),
        )
        self.assertEqual(case["target"].shape, (15, 16, 16))
        self.assertEqual(
            case["metadata"]["spatial_index"],
            int(fields["spatial_index"][0]),
        )

    def test_figure_dump_disabled_by_default(self):
        self.args.figure_samples = 0
        a5, old_iafno, ensemble = self._samplers()
        evaluate_protocol(
            self.args, self.payload, a5, old_iafno, ensemble,
            self.output_dir,
        )
        self.assertFalse(os.path.isfile(
            os.path.join(self.output_dir, "figure_fields.npz")
        ))

    def test_centered_noise_improves_crps_over_the_noisier_row(self):
        a5, old_iafno, ensemble = self._samplers()
        report = evaluate_protocol(
            self.args, self.payload, a5, old_iafno, ensemble,
            self.output_dir,
        )
        self.assertLess(
            report["methods"]["A5_centered_DiAFNO"]["overall"]["crps"],
            report["methods"]["old_DiAFNO"]["overall"]["crps"],
        )

    def test_crps_best_row_is_added_only_when_supplied(self):
        self.args.centered_crps_checkpoint = self._checkpoint(
            "centered_crps.pth"
        )
        a5, old_iafno, ensemble = self._samplers()
        ensemble["A5_centered_DiAFNO_CRPS"] = FakeSampler(
            self.dataset, self.args.centered_crps_checkpoint,
            noise=0.2, seed_offset=0,
        )
        report = evaluate_protocol(
            self.args, self.payload, a5, old_iafno, ensemble,
            self.output_dir,
        )
        self.assertEqual(
            tuple(report["methods"]),
            (
                "A5_centered_DiAFNO", "A5_centered_DiAFNO_CRPS", "A5",
                "old_IAFNO", "old_DiAFNO", "persistence",
            ),
        )


if __name__ == "__main__":
    unittest.main()
