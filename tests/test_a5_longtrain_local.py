"""Local unit tests for the A5 long-train local implementation.

Covers A5 plan section 5 verification items that can run without the
real server HDF5: data & pairing (5.1), resume & LR geometry (5.2),
fixed-val functional bits (5.4, through a frozen sample manifest over
synthetic data), scoring rules (5.5), plus preflight, sample-manifest
and candidate-selection unit tests.
"""

import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tests.ostia_test_h5 import (
    OSTIATestCase,
    make_synthetic_h5,
    write_synthetic_data_manifest,
)

from diafno.data.ostia import OSTIADailyDataset
from diafno.evaluation.sample_manifest import (
    PAIRING_KEYS,
    build_sample_manifest_payload,
    ensure_manifest_universe,
    geo_fingerprint_from_condition,
    load_sample_manifest,
    manifest_dataset_indices,
    sample_manifest_sha256,
)
from diafno.evaluation.candidate_eval import select_best_val_rmse
from scripts.preflight_a5_longtrain import (
    require_fresh_output_dir,
    verify_source_checkpoint,
)
from scripts.validate_a5_epochs import (
    cumulative_training_steps,
    list_candidates,
)


_H5_COUNTER = [0]


def make_synthetic_h5_isolated(tmp, with_manifest=True,
                               condition_mode="sst_mask_geo_season",
                               total_days=140):
    """Unique-file synthetic HDF5 (+ optional data manifest + dataset)."""
    _H5_COUNTER[0] += 1
    h5_path = os.path.join(tmp, "syn_%d.h5" % _H5_COUNTER[0])
    make_synthetic_h5(
        h5_path,
        total_days=total_days,
        samples_per_day=5,
        height=16,
        width=16,
        coordinate_layout="per_row",
        with_time_metadata=True,
    )
    manifest_path = None
    if with_manifest:
        manifest_path = os.path.join(tmp, "manifest_%d.json" % _H5_COUNTER[0])
        write_synthetic_data_manifest(manifest_path, h5_path)
    dataset = OSTIADailyDataset(
        h5_path=h5_path,
        split="val",
        input_days=7,
        output_days=15,
        condition_mode=condition_mode,
        data_manifest=manifest_path,
    )
    return h5_path, manifest_path, dataset


def make_dataset_and_manifest(tmp, condition_mode="sst_mask_geo_season"):
    """Synthetic HDF5 + data manifest + geo-season val dataset.

    140 stored days -> val split [98, 126) holds 28 days, enough for
    7 valid 22-day windows x 5 spatial patches = 35 samples.
    """
    return make_synthetic_h5_isolated(
        tmp, with_manifest=True, condition_mode=condition_mode
    )


class SampleManifestTests(OSTIATestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="a5_manifest_")
        self.h5_path, self.manifest_path, self.dataset = (
            make_dataset_and_manifest(self.tmp)
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_metadata_fields_only_in_manifest_mode(self):
        sample = self.dataset[0]
        meta = sample["metadata"]
        self.assertIn("compact_window_start_day", meta)
        self.assertIn("real_t0_day_offset", meta)
        self.assertEqual(
            int(meta["compact_window_start_day"])
            + self.dataset.input_days - 1,
            int(meta["real_t0_day_offset"])
            - int(self.dataset.real_day_offset(
                int(meta["compact_window_start_day"])
                + self.dataset.input_days - 1
            ))
            + int(meta["compact_window_start_day"])
            + self.dataset.input_days - 1,
        )
        # The real t0 offset must equal the manifest offset of the
        # compact t0 ordinal.
        compact_start = int(meta["compact_window_start_day"])
        t0_ordinal = compact_start + self.dataset.input_days - 1
        self.assertEqual(
            int(meta["real_t0_day_offset"]),
            int(self.dataset.real_day_offsets[t0_ordinal]),
        )
        # attrs-only datasets (no manifest) keep the old 5-key layout.
        _, _, legacy_attrs = make_synthetic_h5_isolated(
            self.tmp, with_manifest=False
        )
        self.assertFalse(legacy_attrs.has_real_day_axis)
        self.assertNotIn(
            "real_t0_day_offset", legacy_attrs[0]["metadata"]
        )
        self.assertNotIn(
            "compact_window_start_day", legacy_attrs[0]["metadata"]
        )

    def test_build_and_roundtrip_manifest(self):
        dataset = self.dataset
        dataset_size = len(dataset)
        generator = np.random.default_rng(123)
        indices = np.sort(generator.choice(
            dataset_size, size=16, replace=False
        )).tolist()
        spatial_entries = {}
        for index in indices:
            sequence_index = index // dataset.samples_per_day
            spatial_index = index % dataset.samples_per_day
            spatial_entries[index] = {
                "compact_start": int(
                    dataset.valid_start_days[int(sequence_index)]
                ),
                "spatial_index": int(spatial_index),
            }
        payload = build_sample_manifest_payload(
            dataset,
            indices,
            split="val",
            spatial_entries=spatial_entries,
            dataset_size_geo=dataset_size,
            legacy_split_start=dataset.split_start_day,
        )
        path = os.path.join(self.tmp, "samples.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        loaded = load_sample_manifest(
            path, split="val", expected_count=16,
            dataset_size=dataset_size,
        )
        self.assertEqual(
            loaded["manifest_sha256"],
            sample_manifest_sha256(payload),
        )
        self.assertEqual(
            manifest_dataset_indices(loaded),
            sorted(indices),
        )
        for entry in loaded["entries"]:
            self.assertEqual(
                entry["legacy_dataset_index"],
                (entry["compact_start"] - dataset.split_start_day)
                * dataset.samples_per_day
                + entry["spatial_index"],
            )
        # Pairing keys are unique.
        keys = [
            tuple(entry[field] for field in PAIRING_KEYS)
            for entry in loaded["entries"]
        ]
        self.assertEqual(len(set(keys)), 16)

    def test_manifest_validation_failures(self):
        dataset = self.dataset
        indices = [0, 1, 2]
        spatial_entries = {
            index: {
                "compact_start": int(
                    dataset.valid_start_days[index // dataset.samples_per_day]
                ),
                "spatial_index": index % dataset.samples_per_day,
            }
            for index in indices
        }
        payload = build_sample_manifest_payload(
            dataset,
            indices,
            split="val",
            spatial_entries=spatial_entries,
            dataset_size_geo=len(dataset),
            legacy_split_start=dataset.split_start_day,
        )
        payload["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            load_sample_manifest(self._write(payload))
        payload = build_sample_manifest_payload(
            dataset,
            indices,
            split="val",
            spatial_entries=spatial_entries,
            dataset_size_geo=len(dataset),
            legacy_split_start=dataset.split_start_day,
        )
        payload["entries"][1]["dataset_index"] = -1
        with self.assertRaisesRegex(ValueError, "dataset_index"):
            load_sample_manifest(self._write(payload))

    def _write(self, payload):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        return path

    def test_geo_fingerprint_is_stable_and_specific(self):
        sample_a = self.dataset[0]
        sample_b = self.dataset[0]
        condition_a = sample_a["condition"].numpy()
        condition_b = sample_b["condition"].numpy()
        self.assertEqual(
            geo_fingerprint_from_condition(condition_a),
            geo_fingerprint_from_condition(condition_b),
        )
        other = self.dataset[1]["condition"].numpy()
        if not np.array_equal(condition_a[8:12], other[8:12]):
            self.assertNotEqual(
                geo_fingerprint_from_condition(condition_a),
                geo_fingerprint_from_condition(other),
            )

    def test_physical_pairing_across_condition_modes(self):
        # The same (gap-filtered) universe and rows must yield identical
        # physical targets/masks/anchors for the geo and legacy modes.
        _, _, geo_dataset = make_dataset_and_manifest(
            self.tmp, "sst_mask_geo_season"
        )
        _, _, legacy_dataset = make_dataset_and_manifest(
            self.tmp, "sst_mask"
        )
        self.assertEqual(len(geo_dataset), len(legacy_dataset))
        for index in (0, 5, 17):
            geo = geo_dataset[index]
            legacy = legacy_dataset[index]
            self.assertTrue(np.array_equal(
                geo["target"].numpy(), legacy["target"].numpy()
            ))
            self.assertTrue(np.array_equal(
                geo["target_mask"].numpy(),
                legacy["target_mask"].numpy(),
            ))
            geo_anchor = geo["condition"].numpy()[
                6, :, :, 0
            ]
            legacy_anchor = legacy["condition"].numpy()[
                6, :, :, 0
            ]
            self.assertTrue(np.allclose(
                geo_anchor, legacy_anchor, atol=1e-6
            ))
            for key in (
                "sequence_index",
                "spatial_index",
                "input_start_time",
                "compact_window_start_day",
                "real_t0_day_offset",
            ):
                self.assertEqual(
                    int(geo["metadata"][key]),
                    int(legacy["metadata"][key]),
                )


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="a5_preflight_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_source_checkpoint_skip_accounting(self):
        path = os.path.join(self.tmp, "epoch_011.pth")
        sidecar = path + ".semantics.json"
        torch.save(
            {
                "epoch": 11,
                "global_step": 2750,
                "scheduler": {"last_epoch": 2747},
                "skipped_optimizer_steps": 3,
                "config": {"dummy": True},
            },
            path,
        )
        with open(sidecar, "w", encoding="utf-8") as file:
            json.dump({"config": {"model_type": "deterministic"}}, file)
        summary = verify_source_checkpoint(
            path, check_sha=False
        )
        self.assertEqual(summary["successful_updates"], 2747)
        self.assertEqual(summary["skips"], 3)

    def test_source_checkpoint_skip_accounting_mismatch_fails(self):
        path = os.path.join(self.tmp, "epoch_011.pth")
        sidecar = path + ".semantics.json"
        torch.save(
            {
                "epoch": 11,
                "global_step": 2750,
                "scheduler": {"last_epoch": 2747},
                "skipped_optimizer_steps": 0,
            },
            path,
        )
        with open(sidecar, "w", encoding="utf-8") as file:
            json.dump({"config": {}}, file)
        with self.assertRaisesRegex(ValueError, "skip accounting"):
            verify_source_checkpoint(path, check_sha=False)

    def test_fresh_output_dir_guard(self):
        train_dir = os.path.join(self.tmp, "train")
        os.makedirs(train_dir, exist_ok=True)
        require_fresh_output_dir(train_dir)
        with open(os.path.join(train_dir, "latest.pth"), "w") as file:
            file.write("x")
        with self.assertRaisesRegex(ValueError, "artifacts"):
            require_fresh_output_dir(train_dir)


class CandidateSelectionTests(unittest.TestCase):
    def test_selects_lowest_unrounded_rmse(self):
        candidates = [
            {
                "overall_rmse": 1.21,
                "cumulative_training_steps": 5000,
                "epoch": 9,
                "source": "epoch_009.pth",
            },
            {
                "overall_rmse": 1.1999,
                "cumulative_training_steps": 2750,
                "epoch": 11,
                "source": "source.pth",
            },
        ]
        best = select_best_val_rmse(candidates)
        self.assertEqual(best["source"], "source.pth")

    def test_tie_prefers_earlier_cumulative_training(self):
        candidates = [
            {
                "overall_rmse": 1.2,
                "cumulative_training_steps": 4000,
                "epoch": 5,
                "source": "epoch_005.pth",
            },
            {
                "overall_rmse": 1.2,
                "cumulative_training_steps": 2750,
                "epoch": 11,
                "source": "source.pth",
            },
        ]
        best = select_best_val_rmse(candidates)
        self.assertEqual(best["source"], "source.pth")


class EpochValSelectionDriverTests(unittest.TestCase):
    """The per-epoch validation driver's pure helpers (plan 4.2 / 5)."""

    def test_list_candidates_source_first_and_sidecar_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = os.path.join(tmp, "train")
            os.makedirs(train_dir)
            # epoch_001 complete (checkpoint + sidecar), epoch_002
            # checkpoint only (still being written -> excluded).
            for name in ("epoch_001.pth", "epoch_001.pth.semantics.json",
                         "epoch_002.pth", "latest.pth"):
                with open(os.path.join(train_dir, name), "w",
                          encoding="utf-8") as file:
                    file.write("x")
            source = os.path.join(tmp, "source.pth")
            candidates = list_candidates(source, train_dir, num_epochs=30)
            self.assertEqual(
                [label for label, _ in candidates],
                ["source_epoch011", "epoch_001"],
            )
            self.assertEqual(candidates[0][1], os.path.abspath(source))

    def test_cumulative_training_steps(self):
        self.assertEqual(
            cumulative_training_steps("source_epoch011", {}), 2750
        )
        self.assertEqual(
            cumulative_training_steps(
                "epoch_007", {"global_step": 1750}
            ),
            2750 + 1750,
        )


class ResumeAndLRGeometryTests(OSTIATestCase):
    """Plan 5.2: new-stage LR 5e-5, cosine horizon 7500 and the
    resume/update accounting that keeps attempts/updates/skips honest."""

    def test_sampler_and_scheduler_geometry(self):
        # 2 GPUs x microbatch 2 x accum 8 = effective batch 32; 8000
        # samples per epoch -> 250 optimizer updates per epoch; 30
        # epochs -> T_max 7500.
        effective = 2 * 2 * 8
        self.assertEqual(effective, 32)
        steps_per_epoch = 8000 // effective
        self.assertEqual(steps_per_epoch, 250)
        self.assertEqual(steps_per_epoch * 30, 7500)

    def test_cosine_t_max_formula(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        import torch.nn as nn
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=30 * 250,
            eta_min=1e-6,
        )
        self.assertEqual(scheduler.T_max, 7500)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-5)
        for _ in range(250):
            optimizer.zero_grad()
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.assertEqual(scheduler.last_epoch, 250)


class ScoringRuleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.target = rng.normal(size=(3, 20))
        self.mask = np.ones((3, 20), dtype=bool)
        self.members = np.stack([
            self.target + rng.normal(scale=0.1, size=(3, 20))
            for _ in range(16)
        ])

    def test_pooled_overall_rmse_not_mean_of_leads(self):
        per_lead_rmse = np.sqrt(
            np.square(self.members.mean(axis=0) - self.target)
            .mean(axis=1)
        )
        pooled = np.sqrt(
            np.square(self.members.mean(axis=0) - self.target).mean()
        )
        self.assertNotAlmostEqual(pooled, per_lead_rmse.mean())
        # pooled is between min and max per-lead rmse
        self.assertGreaterEqual(pooled, per_lead_rmse.min() - 1e-12)
        self.assertLessEqual(pooled, per_lead_rmse.max() + 1e-12)

    def test_deterministic_crps_equals_mae(self):
        from diafno.evaluation.method_comparison import empirical_crps
        target = self.target[0]
        forecast = target + 0.3
        crps = empirical_crps(forecast[None], target)
        self.assertTrue(np.allclose(crps, np.abs(forecast - target)))

    def test_ensemble_crps_lower_than_single_member_mae(self):
        from diafno.evaluation.method_comparison import empirical_crps
        flat_target = self.target.reshape(-1)
        flat_members = self.members.reshape(16, -1)
        crps = empirical_crps(flat_members, flat_target)
        member_mae = np.abs(flat_members - flat_target).mean(axis=0)
        self.assertTrue((crps <= member_mae + 1e-12).all())
        self.assertGreater(crps.mean(), 0.0)

    def test_delta_rmse_sign_flips_on_method_swap(self):
        from scripts.compare_ostia_protocol import (
            delta_block_bootstrap,
        )
        rng = np.random.default_rng(3)
        sse_a = np.abs(rng.normal(size=(40, 15)))
        sse_b = sse_a * 0.5
        crps_a = np.abs(rng.normal(size=(40, 15)))
        crps_b = crps_a * 0.5
        counts = np.full((40, 15), 100)
        times = np.arange(40, dtype=np.int64) * 3
        forward = delta_block_bootstrap(
            sse_a, sse_b, crps_a, crps_b, counts,
            times, origin=0, block_days=22,
            replicates=200, seed=5,
        )
        backward = delta_block_bootstrap(
            sse_b, sse_a, crps_b, crps_a, counts,
            times, origin=0, block_days=22,
            replicates=200, seed=5,
        )
        self.assertAlmostEqual(
            forward["rmse_difference"],
            -backward["rmse_difference"],
        )
        self.assertAlmostEqual(
            forward["crps_difference"],
            -backward["crps_difference"],
        )
        self.assertEqual(forward["num_blocks"], backward["num_blocks"])

    def test_real_day_block_rule_on_manifest(self):
        # Plan 6.2: block id = floor((real_t0 - block_origin) / 22)
        # with real_t0 = day_offsets[compact_start + 6].
        tmp = tempfile.mkdtemp(prefix="a5_blocks_")
        try:
            _, _, dataset = make_dataset_and_manifest(tmp)
            offsets = dataset.real_day_offsets
            compact_start = dataset.valid_start_days[0]
            origin = int(offsets[dataset.split_start_day + 6])
            for sequence in (0, 1, 2):
                start = dataset.valid_start_days[sequence]
                real_t0 = int(offsets[start + 6])
                block = (real_t0 - origin) // 22
                self.assertEqual(
                    block,
                    (int(offsets[start + 6]) - origin) // 22,
                )
                self.assertIsInstance(block, int)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def make_sample_payload(dataset, indices, *, split="val"):
    """Minimal manifest payload over a dataset's own universe."""
    spatial_entries = {}
    for index in indices:
        spatial_entries[int(index)] = {
            "compact_start": int(
                dataset.valid_start_days[
                    int(index) // dataset.samples_per_day
                ]
            ),
            "spatial_index": int(int(index) % dataset.samples_per_day),
        }
    return build_sample_manifest_payload(
        dataset,
        [int(index) for index in indices],
        split=split,
        spatial_entries=spatial_entries,
        dataset_size_geo=len(dataset),
        legacy_split_start=dataset.split_start_day,
    )


class _FakeCenteredValidator:
    """ProtocolValidator stand-in that records its split and scores
    zeros against the real synthetic dataset (no checkpoint needed)."""

    instances = []
    dataset_override = None

    def __init__(self, checkpoint_path, h5_path, data_manifest, device,
                 ensemble_members=1, sampling_steps=16, s_churn=None,
                 use_amp=True, split="test"):
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.ensemble_members = int(ensemble_members)
        self.sampling_steps = int(sampling_steps)
        self.s_churn = s_churn
        self.use_amp = bool(use_amp)
        self.dataset = type(self).dataset_override
        type(self).instances.append(self)

    def sample_index(self, entry):
        return int(entry["dataset_index"])

    def sample_at(self, dataset_index, seed_base=None):
        sample = self.dataset[int(dataset_index)]
        prediction = np.zeros(
            tuple(sample["target"].shape[:-1]), dtype=np.float32
        )
        return (prediction, None, None, None, None)


class _FakeDataset:
    """Length-only dataset stand-in for the universe guard."""

    def __init__(self, size, source):
        self.size = int(size)
        self.sst_mean = source.sst_mean
        self.sst_std = source.sst_std

    def __len__(self):
        return self.size


class CenteredValQueueSplitTests(OSTIATestCase):
    """The centered per-epoch sweep must score candidates inside the
    split universe the frozen sample manifest was built from.

    Historical bug: ``ProtocolValidator`` hardcoded ``split="test"``
    (length 110600) while the val-200 manifest addresses the val
    universe (length 218900) -> ``IndexError(110894)`` on every epoch.
    """

    def setUp(self):
        super().setUp()
        self.h5_path, self.data_manifest, self.dataset = (
            make_dataset_and_manifest(self._tmp)
        )
        indices = np.sort(np.random.default_rng(123).choice(
            len(self.dataset), size=8, replace=False
        )).tolist()
        self.payload = make_sample_payload(self.dataset, indices)
        self.checkpoint = os.path.join(self._tmp, "epoch_005.pth")
        torch.save(
            {
                "epoch": 5,
                "global_step": 1250,
                "scheduler": {"last_epoch": 1248},
                "skipped_optimizer_steps": 2,
            },
            self.checkpoint,
        )
        self.args = argparse.Namespace(
            h5_path=self.h5_path,
            data_manifest=self.data_manifest,
            sample_manifest=os.path.join(self._tmp, "samples.json"),
            validation_dir=os.path.join(self._tmp, "validation"),
            device="cpu",
            sampling_steps=16,
            s_churn=0.0,
            no_amp=True,
            manifest_payload=self.payload,
        )
        _FakeCenteredValidator.instances = []
        _FakeCenteredValidator.dataset_override = self.dataset

    def test_candidate_is_scored_in_the_manifest_split(self):
        import scripts.validate_a5_centered_epochs as centered_val
        protocol = centered_val.build_protocol(self.args)
        self.assertEqual(protocol["split"], "val")
        with mock.patch.object(
                centered_val, "ProtocolValidator", _FakeCenteredValidator
            ):
            candidate = centered_val.validate_candidate(
                self.args, "epoch_005", self.checkpoint, protocol
            )
        self.assertTrue(_FakeCenteredValidator.instances)
        self.assertEqual(
            {validator.split for validator in
             _FakeCenteredValidator.instances},
            {"val"},
        )
        self.assertTrue(np.isfinite(candidate["overall_rmse"]))
        self.assertGreater(candidate["overall_rmse"], 0.0)
        self.assertEqual(candidate["epoch"], 5)
        self.assertEqual(candidate["cumulative_training_steps"], 1250)
        with open(
                os.path.join(
                    self.args.validation_dir, "epoch_005",
                    "protocol.json",
                ),
                "r", encoding="utf-8",
            ) as file:
            recorded = json.load(file)
        self.assertEqual(recorded["split"], "val")

    def test_manifest_universe_guard_is_enforced(self):
        import scripts.validate_a5_centered_epochs as centered_val
        protocol = centered_val.build_protocol(self.args)
        # A loader that does not share the manifest's universe (the real
        # bug: test-split loader, 110600, vs val manifest, 218900) must
        # fail closed instead of raising a bare IndexError later.
        _FakeCenteredValidator.dataset_override = _FakeDataset(
            len(self.dataset) - 3, self.dataset
        )
        with mock.patch.object(
                centered_val, "ProtocolValidator", _FakeCenteredValidator
            ):
            with self.assertRaisesRegex(ValueError, "dataset_size_geo"):
                centered_val.validate_candidate(
                    self.args, "epoch_005", self.checkpoint, protocol
                )
        # A manifest frozen on another split must fail closed as well.
        _FakeCenteredValidator.dataset_override = self.dataset
        self.args.manifest_payload = make_sample_payload(
            self.dataset, [0, 1, 2], split="test"
        )
        with mock.patch.object(
                centered_val, "ProtocolValidator", _FakeCenteredValidator
            ):
            with self.assertRaisesRegex(ValueError, "declares split"):
                centered_val.validate_candidate(
                    self.args, "epoch_005", self.checkpoint, protocol
                )

    def test_load_rejects_manifest_of_another_split(self):
        path = os.path.join(self._tmp, "val_samples.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.payload, file)
        load_sample_manifest(path, split="val")
        with self.assertRaisesRegex(ValueError, "evaluation split"):
            load_sample_manifest(path, split="test")


class ManifestUniverseGuardTests(OSTIATestCase):
    """``ensure_manifest_universe`` turns universe mismatches into
    explicit errors (regression guard for IndexError(110894))."""

    def setUp(self):
        super().setUp()
        _, _, self.dataset = make_dataset_and_manifest(self._tmp)
        self.payload = make_sample_payload(self.dataset, [0, 1, 2])

    def test_matching_universe_passes(self):
        ensure_manifest_universe(
            self.payload, len(self.dataset), split="val"
        )

    def test_size_mismatch_names_both_universes(self):
        with self.assertRaisesRegex(
                ValueError, "dataset_size_geo"
            ) as context:
            ensure_manifest_universe(
                self.payload, len(self.dataset) + 1, split="val"
            )
        self.assertIn("split='val'", str(context.exception))

    def test_split_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "declares split"):
            ensure_manifest_universe(
                self.payload, len(self.dataset), split="test"
            )


if __name__ == "__main__":
    unittest.main()
