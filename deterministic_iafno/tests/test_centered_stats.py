# 用途：验证创新统计量和来源校验规则。
import json
import os
import shutil
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

from deterministic_iafno.centered_stats import (
    CENTERED_TARGET_SPACE,
    LOCKED_MEAN_CHECKPOINT_SHA256,
    cross_check_mean_sidecar,
    indices_sha256,
    sha256_hex_file,
    validate_centered_fresh_inputs,
    validate_centered_stats_payload,
)
from deterministic_iafno.compute_centered_stats import (
    compute_centered_stats,
)


def valid_payload(**overrides):
    payload = {
        "schema_version": 1,
        "split": "train",
        "target_space": CENTERED_TARGET_SPACE,
        "input_days": 7,
        "output_days": 15,
        "condition_mode": "sst_mask",
        "num_samples": 8192,
        "dataset_size": 786100,
        "selection": "evenly_spaced_sequence_spatial_chunk_blocks",
        "indices_sha256": "c" * 64,
        "mean_checkpoint": "/fake/frozen_mean.pth",
        "mean_checkpoint_sha256": LOCKED_MEAN_CHECKPOINT_SHA256,
        "mean_semantics_sha256": "d" * 64,
        "mean_lead_mean": [float(value) for value in range(15)],
        "mean_lead_std": [1.0 + value for value in range(15)],
        "sst_mean": 290.75,
        "sst_std": 9.57,
        "lead_mean": [float(value) for value in range(15)],
        "lead_std": [1.0 + value for value in range(15)],
        "overall_innovation_std": 1.0,
        "valid_pixels": [1000] * 15,
    }
    payload.update(overrides)
    return payload


class CenteredStatsValidatorTests(unittest.TestCase):
    def test_valid_payload_passes(self):
        validated = validate_centered_stats_payload(
            valid_payload(),
            target_chans=15,
            input_days=7,
            output_days=15,
        )
        self.assertEqual(len(validated["lead_mean"]), 15)
        self.assertEqual(len(validated["lead_std"]), 15)
        self.assertEqual(len(validated["mean_lead_std"]), 15)

    def test_wrong_split_fails(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            validate_centered_stats_payload(
                valid_payload(split="val"), 15, 7, 15
            )

    def test_missing_or_wrong_target_space_fails(self):
        payload = valid_payload()
        del payload["target_space"]
        with self.assertRaisesRegex(ValueError, "target_space"):
            validate_centered_stats_payload(payload, 15, 7, 15)
        with self.assertRaisesRegex(ValueError, "target_space"):
            validate_centered_stats_payload(
                valid_payload(target_space="normalized_residual"),
                15,
                7,
                15,
            )

    def test_wrong_lead_count_fails(self):
        payload = valid_payload()
        payload["lead_std"] = [1.0] * 14
        with self.assertRaisesRegex(ValueError, "lead_std"):
            validate_centered_stats_payload(payload, 15, 7, 15)

    def test_nonpositive_or_nonfinite_std_fails(self):
        payload = valid_payload(lead_std=[0.0] + [1.0] * 14)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_centered_stats_payload(payload, 15, 7, 15)
        payload = valid_payload(
            lead_std=[float("nan")] + [1.0] * 14
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_centered_stats_payload(payload, 15, 7, 15)
        payload = valid_payload(
            lead_std=[float("inf")] + [1.0] * 14
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_centered_stats_payload(payload, 15, 7, 15)

    def test_wrong_mean_sha_fails(self):
        with self.assertRaisesRegex(ValueError, "locked frozen mean"):
            validate_centered_stats_payload(
                valid_payload(mean_checkpoint_sha256="0" * 64),
                15,
                7,
                15,
            )

    def test_missing_provenance_hashes_fail(self):
        payload = valid_payload()
        del payload["indices_sha256"]
        with self.assertRaisesRegex(ValueError, "indices_sha256"):
            validate_centered_stats_payload(payload, 15, 7, 15)
        payload = valid_payload()
        del payload["mean_semantics_sha256"]
        with self.assertRaisesRegex(ValueError, "mean_semantics_sha256"):
            validate_centered_stats_payload(payload, 15, 7, 15)

    def test_val_test_keys_rejected(self):
        for forbidden in ("val", "test", "val_indices", "test_indices",
                          "validation", "test_metadata"):
            payload = valid_payload()
            payload[forbidden] = []
            with self.assertRaisesRegex(ValueError, "must not contain"):
                validate_centered_stats_payload(payload, 15, 7, 15)

    def test_wrong_days_and_condition_mode_fail(self):
        with self.assertRaisesRegex(ValueError, "input_days"):
            validate_centered_stats_payload(
                valid_payload(input_days=14), 15, 7, 15
            )
        with self.assertRaisesRegex(ValueError, "condition_mode"):
            validate_centered_stats_payload(
                valid_payload(condition_mode="sst"), 15, 7, 15
            )

    def test_indices_sha256_deterministic(self):
        indices = np.arange(4096, dtype=np.int64)
        self.assertEqual(
            indices_sha256(indices),
            indices_sha256(indices),
        )
        self.assertEqual(len(indices_sha256(indices)), 64)


class DatasetStub:
    normalization = {
        "sst_mean": 290.7488927184541,
        "sst_std": 9.57073350168232,
    }


class CenteredStatsEndToEndTests(unittest.TestCase):
    """Synthetic HDF5 + synthetic frozen-mean checkpoint end-to-end."""

    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(tests_dir, ".tmp_centered_stats")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.h5_path = os.path.join(self.tmp_dir, "synthetic.h5")
        self.mean_path = os.path.join(
            self.tmp_dir, "frozen_mean.pth"
        )
        self.stats_path = os.path.join(
            self.tmp_dir, "centered_stats.json"
        )
        self._write_h5()
        self._write_mean_checkpoint()
        self.mean_sha = sha256_hex_file(self.mean_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_h5(self):
        days = 40
        per_day = 100
        total = days * per_day
        rng = np.random.default_rng(12345)
        base = 290.0 + 2.0 * np.sin(np.arange(days) / 5.0)
        sst = np.zeros((total, 1, 8, 8), dtype=np.float32)
        for day in range(days):
            sst[day * per_day:(day + 1) * per_day, 0] = base[day]
        sst += rng.normal(0, 0.1, size=sst.shape).astype(np.float32)
        mask = np.zeros((total, 8, 8), dtype=np.uint8)
        time = np.repeat(
            np.arange(days, dtype=np.int64), per_day
        )
        with h5py.File(self.h5_path, "w") as file:
            file.create_dataset(
                "sst", data=sst, chunks=(per_day, 1, 8, 8)
            )
            file.create_dataset("mask", data=mask, chunks=(per_day, 8, 8))
            file.create_dataset("lat", data=np.zeros((8, 8)))
            file.create_dataset("lon", data=np.zeros((8, 8)))
            file.create_dataset("time", data=time)
            file.attrs["sst_mean"] = float(sst.mean())
            file.attrs["sst_std"] = float(sst.std())

    def _write_mean_checkpoint(self):
        from diafno.training.artifacts import CheckpointManager
        from diafno.training.config import OSTIATrainingConfig

        config = OSTIATrainingConfig()
        config.output_dir = self.tmp_dir
        config.model.model_type = "deterministic"
        config.model.target_mode = "residual"
        config.model.target_scaling = "lead_standardized"
        config.model.image_size = (8, 8, 1)
        config.model.patch_size = (2, 2, 1)
        config.model.embed_dim = 8
        config.model.num_blocks = 2
        config.model.explicit_layer = 1
        config.model.implicit_layer = 1
        config.model.hidden_size_factor = 2
        config.model.cond_chans = 8
        config.model.target_chans = 15
        config.model.lead_mean = tuple(
            float(value) for value in range(15)
        )
        config.model.lead_std = tuple(
            1.0 + value for value in range(15)
        )
        manager = CheckpointManager(config)
        model = config.model.build_model(torch.device("cpu"))
        optimizer = AdamW(model.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        scaler = GradScaler("cuda", enabled=False)
        random_state = CheckpointManager.capture_random_state()
        manager.save(
            self.mean_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=15,
            global_step=10,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[random_state],
        )

    def _patch_locked_sha(self):
        return mock.patch(
            "deterministic_iafno.centered_stats."
            "LOCKED_MEAN_CHECKPOINT_SHA256",
            self.mean_sha,
        )

    def test_end_to_end_stats_payload_and_determinism(self):
        with self._patch_locked_sha(), mock.patch(
                "deterministic_iafno.compute_centered_stats."
                "LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            payload, _ = compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=64,
                batch_size=32,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
            )
        self.assertEqual(payload["split"], "train")
        self.assertEqual(
            payload["target_space"],
            CENTERED_TARGET_SPACE,
        )
        self.assertEqual(payload["num_samples"], 64)
        self.assertEqual(payload["input_days"], 7)
        self.assertEqual(payload["output_days"], 15)
        self.assertEqual(payload["condition_mode"], "sst_mask")
        self.assertEqual(payload["mean_checkpoint_sha256"], self.mean_sha)
        self.assertEqual(len(payload["lead_mean"]), 15)
        self.assertEqual(len(payload["lead_std"]), 15)
        self.assertTrue(all(
            value > 0.0 for value in payload["lead_std"]
        ))
        self.assertTrue(all(
            np.isfinite(value) for value in payload["lead_std"]
        ))
        self.assertEqual(len(payload["valid_pixels"]), 15)
        self.assertTrue(all(
            count > 0 for count in payload["valid_pixels"]
        ))
        self.assertTrue(np.isfinite(
            payload["overall_innovation_std"]
        ))
        self.assertGreater(payload["overall_innovation_std"], 0.0)
        # Independent recomputation: the payload lead stats must match
        # a direct numpy accumulation over the same deterministic
        # indices (verifies the tool's algebra end to end).
        from deterministic_iafno.compute_centered_stats import (
            load_frozen_mean,
        )
        from deterministic_iafno.compute_lead_stats import (
            build_chunk_aware_indices,
        )
        from diafno.data.ostia import OSTIADailyDataset
        dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask",
        )
        with self._patch_locked_sha(), mock.patch(
                "deterministic_iafno.compute_centered_stats."
                "LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            mean_model, _ = load_frozen_mean(
                self.mean_path, torch.device("cpu")
            )
        indices = build_chunk_aware_indices(dataset, 64)
        counts = np.zeros(15, dtype=np.float64)
        totals = np.zeros(15, dtype=np.float64)
        squares = np.zeros(15, dtype=np.float64)
        for start in range(0, len(indices), 32):
            samples = dataset.__getitems__(
                indices[start:start + 32].tolist()
            )
            condition = torch.stack([
                sample["condition"] for sample in samples
            ])
            target = torch.stack([
                sample["target"] for sample in samples
            ]).numpy()
            target_mask = torch.stack([
                sample["target_mask"] for sample in samples
            ]).numpy()
            anchor = condition[
                :, 6:7
            ].numpy()
            residual = target - anchor
            with torch.no_grad():
                mu = mean_model.predict(condition).numpy()
            innovation = residual - mu
            for lead in range(15):
                values = innovation[:, lead][target_mask[:, lead] > 0]
                counts[lead] += values.size
                totals[lead] += values.sum()
                squares[lead] += np.square(values).sum()
        means = totals / counts
        stds = np.sqrt(np.maximum(
            squares / counts - means * means, 0.0
        ))
        self.assertTrue(np.allclose(
            payload["lead_mean"], means, atol=1e-5
        ))
        self.assertTrue(np.allclose(
            payload["lead_std"], stds, atol=1e-5
        ))
        with self._patch_locked_sha():
            validate_centered_stats_payload(payload, 15, 7, 15)
        # Byte-identical recomputation on the same machine.
        with self._patch_locked_sha(), mock.patch(
                "deterministic_iafno.compute_centered_stats."
                "LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            payload_again, _ = compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=64,
                batch_size=32,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
            )
        self.assertEqual(
            json.dumps(payload, sort_keys=True),
            json.dumps(payload_again, sort_keys=True),
        )

    def test_mean_sha_mismatch_fails_closed(self):
        # No patch: the real file SHA cannot equal the locked identity.
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=16,
                batch_size=32,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
            )

    def test_fresh_input_validation_positive_and_negative(self):
        with self._patch_locked_sha(), mock.patch(
                "deterministic_iafno.compute_centered_stats."
                "LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            payload, _ = compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=32,
                batch_size=32,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
            )
            with open(self.stats_path, "w", encoding="utf-8") as file:
                json.dump(payload, file)
        from diafno.models.config import OSTIAModelConfig
        model_config = OSTIAModelConfig(
            image_size=(8, 8, 1),
            patch_size=(2, 2, 1),
            embed_dim=8,
            num_blocks=2,
            explicit_layer=1,
            implicit_layer=1,
            hidden_size_factor=2,
            cond_chans=8,
            target_chans=15,
            target_mode="residual",
            model_type="centered_diffusion",
            target_scaling="lead_standardized",
            sigma_data=1.0,
            lead_mean=tuple(payload["lead_mean"]),
            lead_std=tuple(payload["lead_std"]),
            mean_lead_mean=tuple(payload["mean_lead_mean"]),
            mean_lead_std=tuple(payload["mean_lead_std"]),
        )
        with self._patch_locked_sha():
            validated, immutable = validate_centered_fresh_inputs(
                self.mean_path,
                self.stats_path,
                model_config,
            )
        self.assertEqual(validated["lead_mean"],
                         tuple(payload["lead_mean"]))
        self.assertEqual(immutable["model_type"], "deterministic")
        self.assertEqual(immutable["target_mode"], "residual")
        self.assertEqual(immutable["target_scaling"],
                         "lead_standardized")
        self.assertEqual(immutable["input_days"], 7)
        self.assertEqual(immutable["output_days"], 15)
        # A missing sidecar fails closed.
        sidecar_path = self.mean_path + ".semantics.json"
        backup = sidecar_path + ".bak"
        os.replace(sidecar_path, backup)
        try:
            with self._patch_locked_sha():
                with self.assertRaisesRegex(ValueError, "sidecar"):
                    validate_centered_fresh_inputs(
                        self.mean_path,
                        self.stats_path,
                        model_config,
                    )
        finally:
            os.replace(backup, sidecar_path)
        # Architecture mismatch fails closed.
        model_config.image_size = (16, 16, 1)
        with self._patch_locked_sha():
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_centered_fresh_inputs(
                    self.mean_path,
                    self.stats_path,
                    model_config,
                )

    def test_mean_sidecar_lead_stats_mismatch_fails(self):
        with self._patch_locked_sha():
            immutable = {
                "model_type": "deterministic",
                "target_mode": "residual",
                "target_scaling": "lead_standardized",
                "input_days": 7,
                "output_days": 15,
                "lead_mean": [float(value) for value in range(15)],
                "lead_std": [1.0 + value for value in range(15)],
            }
            stats = valid_payload(
                mean_lead_mean=[999.0] * 15,
                mean_checkpoint_sha256=self.mean_sha,
            )
            # A sidecar whose lead stats disagree with the stats JSON
            # must be rejected by cross_check_mean_sidecar.
            sidecar_path = self.mean_path + ".semantics.json"
            backup = sidecar_path + ".bak"
            if os.path.isfile(sidecar_path):
                os.replace(sidecar_path, backup)
            try:
                with open(sidecar_path, "w", encoding="utf-8") as file:
                    json.dump({
                        "schema_version": 4,
                        "semantic_manifest": {
                            "schema_version": 4,
                            "immutable": immutable,
                        },
                    }, file)
                with self.assertRaisesRegex(ValueError, "lead_mean"):
                    cross_check_mean_sidecar(stats, self.mean_path)
            finally:
                if os.path.isfile(sidecar_path):
                    os.remove(sidecar_path)
                if os.path.isfile(backup):
                    os.replace(backup, sidecar_path)


if __name__ == "__main__":
    unittest.main()
