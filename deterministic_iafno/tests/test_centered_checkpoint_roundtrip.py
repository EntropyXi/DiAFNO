# 用途：验证centered 模型保存与恢复的一致性。
import os
import shutil
import unittest

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diafno.inference.model import InferenceModelLoader
from diafno.models.config import OSTIAModelConfig
from diafno.training.artifacts import CheckpointManager
from diafno.training.config import (
    OSTIATrainingConfig,
    default_training_model,
)
from deterministic_iafno.checkpoint_semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    build_semantic_manifest,
    load_semantic_sidecar,
    restore_resume_semantics,
)
from deterministic_iafno.centered_diffusion import (
    FrozenMeanCenteredDiffusion,
)


class DatasetStub:
    normalization = {
        "sst_mean": 290.7488927184541,
        "sst_std": 9.57073350168232,
    }


def centered_config():
    config = OSTIATrainingConfig()
    config.output_dir = None
    config.model.model_type = "centered_diffusion"
    config.model.target_mode = "residual"
    config.model.target_scaling = "lead_standardized"
    config.model.sigma_data = 1.0
    config.model.image_size = (8, 8, 1)
    config.model.patch_size = (2, 2, 1)
    config.model.embed_dim = 8
    config.model.num_blocks = 2
    config.model.explicit_layer = 1
    config.model.implicit_layer = 1
    config.model.hidden_size_factor = 2
    config.model.cond_chans = 3
    config.model.target_chans = 2
    config.model.lead_mean = (0.1, -0.1)
    config.model.lead_std = (1.0, 2.0)
    config.model.mean_lead_mean = (0.2, 0.3)
    config.model.mean_lead_std = (1.0, 1.5)
    config.model.mean_checkpoint_sha256 = "ab" * 32
    config.model.mean_semantics_sha256 = "cd" * 32
    return config


class CenteredCheckpointRoundTripTests(unittest.TestCase):
    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(tests_dir, ".tmp_centered_ckpt")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(
            self.tmp_dir, "latest.pth"
        )
        self.config = centered_config()
        self.config.output_dir = self.tmp_dir

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _save(self, config=None):
        config = self.config if config is None else config
        manager = CheckpointManager(config)
        model = config.model.build_model(torch.device("cpu"))
        optimizer = AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=2e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        scaler = GradScaler("cuda", enabled=False)
        random_state = CheckpointManager.capture_random_state()
        manager.save(
            self.checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=1,
            global_step=7,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[random_state],
            skipped_optimizer_steps=2,
            skipped_optimizer_step_numbers=[3, 6],
        )
        return model

    def test_schema4_sidecar_carries_centered_semantics(self):
        self._save()
        sidecar = load_semantic_sidecar(self.checkpoint_path)
        self.assertIsNotNone(sidecar)
        self.assertEqual(
            sidecar["schema_version"], CHECKPOINT_SCHEMA_VERSION
        )
        self.assertEqual(
            sidecar["semantic_manifest"]["schema_version"],
            CHECKPOINT_SCHEMA_VERSION,
        )
        immutable = sidecar["semantic_manifest"]["immutable"]
        self.assertEqual(
            immutable["model_type"], "centered_diffusion"
        )
        self.assertEqual(immutable["target_mode"], "residual")
        self.assertEqual(
            immutable["target_scaling"], "lead_standardized"
        )
        self.assertEqual(immutable["split"], "train")
        self.assertEqual(immutable["condition_mode"], "sst_mask")
        self.assertEqual(immutable["sigma_data"], 1.0)
        self.assertEqual(immutable["lead_mean"], [0.1, -0.1])
        self.assertEqual(immutable["lead_std"], [1.0, 2.0])
        self.assertEqual(
            immutable["mean_lead_mean"], [0.2, 0.3]
        )
        self.assertEqual(
            immutable["mean_lead_std"], [1.0, 1.5]
        )
        self.assertEqual(
            immutable["mean_checkpoint_sha256"], "ab" * 32
        )
        self.assertEqual(
            immutable["mean_semantics_sha256"], "cd" * 32
        )

    def test_checkpoint_contains_mean_weights_and_skip_counters(self):
        self._save()
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu",
            weights_only=False,
        )
        keys = set(checkpoint["model"].keys())
        self.assertTrue(any(
            key.startswith("mean_model.") for key in keys
        ))
        self.assertEqual(checkpoint["skipped_optimizer_steps"], 2)
        self.assertEqual(
            checkpoint["skipped_optimizer_step_numbers"], [3, 6]
        )

    def test_inference_rebuilds_without_mean_path(self):
        # The centered checkpoint is self-contained: no external mean
        # file exists anywhere in this test, yet the loader must
        # rebuild the wrapper (mean weights + stats) and sample.
        self._save()
        model, model_config, steps, normalization = (
            InferenceModelLoader.load(
                self.checkpoint_path,
                torch.device("cpu"),
            )
        )
        self.assertIsInstance(
            model, FrozenMeanCenteredDiffusion
        )
        self.assertEqual(
            model_config.model_type, "centered_diffusion"
        )
        self.assertEqual(steps, model_config.sampling_steps)
        self.assertEqual(normalization["sst_mean"], 290.7488927184541)
        for parameter in model.mean_model.parameters():
            self.assertFalse(parameter.requires_grad)
        condition = torch.randn(1, 3, 8, 8, 1)
        forecast = model.sample(
            condition, num_sample_steps=4, seed=1
        )
        self.assertEqual(tuple(forecast.shape), (1, 2, 8, 8, 1))
        self.assertTrue(torch.isfinite(forecast).all())

    def test_bare_resume_restores_centered_semantics(self):
        self._save()
        sidecar = load_semantic_sidecar(self.checkpoint_path)
        from dataclasses import asdict
        current = OSTIATrainingConfig()
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        notices = restore_resume_semantics(
            sidecar, current, defaults
        )
        self.assertEqual(
            current.model.model_type, "centered_diffusion"
        )
        self.assertEqual(current.model.sigma_data, 1.0)
        self.assertEqual(current.model.lead_mean, (0.1, -0.1))
        self.assertEqual(current.model.lead_std, (1.0, 2.0))
        self.assertEqual(
            current.model.mean_lead_mean, (0.2, 0.3)
        )
        self.assertEqual(
            current.model.mean_lead_std, (1.0, 1.5)
        )
        self.assertEqual(
            current.model.mean_checkpoint_sha256, "ab" * 32
        )
        self.assertTrue(any(
            "restored immutable semantics" in notice
            for notice in notices
        ))

    def test_explicit_centered_conflict_fails_closed(self):
        self._save()
        sidecar = load_semantic_sidecar(self.checkpoint_path)
        from dataclasses import asdict
        current = OSTIATrainingConfig()
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        with self.assertRaisesRegex(
                ValueError, "immutable semantic conflict"
            ):
            restore_resume_semantics(
                sidecar,
                current,
                defaults,
                explicit_fields={"mean_checkpoint_sha256"},
            )

    def test_schema3_legacy_manifest_still_validates(self):
        # A schema-3 saved manifest must stay readable: fields the old
        # checkpoint lacks are not compared against the new config.
        legacy_manifest = build_semantic_manifest(
            OSTIATrainingConfig(), world_size=1
        )
        legacy_manifest["schema_version"] = 3
        immutable = legacy_manifest["immutable"]
        for field in (
                "mean_lead_mean",
                "mean_lead_std",
                "mean_checkpoint_sha256",
                "mean_semantics_sha256",
            ):
            immutable.pop(field, None)
        checkpoint = {"semantic_manifest": legacy_manifest}
        from deterministic_iafno.checkpoint_semantics import (
            validate_semantic_manifest,
        )
        warnings = validate_semantic_manifest(
            checkpoint,
            OSTIATrainingConfig(),
            world_size=1,
        )
        self.assertEqual(warnings, [])

    def test_roundtrip_load_restores_weights_and_skip_counters(self):
        original = self._save()
        manager = CheckpointManager(self.config)
        fresh = self.config.model.build_model(torch.device("cpu"))
        optimizer = AdamW(
            [
                parameter
                for parameter in fresh.parameters()
                if parameter.requires_grad
            ],
            lr=2e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-6
        )
        scaler = GradScaler("cuda", enabled=False)
        checkpoint = manager.load(
            self.checkpoint_path,
            fresh,
            optimizer,
            scheduler,
            scaler,
            torch.device("cpu"),
            rank=0,
            world_size=1,
        )
        self.assertEqual(checkpoint["global_step"], 7)
        self.assertEqual(
            checkpoint["skipped_optimizer_steps"], 2
        )
        original_state = original.state_dict()
        for key, value in fresh.state_dict().items():
            self.assertTrue(torch.allclose(
                value, original_state[key]
            ))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in fresh.mean_model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
