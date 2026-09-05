# 用途：验证模型及优化器 checkpoint 的保存恢复。
import os
import shutil
import unittest

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

from diafno.training.artifacts import CheckpointManager
from diafno.training.config import OSTIATrainingConfig
from deterministic_iafno.checkpoint_semantics import (
    load_semantic_sidecar,
    resolve_sidecar_path,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class DatasetStub:
    normalization = {
        "sst_mean": 290.7488927184541,
        "sst_std": 9.57073350168232,
    }


class CheckpointRoundTripTests(unittest.TestCase):
    """Save/load round trip through CheckpointManager with the semantic
    manifest and the per-checkpoint sidecar file."""

    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(tests_dir, ".tmp_ckpt")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(
            self.tmp_dir,
            "latest.pth",
        )
        self.config = OSTIATrainingConfig()
        self.config.output_dir = self.tmp_dir

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_load_round_trip(self):
        manager = CheckpointManager(self.config)

        model = TinyNet()
        optimizer = AdamW(model.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6,
        )
        scaler = GradScaler("cuda", enabled=False)
        random_state = CheckpointManager.capture_random_state()
        manager.save(
            self.checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=3,
            global_step=42,
            train_loss=0.123,
            dataset=DatasetStub(),
            random_states=[random_state],
        )
        self.assertTrue(os.path.isfile(self.checkpoint_path))

        sidecar_path = resolve_sidecar_path(self.checkpoint_path)
        self.assertIsNotNone(sidecar_path)
        self.assertTrue(
            sidecar_path.endswith("latest.pth.semantics.json")
        )
        sidecar = load_semantic_sidecar(self.checkpoint_path)
        self.assertIsNotNone(sidecar)
        manifest = sidecar["semantic_manifest"]
        self.assertEqual(
            manifest["immutable"]["target_mode"],
            "residual",
        )
        self.assertEqual(
            manifest["immutable"]["model_type"],
            "diffusion",
        )
        self.assertEqual(
            manifest["immutable"]["target_chans"],
            15,
        )

        # Load into a fresh model of the same class.
        fresh_model = TinyNet()
        fresh_optimizer = AdamW(
            fresh_model.parameters(),
            lr=2e-4,
        )
        fresh_scheduler = CosineAnnealingLR(
            fresh_optimizer,
            T_max=10,
            eta_min=1e-6,
        )
        fresh_scaler = GradScaler("cuda", enabled=False)
        loaded = manager.load(
            self.checkpoint_path,
            fresh_model,
            fresh_optimizer,
            fresh_scheduler,
            fresh_scaler,
            torch.device("cpu"),
            rank=0,
            world_size=1,
        )
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["global_step"], 42)
        self.assertTrue(torch.allclose(
            fresh_model.linear.weight,
            model.linear.weight,
        ))
        self.assertEqual(
            fresh_optimizer.param_groups[0]["lr"],
            optimizer.param_groups[0]["lr"],
        )

    def test_sidecar_resume_restore_round_trip(self):
        # A deterministic lead-standardized run: bare resume must
        # restore its model_type/target_scaling/lead stats from the
        # sidecar, exactly like the trainer does.
        self.config.model.model_type = "deterministic"
        self.config.model.target_scaling = "lead_standardized"
        self.config.model.lead_mean = tuple(
            float(value) for value in range(15)
        )
        self.config.model.lead_std = tuple(
            1.0 + value for value in range(15)
        )
        manager = CheckpointManager(self.config)
        model = TinyNet()
        optimizer = AdamW(model.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        scaler = GradScaler("cuda", enabled=False)
        random_state = CheckpointManager.capture_random_state()
        manager.save(
            self.checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=1,
            global_step=10,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[random_state],
        )

        from dataclasses import asdict
        from diafno.training.config import default_training_model
        from deterministic_iafno.checkpoint_semantics import (
            restore_resume_semantics,
        )

        sidecar = load_semantic_sidecar(self.checkpoint_path)
        resume_config = OSTIATrainingConfig()
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        restore_resume_semantics(
            sidecar,
            resume_config,
            defaults,
        )
        self.assertEqual(
            resume_config.model.model_type,
            "deterministic",
        )
        self.assertEqual(
            resume_config.model.target_scaling,
            "lead_standardized",
        )
        self.assertEqual(len(resume_config.model.lead_mean), 15)

    def test_reviewed_optimizer_and_schedule_override_is_applied(self):
        manager = CheckpointManager(self.config)
        model = TinyNet()
        optimizer = AdamW(model.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6,
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
            global_step=2,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[random_state],
        )

        self.config.learning_rate = 1e-3
        self.config.num_epochs = 50
        self.config.allow_resume_override = True
        fresh_model = TinyNet()
        fresh_optimizer = AdamW(
            fresh_model.parameters(),
            lr=self.config.learning_rate,
        )
        fresh_scheduler = CosineAnnealingLR(
            fresh_optimizer,
            T_max=20,
            eta_min=self.config.min_learning_rate,
        )
        fresh_scaler = GradScaler("cuda", enabled=False)
        manager.load(
            self.checkpoint_path,
            fresh_model,
            fresh_optimizer,
            fresh_scheduler,
            fresh_scaler,
            torch.device("cpu"),
            rank=0,
            world_size=1,
        )
        self.assertEqual(
            fresh_optimizer.param_groups[0]["lr"],
            self.config.learning_rate,
        )
        self.assertEqual(fresh_scheduler.T_max, 20)
        self.assertEqual(
            fresh_scheduler.base_lrs,
            [self.config.learning_rate],
        )


if __name__ == "__main__":
    unittest.main()
