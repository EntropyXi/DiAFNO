import os
import shutil
import unittest

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diafno.training.artifacts import CheckpointManager, TrainingHistory
from diafno.training.config import OSTIATrainingConfig
from diafno.training.runtime import DistributedRuntime
from diafno.training.trainer import OSTIATrainer


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def build_trainer(tmp_dir, model=None):
    trainer = OSTIATrainer.__new__(OSTIATrainer)
    config = OSTIATrainingConfig()
    config.output_dir = tmp_dir
    trainer.config = config
    trainer.runtime = DistributedRuntime()
    trainer.model = TinyLinear() if model is None else model
    trainable = [
        parameter
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    ]
    trainer.optimizer = AdamW(trainable, lr=1e-3)
    trainer.scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=10,
        eta_min=1e-6,
    )
    trainer.scaler = GradScaler("cuda", enabled=False)
    trainer.amp_enabled = False
    trainer.global_step = 0
    trainer.skipped_optimizer_steps = 0
    trainer.skipped_optimizer_step_numbers = []
    trainer._mean_grad_asserted = False
    trainer.history = TrainingHistory(tmp_dir, max_grad_norm=1.0)
    trainer.checkpoints = CheckpointManager(config)
    return trainer


class AmpOverflowSkipTests(unittest.TestCase):
    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(tests_dir, ".tmp_amp_skip")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def backward_once(self, trainer, seed=3):
        torch.manual_seed(seed)
        loss = trainer.model(
            torch.randn(2, 4)
        ).sum()
        trainer.scaler.scale(loss).backward()
        return loss

    def test_overflow_detector_flags_nonfinite_grads(self):
        trainer = build_trainer(self.tmp_dir)
        self.backward_once(trainer)
        self.assertFalse(trainer._detect_grad_overflow())
        for parameter in trainer.optimizer.param_groups[0]["params"]:
            parameter.grad.add_(float("inf"))
        self.assertTrue(trainer._detect_grad_overflow())

    def test_overflow_skips_optimizer_and_scheduler(self):
        trainer = build_trainer(self.tmp_dir)
        self.backward_once(trainer)
        before = {
            id(parameter): parameter.detach().clone()
            for parameter in trainer.optimizer.param_groups[0]["params"]
        }
        original_detect = trainer._detect_grad_overflow
        trainer._detect_grad_overflow = lambda: True
        try:
            trainer._optimizer_update()
        finally:
            trainer._detect_grad_overflow = original_detect
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(trainer.skipped_optimizer_steps, 1)
        self.assertEqual(
            trainer.skipped_optimizer_step_numbers, [1]
        )
        self.assertEqual(
            trainer.scheduler.last_epoch, 0
        )
        for parameter in trainer.optimizer.param_groups[0]["params"]:
            self.assertTrue(torch.equal(
                parameter.detach(), before[id(parameter)]
            ))
        self.assertEqual(
            trainer.history.skipped_optimizer_steps, [1]
        )
        self.assertEqual(trainer.history.gradient_steps, [])

    def test_clean_step_advances_optimizer_and_scheduler(self):
        trainer = build_trainer(self.tmp_dir)
        self.backward_once(trainer)
        before = {
            id(parameter): parameter.detach().clone()
            for parameter in trainer.optimizer.param_groups[0]["params"]
        }
        trainer._optimizer_update()
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(trainer.skipped_optimizer_steps, 0)
        self.assertEqual(
            trainer.skipped_optimizer_step_numbers, []
        )
        self.assertEqual(trainer.scheduler.last_epoch, 1)
        changed = [
            id(parameter)
            for parameter in trainer.optimizer.param_groups[0]["params"]
            if not torch.equal(
                parameter.detach(), before[id(parameter)]
            )
        ]
        self.assertTrue(changed)
        self.assertEqual(trainer.history.gradient_steps, [1])
        self.assertEqual(
            trainer.history.skipped_optimizer_steps, []
        )

    def test_global_step_tracks_updates_even_when_skipped(self):
        trainer = build_trainer(self.tmp_dir)
        original_detect = trainer._detect_grad_overflow
        trainer._detect_grad_overflow = lambda: True
        try:
            for _ in range(3):
                trainer.optimizer.zero_grad(set_to_none=True)
                self.backward_once(trainer)
                trainer._optimizer_update()
        finally:
            trainer._detect_grad_overflow = original_detect
        self.assertEqual(trainer.global_step, 3)
        self.assertEqual(trainer.skipped_optimizer_steps, 3)
        self.assertEqual(
            trainer.skipped_optimizer_step_numbers, [1, 2, 3]
        )
        self.assertEqual(trainer.scheduler.last_epoch, 0)

    def test_mean_grad_assertion_after_first_step(self):
        mean = nn.Linear(2, 2)
        mean.requires_grad_(False)
        mean.eval()
        mean.weight.grad = torch.ones_like(mean.weight)
        trainer = build_trainer(self.tmp_dir)
        trainer.model.mean_model = mean
        with self.assertRaises(AssertionError):
            trainer._assert_mean_frozen_grads()

    def test_optimizer_excludes_frozen_mean(self):
        from deterministic_iafno.centered_diffusion import (
            FrozenMeanCenteredDiffusion,
        )
        from deterministic_iafno.tests.test_centered_diffusion import (
            RecordingDiffusion,
            RecordingMean,
        )
        wrapper = FrozenMeanCenteredDiffusion(
            RecordingMean((0.0, 0.0)),
            RecordingDiffusion(),
            lead_mean=(0.0, 0.0),
            lead_std=(1.0, 1.0),
        )
        config = OSTIATrainingConfig()
        config.output_dir = self.tmp_dir
        config.model.model_type = "centered_diffusion"
        config.model.image_size = (2, 2, 1)
        config.model.patch_size = (1, 1, 1)
        config.model.embed_dim = 4
        config.model.num_blocks = 2
        config.model.explicit_layer = 1
        config.model.implicit_layer = 1
        config.model.hidden_size_factor = 2
        config.model.cond_chans = 2
        config.model.target_chans = 2
        config.model.target_mode = "residual"
        config.model.target_scaling = "lead_standardized"
        config.model.sigma_data = 1.0
        config.model.lead_mean = (0.0, 0.0)
        config.model.lead_std = (1.0, 1.0)
        config.model.mean_lead_mean = (0.0, 0.0)
        config.model.mean_lead_std = (1.0, 1.0)
        # resume_path not None skips the fresh-run mean file load
        config.resume_path = "latest"
        trainer = OSTIATrainer.__new__(OSTIATrainer)
        trainer.config = config
        trainer.runtime = DistributedRuntime()
        trainer.checkpoints = CheckpointManager(config)
        trainer.model = wrapper
        trainer._reassert_frozen_mean()
        mean_params = {
            id(parameter)
            for parameter in wrapper.mean_model.parameters()
        }
        optimizer = AdamW(
            [
                parameter
                for parameter in wrapper.parameters()
                if parameter.requires_grad
            ],
            lr=1e-3,
        )
        self.assertTrue(all(
            id(parameter) not in mean_params
            for parameter in optimizer.param_groups[0]["params"]
        ))
        self.assertFalse(wrapper.mean_model.training)


if __name__ == "__main__":
    unittest.main()
