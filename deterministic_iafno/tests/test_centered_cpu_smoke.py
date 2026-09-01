import os
import shutil
import unittest

import torch

from diafno.training.artifacts import CheckpointManager
from diafno.training.config import OSTIATrainingConfig
from diafno.training.runtime import DistributedRuntime
from diafno.training.trainer import OSTIATrainer
from deterministic_iafno.checkpoint_semantics import (
    load_semantic_sidecar,
)


class FakeSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class FakeLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class DatasetStub:
    normalization = {
        "sst_mean": 290.7488927184541,
        "sst_std": 9.57073350168232,
    }


def centered_train_config(output_dir):
    config = OSTIATrainingConfig()
    config.output_dir = output_dir
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
    config.model.input_days = 3
    config.model.output_days = 2
    config.model.cond_chans = 4
    config.model.target_chans = 2
    config.model.lead_mean = (0.1, -0.1)
    config.model.lead_std = (1.0, 2.0)
    config.model.mean_lead_mean = (0.0, 0.1)
    config.model.mean_lead_std = (1.0, 1.5)
    config.model.mean_checkpoint_sha256 = "ab" * 32
    config.model.mean_semantics_sha256 = "cd" * 32
    config.num_epochs = 2
    config.batch_per_gpu = 2
    config.gradient_accumulation = 1
    config.checkpoint_interval = 1
    config.use_amp = False
    # Skip the fresh-run mean file load in this smoke: the wrapper is
    # built with the stats above and the mean stays frozen regardless.
    config.resume_path = "latest"
    return config


class CenteredCpuSmokeTests(unittest.TestCase):
    """Trainer-level tiny CPU smoke: real training loop, real AMP
    guard, real checkpoint save with schema-4 semantics."""

    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(tests_dir, ".tmp_centered_smoke")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_trainer(self, config, batches):
        trainer = OSTIATrainer.__new__(OSTIATrainer)
        trainer.config = config
        trainer.runtime = DistributedRuntime()
        trainer.data = type(
            "FakeData",
            (),
            {
                "sampler": FakeSampler(),
                "loader": FakeLoader(batches),
                "dataset": DatasetStub(),
            },
        )()
        trainer.skipped_optimizer_steps = 0
        trainer.skipped_optimizer_step_numbers = []
        trainer._mean_grad_asserted = False
        trainer.start_epoch = 0
        trainer.global_step = 0
        trainer.checkpoints = CheckpointManager(config)
        trainer.history = __import__(
            "diafno.training.artifacts", fromlist=["TrainingHistory"]
        ).TrainingHistory(config.output_dir, config.max_grad_norm)
        trainer._build_training_components()
        return trainer

    @staticmethod
    def batches(count=4, batch=2):
        torch.manual_seed(7)
        return [
            {
                "condition": torch.randn(batch, 4, 8, 8, 1),
                "target": torch.randn(batch, 2, 8, 8, 1),
                "target_mask": torch.ones(batch, 2, 8, 8, 1),
            }
            for _ in range(count)
        ]

    def test_train_epoch_smoke(self):
        config = centered_train_config(self.tmp_dir)
        trainer = self._build_trainer(config, self.batches())
        model = trainer.model
        diffusion_before = {
            name: parameter.detach().clone()
            for name, parameter in model.diffusion.named_parameters()
        }
        epoch_loss, epoch_batches = trainer._train_epoch(0)
        self.assertEqual(epoch_batches, 4)
        self.assertTrue(torch.isfinite(torch.tensor(epoch_loss)))
        self.assertEqual(trainer.global_step, 4)
        self.assertEqual(trainer.skipped_optimizer_steps, 0)
        self.assertEqual(trainer.skipped_optimizer_step_numbers, [])
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.mean_model.parameters()
        ))
        updated = any(
            not torch.equal(parameter, diffusion_before[name])
            for name, parameter in model.diffusion.named_parameters()
        )
        self.assertTrue(updated)
        self.assertFalse(model.mean_model.training)
        self.assertEqual(trainer.data.sampler.epochs, [0])

    def test_epoch_checkpoint_and_sidecar(self):
        config = centered_train_config(self.tmp_dir)
        trainer = self._build_trainer(config, self.batches())
        epoch_loss, _ = trainer._train_epoch(0)
        mean_loss = trainer._mean_train_loss(epoch_loss, 4)
        trainer._finish_epoch(0, mean_loss)
        checkpoint_path = os.path.join(self.tmp_dir, "latest.pth")
        self.assertTrue(os.path.isfile(checkpoint_path))
        sidecar = load_semantic_sidecar(checkpoint_path)
        self.assertIsNotNone(sidecar)
        immutable = sidecar["semantic_manifest"]["immutable"]
        self.assertEqual(
            immutable["model_type"], "centered_diffusion"
        )
        self.assertEqual(immutable["sigma_data"], 1.0)
        self.assertEqual(immutable["split"], "train")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.assertEqual(checkpoint["global_step"], 4)
        self.assertEqual(checkpoint["skipped_optimizer_steps"], 0)
        self.assertEqual(
            checkpoint["skipped_optimizer_step_numbers"], []
        )
        sample = trainer.model.sample(
            self.batches(count=1, batch=1)[0]["condition"],
            num_sample_steps=4,
            seed=1,
        )
        self.assertEqual(tuple(sample.shape), (1, 2, 8, 8, 1))
        self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
