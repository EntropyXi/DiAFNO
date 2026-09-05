# 用途：验证centered 配置与命令行参数校验。
import json
import os
import unittest

from diafno.training.config import (
    build_parser,
    merge_config_json,
    training_config_from_args,
)
from deterministic_iafno.centered_stats import (
    LOCKED_MEAN_CHECKPOINT_SHA256,
)


def valid_centered_payload():
    return {
        "schema_version": 1,
        "split": "train",
        "target_space": "normalized_centered_residual",
        "input_days": 7,
        "output_days": 15,
        "condition_mode": "sst_mask",
        "num_samples": 8192,
        "dataset_size": 786100,
        "selection": "evenly_spaced_sequence_spatial_chunk_blocks",
        "indices_sha256": "a" * 64,
        "mean_checkpoint": "/fake/frozen_mean.pth",
        "mean_checkpoint_sha256": LOCKED_MEAN_CHECKPOINT_SHA256,
        "mean_semantics_sha256": "b" * 64,
        "mean_lead_mean": [float(value) for value in range(15)],
        "mean_lead_std": [1.0 + value for value in range(15)],
        "sst_mean": 290.75,
        "sst_std": 9.57,
        "lead_mean": [float(value) for value in range(15)],
        "lead_std": [1.0 + value for value in range(15)],
        "overall_innovation_std": 1.0,
        "valid_pixels": [1000] * 15,
    }


class CenteredConfigCliTests(unittest.TestCase):
    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.stats_path = os.path.join(
            tests_dir, ".tmp_centered_stats_cli.json"
        )
        with open(self.stats_path, "w", encoding="utf-8") as file:
            json.dump(valid_centered_payload(), file)

    def tearDown(self):
        if os.path.isfile(self.stats_path):
            os.remove(self.stats_path)

    def parse(self, *cli):
        return build_parser().parse_args(list(cli))

    def test_fresh_centered_requires_mean_checkpoint(self):
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--centered-stats", self.stats_path,
            "--sigma-data", "1.0",
        )
        with self.assertRaisesRegex(ValueError, "--mean-checkpoint"):
            training_config_from_args(args)

    def test_fresh_centered_requires_stats(self):
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--mean-checkpoint", "/fake/frozen_mean.pth",
            "--sigma-data", "1.0",
        )
        with self.assertRaisesRegex(ValueError, "--centered-stats"):
            training_config_from_args(args)

    def test_factory_sigma_data_rejected_for_centered(self):
        # No --sigma-data: the 0.15 factory default must fail closed.
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--mean-checkpoint", "/fake/frozen_mean.pth",
            "--centered-stats", self.stats_path,
        )
        with self.assertRaisesRegex(ValueError, "sigma_data=1.0"):
            training_config_from_args(args)

    def test_nonunit_sigma_data_rejected_for_centered(self):
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--mean-checkpoint", "/fake/frozen_mean.pth",
            "--centered-stats", self.stats_path,
            "--sigma-data", "0.15",
        )
        with self.assertRaisesRegex(ValueError, "sigma_data=1.0"):
            training_config_from_args(args)

    def test_centered_rejects_init_from_and_lead_stats(self):
        base = (
            "--model-type", "centered_diffusion",
            "--mean-checkpoint", "/fake/frozen_mean.pth",
            "--centered-stats", self.stats_path,
            "--sigma-data", "1.0",
        )
        with self.assertRaisesRegex(ValueError, "--init-from"):
            training_config_from_args(
                self.parse(*base, "--init-from", "/fake/old.pth")
            )
        with self.assertRaisesRegex(ValueError, "--lead-stats"):
            training_config_from_args(
                self.parse(*base, "--lead-stats", "/fake/old.json")
            )

    def test_fresh_centered_config_populates_model_fields(self):
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--mean-checkpoint", "/fake/frozen_mean.pth",
            "--centered-stats", self.stats_path,
            "--sigma-data", "1.0",
        )
        config = training_config_from_args(args)
        self.assertEqual(
            config.model.model_type, "centered_diffusion"
        )
        self.assertEqual(config.model.sigma_data, 1.0)
        self.assertEqual(config.model.lead_mean, tuple(
            float(value) for value in range(15)
        ))
        self.assertEqual(
            config.model.mean_lead_std,
            tuple(1.0 + value for value in range(15)),
        )
        self.assertEqual(
            config.model.mean_checkpoint_sha256,
            LOCKED_MEAN_CHECKPOINT_SHA256,
        )
        self.assertEqual(len(config.model.mean_semantics_sha256), 64)
        self.assertIn(
            "lead_std", config.explicit_resume_fields
        )
        self.assertIn(
            "mean_checkpoint_sha256",
            config.explicit_resume_fields,
        )

    def test_centered_resume_does_not_require_fresh_paths(self):
        args = self.parse(
            "--model-type", "centered_diffusion",
            "--resume",
            "--sigma-data", "1.0",
        )
        config = training_config_from_args(args)
        self.assertEqual(config.resume_path, "latest")
        self.assertIsNone(config.mean_checkpoint_path)
        self.assertIsNone(config.centered_stats_path)

    def test_config_json_merge_and_override_notes(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".tmp_config_json.json",
        )
        payload = {
            "model_type": "centered_diffusion",
            "sigma_data": 1.0,
            "batch_per_gpu": 8,
        }
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        try:
            args = self.parse(
                "--config", config_path,
                "--batch-per-gpu", "16",
            )
            overrides = merge_config_json(args, config_path)
            self.assertEqual(args.model_type, "centered_diffusion")
            self.assertEqual(args.sigma_data, 1.0)
            self.assertEqual(args.batch_per_gpu, 16)
            self.assertEqual(len(overrides), 1)
            self.assertIn("batch_per_gpu", overrides[0])
        finally:
            if os.path.isfile(config_path):
                os.remove(config_path)

    def test_config_json_unknown_keys_rejected(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".tmp_config_json_bad.json",
        )
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump({"not_a_field": 1}, file)
        try:
            args = self.parse("--config", config_path)
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                merge_config_json(args, config_path)
        finally:
            if os.path.isfile(config_path):
                os.remove(config_path)


if __name__ == "__main__":
    unittest.main()
