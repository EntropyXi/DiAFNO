import unittest
from copy import deepcopy

from diafno.training.config import OSTIATrainingConfig
from deterministic_iafno.checkpoint_semantics import (
    build_semantic_manifest,
    validate_semantic_manifest,
)


class CheckpointSemanticTests(unittest.TestCase):
    def setUp(self):
        self.config = OSTIATrainingConfig()

    def checkpoint(self, config=None, world_size=2):
        config = self.config if config is None else config
        return {
            "semantic_manifest": build_semantic_manifest(
                config,
                world_size=world_size,
            )
        }

    def test_matching_manifest_passes(self):
        warnings = validate_semantic_manifest(
            self.checkpoint(),
            self.config,
            world_size=2,
        )
        self.assertEqual(warnings, [])

    def test_training_noise_mismatch_fails(self):
        checkpoint = self.checkpoint()
        changed = deepcopy(self.config)
        changed.model.p_mean = -0.5
        with self.assertRaisesRegex(
                ValueError,
                "immutable semantic mismatch",
            ):
            validate_semantic_manifest(
                checkpoint,
                changed,
                world_size=2,
            )

        changed = deepcopy(self.config)
        changed.model.p_std = 0.8
        with self.assertRaisesRegex(
                ValueError,
                "immutable semantic mismatch",
            ):
            validate_semantic_manifest(
                checkpoint,
                changed,
                world_size=2,
            )

    def test_sampler_mismatch_warns(self):
        checkpoint = self.checkpoint()
        changed = deepcopy(self.config)
        changed.model.sampling_steps = 32
        changed.model.rho = 5.0
        warnings = validate_semantic_manifest(
            checkpoint,
            changed,
            world_size=2,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("sampler profile differs", warnings[0])

    def test_effective_batch_mismatch_requires_override(self):
        checkpoint = self.checkpoint()
        with self.assertRaisesRegex(
                ValueError,
                "training compatibility mismatch",
            ):
            validate_semantic_manifest(
                checkpoint,
                self.config,
                world_size=1,
            )
        warnings = validate_semantic_manifest(
            checkpoint,
            self.config,
            world_size=1,
            allow_compatible_override=True,
        )
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
