# 用途：验证条件通道配置、checkpoint 语义及模型前后向。
"""Condition-schema config, checkpoint contracts and CPU forward/
backward coverage for the three model types (plan section 6.2)."""

import json
import os
import unittest
from dataclasses import asdict

import numpy as np
import torch

from diafno.data.condition_schema import (
    GEO_STATIC_CHANNEL_NAMES,
    VALID_MASK_CHANNEL_NAME,
    condition_channel_names,
    condition_chans,
    condition_schema_version_for,
    resolve_condition_mode,
)
from diafno.data.ostia import (
    OSTIADailyDataset,
    copy_dataset_provenance,
    verify_checkpoint_data_contract,
)
from diafno.inference.model import InferenceModelLoader
from diafno.models.config import OSTIAModelConfig
from diafno.training.artifacts import CheckpointManager
from diafno.training.config import (
    OSTIATrainingConfig,
    build_parser,
    default_training_model,
    merge_config_json,
    training_config_from_args,
)
from diafno.training.data import OSTIATrainingData
from diafno.training.runtime import DistributedRuntime
from diafno.training.trainer import OSTIATrainer
from deterministic_iafno.checkpoint_semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    build_semantic_manifest,
    load_semantic_sidecar,
    restore_resume_semantics,
    validate_semantic_manifest,
)

from .ostia_test_h5 import (
    OSTIATestCase,
    make_synthetic_h5,
)


def tiny_model_config(
        mode="sst_mask",
        patch_size=(2, 2, 1),
        num_blocks=2,
        implicit_layer=1,
        image_size=(8, 8, 1),
        input_days=3,
        output_days=2,
        model_type="deterministic",
        target_scaling="raw",
        sigma_data=1.0,
    ):
    config = OSTIAModelConfig(
        input_days=input_days,
        output_days=output_days,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=16,
        num_blocks=num_blocks,
        explicit_layer=1,
        implicit_layer=implicit_layer,
        hidden_size_factor=2,
        sampling_steps=4,
        sigma_data=sigma_data,
        sigma_max=4.0,
        sigma_min=0.002,
        p_mean=-1.2,
        p_std=1.2,
        rho=7.0,
        target_mode="residual",
        model_type=model_type,
        target_scaling=target_scaling,
    )
    config.adopt_condition_mode(mode)
    return config


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
        "sst_mean": 280.0,
        "sst_std": 10.0,
    }


def build_trainer(config, tmp_dir, loader_len=1):
    config.output_dir = tmp_dir
    trainer = OSTIATrainer.__new__(OSTIATrainer)
    trainer.config = config
    trainer.runtime = DistributedRuntime()
    trainer.data = type(
        "FakeData",
        (),
        {
            "sampler": FakeSampler(),
            "loader": FakeLoader([None] * loader_len),
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
    ).TrainingHistory(tmp_dir, config.max_grad_norm)
    trainer._build_training_components()
    return trainer


class ConditionSchemaTableTests(unittest.TestCase):
    def test_mode_channel_counts(self):
        self.assertEqual(condition_chans("sst", 7), 7)
        self.assertEqual(condition_chans("sst_mask", 7), 8)
        self.assertEqual(condition_chans("sst_mask_geo_season", 7), 14)
        self.assertEqual(condition_schema_version_for("sst"), 1)
        self.assertEqual(condition_schema_version_for("sst_mask"), 1)
        self.assertEqual(
            condition_schema_version_for("sst_mask_geo_season"), 2
        )

    def test_fixed_order_and_names(self):
        names = condition_channel_names("sst_mask_geo_season", 7)
        self.assertEqual(
            names,
            (
                "sst_tminus6",
                "sst_tminus5",
                "sst_tminus4",
                "sst_tminus3",
                "sst_tminus2",
                "sst_tminus1",
                "sst_t0",
                VALID_MASK_CHANNEL_NAME,
                *GEO_STATIC_CHANNEL_NAMES,
            ),
        )

    def test_names_scale_with_input_days(self):
        self.assertEqual(
            condition_channel_names("sst_mask", 3),
            ("sst_tminus2", "sst_tminus1", "sst_t0",
             VALID_MASK_CHANNEL_NAME),
        )

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "condition_mode"):
            condition_channel_names("seasonal", 7)


class ModelConditionSchemaTests(unittest.TestCase):
    def test_legacy_default_config_validates(self):
        config = OSTIAModelConfig()
        config.validate_condition_schema()
        self.assertEqual(config.cond_chans, 8)
        self.assertEqual(config.condition_mode, "sst_mask")
        self.assertEqual(config.condition_schema_version, 1)

    def test_adopt_makes_schema_canonical(self):
        config = OSTIAModelConfig()
        config.adopt_condition_mode("sst_mask_geo_season")
        self.assertEqual(config.cond_chans, 14)
        self.assertEqual(config.condition_schema_version, 2)
        self.assertEqual(
            config.condition_channel_names,
            condition_channel_names("sst_mask_geo_season", 7),
        )
        config.validate_condition_schema()
        config.adopt_condition_mode("sst_mask")
        self.assertEqual(config.cond_chans, 8)

    def test_hand_written_cond_chans_mismatch_fails(self):
        config = OSTIAModelConfig(
            condition_mode="sst_mask_geo_season",
            condition_schema_version=2,
            cond_chans=8,  # stale 8-channel value
        )
        with self.assertRaisesRegex(ValueError, "cond_chans=14"):
            config.validate_condition_schema()

    def test_schema_version_mismatch_fails(self):
        config = OSTIAModelConfig(
            condition_mode="sst_mask_geo_season",
            condition_schema_version=1,
            cond_chans=14,
        )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            config.validate_condition_schema()

    def test_hand_edited_channel_names_fail(self):
        config = OSTIAModelConfig(
            condition_mode="sst_mask_geo_season",
            condition_schema_version=2,
            cond_chans=14,
            condition_channel_names=(
                "sst_tminus6",
                *("x",) * 13,
            ),
        )
        with self.assertRaisesRegex(ValueError, "channel_names"):
            config.validate_condition_schema()

    def test_build_model_rejects_8_channel_geo_config(self):
        config = OSTIAModelConfig(
            condition_mode="sst_mask_geo_season",
            cond_chans=8,
        )
        with self.assertRaisesRegex(ValueError, "cond_chans=14"):
            config.build_model("cpu")

    def test_checkpoint_roundtrip_legacy_and_new(self):
        legacy_payload = {
            "input_days": 7,
            "output_days": 15,
            "cond_chans": 8,
            "target_chans": 15,
            "image_size": [448, 448, 1],
            "patch_size": [8, 8, 1],
            "embed_dim": 128,
            "num_blocks": 8,
            "explicit_layer": 4,
            "implicit_layer": 2,
            "hidden_size_factor": 4,
            "target_mode": "absolute",
            "model_type": "diffusion",
            "target_scaling": "raw",
        }
        restored = OSTIAModelConfig.from_checkpoint(legacy_payload)
        self.assertEqual(restored.condition_mode, "sst_mask")
        self.assertEqual(restored.condition_schema_version, 1)
        self.assertIsNone(restored.condition_channel_names)
        self.assertEqual(restored.cond_chans, 8)
        restored.validate_condition_schema()

        new = OSTIAModelConfig()
        new.adopt_condition_mode("sst_mask_geo_season")
        new.calendar_encoding = "standard"
        new.time_units_reference = "days since 2020-01-01"
        new.geospatial_summary = {
            "encoding": "sin_cos_radians",
            "resolved_units": "degrees",
            "lat_shape": [448],
            "lon_shape": [448],
        }
        new.validate_condition_schema()
        payload = new.to_checkpoint()
        restored_new = OSTIAModelConfig.from_checkpoint(payload)
        self.assertEqual(restored_new, new)
        self.assertEqual(
            restored_new.condition_channel_names,
            tuple(payload["condition_channel_names"]),
        )

    def test_resolve_condition_mode_fail_closed(self):
        self.assertEqual(
            resolve_condition_mode(None, "sst_mask_geo_season", "x"),
            "sst_mask_geo_season",
        )
        self.assertEqual(
            resolve_condition_mode("sst_mask", "sst_mask", "x"),
            "sst_mask",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_condition_mode(
                "sst_mask",
                "sst_mask_geo_season",
                "validation",
            )


class FourteenChannelForwardBackwardTests(OSTIATestCase):
    def _assert_finite_training_step(self, config):
        device = torch.device("cpu")
        model = config.build_model(device)
        self.assertTrue(torch.isfinite(
            torch.stack(
                [
                    parameter.detach().float().norm()
                    for parameter in model.parameters()
                ]
            )
        ).all())
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=1e-3,
        )
        condition = torch.randn(2, config.cond_chans, *config.image_size)
        target = torch.randn(
            2, config.target_chans, *config.image_size
        )
        target_mask = torch.ones_like(target)
        loss = model(target, condition, target_mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(
            gradient is not None
            and torch.isfinite(gradient).all()
            for gradient in gradients
        ))
        optimizer.step()
        return model

    def _geo_wide_config(self, model_type, target_scaling,
                         sigma_data=1.0):
        config = OSTIAModelConfig(
            input_days=7,
            output_days=15,
            image_size=(16, 16, 1),
            patch_size=(4, 4, 1),
            embed_dim=16,
            num_blocks=4,
            explicit_layer=2,
            implicit_layer=2,
            hidden_size_factor=2,
            sampling_steps=4,
            sigma_data=sigma_data,
            sigma_max=4.0,
            sigma_min=0.002,
            p_mean=-1.2,
            p_std=1.2,
            rho=7.0,
            target_mode="residual",
            model_type=model_type,
            target_scaling=target_scaling,
            lead_mean=tuple(0.01 * value for value in range(15)),
            lead_std=tuple(1.0 + value for value in range(15)),
            mean_lead_mean=tuple(
                -0.01 * value for value in range(15)
            ),
            mean_lead_std=tuple(1.0 + value for value in range(15)),
            mean_checkpoint_sha256="ab" * 32,
            mean_semantics_sha256="cd" * 32,
        )
        config.adopt_condition_mode("sst_mask_geo_season")
        self.assertEqual(config.cond_chans, 14)
        config.validate_condition_schema()
        return config

    def test_deterministic_14_channel_forward_backward(self):
        self._assert_finite_training_step(self._geo_wide_config(
            "deterministic", "raw"
        ))

    def test_diffusion_14_channel_forward_backward(self):
        config = self._geo_wide_config(
            "diffusion", "raw", sigma_data=0.15
        )
        self._assert_finite_training_step(config)

    def test_centered_diffusion_14_channel_forward_backward(self):
        config = self._geo_wide_config(
            "centered_diffusion",
            "lead_standardized",
            sigma_data=1.0,
        )
        self._assert_finite_training_step(config)

    def test_legacy_8_channel_still_trains(self):
        config = tiny_model_config(mode="sst_mask")
        self.assertEqual(config.cond_chans, 4)
        model = self._assert_finite_training_step(config)
        self.assertIsNotNone(model)


class CheckpointContractTests(OSTIATestCase):
    def _trainer_config(self, mode="sst_mask", **kwargs):
        config = OSTIATrainingConfig()
        config.use_amp = False
        config.condition_mode = mode
        config.model = tiny_model_config(mode=mode, **kwargs)
        return config

    def _save_checkpoint(self, config, name="latest.pth",
                         epoch=3, global_step=250):
        trainer = build_trainer(config, self._tmp)
        manager = trainer.checkpoints
        random_state = manager.capture_random_state()
        path = os.path.join(self._tmp, name)
        manager.save(
            path,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.scaler,
            epoch=epoch,
            global_step=global_step,
            train_loss=0.25,
            dataset=DatasetStub(),
            random_states=[random_state],
            skipped_optimizer_steps=1,
            skipped_optimizer_step_numbers=[17],
        )
        return path

    def test_checkpoint_and_sidecar_persist_condition_schema(self):
        config = self._trainer_config(mode="sst_mask_geo_season")
        config.model.calendar_encoding = "standard"
        config.model.time_units_reference = "days since 2020-01-01"
        config.model.geospatial_summary = {"resolved_units": "degrees"}
        path = self._save_checkpoint(config)
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
        model_payload = checkpoint["config"]
        self.assertEqual(
            model_payload["condition_mode"], "sst_mask_geo_season"
        )
        self.assertEqual(model_payload["condition_schema_version"], 2)
        self.assertEqual(model_payload["cond_chans"], 10)
        self.assertEqual(len(model_payload["condition_channel_names"]),
                         10)
        self.assertEqual(model_payload["calendar_encoding"], "standard")
        self.assertEqual(model_payload["geospatial_summary"],
                         {"resolved_units": "degrees"})
        sidecar = load_semantic_sidecar(path)
        self.assertIsNotNone(sidecar)
        self.assertEqual(sidecar["schema_version"], 5)
        self.assertEqual(
            sidecar["config"]["condition_mode"],
            "sst_mask_geo_season",
        )
        immutable = sidecar["semantic_manifest"]["immutable"]
        self.assertEqual(immutable["cond_chans"], 10)
        self.assertEqual(immutable["condition_mode"],
                         "sst_mask_geo_season")
        self.assertEqual(
            immutable["condition_schema_version"], 2
        )
        self.assertEqual(
            immutable["condition_channel_names"],
            list(condition_channel_names(
                "sst_mask_geo_season", 3
            )),
        )
        self.assertEqual(immutable["calendar_encoding"], "standard")
        self.assertEqual(
            immutable["geospatial_summary"],
            {"resolved_units": "degrees"},
        )

    def test_resume_restores_schema_and_step_continuity(self):
        config = self._trainer_config(mode="sst_mask_geo_season")
        path = self._save_checkpoint(
            config, epoch=3, global_step=250
        )
        resumed = self._trainer_config(mode="sst_mask_geo_season")
        resumed.resume_path = path
        trainer = build_trainer(resumed, self._tmp)
        trainer._restore_resume_semantics()
        trainer._resume_training()
        self.assertEqual(trainer.start_epoch, 3)
        self.assertEqual(trainer.global_step, 250)
        self.assertEqual(trainer.skipped_optimizer_steps, 1)
        self.assertEqual(
            trainer.skipped_optimizer_step_numbers, [17]
        )
        self.assertEqual(
            trainer.config.model.cond_chans, 10
        )

    def test_8_channel_checkpoint_cannot_resume_14_channel(self):
        checkpoint_path = self._save_checkpoint(
            self._trainer_config(mode="sst_mask")
        )
        current = self._trainer_config(mode="sst_mask_geo_season")
        current.resume_path = checkpoint_path
        trainer = build_trainer(current, self._tmp)
        state_dict_calls = []

        def spying_load_state_dict(state_dict, strict=True):
            state_dict_calls.append(state_dict)

        model = trainer.checkpoints.unwrap_model(trainer.model)
        model.load_state_dict = spying_load_state_dict
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            trainer.checkpoints.load(
                checkpoint_path,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.runtime.device,
                trainer.runtime.rank,
                trainer.runtime.world_size,
            )
        # The failure happened before any weights were loaded.
        self.assertEqual(state_dict_calls, [])

    def test_architecture_mismatch_cannot_resume(self):
        checkpoint_path = self._save_checkpoint(
            self._trainer_config(mode="sst_mask_geo_season",
                                 patch_size=(2, 2, 1))
        )
        current = self._trainer_config(
            mode="sst_mask_geo_season",
            patch_size=(4, 4, 1),  # different patch layout
        )
        current.resume_path = checkpoint_path
        trainer = build_trainer(current, self._tmp)
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            trainer.checkpoints.load(
                checkpoint_path,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.runtime.device,
                trainer.runtime.rank,
                trainer.runtime.world_size,
            )

    def test_implicit_layer_mismatch_cannot_resume(self):
        checkpoint_path = self._save_checkpoint(
            self._trainer_config(mode="sst_mask_geo_season",
                                 implicit_layer=1)
        )
        current = self._trainer_config(
            mode="sst_mask_geo_season",
            implicit_layer=2,
        )
        current.resume_path = checkpoint_path
        trainer = build_trainer(current, self._tmp)
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            trainer.checkpoints.load(
                checkpoint_path,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.runtime.device,
                trainer.runtime.rank,
                trainer.runtime.world_size,
            )

    def test_manifest_rejects_8_vs_14_and_patch_mismatch(self):
        saved = self._trainer_config(mode="sst_mask")
        current = self._trainer_config(mode="sst_mask_geo_season")
        checkpoint = {
            "semantic_manifest": build_semantic_manifest(
                saved, world_size=1
            )
        }
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            validate_semantic_manifest(
                checkpoint,
                current,
                world_size=1,
            )
        patch_mismatch = self._trainer_config(
            mode="sst_mask_geo_season", patch_size=(4, 4, 1)
        )
        checkpoint = {
            "semantic_manifest": build_semantic_manifest(
                self._trainer_config(mode="sst_mask_geo_season"),
                world_size=1,
            )
        }
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            validate_semantic_manifest(
                checkpoint,
                patch_mismatch,
                world_size=1,
            )

    def test_bare_resume_conflict_is_fail_closed(self):
        checkpoint_config = self._trainer_config(
            mode="sst_mask_geo_season"
        )
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config, world_size=1
            )
        }
        current = self._trainer_config(mode="sst_mask_geo_season")
        current.model.cond_chans = 8  # hand-typed stale value
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        notices = __import__(
            "deterministic_iafno.checkpoint_semantics",
            fromlist=["restore_resume_semantics"],
        ).restore_resume_semantics(
            sidecar, current, defaults
        )
        self.assertEqual(current.model.cond_chans, 10)
        self.assertTrue(any(
            "restored immutable semantics" in notice
            for notice in notices
        ))


    def test_summary_mismatch_fails_before_state_dict_load(self):
        """Condition-schema provenance is part of the immutable
        semantics: a checkpoint whose geospatial summary differs from
        the current config fails in CheckpointManager.load before any
        weight is loaded."""
        config = self._trainer_config(mode="sst_mask_geo_season")
        config.model.calendar_encoding = "standard"
        config.model.time_units_reference = "days since 2020-01-01"
        config.model.geospatial_summary = {
            "lat_sha256": "aa" * 32,
            "lon_sha256": "bb" * 32,
            "lat_min": -80.0,
        }
        checkpoint_path = self._save_checkpoint(config)
        tampered = self._trainer_config(mode="sst_mask_geo_season")
        tampered.resume_path = checkpoint_path
        tampered.model.calendar_encoding = "standard"
        tampered.model.time_units_reference = "days since 2020-01-01"
        tampered.model.geospatial_summary = {
            # Same shape, provably different coordinates.
            "lat_sha256": "ee" * 32,
            "lon_sha256": "bb" * 32,
            "lat_min": -80.0,
        }
        trainer = build_trainer(tampered, self._tmp)
        state_dict_calls = []

        def spying_load_state_dict(state_dict, strict=True):
            state_dict_calls.append(state_dict)

        model = trainer.checkpoints.unwrap_model(trainer.model)
        model.load_state_dict = spying_load_state_dict
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            trainer.checkpoints.load(
                checkpoint_path,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.scaler,
                trainer.runtime.device,
                trainer.runtime.rank,
                trainer.runtime.world_size,
            )
        self.assertEqual(state_dict_calls, [])


class DataSetupAdoptionTests(OSTIATestCase):
    def _config_with_file(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("adopt.h5"),
            total_days=300,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=5,
        )
        config = OSTIATrainingConfig()
        config.train_h5_path = h5_path
        config.condition_mode = "sst_mask_geo_season"
        config.num_workers = 0
        config.batch_per_gpu = 2
        config.samples_per_epoch = 64
        config.gradient_accumulation = 1
        config.model = tiny_model_config(
            mode="sst_mask_geo_season",
            input_days=7,
            output_days=15,
            image_size=(8, 8, 1),
        )
        return config

    def test_data_setup_adopts_schema_and_provenance(self):
        config = self._config_with_file()
        runtime = DistributedRuntime()
        data = OSTIATrainingData(config, runtime).setup()
        self.assertEqual(data.dataset.condition_chans, 14)
        self.assertEqual(config.model.condition_mode,
                         "sst_mask_geo_season")
        self.assertEqual(config.model.cond_chans, 14)
        self.assertEqual(
            config.model.calendar_encoding, "standard"
        )
        self.assertIsNotNone(config.model.geospatial_summary)
        self.assertEqual(config.model.geospatial_summary[
            "resolved_units"], "degrees")
        # The data contract round-trips into a model config.
        payload = config.model.to_checkpoint()
        restored = OSTIAModelConfig.from_checkpoint(payload)
        restored.validate_condition_schema()
        verify_checkpoint_data_contract(
            data.dataset, restored
        )


class ConfigJsonContractTests(OSTIATestCase):
    def _lead_stats(self, target_chans=15):
        path = os.path.join(self._tmp, "lead_stats.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 1,
                "target_space": "normalized_residual",
                "split": "train",
                "selection": "evenly_spaced_dataset_indices",
                "num_samples": 4096,
                "dataset_size": 786100,
                "input_days": 7,
                "output_days": 15,
                "condition_mode": "sst_mask_geo_season",
                "sst_mean": 280.0,
                "sst_std": 10.0,
                "lead_mean": [0.01 * value
                              for value in range(target_chans)],
                "lead_std": [1.0 + value
                             for value in range(target_chans)],
            }, file)
        return path

    def test_a1_json_expands_geo_14_channel_model(self):
        stats = self._lead_stats()
        args = build_parser().parse_args(
            [
                "--config",
                os.path.join(
                    os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    "configs",
                    "ostia_ablation_A1_geo_p8_b8_i2.json",
                ),
                "--lead-stats",
                stats,
            ]
        )
        merge_config_json(args, args.config)
        config = training_config_from_args(args)
        self.assertEqual(
            config.condition_mode, "sst_mask_geo_season"
        )
        self.assertEqual(config.model.condition_mode,
                         "sst_mask_geo_season")
        self.assertEqual(config.model.cond_chans, 14)
        self.assertEqual(config.model.patch_size, (8, 8, 1))
        self.assertEqual(config.model.num_blocks, 8)
        self.assertEqual(config.model.implicit_layer, 2)
        config.model.validate_condition_schema()
        self.assertIn("patch_size", config.explicit_resume_fields)

    def test_a5_json_expands_implicit4(self):
        stats = self._lead_stats()
        args = build_parser().parse_args(
            [
                "--config",
                os.path.join(
                    os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    "configs",
                    "ostia_ablation_A5_geo_p4_best_i4.json",
                ),
                "--lead-stats",
                stats,
            ]
        )
        merge_config_json(args, args.config)
        config = training_config_from_args(args)
        self.assertEqual(config.model.patch_size, (4, 4, 1))
        self.assertEqual(config.model.num_blocks, 2)
        self.assertEqual(config.model.implicit_layer, 4)
        config.model.validate_condition_schema()

    def test_malformed_patch_size_rejected(self):
        args = build_parser().parse_args([])
        args.patch_size = [4, 4]
        with self.assertRaisesRegex(ValueError, "3-element"):
            training_config_from_args(args)

    def test_nonpositive_blocks_rejected(self):
        args = build_parser().parse_args([])
        args.num_blocks = 0
        with self.assertRaisesRegex(ValueError, "num_blocks"):
            training_config_from_args(args)

    def test_cli_architecture_flags_exposed(self):
        args = build_parser().parse_args(
            [
                "--patch-size", "4", "4", "1",
                "--num-blocks", "2",
                "--implicit-layer", "4",
            ]
        )
        config = training_config_from_args(args)
        self.assertEqual(config.model.patch_size, (4, 4, 1))
        self.assertEqual(config.model.num_blocks, 2)
        self.assertEqual(config.model.implicit_layer, 4)
        config.model.validate_condition_schema()
        self.assertIn("patch_size", config.explicit_resume_fields)
        self.assertIn("num_blocks", config.explicit_resume_fields)
        self.assertIn("implicit_layer",
                      config.explicit_resume_fields)
        # Defaults untouched without flags.
        bare = training_config_from_args(
            build_parser().parse_args([])
        )
        self.assertEqual(bare.model.patch_size, (8, 8, 1))
        self.assertEqual(bare.model.num_blocks, 8)
        self.assertEqual(bare.model.implicit_layer, 2)
        self.assertFalse(
            {"patch_size", "num_blocks", "implicit_layer"} &
            set(bare.explicit_resume_fields)
        )

    def test_cli_arch_flags_win_over_config_json(self):
        stats = self._lead_stats()
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
            "configs",
            "ostia_ablation_A2_geo_p4_b8_i2.json",
        )
        args = build_parser().parse_args(
            [
                "--config", config_path,
                "--lead-stats", stats,
                "--num-blocks", "2",
            ]
        )
        notes = merge_config_json(args, args.config)
        config = training_config_from_args(args)
        self.assertEqual(config.model.num_blocks, 2)
        self.assertTrue(any(
            "num_blocks" in note and "overrides" in note
            for note in notes
        ))


class ValidationInferenceContractTests(OSTIATestCase):
    def setUp(self):
        super().setUp()
        self.h5_path = make_synthetic_h5(
            self.tmp_path("geo.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
        )
        dataset = __import__(
            "diafno.data.ostia", fromlist=["OSTIADailyDataset"]
        ).OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        model_config = tiny_model_config(
            mode="sst_mask_geo_season",
            image_size=(8, 10, 1),
            patch_size=(2, 2, 1),
            input_days=7,
            output_days=15,
        )
        copy_dataset_provenance(model_config, dataset)
        model_config.validate_condition_schema()
        model = model_config.build_model("cpu")
        self.checkpoint_path = os.path.join(
            self._tmp, "model.pth"
        )
        torch.save(
            {
                "config": model_config.to_checkpoint(),
                "model": model.state_dict(),
                "normalization": {
                    "sst_mean": 280.0,
                    "sst_std": 10.0,
                },
            },
            self.checkpoint_path,
        )

    def test_loader_restores_geo_contract_from_checkpoint(self):
        model, model_config, steps, normalization = (
            InferenceModelLoader.load(
                self.checkpoint_path,
                torch.device("cpu"),
            )
        )
        self.assertEqual(
            model_config.condition_mode, "sst_mask_geo_season"
        )
        self.assertEqual(model_config.cond_chans, 14)
        self.assertIsNotNone(model_config.geospatial_summary)
        dataset = __import__(
            "diafno.data.ostia", fromlist=["OSTIADailyDataset"]
        ).OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            input_days=7,
            output_days=15,
            condition_mode=model_config.condition_mode,
        )
        verify_checkpoint_data_contract(dataset, model_config)

    def test_geo_contract_mismatch_fails_closed(self):
        model_config = OSTIAModelConfig.from_checkpoint(
            torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )["config"]
        )
        model_config.geospatial_summary = {
            "resolved_units": "radians",  # provably different file
        }
        dataset = __import__(
            "diafno.data.ostia", fromlist=["OSTIADailyDataset"]
        ).OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        with self.assertRaisesRegex(
                ValueError, "does not match the current HDF5"
            ):
            verify_checkpoint_data_contract(dataset, model_config)
        # A checkpoint without the geo provenance also fails closed.
        model_config = OSTIAModelConfig.from_checkpoint(
            torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )["config"]
        )
        model_config.geospatial_summary = None
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_checkpoint_data_contract(dataset, model_config)

    def test_legacy_checkpoint_on_geo_file_contract_is_noop(self):
        dataset = __import__(
            "diafno.data.ostia", fromlist=["OSTIADailyDataset"]
        ).OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            condition_mode="sst_mask",
        )
        legacy = OSTIAModelConfig()
        legacy.validate_condition_schema()
        verify_checkpoint_data_contract(dataset, legacy)


class EndToEndGeoSeasonTrainerSmokeTests(OSTIATestCase):
    """Full trainer path on a tiny synthetic geo-season HDF5 (CPU):
    data setup adopts the schema, the loader builds 14-channel
    conditions and one epoch trains + checkpoints."""

    def test_trainer_epoch_end_to_end_on_geo_h5(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("e2e.h5"),
            total_days=300,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=5,
        )
        config = OSTIATrainingConfig()
        config.train_h5_path = h5_path
        config.output_dir = self.tmp_path("run")
        config.condition_mode = "sst_mask_geo_season"
        config.num_workers = 0
        config.batch_per_gpu = 4
        config.gradient_accumulation = 1
        config.samples_per_epoch = 32
        config.num_epochs = 1
        config.checkpoint_interval = 1
        config.use_amp = False
        config.model = OSTIAModelConfig(
            input_days=7,
            output_days=15,
            image_size=(8, 10, 1),
            patch_size=(2, 2, 1),
            embed_dim=16,
            num_blocks=4,
            explicit_layer=1,
            implicit_layer=1,
            hidden_size_factor=2,
            sampling_steps=4,
            sigma_data=1.0,
            sigma_max=4.0,
            sigma_min=0.002,
            p_mean=-1.2,
            p_std=1.2,
            rho=7.0,
            target_mode="residual",
            model_type="deterministic",
            target_scaling="raw",
        )
        config.model.adopt_condition_mode(
            "sst_mask_geo_season"
        )
        trainer = OSTIATrainer(config)
        try:
            trainer.train()
        finally:
            trainer.runtime.cleanup()
        self.assertEqual(trainer.global_step, 8)
        self.assertEqual(
            trainer.skipped_optimizer_steps, 0
        )
        checkpoint_path = os.path.join(
            config.output_dir, "latest.pth"
        )
        self.assertTrue(os.path.isfile(checkpoint_path))
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            checkpoint["config"]["condition_mode"],
            "sst_mask_geo_season",
        )
        self.assertEqual(checkpoint["config"]["cond_chans"], 14)
        self.assertIsNotNone(
            checkpoint["config"]["geospatial_summary"]
        )
        self.assertIsNotNone(
            checkpoint["config"]["time_axis_summary"]
        )
        self.assertIsNotNone(
            checkpoint["config"]["calendar_encoding"]
        )
        self.assertTrue(os.path.isfile(
            checkpoint_path + ".semantics.json"
        ))
        history_path = os.path.join(
            config.output_dir, "training_curves.npz"
        )
        self.assertTrue(os.path.isfile(history_path))
        with np.load(history_path) as history:
            self.assertEqual(
                int(history["loss_steps"][-1]),
                trainer.global_step,
            )

    def test_stage1_style_resume_epoch_budget_continuity(self):
        """Resume horizon in the exact stage-1 shape (scaled down).

        Stage 1 runs 5 epochs x 10 optimizer steps and resumes with
        num_epochs=6 and the same 10 steps/epoch, so the restored
        scheduler (last epoch = 50) keeps exactly 10 more steps
        (T_max 60).  Here the same arithmetic is exercised on CPU with
        2 steps/epoch: 5 epochs -> global_step 10, resume num_epochs=6
        -> exactly one more epoch -> global_step 12.
        """
        h5_path = make_synthetic_h5(
            self.tmp_path("budget.h5"),
            total_days=300,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=5,
        )

        def make_config(output_dir, num_epochs, resume_path=None):
            config = OSTIATrainingConfig()
            config.train_h5_path = h5_path
            config.output_dir = output_dir
            config.condition_mode = "sst_mask_geo_season"
            config.num_workers = 0
            config.batch_per_gpu = 4
            config.gradient_accumulation = 1
            config.samples_per_epoch = 8  # 2 optimizer steps/epoch
            config.num_epochs = num_epochs
            config.checkpoint_interval = 1
            config.use_amp = False
            config.allow_resume_override = (
                resume_path is not None
            )
            config.resume_path = resume_path
            config.model = OSTIAModelConfig(
                input_days=7,
                output_days=15,
                image_size=(8, 10, 1),
                patch_size=(2, 2, 1),
                embed_dim=16,
                num_blocks=4,
                explicit_layer=1,
                implicit_layer=1,
                hidden_size_factor=2,
                sampling_steps=4,
                sigma_data=1.0,
                sigma_max=4.0,
                sigma_min=0.002,
                p_mean=-1.2,
                p_std=1.2,
                rho=7.0,
                target_mode="residual",
                model_type="deterministic",
                target_scaling="raw",
            )
            config.model.adopt_condition_mode(
                "sst_mask_geo_season"
            )
            return config

        run1_dir = self.tmp_path("run1")
        first = make_config(run1_dir, num_epochs=5)
        trainer1 = OSTIATrainer(first)
        try:
            trainer1.train()
        finally:
            trainer1.runtime.cleanup()
        # 5 epochs x 2 steps = 10 optimizer steps (50 in stage-1 terms).
        self.assertEqual(trainer1.global_step, 10)
        checkpoint1 = os.path.join(run1_dir, "latest.pth")
        saved1 = torch.load(
            checkpoint1, map_location="cpu", weights_only=False
        )
        self.assertEqual(saved1["epoch"], 5)
        self.assertEqual(saved1["global_step"], 10)

        resume_dir = self.tmp_path("resume2")
        resumed = make_config(
            resume_dir,
            num_epochs=6,  # start_epoch 5 -> exactly one more epoch
            resume_path=checkpoint1,
        )
        trainer2 = OSTIATrainer(resumed)
        try:
            trainer2.train()
        finally:
            trainer2.runtime.cleanup()
        # The resume leg really ran one epoch and continued the global
        # step counter (stage-1 terms: 50 -> 60).
        self.assertEqual(trainer2.start_epoch, 5)
        self.assertEqual(trainer2.global_step, 12)
        saved2 = torch.load(
            os.path.join(resume_dir, "latest.pth"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(saved2["epoch"], 6)
        self.assertEqual(saved2["global_step"], 12)
        self.assertEqual(
            saved2["config"]["condition_mode"],
            "sst_mask_geo_season",
        )


class ManifestVersionAndCompatibilityTests(OSTIATestCase):
    """Schema-version 5 manifest: the condition-schema fields are
    immutable semantics; v4 sidecars keep validating on the fields
    they actually store."""

    def _geo_config(self):
        config = OSTIATrainingConfig()
        config.condition_mode = "sst_mask_geo_season"
        model = tiny_model_config(mode="sst_mask_geo_season")
        model.calendar_encoding = "standard"
        model.time_units_reference = "days since 2020-01-01"
        model.geospatial_summary = {
            "encoding": "sin_cos_radians",
            "resolved_units": "degrees",
            "lat_shape": [8],
            "lon_shape": [10],
            "nonfinite": 0,
            "layout": "full_grid_1d_row_aligned",
            "lat_min": -80.0,
            "lat_max": 82.0,
            "lon_min": -177.0,
            "lon_max": 177.0,
            "lat_sha256": "ab" * 32,
            "lon_sha256": "cd" * 32,
            "digest_spec": "sha256_le_f8_raw_order",
        }
        config.model = model
        return config

    def test_current_manifest_version_is_five(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 5)
        manifest = build_semantic_manifest(
            self._geo_config(), world_size=1
        )
        self.assertEqual(manifest["schema_version"], 5)

    def test_manifest_immutable_carries_condition_schema(self):
        manifest = build_semantic_manifest(
            self._geo_config(), world_size=1
        )
        immutable = manifest["immutable"]
        for field in (
                "condition_mode",
                "condition_schema_version",
                "condition_channel_names",
                "calendar_encoding",
                "time_units_reference",
                "geospatial_summary",
                "time_axis_summary",
                "data_manifest_sha256",
            ):
            self.assertIn(field, immutable)
        self.assertEqual(immutable["condition_mode"],
                         "sst_mask_geo_season")
        self.assertEqual(immutable["condition_schema_version"], 2)
        self.assertEqual(
            immutable["condition_channel_names"],
            list(condition_channel_names(
                "sst_mask_geo_season", 3
            )),
        )
        self.assertEqual(immutable["calendar_encoding"], "standard")
        self.assertEqual(
            immutable["geospatial_summary"]["lat_sha256"],
            "ab" * 32,
        )

    def test_v4_sidecar_without_new_fields_still_validates(self):
        config = OSTIATrainingConfig()
        config.model = tiny_model_config(mode="sst_mask")
        manifest = build_semantic_manifest(config, world_size=1)
        # Emulate a schema-4 checkpoint written by the old code: no
        # condition-schema fields beyond the old ones.
        old = json.loads(json.dumps(manifest))
        old["schema_version"] = 4
        for field in (
                "condition_schema_version",
                "condition_channel_names",
                "calendar_encoding",
                "time_units_reference",
                "geospatial_summary",
            ):
            old["immutable"].pop(field, None)
        warnings = validate_semantic_manifest(
            {"semantic_manifest": old},
            config,
            world_size=1,
        )
        self.assertEqual(warnings, [])

    def test_v4_sidecar_still_rejects_channel_mismatch(self):
        legacy = OSTIATrainingConfig()
        legacy.model = tiny_model_config(mode="sst_mask")
        manifest = build_semantic_manifest(legacy, world_size=1)
        old = json.loads(json.dumps(manifest))
        old["schema_version"] = 4
        for field in (
                "condition_schema_version",
                "condition_channel_names",
                "calendar_encoding",
                "time_units_reference",
                "geospatial_summary",
            ):
            old["immutable"].pop(field, None)
        current = self._geo_config()
        with self.assertRaisesRegex(ValueError, "immutable semantic"):
            validate_semantic_manifest(
                {"semantic_manifest": old},
                current,
                world_size=1,
            )

    def test_restore_converts_condition_names_to_tuple(self):
        config = self._geo_config()
        manifest = build_semantic_manifest(config, world_size=1)
        sidecar = {"semantic_manifest": json.loads(
            json.dumps(manifest))}
        current = OSTIATrainingConfig()
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        notices = restore_resume_semantics(
            sidecar, current, defaults
        )
        names = current.model.condition_channel_names
        self.assertIsInstance(names, tuple)
        self.assertEqual(
            names,
            condition_channel_names("sst_mask_geo_season", 3),
        )
        self.assertEqual(
            current.model.condition_schema_version, 2
        )
        self.assertEqual(current.model.cond_chans, 10)
        self.assertEqual(current.model.calendar_encoding,
                         "standard")
        self.assertEqual(current.model.time_units_reference,
                         "days since 2020-01-01")
        self.assertEqual(current.condition_mode,
                         "sst_mask_geo_season")
        self.assertTrue(any(
            "restored immutable semantics" in item
            for item in notices
        ))

    def test_explicit_conflict_for_new_fields_fails_closed(self):
        config = self._geo_config()
        manifest = build_semantic_manifest(config, world_size=1)
        sidecar = {"semantic_manifest": json.loads(
            json.dumps(manifest))}
        current = OSTIATrainingConfig()
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        with self.assertRaisesRegex(ValueError,
                                    "explicitly set condition"):
            restore_resume_semantics(
                sidecar,
                current,
                defaults,
                explicit_fields={"condition_channel_names"},
            )


class DataSetupProvenanceCompareTests(OSTIATestCase):
    """data.setup must compare restored checkpoint provenance with the
    current HDF5 instead of silently overwriting it (resume) and only
    write provenance on fresh runs."""

    def _base_config(self, h5_path, output_dir):
        config = OSTIATrainingConfig()
        config.train_h5_path = h5_path
        config.output_dir = output_dir
        config.condition_mode = "sst_mask_geo_season"
        config.num_workers = 0
        config.batch_per_gpu = 2
        config.samples_per_epoch = 64
        config.gradient_accumulation = 1
        config.model = tiny_model_config(
            mode="sst_mask_geo_season",
            input_days=7,
            output_days=15,
            image_size=(8, 10, 1),
        )
        return config

    def _files(self, name_a="a.h5", name_b="b.h5"):
        a = make_synthetic_h5(
            self.tmp_path(name_a),
            total_days=300,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=5,
        )
        b = make_synthetic_h5(
            self.tmp_path(name_b),
            total_days=300,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=5,
            # Same shape, provably different coordinates.
            lat=np.linspace(-80.5, 82.5, 8),
        )
        return a, b

    def test_resume_with_rotated_file_fails_closed(self):
        file_a, file_b = self._files()
        dataset_a = OSTIADailyDataset(
            h5_path=file_a,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        config = self._base_config(file_b, self.tmp_path("out"))
        config.resume_path = "latest"
        copy_dataset_provenance(config.model, dataset_a)
        runtime = DistributedRuntime()
        with self.assertRaisesRegex(
                ValueError, "checkpoint provenance does not match"
            ):
            OSTIATrainingData(config, runtime).setup()
        # Nothing was overwritten by the failed attempt.
        self.assertEqual(
            config.model.geospatial_summary,
            dataset_a.geospatial_summary,
        )

    def test_resume_with_matching_file_passes(self):
        file_a, _ = self._files()
        dataset_a = OSTIADailyDataset(
            h5_path=file_a,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        config = self._base_config(file_a, self.tmp_path("out"))
        config.resume_path = "latest"
        copy_dataset_provenance(config.model, dataset_a)
        runtime = DistributedRuntime()
        data = OSTIATrainingData(config, runtime).setup()
        self.assertEqual(
            config.model.geospatial_summary,
            dataset_a.geospatial_summary,
        )
        self.assertIsNotNone(data.dataset)

    def test_resume_missing_provenance_fails_closed(self):
        file_a, _ = self._files()
        config = self._base_config(file_a, self.tmp_path("out"))
        config.resume_path = "latest"
        # A geo checkpoint with nothing recorded cannot prove the
        # contract; resuming must fail instead of guessing.
        copy_dataset_provenance(config.model, OSTIADailyDataset(
            h5_path=file_a,
            split="val",
            condition_mode="sst_mask_geo_season",
        ))
        config.model.calendar_encoding = None
        config.model.time_units_reference = None
        config.model.geospatial_summary = None
        config.model.time_axis_summary = None
        config.model.data_manifest_sha256 = None
        runtime = DistributedRuntime()
        with self.assertRaisesRegex(
                ValueError, "no recorded provenance"
            ):
            OSTIATrainingData(config, runtime).setup()

    def test_fresh_run_writes_current_provenance(self):
        file_a, _ = self._files()
        config = self._base_config(file_a, self.tmp_path("out"))
        config.resume_path = None
        runtime = DistributedRuntime()
        data = OSTIATrainingData(config, runtime).setup()
        self.assertEqual(
            config.model.geospatial_summary,
            data.dataset.geospatial_summary,
        )
        self.assertEqual(
            config.model.calendar_encoding,
            data.dataset.calendar_encoding,
        )


class GeospatialCoordinateContractTests(OSTIATestCase):
    """Coordinate digests pin lat/lon values, not only units/shapes."""

    def _write_checkpoint(self, h5_path):
        dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        model_config = tiny_model_config(
            mode="sst_mask_geo_season",
            image_size=(*dataset.image_shape, 1),
            patch_size=(2, 2, 1),
            input_days=7,
            output_days=15,
        )
        copy_dataset_provenance(model_config, dataset)
        model = model_config.build_model("cpu")
        path = os.path.join(self._tmp, "geo_model.pth")
        torch.save(
            {
                "config": model_config.to_checkpoint(),
                "model": model.state_dict(),
                "normalization": {
                    "sst_mean": 280.0,
                    "sst_std": 10.0,
                },
            },
            path,
        )
        return path

    def test_digest_changes_with_coordinates(self):
        file_a = make_synthetic_h5(
            self.tmp_path("a.h5"), total_days=240, height=8,
            width=10, first_time=30,
        )
        file_b = make_synthetic_h5(
            self.tmp_path("b.h5"), total_days=240, height=8,
            width=10, first_time=30,
            lat=np.linspace(-79.0, 83.0, 8),
        )
        dataset_a = OSTIADailyDataset(
            h5_path=file_a, split="val",
            condition_mode="sst_mask_geo_season",
        )
        dataset_b = OSTIADailyDataset(
            h5_path=file_b, split="val",
            condition_mode="sst_mask_geo_season",
        )
        summary_a = dataset_a.geospatial_summary
        summary_b = dataset_b.geospatial_summary
        self.assertEqual(
            summary_a["lat_shape"], summary_b["lat_shape"]
        )
        self.assertEqual(
            summary_a["resolved_units"],
            summary_b["resolved_units"],
        )
        self.assertNotEqual(
            summary_a["lat_sha256"], summary_b["lat_sha256"]
        )
        self.assertNotEqual(summary_a["lat_min"], summary_b["lat_min"])
        # The longitude axis is untouched: its digest stays identical.
        self.assertEqual(
            summary_a["lon_sha256"], summary_b["lon_sha256"]
        )
        # Digests are exact and deterministic across instances.
        again = OSTIADailyDataset(
            h5_path=file_a, split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            again.geospatial_summary["lat_sha256"],
            summary_a["lat_sha256"],
        )

    def test_coordinate_change_fails_validation_contract(self):
        file_a = make_synthetic_h5(
            self.tmp_path("a.h5"), total_days=240, height=8,
            width=10, first_time=30,
        )
        checkpoint = self._write_checkpoint(file_a)
        model_config = OSTIAModelConfig.from_checkpoint(
            torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )["config"]
        )
        # Same shape, same file attrs, different coordinates: the
        # checkpoint digest must not match the current HDF5.
        file_b = make_synthetic_h5(
            self.tmp_path("b.h5"), total_days=240, height=8,
            width=10, first_time=30,
            lat=np.linspace(-79.0, 83.0, 8),
        )
        dataset_b = OSTIADailyDataset(
            h5_path=file_b, split="val",
            condition_mode="sst_mask_geo_season",
        )
        with self.assertRaisesRegex(
                ValueError, "does not match the current HDF5"
            ):
            verify_checkpoint_data_contract(dataset_b, model_config)
        # And the identical file still passes with the same checkpoint.
        dataset_a = OSTIADailyDataset(
            h5_path=file_a, split="val",
            condition_mode="sst_mask_geo_season",
        )
        verify_checkpoint_data_contract(dataset_a, model_config)


if __name__ == "__main__":
    unittest.main()
