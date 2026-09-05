# 用途：验证续训时配置和训练状态的恢复。
import json
import os
import tempfile
import unittest
from copy import deepcopy

from diafno.training.config import (
    OSTIATrainingConfig,
    build_parser,
    default_training_model,
    training_config_from_args,
)
from deterministic_iafno.checkpoint_semantics import (
    build_semantic_manifest,
    restore_resume_semantics,
    validate_semantic_manifest,
)
from dataclasses import asdict


# 用途：构造测试默认参数集的辅助函数。
# 参数：见签名；输出 默认参数 dict。
def build_defaults():
    defaults = dict(asdict(default_training_model()))
    defaults["split"] = "train"
    defaults["condition_mode"] = "sst_mask"
    return defaults


class ResumeRestoreTests(unittest.TestCase):
    # 用途：验证裸 resume 恢复全部不可变语义。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_bare_resume_restores_immutable_semantics(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.sigma_max = 1.0
        checkpoint_config.model.sigma_min = 0.0005
        checkpoint_config.model.p_mean = -3.0
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=2,
            )
        }
        current = OSTIATrainingConfig()
        notices = restore_resume_semantics(
            sidecar,
            current,
            build_defaults(),
        )
        self.assertEqual(current.model.sigma_max, 1.0)
        self.assertEqual(current.model.sigma_min, 0.0005)
        self.assertEqual(current.model.p_mean, -3.0)
        self.assertTrue(
            any("restored immutable semantics" in notice
                for notice in notices)
        )

    # 用途：验证显式语义冲突时 fail-closed。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_explicit_conflict_fails_closed(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.p_mean = -3.0
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=2,
            )
        }
        current = OSTIATrainingConfig()
        current.model.p_mean = -2.3
        with self.assertRaisesRegex(
                ValueError,
                "immutable semantic conflict",
            ):
            restore_resume_semantics(
                sidecar,
                current,
                build_defaults(),
            )

    # 用途：验证等于默认值的显式 CLI 参数不被当作未提供。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_explicit_default_value_is_not_mistaken_for_bare_cli(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.p_mean = -3.0
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=1,
            )
        }
        current = OSTIATrainingConfig()
        self.assertEqual(current.model.p_mean, -1.2)
        with self.assertRaisesRegex(ValueError, "explicitly set p_mean"):
            restore_resume_semantics(
                sidecar,
                current,
                build_defaults(),
                explicit_fields={"p_mean"},
            )

    # 用途：验证显式采样参数在警告下优先生效。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_explicit_sampler_default_wins_with_warning(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.sigma_max = 1.0
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=1,
            )
        }
        current = OSTIATrainingConfig()
        notices = restore_resume_semantics(
            sidecar,
            current,
            build_defaults(),
            explicit_fields={"sigma_max"},
        )
        self.assertEqual(current.model.sigma_max, 80)
        self.assertTrue(any("explicit CLI" in item for item in notices))

    # 用途：验证采样参数冲突只警告不报错。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_sampler_conflict_is_warning_not_error(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.sampling_steps = 16
        checkpoint_config.model.sigma_max = 1.0
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=2,
            )
        }
        current = OSTIATrainingConfig()
        current.model.sampling_steps = 32
        notices = restore_resume_semantics(
            sidecar,
            current,
            build_defaults(),
        )
        self.assertEqual(current.model.sampling_steps, 32)
        self.assertTrue(
            any("sampler profile" in notice for notice in notices)
        )
        self.assertTrue(
            any("sigma_max" in notice for notice in notices)
        )
        self.assertEqual(current.model.sigma_max, 1.0)

    # 用途：验证裸 resume 恢复 lead stats 语义。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_lead_stats_restored_on_bare_resume(self):
        checkpoint_config = OSTIATrainingConfig()
        checkpoint_config.model.model_type = "deterministic"
        checkpoint_config.model.target_scaling = "lead_standardized"
        checkpoint_config.model.lead_mean = tuple(
            float(value) for value in range(15)
        )
        checkpoint_config.model.lead_std = tuple(
            1.0 + value for value in range(15)
        )
        sidecar = {
            "semantic_manifest": build_semantic_manifest(
                checkpoint_config,
                world_size=1,
            )
        }
        current = OSTIATrainingConfig()
        restore_resume_semantics(
            sidecar,
            current,
            build_defaults(),
        )
        self.assertEqual(current.model.model_type, "deterministic")
        self.assertEqual(
            current.model.target_scaling,
            "lead_standardized",
        )
        self.assertEqual(len(current.model.lead_mean), 15)
        self.assertEqual(len(current.model.lead_std), 15)

    # 用途：验证 epoch 预算不一致需审核覆盖。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_num_epochs_mismatch_requires_reviewed_override(self):
        checkpoint = {
            "semantic_manifest": build_semantic_manifest(
                OSTIATrainingConfig(),
                world_size=2,
            )
        }
        changed = OSTIATrainingConfig()
        changed.num_epochs = 40
        with self.assertRaisesRegex(
                ValueError,
                "training compatibility mismatch",
            ):
            validate_semantic_manifest(
                checkpoint,
                changed,
                world_size=2,
            )
        warnings = validate_semantic_manifest(
            checkpoint,
            changed,
            world_size=2,
            allow_compatible_override=True,
        )
        self.assertTrue(
            any("explicitly accepted" in warning
                for warning in warnings)
        )

    # 用途：测试辅助函数。
    # 参数：见函数签名（测试夹具辅助）；输出 测试用构造对象。
    def test_lr_mismatch_requires_override(self):
        checkpoint = {
            "semantic_manifest": build_semantic_manifest(
                OSTIATrainingConfig(),
                world_size=2,
            )
        }
        changed = OSTIATrainingConfig()
        changed.learning_rate = 1e-3
        with self.assertRaisesRegex(
                ValueError,
                "training compatibility mismatch",
            ):
            validate_semantic_manifest(
                checkpoint,
                changed,
                world_size=2,
            )

    # 用途：验证 CLI lead stats 路径的保存-恢复往返。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_cli_lead_stats_path_roundtrip(self):
        payload = {
            "schema_version": 1,
            "target_space": "normalized_residual",
            "split": "train",
            "selection": "evenly_spaced_dataset_indices",
            "num_samples": 4096,
            "dataset_size": 786100,
            "input_days": 7,
            "output_days": 15,
            "sst_mean": 290.7488927184541,
            "sst_std": 9.57073350168232,
            "lead_mean": [float(value) for value in range(15)],
            "lead_std": [1.0 + value for value in range(15)],
        }
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(
            tests_dir,
            ".tmp_lead_stats.json",
        )
        with open(stats_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        try:
            args = build_parser().parse_args([
                "--lead-stats", stats_path,
                "--model-type", "deterministic",
                "--target-scaling", "lead_standardized",
            ])
            config = training_config_from_args(args)
            self.assertEqual(config.model.model_type, "deterministic")
            self.assertEqual(len(config.model.lead_mean), 15)
            self.assertEqual(len(config.model.lead_std), 15)
            self.assertEqual(config.model.lead_std[0], 1.0)
        finally:
            if os.path.isfile(stats_path):
                os.remove(stats_path)

    # 用途：验证 CLI 的 lead_standardized 语义必须提供统计量。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_cli_lead_standardized_requires_stats(self):
        args = build_parser().parse_args([
            "--model-type", "deterministic",
            "--target-scaling", "lead_standardized",
        ])
        with self.assertRaisesRegex(ValueError, "--lead-stats"):
            training_config_from_args(args)

    # 用途：验证 CLI 记录显式给出的 resume 覆盖字段。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_cli_records_explicit_resume_fields(self):
        bare = training_config_from_args(
            build_parser().parse_args(["--resume"])
        )
        self.assertEqual(bare.explicit_resume_fields, ())

        explicit = training_config_from_args(
            build_parser().parse_args([
                "--resume",
                "--p-mean", "-1.2",
                "--sigma-max", "80",
            ])
        )
        self.assertEqual(
            set(explicit.explicit_resume_fields),
            {"p_mean", "sigma_max"},
        )


if __name__ == "__main__":
    unittest.main()
