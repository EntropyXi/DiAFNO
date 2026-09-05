# 用途：验证checkpoint 不可变字段与兼容性约束。
import unittest
from copy import deepcopy

from diafno.training.config import OSTIATrainingConfig
from deterministic_iafno.checkpoint_semantics import (
    build_semantic_manifest,
    validate_semantic_manifest,
)


class CheckpointSemanticTests(unittest.TestCase):
    # 用途：每个测试前的夹具准备。
    # 参数：无输入；输出 无。
    def setUp(self):
        self.config = OSTIATrainingConfig()

    # 用途：构造测试用 checkpoint 载荷的辅助函数。
    # 参数：见签名；输出 checkpoint dict。
    def checkpoint(self, config=None, world_size=2):
        config = self.config if config is None else config
        return {
            "semantic_manifest": build_semantic_manifest(
                config,
                world_size=world_size,
            )
        }

    # 用途：验证匹配的语义清单通过校验。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_matching_manifest_passes(self):
        warnings = validate_semantic_manifest(
            self.checkpoint(),
            self.config,
            world_size=2,
        )
        self.assertEqual(warnings, [])

    # 用途：验证训练噪声语义不一致时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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

    # 用途：验证采样参数不一致时仅警告。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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

    # 用途：验证有效 batch 不一致必须显式覆盖。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
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
