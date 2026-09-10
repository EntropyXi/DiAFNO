# 用途：A5-centered（14 通道 geo-season + v2 统计协议）本地 E2E 测试。
"""Local end-to-end tests for the A5-centered DiAFNO protocol.

Covers the plan's local checks that can run without the real server
artifacts: a synthetic geo-season HDF5 + data manifest, a tiny
deterministic geo-season "A5-like" mean checkpoint (built through the
real CheckpointManager so its semantic sidecar proves the geo-season
contract), a v2 centered-stats payload validated against it, the
per-rank fresh-run validation, and the centered wrapper behaviour:
frozen mean strict load + zero gradients + unchanged weights, diffusion
parameters actually training, finite sampling in residual space and the
zero-innovation reconstruction ``inverse_innovation(-m/s) == 0``.

The v2 frozen-mean identity constant is patched to the synthetic mean
file SHA (exactly like the v1 tests patch the legacy lock), because the
real A5 checkpoint only exists on the server.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from deterministic_iafno import centered_stats as centered_stats_module
from deterministic_iafno.checkpoint_semantics import (
    load_semantic_sidecar,
)
from deterministic_iafno.centered_stats import (
    CENTERED_TARGET_SPACE,
    V2_SIDECAR_PRESENCE_FIELDS,
    sha256_hex_file,
    sha256_of_normalized,
    validate_centered_fresh_inputs,
    validate_centered_stats_payload,
)
from diafno.data.ostia import (
    OSTIADailyDataset,
    copy_dataset_provenance,
)
from diafno.models.config import OSTIAModelConfig
from diafno.training.artifacts import CheckpointManager
from diafno.training.config import OSTIATrainingConfig
from tests.ostia_test_h5 import (
    make_synthetic_h5,
    write_synthetic_data_manifest,
)


class DatasetStub:
    normalization = {
        "sst_mean": 290.7488927184541,
        "sst_std": 9.57073350168232,
    }


class A5CenteredLocalTests(unittest.TestCase):
    """Synthetic geo-season A5-mean + v2 stats + centered wrapper."""

    # 用途：夹具准备：合成 geo HDF5 + manifest + 微缩 A5 型确定性均值 checkpoint。
    # 参数：无输入；输出 无。
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.tmp_dir, "geo.h5")
        make_synthetic_h5(
            self.h5_path,
            total_days=140,
            samples_per_day=5,
            height=16,
            width=16,
            coordinate_layout="per_row",
            with_time_metadata=True,
        )
        self.manifest_path = os.path.join(self.tmp_dir, "manifest.json")
        write_synthetic_data_manifest(self.manifest_path, self.h5_path)
        self.dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask_geo_season",
            data_manifest=self.manifest_path,
        )
        self.mean_path = os.path.join(self.tmp_dir, "mean.pth")
        self.stats_path = os.path.join(self.tmp_dir, "stats.json")
        self._write_mean_checkpoint()
        self.mean_sha = sha256_hex_file(self.mean_path)
        self.sidecar_immutable = load_semantic_sidecar(
            self.mean_path
        )["semantic_manifest"]["immutable"]

    # 用途：夹具清理。
    # 参数：无输入；输出 无。
    def tearDown(self):
        self.dataset.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # 用途：构造并保存一个 14 通道 geo-season 确定性（A5 型）均值 checkpoint。
    # 参数：无输入；输出 无（写 self.mean_path 及 sidecar）。
    def _write_mean_checkpoint(self):
        config = OSTIATrainingConfig()
        config.output_dir = self.tmp_dir
        config.condition_mode = "sst_mask_geo_season"
        model = config.model
        model.adopt_condition_mode("sst_mask_geo_season")
        model.image_size = (16, 16, 1)
        model.patch_size = (2, 2, 1)
        model.embed_dim = 8
        model.num_blocks = 2
        model.explicit_layer = 1
        model.implicit_layer = 1
        model.hidden_size_factor = 2
        model.model_type = "deterministic"
        model.target_mode = "residual"
        model.target_scaling = "lead_standardized"
        model.lead_mean = tuple(float(value) for value in range(15))
        model.lead_std = tuple(1.0 + value for value in range(15))
        copy_dataset_provenance(model, self.dataset)
        built = model.build_model(torch.device("cpu"))
        manager = CheckpointManager(config)
        optimizer = AdamW(built.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        scaler = GradScaler("cuda", enabled=False)
        manager.save(
            self.mean_path,
            built,
            optimizer,
            scheduler,
            scaler,
            epoch=11,
            global_step=100,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[CheckpointManager.capture_random_state()],
        )

    # 用途：构造 v2 centered stats 载荷（指向合成均值）。
    # 参数：无输入；输出 载荷 dict（同时写入 self.stats_path）。
    def _v2_stats_payload(self):
        payload = {
            "schema_version": 2,
            "split": "train",
            "target_space": CENTERED_TARGET_SPACE,
            "input_days": 7,
            "output_days": 15,
            "condition_mode": "sst_mask_geo_season",
            "num_samples": 64,
            "dataset_size": len(self.dataset),
            "selection": "test_selection",
            "indices_sha256": "a" * 64,
            "mean_checkpoint": self.mean_path,
            "mean_checkpoint_sha256": self.mean_sha,
            "mean_semantics_sha256": sha256_of_normalized(
                self.sidecar_immutable
            ),
            "data_manifest_sha256": self.dataset.data_manifest_sha256,
            "mean_lead_mean": list(self.sidecar_immutable["lead_mean"]),
            "mean_lead_std": list(self.sidecar_immutable["lead_std"]),
            "sst_mean": 290.7488927184541,
            "sst_std": 9.57073350168232,
            "lead_mean": [float(value) / 10 for value in range(15)],
            "lead_std": [1.0 + float(value) / 10 for value in range(15)],
            "overall_innovation_std": 1.2,
            "valid_pixels": [100] * 15,
            "h5_path": self.h5_path,
        }
        with open(self.stats_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return payload

    # 用途：基于均值 sidecar 构造 centered_diffusion 模型配置。
    # 参数：无输入；输出 OSTIAModelConfig。
    def _centered_model_config(self):
        sidecar_config = load_semantic_sidecar(self.mean_path)["config"]
        model = OSTIAModelConfig.from_checkpoint(dict(sidecar_config))
        model.model_type = "centered_diffusion"
        model.target_mode = "residual"
        model.target_scaling = "lead_standardized"
        model.sigma_data = 1.0
        payload = self._v2_stats_payload()
        model.lead_mean = tuple(payload["lead_mean"])
        model.lead_std = tuple(payload["lead_std"])
        model.mean_lead_mean = tuple(payload["mean_lead_mean"])
        model.mean_lead_std = tuple(payload["mean_lead_std"])
        model.mean_checkpoint_sha256 = self.mean_sha
        model.mean_semantics_sha256 = sha256_of_normalized(
            self.sidecar_immutable
        )
        return model

    # 用途：均值 sidecar 证明了 geo-season 契约且 v2 载荷可校验。
    # 参数：无输入；输出 无（断言）。
    def test_mean_sidecar_proves_geo_season_contract(self):
        self.assertEqual(
            self.sidecar_immutable.get("condition_mode"),
            "sst_mask_geo_season",
        )
        self.assertEqual(
            self.sidecar_immutable.get("cond_chans"), 14
        )
        for field in V2_SIDECAR_PRESENCE_FIELDS:
            self.assertIsNotNone(
                self.sidecar_immutable.get(field),
                f"mean sidecar immutable lacks {field}",
            )

    # 用途：v2 载荷（schema 2 + geo-season + 合成均值身份补丁）通过载荷校验。
    # 参数：无输入；输出 无（断言）。
    def test_v2_payload_validates_with_synthetic_mean_lock(self):
        payload = self._v2_stats_payload()
        with mock.patch.object(
                centered_stats_module,
                "A5_LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            validated = validate_centered_stats_payload(
                payload, 15, 7, 15
            )
        self.assertEqual(
            validated["mean_checkpoint_sha256"], self.mean_sha
        )

    # 用途：fresh-run 校验（均值文件 + stats + centered 配置）在 v2 下通过。
    # 参数：无输入；输出 无（断言）。
    def test_fresh_inputs_v2_arch_match_passes(self):
        self._v2_stats_payload()
        model = self._centered_model_config()
        with mock.patch.object(
                centered_stats_module,
                "A5_LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            validated, immutable = validate_centered_fresh_inputs(
                self.mean_path, self.stats_path, model
            )
        self.assertEqual(
            validated["mean_checkpoint_sha256"], self.mean_sha
        )
        self.assertEqual(
            immutable["condition_mode"], "sst_mask_geo_season"
        )

    # 用途：wrapper 均值 strict 加载/冻结/不变，扩散参数实际更新，采样有限且可重建。
    # 参数：无输入；输出 无（断言）。
    def test_wrapper_mean_frozen_diffusion_trains(self):
        model = self._centered_model_config()
        wrapper = model.build_model(torch.device("cpu"))
        checkpoint = torch.load(
            self.mean_path,
            map_location="cpu",
            weights_only=False,
        )
        wrapper.mean_model.load_state_dict(
            checkpoint["model"], strict=True
        )
        before = {
            key: value.clone()
            for key, value in wrapper.mean_model.state_dict().items()
        }
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in wrapper.mean_model.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in wrapper.diffusion.parameters()
            )
        )
        sample = self.dataset[0]
        condition = torch.from_numpy(
            sample["condition"].numpy()
        )[None].float()
        target = torch.from_numpy(sample["target"].numpy())[None].float()
        mask = torch.from_numpy(
            sample["target_mask"].numpy()
        )[None]
        residual = (target - condition[:, 6:7]).float()
        loss = wrapper(residual, condition, mask)
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in wrapper.mean_model.parameters()
            ),
            "frozen mean produced gradients",
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in wrapper.diffusion.parameters()
            ),
            "diffusion parameters received no gradients",
        )
        with torch.no_grad():
            forecast = wrapper.sample(
                condition, num_sample_steps=2, seed=1
            )
        self.assertEqual(
            tuple(forecast.shape), tuple(residual.shape)
        )
        self.assertTrue(torch.isfinite(forecast).all())
        for key, value in wrapper.mean_model.state_dict().items():
            self.assertTrue(
                torch.equal(value, before[key]),
                f"frozen mean weight {key} changed after backward",
            )

    # 用途：零 innovation 重建：inverse_innovation(-m/s) 严格回到 0（r=mu）。
    # 参数：无输入；输出 无（断言）。
    def test_zero_innovation_reconstruction(self):
        model = self._centered_model_config()
        wrapper = model.build_model(torch.device("cpu"))
        zero_innovation = wrapper.inverse_innovation(
            -wrapper.innovation_mean / wrapper.innovation_std
        )
        self.assertTrue(
            torch.allclose(
                zero_innovation,
                torch.zeros_like(zero_innovation),
                atol=1e-6,
            )
        )


class A5CenteredPreflightTests(unittest.TestCase):
    """Pure-function preflight checks on the synthetic geo fixture."""

    # 用途：夹具准备（与本地 E2E 测试同构的合成均值）。
    # 参数：无输入；输出 无。
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.h5_path = os.path.join(self.tmp_dir, "geo.h5")
        make_synthetic_h5(
            self.h5_path,
            total_days=140,
            samples_per_day=5,
            height=16,
            width=16,
            coordinate_layout="per_row",
            with_time_metadata=True,
        )
        self.manifest_path = os.path.join(self.tmp_dir, "manifest.json")
        write_synthetic_data_manifest(self.manifest_path, self.h5_path)
        self.dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask_geo_season",
            data_manifest=self.manifest_path,
        )
        config = OSTIATrainingConfig()
        config.output_dir = self.tmp_dir
        config.condition_mode = "sst_mask_geo_season"
        model = config.model
        model.adopt_condition_mode("sst_mask_geo_season")
        model.image_size = (16, 16, 1)
        model.patch_size = (2, 2, 1)
        model.embed_dim = 8
        model.num_blocks = 2
        model.explicit_layer = 1
        model.implicit_layer = 1
        model.hidden_size_factor = 2
        model.model_type = "deterministic"
        model.target_mode = "residual"
        model.target_scaling = "lead_standardized"
        model.lead_mean = tuple(float(value) for value in range(15))
        model.lead_std = tuple(1.0 + value for value in range(15))
        copy_dataset_provenance(model, self.dataset)
        self.mean_path = os.path.join(self.tmp_dir, "mean.pth")
        built = model.build_model(torch.device("cpu"))
        manager = CheckpointManager(config)
        optimizer = AdamW(built.parameters(), lr=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        manager.save(
            self.mean_path,
            built,
            optimizer,
            scheduler,
            GradScaler("cuda", enabled=False),
            epoch=11,
            global_step=100,
            train_loss=0.5,
            dataset=DatasetStub(),
            random_states=[CheckpointManager.capture_random_state()],
        )
        from scripts.preflight_a5_centered import (
            mean_sidecar_immutable,
        )
        self.immutable, _ = mean_sidecar_immutable(self.mean_path)
        self.mean_sha = sha256_hex_file(self.mean_path)

    # 用途：清理。
    # 参数：无输入；输出 无。
    def tearDown(self):
        self.dataset.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # 用途：合成均值通过 preflight 的冻结均值函数（替换为合成文件 SHA 后）。
    # 参数：无输入；输出 无（断言）。
    def test_v2_sidecar_contract_function(self):
        from scripts.preflight_a5_centered import (
            verify_frozen_mean,
        )
        with mock.patch(
                "scripts.preflight_a5_centered.A5_MEAN_SHA256",
                sha256_hex_file(self.mean_path),
            ):
            immutable = verify_frozen_mean(self.mean_path)
        self.assertEqual(
            immutable["condition_mode"], "sst_mask_geo_season"
        )
        self.assertEqual(immutable["cond_chans"], 14)

    # 用途：架构核对：匹配的载荷通过，patch 不符则拒绝。
    # 参数：无输入；输出 无（断言）。
    def test_arch_vs_mean_sidecar(self):
        from scripts.preflight_a5_centered import (
            verify_arch_vs_mean_sidecar,
        )
        payload = {
            "model_type": "centered_diffusion",
            "condition_mode": "sst_mask_geo_season",
            "sigma_data": 1.0,
            "output_days": 15,
            "input_days": 7,
            "image_size": [16, 16, 1],
            "patch_size": [2, 2, 1],
            "embed_dim": 8,
            "num_blocks": 2,
            "explicit_layer": 1,
            "implicit_layer": 1,
            "hidden_size_factor": 2,
        }
        verify_arch_vs_mean_sidecar(payload, self.immutable)
        bad = dict(payload)
        bad["patch_size"] = [4, 4, 1]
        with self.assertRaisesRegex(ValueError, "patch_size"):
            verify_arch_vs_mean_sidecar(bad, self.immutable)

    # 用途：v2 统计计算用嵌套前缀方案：selection/num_samples/indices_sha 正确。
    # 参数：无输入；输出 无（断言）。
    def test_compute_stats_v2_nested_prefix(self):
        from deterministic_iafno.compute_centered_stats import (
            build_chunk_aware_indices,
            compute_centered_stats,
        )
        from deterministic_iafno.centered_stats import (
            indices_sha256,
        )
        with mock.patch.object(
                centered_stats_module,
                "A5_LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ), mock.patch(
                "deterministic_iafno.compute_centered_stats."
                "A5_LOCKED_MEAN_CHECKPOINT_SHA256",
                self.mean_sha,
            ):
            payload8, _ = compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=8,
                batch_size=8,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
                data_manifest=self.manifest_path,
            )
            payload16, _ = compute_centered_stats(
                h5_path=self.h5_path,
                mean_checkpoint_path=self.mean_path,
                num_samples=16,
                batch_size=8,
                input_days=7,
                output_days=15,
                device=torch.device("cpu"),
                use_amp=False,
                data_manifest=self.manifest_path,
            )
        # compute_centered_stats self-validated both payloads under the
        # patched lock; only assert protocol fields out here.
        self.assertEqual(payload8["schema_version"], 2)
        self.assertEqual(
            payload8["selection"], "nested_chunk_aware_prefix"
        )
        self.assertEqual(payload16["selection"], "nested_chunk_aware_prefix")
        master = build_chunk_aware_indices(
            self.dataset, min(65536, len(self.dataset))
        )
        self.assertEqual(
            payload8["indices_sha256"], indices_sha256(master[:8])
        )
        self.assertEqual(
            payload16["indices_sha256"], indices_sha256(master[:16])
        )


class DualBestSelectionTests(unittest.TestCase):
    """Dual-metric selection helpers (plan 7.2)."""

    # 用途：构造候选。
    # 参数：输入 label、rmse、crps、steps；输出 候选 dict。
    @staticmethod
    def candidate(label, rmse, crps, steps):
        return {
            "label": label,
            "source": label + ".pth",
            "overall_rmse": rmse,
            "overall_crps": crps,
            "cumulative_training_steps": steps,
        }

    # 用途：RMSE 与 CRPS 各选各的最优，互不干扰。
    # 参数：无输入；输出 无（断言）。
    def test_dual_metric_selection(self):
        from scripts.validate_a5_centered_epochs import select_best
        candidates = [
            self.candidate("epoch_010", 0.60, 0.40, 2500),
            self.candidate("epoch_020", 0.58, 0.35, 5000),
            self.candidate("epoch_030", 0.59, 0.33, 7500),
        ]
        best_rmse = select_best(candidates, "overall_rmse")
        best_crps = select_best(candidates, "overall_crps")
        self.assertEqual(best_rmse["label"], "epoch_020")
        self.assertEqual(best_crps["label"], "epoch_030")

    # 用途：同分取较早累计步数。
    # 参数：无输入；输出 无（断言）。
    def test_tie_prefers_earlier_steps(self):
        from scripts.validate_a5_centered_epochs import select_best
        candidates = [
            self.candidate("epoch_020", 0.58, 0.35, 5000),
            self.candidate("epoch_030", 0.58, 0.35, 7500),
        ]
        self.assertEqual(
            select_best(candidates, "overall_rmse")["label"], "epoch_020"
        )
        self.assertEqual(
            select_best(candidates, "overall_crps")["label"], "epoch_020"
        )


class StabilitySummaryTests(unittest.TestCase):
    """Pure stability-decision function (plan 6.1 thresholds)."""

    # 用途：构造统计载荷的最小结构（仅 stability_summary 所需字段）。
    # 参数：输入 num_samples、lead_mean、lead_std；输出 载荷 dict。
    @staticmethod
    def payload(num_samples, lead_mean, lead_std):
        return {
            "num_samples": num_samples,
            "lead_mean": lead_mean,
            "lead_std": lead_std,
            "valid_pixels": [100] * 15,
        }

    # 用途：逐 lead 变化都在阈值内 -> 稳定。
    # 参数：无输入；输出 无（断言）。
    def test_small_changes_stable(self):
        from scripts.centered_stats_stability import (
            stability_summary,
        )
        prev = self.payload(
            8192, [0.0] * 15, [1.0 + 0.01 * i for i in range(15)]
        )
        cur = self.payload(
            16384,
            [0.001] * 15,
            [1.0 * 1.005 + 0.01 * i for i in range(15)],
        )
        stable, details = stability_summary(prev, cur)
        self.assertTrue(stable)
        self.assertTrue(details["stable"])

    # 用途：任一 lead 的 std 相对变化超 2% -> 不稳定。
    # 参数：无输入；输出 无（断言）。
    def test_large_std_change_unstable(self):
        from scripts.centered_stats_stability import (
            stability_summary,
        )
        cur_std = [1.0 + 0.01 * i for i in range(15)]
        prev_std = list(cur_std)
        prev_std[7] = cur_std[7] * 0.95  # 5% relative change on lead 7
        stable, details = stability_summary(
            self.payload(8192, [0.0] * 15, prev_std),
            self.payload(16384, [0.0] * 15, cur_std),
        )
        self.assertFalse(stable)
        self.assertGreater(details["rel_std_change"][7], 0.02)

    # 用途：长度不一致 -> 抛错。
    # 参数：无输入；输出 无（断言）。
    def test_length_mismatch_raises(self):
        from scripts.centered_stats_stability import (
            stability_summary,
        )
        with self.assertRaisesRegex(ValueError, "disagree in length"):
            stability_summary(
                self.payload(8192, [0.0] * 15, [1.0] * 15),
                self.payload(16384, [0.0] * 14, [1.0] * 14),
            )


class CandidateCadenceTests(unittest.TestCase):
    """list_candidates cadence (--every) helper."""

    # 用途：每 K 轮 + 末轮恒在。
    # 参数：无输入；输出 无（断言）。
    def test_every_and_last(self):
        from scripts.validate_a5_centered_epochs import list_candidates
        with tempfile.TemporaryDirectory() as tmp:
            for epoch in range(1, 31):
                with open(
                        os.path.join(tmp, f"epoch_{epoch:03d}.pth"), "w"
                    ) as file:
                    file.write("x")
                with open(
                        os.path.join(
                            tmp, f"epoch_{epoch:03d}.pth.semantics.json"
                        ),
                        "w",
                    ) as file:
                    file.write("{}")
            labels = [label for label, _ in list_candidates(tmp, 30, 5)]
        self.assertEqual(
            labels,
            ["epoch_005", "epoch_010", "epoch_015", "epoch_020",
             "epoch_025", "epoch_030"],
        )


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
