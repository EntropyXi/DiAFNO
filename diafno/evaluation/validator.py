# 用途：执行模型或 persistence 预测、条件消融及分 lead 评分。
import json
import os

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..data.condition_schema import resolve_condition_mode
from ..data.ostia import (
    OSTIADailyDataset,
    verify_checkpoint_data_contract,
)
from ..inference.model import InferenceModelLoader
from .bootstrap import paired_temporal_block_bootstrap
from .metrics import RunningSSTMetrics, persistence_skill
from .sample_manifest import (
    load_sample_manifest,
    manifest_dataset_indices,
)


class OSTIAValidator:
    # 用途：初始化验证器：保存配置并准备指标累计器与结果容器。
    # 参数：输入 config（验证配置对象）；输出 无。
    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
        self.model_config = None
        self.sampling_steps = None
        self.normalization = None
        self.dataset = None
        self.loader = None
        self.amp_enabled = False
        self.sample_manifest = None
        self._bootstrap_time_axis = None

    # 用途：构造固定可复现的验证样本索引（均匀抽样或指定子集）。
    # 参数：无输入（读配置）；输出 样本索引数组。
    def _build_indices(self):
        dataset_size = len(self.dataset)
        if self.config.max_samples is None:
            return None
        if self.config.max_samples < 1:
            raise ValueError("max_samples must be positive")
        sample_count = min(
            self.config.max_samples,
            dataset_size
        )
        generator = np.random.default_rng(
            self.config.seed
        )
        return np.sort(
            generator.choice(
                dataset_size,
                size=sample_count,
                replace=False
            )
        ).tolist()

    # 用途：核对 checkpoint 归一化统计量与数据集统计量一致（防口径漂移）。
    # 参数：无输入；输出 无（不一致抛 ValueError）。
    def _check_normalization(self):
        if not isinstance(self.normalization, dict):
            return
        checkpoint_mean = self.normalization.get("sst_mean")
        checkpoint_std = self.normalization.get("sst_std")
        if checkpoint_mean is None or checkpoint_std is None:
            return
        if not np.isclose(
                float(checkpoint_mean),
                self.dataset.sst_mean,
                rtol=1e-6,
                atol=1e-6
            ) or not np.isclose(
                float(checkpoint_std),
                self.dataset.sst_std,
                rtol=1e-6,
                atol=1e-6
            ):
            raise ValueError(
                "checkpoint and validation dataset normalization "
                "parameters do not match"
            )

    # 用途：装配验证组件：数据集、模型加载、评估协议与输出路径。
    # 参数：无输入；输出 无（组件写入实例属性）。
    def setup(self):
        if self.config.ensemble_members < 1:
            raise ValueError(
                "ensemble_members must be at least 1"
            )
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.config.paired_bootstrap_replicates < 0:
            raise ValueError(
                "paired_bootstrap_replicates must be non-negative"
            )
        if self.config.bootstrap_block_days < 1:
            raise ValueError("bootstrap_block_days must be positive")
        if not 0.0 < self.config.bootstrap_confidence < 1.0:
            raise ValueError(
                "bootstrap_confidence must be between 0 and 1"
            )
        if (
                self.config.s_churn is not None
                and self.config.s_churn < 0
            ):
            raise ValueError("s_churn must be non-negative")
        device = self.config.device
        if (
                device.startswith("cuda")
                and not torch.cuda.is_available()
            ):
            device = "cpu"
        self.device = torch.device(device)
        (
            self.model,
            self.model_config,
            self.sampling_steps,
            self.normalization
        ) = InferenceModelLoader.load(
            checkpoint_path=self.config.checkpoint,
            device=self.device,
            sampling_steps=self.config.sampling_steps
        )
        if self.config.s_churn is not None:
            if not hasattr(self.model, "S_churn"):
                raise ValueError(
                    "--s-churn only applies to diffusion checkpoints"
                )
            self.model.S_churn = self.config.s_churn
        # The condition contract comes from the checkpoint, never from
        # a stale CLI default; an explicit conflicting override is a
        # launch error before any data is read.
        condition_mode = resolve_condition_mode(
            self.config.condition_mode,
            self.model_config.condition_mode,
            "validation",
        )
        # Any checkpoint bound to an upstream data manifest cannot be
        # validated without the same gap-filtered time mapping.
        time_source = (
            self.model_config.time_axis_summary or {}
        ).get("source")
        data_manifest = getattr(
            self.config, "data_manifest", None
        )
        if time_source == "data_manifest" and data_manifest is None:
            raise ValueError(
                "this checkpoint was trained with an "
                "upstream data manifest; validation requires the "
                "matching --data-manifest file"
            )
        self.dataset = OSTIADailyDataset(
            h5_path=self.config.h5_path,
            split=self.config.split,
            input_days=self.model_config.input_days,
            output_days=self.model_config.output_days,
            condition_mode=condition_mode,
            data_manifest=data_manifest
        )
        verify_checkpoint_data_contract(
            self.dataset,
            self.model_config,
        )
        self._check_normalization()
        self.sample_manifest = None
        indices = self._build_indices()
        if self.config.sample_manifest is not None:
            payload = load_sample_manifest(
                self.config.sample_manifest,
                split=self.config.split,
                expected_count=(
                    None
                    if self.config.max_samples is None
                    else self.config.max_samples
                ),
                dataset_size=len(self.dataset),
            )
            indices = manifest_dataset_indices(payload)
            self.sample_manifest = {
                "path": os.path.abspath(self.config.sample_manifest),
                "manifest_sha256": payload["manifest_sha256"],
            }
        validation_data = (
            self.dataset
            if indices is None
            else Subset(self.dataset, indices)
        )
        loader_options = {
            "dataset": validation_data,
            "batch_size": self.config.batch_size,
            "shuffle": False,
            "num_workers": self.config.num_workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": False
        }
        if self.config.num_workers > 0:
            loader_options["persistent_workers"] = True
        self.loader = DataLoader(**loader_options)
        self.amp_enabled = (
            self.config.use_amp
            and self.device.type == "cuda"
        )
        return self

    # 用途：按条件消融设置破坏条件（置零/仅 anchor/逆序/打乱，均保留 day-7 anchor）。
    # 参数：输入 condition（原始条件场）；输出 消融后的条件场。
    def _ablate_condition(self, condition):
        mode = self.config.condition_ablation
        if mode == "none":
            return condition
        condition = condition.clone()
        input_days = self.model_config.input_days
        if mode == "anchor_only":
            anchor = condition[:, input_days - 1:input_days]
            condition[:, :input_days] = anchor.repeat(
                1,
                input_days,
                1,
                1,
                1
            )
        elif mode == "reverse_history":
            condition[:, :input_days - 1] = torch.flip(
                condition[:, :input_days - 1],
                dims=(1,)
            )
        elif mode == "shuffle_history":
            if condition.shape[0] < 2:
                raise ValueError(
                    "shuffle_history requires batch_size >= 2"
                )
            condition[:, :input_days - 1] = torch.roll(
                condition[:, :input_days - 1],
                shifts=1,
                dims=0
            )
        elif mode == "zero_sst":
            condition[:, :input_days] = 0
        elif mode == "reverse_sst":
            condition[:, :input_days] = torch.flip(
                condition[:, :input_days],
                dims=(1,)
            )
        return condition

    # 用途：固定 σ 去噪探针：绕过采样器直接测单点去噪能力。
    # 参数：输入 condition（条件场）、target（真值，仅用于形状）、batch_index（批号）；输出 探针预测。
    def _predict_probe(self, condition, target, batch_index):
        """Fixed-sigma denoising probe that bypasses the sampler entirely.

        This probe includes the true target in its noised input.  At tiny
        sigma it is therefore a near-identity numerical check, not an
        estimate of the forecast-time conditional mean.  Residual mode
        re-anchors the probe target and output by the last condition day.
        """
        probe_sigma = (
            self.config.probe_sigma
            if self.config.probe_sigma is not None
            else 0.002
        )
        if self.model_config.model_type != "diffusion":
            raise ValueError(
                "probe mode only applies to diffusion checkpoints"
            )
        input_days = self.model_config.input_days
        anchor = condition[:, input_days - 1:input_days]
        probe_target = target
        if self.model_config.target_mode == "residual":
            probe_target = target - anchor
        predictions = []
        for member_index in range(
                self.config.ensemble_members
            ):
            seed = (
                self.config.seed
                + batch_index * 1000
                + member_index
            )
            generator = torch.Generator(
                device=self.device
            ).manual_seed(seed)
            noise = torch.randn(
                probe_target.shape,
                device=probe_target.device,
                dtype=probe_target.dtype,
                generator=generator
            )
            noised = probe_target + probe_sigma * noise
            denoised = self.model.preconditioned_network_forward(
                noised,
                probe_sigma,
                condition
            )
            if self.model_config.target_mode == "residual":
                denoised = denoised + anchor
            predictions.append(denoised)
        return torch.stack(predictions, dim=0).mean(dim=0)

    # 用途：按模型语义产出预测：确定性直出，或扩散按协议集成采样后取均值。
    # 参数：输入 condition（条件场）、batch_index（批号）；输出 预测场（normalized residual 或绝对空间）。
    def _predict(self, condition, batch_index):
        if self.config.prediction_mode == "persistence":
            last_day = condition[
                :,
                self.model_config.input_days - 1:
                self.model_config.input_days
            ]
            return last_day.repeat(
                1,
                self.model_config.output_days,
                1,
                1,
                1
            )
        if self.config.prediction_mode == "linear_trend":
            input_days = self.model_config.input_days
            history = condition[:, :input_days]
            time = torch.arange(
                input_days,
                device=history.device,
                dtype=history.dtype
            )
            centered_time = time - time.mean()
            slope = (
                history
                * centered_time.view(1, -1, 1, 1, 1)
            ).sum(dim=1, keepdim=True) / (
                centered_time.square().sum().clamp_min(1.0)
            )
            intercept = (
                history.mean(dim=1, keepdim=True)
                - slope * time.mean()
            )
            future_time = torch.arange(
                input_days,
                input_days + self.model_config.output_days,
                device=history.device,
                dtype=history.dtype
            )
            return (
                intercept
                + slope * future_time.view(1, -1, 1, 1, 1)
            )
        if self.model_config.model_type == "deterministic":
            if self.config.ensemble_members != 1:
                raise ValueError(
                    "deterministic checkpoints require "
                    "--ensemble-members 1"
                )
            original_condition = condition
            condition = self._ablate_condition(condition)
            prediction = self.model.predict(condition)
            if self.model_config.target_mode == "residual":
                last_day = original_condition[
                    :,
                    self.model_config.input_days - 1:
                    self.model_config.input_days
                ]
                prediction = prediction + last_day
            return prediction
        original_condition = condition
        condition = self._ablate_condition(condition)
        predictions = []
        for member_index in range(
                self.config.ensemble_members
            ):
            seed = (
                self.config.seed
                + batch_index * 1000
                + member_index
            )
            predictions.append(
                self.model.sample(
                    condition=condition,
                    num_sample_steps=self.sampling_steps,
                    seed=seed
                )
            )
        prediction = torch.stack(
            predictions,
            dim=0
        ).mean(dim=0)
        if self.model_config.target_mode == "residual":
            last_day = original_condition[
                :,
                self.model_config.input_days - 1:
                self.model_config.input_days
            ]
            prediction = prediction + last_day
        return prediction

    @staticmethod
    # 用途：去掉末尾长度为 1 的深度轴（static 方法）。
    # 参数：输入 value（数组）；输出 形状 [B,C,H,W] 的数组。
    def _without_depth_axis(value):
        if value.ndim != 5 or value.shape[-1] != 1:
            raise ValueError(
                "validation tensors must have shape "
                f"[batch,lead,H,W,1], but got {tuple(value.shape)}"
            )
        return value[..., 0]

    # 用途：把模型空间预测反标准化回开尔文并重建绝对 SST（anchor 只加一次）。
    # 参数：输入 value（模型空间值）；输出 物理空间值。
    def _inverse_transform(self, value):
        return (
            value * self.dataset.sst_std
            + self.dataset.sst_mean
        )

    # 用途：把验证指标结果写为 JSON。
    # 参数：输入 result（结果 dict）；输出 无。
    def _save_result(self, result):
        output_dir = os.path.dirname(
            self.config.output_path
        )
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(
                self.config.output_path,
                "w",
                encoding="utf-8"
            ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )

    @torch.no_grad()
    # 用途：验证主循环：逐批预测、累计指标、（可选）条件消融与探针，最后落盘。
    # 参数：无输入；输出 无（结果写 output_path）。
    def run(self):
        self.setup()
        overall = RunningSSTMetrics()
        residual_overall = RunningSSTMetrics()
        persistence_overall = RunningSSTMetrics()
        by_lead = [
            RunningSSTMetrics()
            for _ in range(self.model_config.output_days)
        ]
        residual_by_lead = [
            RunningSSTMetrics()
            for _ in range(self.model_config.output_days)
        ]
        persistence_by_lead = [
            RunningSSTMetrics()
            for _ in range(self.model_config.output_days)
        ]
        num_samples = 0
        paired_model_sse = []
        paired_persistence_sse = []
        paired_valid_counts = []
        paired_initialization_times = []
        progress = tqdm(
            self.loader,
            desc="OSTIA validation"
        )
        for batch_index, batch in enumerate(progress):
            condition = batch["condition"].to(
                self.device,
                non_blocking=True
            )
            target = batch["target"].to(
                self.device,
                non_blocking=True
            )
            target_mask = batch["target_mask"].to(
                self.device,
                non_blocking=True
            )
            with autocast("cuda", enabled=self.amp_enabled):
                if self.config.prediction_mode == "probe":
                    prediction = self._predict_probe(
                        condition,
                        target,
                        batch_index
                    )
                else:
                    prediction = self._predict(
                        condition,
                        batch_index
                    )
            anchor = condition[
                :,
                self.model_config.input_days - 1:
                self.model_config.input_days
            ]
            anchor = anchor.repeat(
                1,
                self.model_config.output_days,
                1,
                1,
                1
            )
            prediction_residual = (
                prediction - anchor
            ) * self.dataset.sst_std
            target_residual = (
                target - anchor
            ) * self.dataset.sst_std
            persistence = anchor
            prediction = self._without_depth_axis(
                self._inverse_transform(prediction)
            ).float().cpu().numpy()
            target = self._without_depth_axis(
                self._inverse_transform(target)
            ).float().cpu().numpy()
            prediction_residual = self._without_depth_axis(
                prediction_residual
            ).float().cpu().numpy()
            target_residual = self._without_depth_axis(
                target_residual
            ).float().cpu().numpy()
            persistence = self._without_depth_axis(
                self._inverse_transform(persistence)
            ).float().cpu().numpy()
            target_mask = self._without_depth_axis(
                target_mask
            ).float().cpu().numpy()
            if self.config.paired_bootstrap_replicates > 0:
                valid = (
                    np.isfinite(prediction)
                    & np.isfinite(target)
                    & np.isfinite(persistence)
                    & (target_mask > 0)
                )
                paired_model_sse.append(
                    np.where(
                        valid,
                        np.square(prediction - target),
                        0.0
                    ).sum(axis=(2, 3), dtype=np.float64)
                )
                paired_persistence_sse.append(
                    np.where(
                        valid,
                        np.square(persistence - target),
                        0.0
                    ).sum(axis=(2, 3), dtype=np.float64)
                )
                paired_valid_counts.append(
                    valid.sum(axis=(2, 3), dtype=np.int64)
                )
                # Prefer the real-day initialization axis when the
                # dataset is bound to an upstream data manifest; the
                # compact axis stays the fallback so legacy runs are
                # unchanged.
                if (
                        getattr(self.dataset, "has_real_day_axis", False)
                        and "real_t0_day_offset" in batch["metadata"]
                    ):
                    paired_initialization_times.append(
                        batch["metadata"]["real_t0_day_offset"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    self._bootstrap_time_axis = "real_day_offset_t0"
                else:
                    paired_initialization_times.append(
                        batch["metadata"]["input_start_time"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    self._bootstrap_time_axis = "compact_input_start"
            overall.update(prediction, target, target_mask)
            residual_overall.update(
                prediction_residual,
                target_residual,
                target_mask
            )
            persistence_overall.update(
                persistence,
                target,
                target_mask
            )
            for lead_index, metrics in enumerate(by_lead):
                metrics.update(
                    prediction[:, lead_index],
                    target[:, lead_index],
                    target_mask[:, lead_index]
                )
                residual_by_lead[lead_index].update(
                    prediction_residual[:, lead_index],
                    target_residual[:, lead_index],
                    target_mask[:, lead_index]
                )
                persistence_by_lead[lead_index].update(
                    persistence[:, lead_index],
                    target[:, lead_index],
                    target_mask[:, lead_index]
                )
            num_samples += prediction.shape[0]
            progress.set_postfix(samples=num_samples)
        overall_result = overall.compute()
        residual_overall_result = residual_overall.compute()
        persistence_overall_result = persistence_overall.compute()
        by_lead_result = {
            str(index + 1): metrics.compute()
            for index, metrics in enumerate(by_lead)
        }
        residual_by_lead_result = {
            str(index + 1): metrics.compute()
            for index, metrics in enumerate(residual_by_lead)
        }
        persistence_by_lead_result = {
            str(index + 1): metrics.compute()
            for index, metrics in enumerate(persistence_by_lead)
        }
        result = {
            "checkpoint": os.path.abspath(
                self.config.checkpoint
            ),
            "split": self.config.split,
            "num_samples": num_samples,
            "prediction_mode": self.config.prediction_mode,
            "probe_sigma": self.config.probe_sigma,
            "sampling_steps": self.sampling_steps,
            "s_churn": getattr(self.model, "S_churn", None),
            "ensemble_members": self.config.ensemble_members,
            "sampler_profile": {
                "model_type": self.model_config.model_type,
                "sampling_steps": self.sampling_steps,
                "sigma_min": self.model_config.sigma_min,
                "sigma_max": self.model_config.sigma_max,
                "rho": self.model_config.rho,
                "s_churn": getattr(self.model, "S_churn", None),
                "ensemble_members": self.config.ensemble_members,
            },
            "condition_ablation": self.config.condition_ablation,
            "seed": self.config.seed,
            "sample_manifest": self.sample_manifest,
            "overall": overall_result,
            "by_lead_day": by_lead_result,
            "residual_overall": residual_overall_result,
            "residual_by_lead_day": residual_by_lead_result,
            "persistence_overall": persistence_overall_result,
            "persistence_by_lead_day": persistence_by_lead_result,
            "persistence_skill": {
                "overall": persistence_skill(
                    overall_result,
                    persistence_overall_result
                ),
                "by_lead_day": {
                    str(index + 1): persistence_skill(
                        by_lead_result[str(index + 1)],
                        persistence_by_lead_result[str(index + 1)]
                    )
                    for index in range(
                        self.model_config.output_days
                    )
                }
            }
        }
        if self.config.paired_bootstrap_replicates > 0:
            result["paired_block_bootstrap"] = (
                paired_temporal_block_bootstrap(
                    np.concatenate(paired_model_sse, axis=0),
                    np.concatenate(paired_persistence_sse, axis=0),
                    np.concatenate(paired_valid_counts, axis=0),
                    np.concatenate(
                        paired_initialization_times,
                        axis=0
                    ),
                    block_days=self.config.bootstrap_block_days,
                    replicates=(
                        self.config.paired_bootstrap_replicates
                    ),
                    confidence_level=(
                        self.config.bootstrap_confidence
                    ),
                    seed=self.config.bootstrap_seed,
                    block_origin_time=(
                        self.dataset.real_day_offset(
                            self.dataset.split_start_day
                            + self.model_config.input_days - 1
                        )
                        if self._bootstrap_time_axis
                        == "real_day_offset_t0"
                        else (
                            self.dataset.first_time
                            + self.dataset.split_start_day
                        )
                    ),
                )
            )
            result["bootstrap_time_axis"] = (
                self._bootstrap_time_axis
            )
        self._save_result(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
