import json
import os

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..data.ostia import OSTIADailyDataset
from ..inference.model import InferenceModelLoader
from .metrics import RunningSSTMetrics


class OSTIAValidator:
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

    def setup(self):
        if self.config.ensemble_members < 1:
            raise ValueError(
                "ensemble_members must be at least 1"
            )
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive")
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
            self.model.S_churn = self.config.s_churn
        self.dataset = OSTIADailyDataset(
            h5_path=self.config.h5_path,
            split=self.config.split,
            input_days=self.model_config.input_days,
            output_days=self.model_config.output_days,
            condition_mode=self.config.condition_mode
        )
        self._check_normalization()
        indices = self._build_indices()
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

    def _ablate_condition(self, condition):
        mode = self.config.condition_ablation
        if mode == "none":
            return condition
        condition = condition.clone()
        input_days = self.model_config.input_days
        if mode == "zero_sst":
            condition[:, :input_days] = 0
        elif mode == "reverse_sst":
            condition[:, :input_days] = torch.flip(
                condition[:, :input_days],
                dims=(1,)
            )
        return condition

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
    def _without_depth_axis(value):
        if value.ndim != 5 or value.shape[-1] != 1:
            raise ValueError(
                "validation tensors must have shape "
                f"[batch,lead,H,W,1], but got {tuple(value.shape)}"
            )
        return value[..., 0]

    def _inverse_transform(self, value):
        return (
            value * self.dataset.sst_std
            + self.dataset.sst_mean
        )

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
    def run(self):
        self.setup()
        overall = RunningSSTMetrics()
        by_lead = [
            RunningSSTMetrics()
            for _ in range(self.model_config.output_days)
        ]
        num_samples = 0
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
                prediction = self._predict(
                    condition,
                    batch_index
                )
            prediction = self._without_depth_axis(
                self._inverse_transform(prediction)
            ).float().cpu().numpy()
            target = self._without_depth_axis(
                self._inverse_transform(target)
            ).float().cpu().numpy()
            target_mask = self._without_depth_axis(
                target_mask
            ).float().cpu().numpy()
            overall.update(prediction, target, target_mask)
            for lead_index, metrics in enumerate(by_lead):
                metrics.update(
                    prediction[:, lead_index],
                    target[:, lead_index],
                    target_mask[:, lead_index]
                )
            num_samples += prediction.shape[0]
            progress.set_postfix(samples=num_samples)
        result = {
            "checkpoint": os.path.abspath(
                self.config.checkpoint
            ),
            "split": self.config.split,
            "num_samples": num_samples,
            "prediction_mode": self.config.prediction_mode,
            "sampling_steps": self.sampling_steps,
            "s_churn": self.model.S_churn,
            "ensemble_members": self.config.ensemble_members,
            "condition_ablation": self.config.condition_ablation,
            "seed": self.config.seed,
            "overall": overall.compute(),
            "by_lead_day": {
                str(index + 1): metrics.compute()
                for index, metrics in enumerate(by_lead)
            }
        }
        self._save_result(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
