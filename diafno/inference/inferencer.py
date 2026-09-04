import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.condition_schema import resolve_condition_mode
from ..data.ostia import (
    OSTIADailyDataset,
    verify_checkpoint_data_contract,
)

from .model import InferenceModelLoader
from .writer import InferenceSampleWriter


class OSTIAInferencer:
    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
        self.model_config = None
        self.normalization = None
        self.sampling_steps = None
        self.dataset = None
        self.loader = None
        self.writer = None
        self.amp_enabled = False

    def setup(self):
        if self.config.ensemble_members < 1:
            raise ValueError(
                "ensemble_members must be at least 1"
            )
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
        # The condition contract comes from the checkpoint, never from
        # a stale CLI default; an explicit conflicting override is a
        # launch error before any data is read.
        condition_mode = resolve_condition_mode(
            self.config.condition_mode,
            self.model_config.condition_mode,
            "inference",
        )
        # Any checkpoint bound to an upstream data manifest cannot be
        # used for inference without that same gap-filtered mapping.
        time_source = (
            self.model_config.time_axis_summary or {}
        ).get("source")
        data_manifest = getattr(
            self.config, "data_manifest", None
        )
        if time_source == "data_manifest" and data_manifest is None:
            raise ValueError(
                "this checkpoint was trained with an "
                "upstream data manifest; inference requires the "
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
        loader_options = {
            "dataset": self.dataset,
            "batch_size": self.config.batch_size,
            "shuffle": False,
            "num_workers": self.config.num_workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": False
        }
        if self.config.num_workers > 0:
            loader_options["persistent_workers"] = True
        self.loader = DataLoader(**loader_options)
        self.writer = InferenceSampleWriter(
            output_dir=self.config.output_dir,
            checkpoint_path=self.config.checkpoint,
            sampling_steps=self.sampling_steps,
            save_members=self.config.save_members,
            compress=self.config.compress
        )
        self.amp_enabled = (
            self.config.use_amp
            and self.device.type == "cuda"
        )
        return self

    @staticmethod
    def unpack_batch(batch):
        if isinstance(batch, dict):
            return (
                batch["condition"],
                batch.get("target"),
                batch.get("target_mask"),
                batch.get("metadata")
            )
        metadata = batch[3] if len(batch) > 3 else None
        return batch[0], batch[1], batch[2], metadata

    def check_condition(self, condition):
        expected_shape = (
            self.model_config.cond_chans,
            *self.model_config.image_size
        )
        if condition.ndim != 5:
            raise ValueError(
                "condition must have shape [B,C,H,W,Z], "
                f"but got {tuple(condition.shape)}"
            )
        if tuple(condition.shape[1:]) != expected_shape:
            raise ValueError(
                "condition shape mismatch: expected "
                f"[B,{','.join(map(str, expected_shape))}], "
                f"but got {tuple(condition.shape)}"
            )

    def _move_batch(
            self,
            condition,
            target,
            target_mask,
        ):
        condition = condition.to(
            self.device,
            non_blocking=True
        )
        if torch.is_tensor(target):
            target = target.to(
                self.device,
                non_blocking=True
            )
        if torch.is_tensor(target_mask):
            target_mask = target_mask.to(
                self.device,
                non_blocking=True
            )
        return condition, target, target_mask

    def inverse_transform(self, value):
        if value is None:
            return None
        transform = getattr(
            self.dataset,
            "inverse_transform_sst",
            None
        )
        if callable(transform):
            return transform(value)
        mean = getattr(self.dataset, "sst_mean", None)
        std = getattr(self.dataset, "sst_std", None)
        if mean is not None and std is not None:
            return value * std + mean
        if isinstance(self.normalization, dict):
            mean = self.normalization.get(
                "sst_mean",
                self.normalization.get("mean")
            )
            std = self.normalization.get(
                "sst_std",
                self.normalization.get("std")
            )
            if mean is not None and std is not None:
                return value * std + mean
        return value

    def _predict_ensemble(
            self,
            condition,
            batch_index,
        ):
        predictions = []
        for member_index in range(
            self.config.ensemble_members
        ):
            seed = (
                self.config.seed
                + batch_index * 1000
                + member_index
            )
            prediction = self.model.sample(
                condition=condition,
                num_sample_steps=self.sampling_steps,
                seed=seed
            )
            predictions.append(prediction)
        members = torch.stack(predictions, dim=0)
        if self.model_config.target_mode == "residual":
            last_day = condition[
                :,
                self.model_config.input_days - 1:
                self.model_config.input_days
            ]
            members = members + last_day.unsqueeze(0)
        return members.mean(dim=0), members

    def _save_batch(
            self,
            start_index,
            prediction,
            target,
            target_mask,
            members,
            metadata,
        ):
        saved = start_index
        for item_index in range(prediction.shape[0]):
            if (
                self.config.max_samples is not None
                and saved >= self.config.max_samples
            ):
                break
            sample_target = (
                target[item_index]
                if target is not None
                else None
            )
            sample_mask = (
                target_mask[item_index]
                if target_mask is not None
                else None
            )
            sample_members = (
                members[:, item_index]
                if self.config.save_members
                else None
            )
            sample_metadata = (
                self.writer.metadata_item(
                    metadata,
                    item_index
                )
            )
            self.writer.save(
                index=saved,
                prediction=prediction[item_index],
                target=sample_target,
                target_mask=sample_mask,
                ensemble_members=sample_members,
                metadata=sample_metadata
            )
            saved += 1
        return saved

    @torch.no_grad()
    def run(self):
        self.setup()
        saved = 0
        progress = tqdm(
            self.loader,
            desc="OSTIA inference"
        )
        for batch_index, batch in enumerate(progress):
            (
                condition,
                target,
                target_mask,
                metadata
            ) = self.unpack_batch(batch)
            (
                condition,
                target,
                target_mask
            ) = self._move_batch(
                condition,
                target,
                target_mask
            )
            self.check_condition(condition)
            with autocast("cuda", enabled=self.amp_enabled):
                prediction, members = (
                    self._predict_ensemble(
                        condition,
                        batch_index
                    )
                )
            prediction = self.inverse_transform(
                prediction
            )
            target = self.inverse_transform(target)
            saved = self._save_batch(
                start_index=saved,
                prediction=prediction,
                target=target,
                target_mask=target_mask,
                members=members,
                metadata=metadata
            )
            progress.set_postfix(saved=saved)
            if (
                self.config.max_samples is not None
                and saved >= self.config.max_samples
            ):
                break
        print(
            f"Saved {saved} samples to "
            f"{self.config.output_dir}"
        )
        return saved
