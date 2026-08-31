import math
import os
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.optim as optim

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .artifacts import CheckpointManager, TrainingHistory
from .config import default_training_model
from .data import OSTIATrainingData
from .runtime import DistributedRuntime, set_random_seed
from ..models.config import OSTIAModelConfig
from deterministic_iafno.checkpoint_semantics import (
    load_semantic_sidecar,
    restore_resume_semantics,
)


class OSTIATrainer:
    def __init__(self, config):
        self.config = config
        self.runtime = DistributedRuntime()
        self.data = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.amp_enabled = False
        self.start_epoch = 0
        self.global_step = 0
        self.history = TrainingHistory(
            config.output_dir,
            config.max_grad_norm
        )
        self.checkpoints = CheckpointManager(config)

    def setup(self):
        if (
                self.config.resume_path is not None
                and self.config.init_from is not None
        ):
            raise ValueError(
                "--resume and --init-from are mutually exclusive"
            )
        self.runtime.setup()
        set_random_seed(
            self.config.seed,
            self.runtime.rank
        )
        if self.runtime.is_main_process:
            os.makedirs(
                self.config.output_dir,
                exist_ok=True
            )
            print("Using GPUs:", self.runtime.world_size)
            print("Device:", self.runtime.device)
        self._restore_resume_semantics()
        self.data = OSTIATrainingData(
            self.config,
            self.runtime
        ).setup()
        self._build_training_components()
        if (
                self.config.resume_path is None
                and self.config.init_from is not None
        ):
            self._init_from_checkpoint()
        self._resume_training()

    def _build_training_components(self):
        self.model = self.config.model.build_model(
            self.runtime.device
        )
        if self.runtime.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.runtime.local_rank],
                output_device=self.runtime.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False
            )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        optimizer_steps_per_epoch = math.ceil(
            len(self.data.loader)
            / self.config.gradient_accumulation
        )
        self.config.optimizer_steps_per_epoch = (
            optimizer_steps_per_epoch
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(
                self.config.num_epochs
                * optimizer_steps_per_epoch
            ),
            eta_min=self.config.min_learning_rate
        )
        self.amp_enabled = (
            self.config.use_amp
            and self.runtime.device.type == "cuda"
        )
        self.scaler = GradScaler(
            "cuda",
            enabled=self.amp_enabled
        )
        self.optimizer.zero_grad(set_to_none=True)

    def _init_from_checkpoint(self):
        """Load model weights only from another checkpoint (e.g. an
        absolute-mode checkpoint reused for residual fine-tuning).

        Optimizer/scheduler/scaler/random states are intentionally NOT
        restored, so training starts from epoch 0 with a fresh schedule.
        """
        init_path = os.path.abspath(self.config.init_from)
        checkpoint = torch.load(
            init_path,
            map_location="cpu",
            weights_only=False
        )
        source_config = OSTIAModelConfig.from_checkpoint(
            checkpoint["config"]
        )
        current = self.config.model
        arch_fields = (
            "input_days",
            "output_days",
            "cond_chans",
            "target_chans",
            "image_size",
            "patch_size",
            "embed_dim",
            "num_blocks",
            "explicit_layer",
            "implicit_layer",
            "hidden_size_factor"
        )
        mismatches = {
            field: (getattr(source_config, field),
                    getattr(current, field))
            for field in arch_fields
            if getattr(source_config, field) != getattr(current, field)
        }
        if mismatches:
            raise ValueError(
                "init-from architecture mismatch for fields "
                f"(source, current): {mismatches}"
            )
        if (
                current.model_type == "diffusion"
                and float(source_config.sigma_data) != float(
                    current.sigma_data
                )
            ):
            raise ValueError(
                "init-from sigma_data mismatch: source "
                f"{source_config.sigma_data} vs current "
                f"{current.sigma_data}; pretrained weights are "
                "incompatible with a different sigma_data"
            )
        try:
            self.checkpoints.unwrap_model(
                self.model
            ).load_state_dict(
                checkpoint["model"],
                strict=True
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"init-from {init_path} state_dict failed to load "
                f"(source target_mode={source_config.target_mode}, "
                f"current target_mode={current.target_mode}): {error}"
            ) from error
        if self.runtime.is_main_process:
            print(
                f"Initialized weights from {init_path} "
                f"(source target_mode={source_config.target_mode}, "
                f"current target_mode={current.target_mode})"
            )
        self.runtime.barrier()

    def _resolve_resume_path(self):
        resume_path = self.config.resume_path
        if resume_path is None:
            return None
        if resume_path == "latest":
            resume_path = os.path.join(
                self.config.output_dir,
                "latest.pth"
            )
        return os.path.abspath(resume_path)

    def _restore_resume_semantics(self):
        """Restore immutable model/data/training-noise semantics from
        the checkpoint semantic sidecar before building the model.

        A bare ``--resume latest`` therefore continues with exactly the
        checkpoint's semantics instead of silently adopting newer CLI
        defaults (the epoch23-style drift).  Legacy checkpoints without
        a sidecar are left to fail-fast validation of the fields they
        actually store.  Runs identically on every DDP rank.
        """
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return
        sidecar = load_semantic_sidecar(resume_path)
        if sidecar is None:
            if self.runtime.is_main_process:
                print(
                    "Resume note: no semantic sidecar for",
                    resume_path,
                    "(legacy checkpoint); only fields stored in its "
                    "model config will be validated",
                )
            return
        defaults = dict(asdict(default_training_model()))
        defaults["split"] = "train"
        defaults["condition_mode"] = "sst_mask"
        notices = restore_resume_semantics(
            sidecar,
            self.config,
            defaults,
            explicit_fields=self.config.explicit_resume_fields,
        )
        if self.runtime.is_main_process:
            for notice in notices:
                print(f"Resume notice: {notice}")

    def _resume_training(self):
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return
        checkpoint = self.checkpoints.load(
            resume_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.runtime.device,
            self.runtime.rank,
            self.runtime.world_size
        )
        self.start_epoch = int(checkpoint["epoch"])
        self.global_step = int(checkpoint["global_step"])
        if self.runtime.is_main_process:
            self.history.load()
            print("Resumed checkpoint:", resume_path)
            print(
                "Starting epoch:",
                self.start_epoch + 1,
                "global step:",
                self.global_step
            )

    @staticmethod
    def unpack_batch(batch):
        if isinstance(batch, dict):
            return (
                batch["condition"],
                batch["target"],
                batch["target_mask"]
            )
        return batch[0], batch[1], batch[2]

    def check_batch(
            self,
            condition,
            target,
            target_mask,
        ):
        model_config = self.config.model
        expected_condition = (
            model_config.cond_chans,
            *model_config.image_size
        )
        expected_target = (
            model_config.target_chans,
            *model_config.image_size
        )
        if condition.ndim != 5:
            raise ValueError(
                f"condition shape error: {condition.shape}"
            )
        if target.ndim != 5:
            raise ValueError(
                f"target shape error: {target.shape}"
            )
        if target_mask.ndim != 5:
            raise ValueError(
                f"target_mask shape error: {target_mask.shape}"
            )
        if condition.shape[1:] != expected_condition:
            raise ValueError(
                f"condition must have shape "
                f"[B,{','.join(map(str, expected_condition))}], "
                f"but got {condition.shape}"
            )
        if target.shape[1:] != expected_target:
            raise ValueError(
                f"target must have shape "
                f"[B,{','.join(map(str, expected_target))}], "
                f"but got {target.shape}"
            )
        if target_mask.shape != target.shape:
            raise ValueError(
                f"target_mask shape {target_mask.shape} "
                f"does not match target shape {target.shape}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains NaN or Inf")
        if not torch.isfinite(target).all():
            raise ValueError("target contains NaN or Inf")
        if not torch.isfinite(target_mask).all():
            raise ValueError("target_mask contains NaN or Inf")

    def _move_batch(self, batch):
        condition, target, target_mask = (
            self.unpack_batch(batch)
        )
        move_options = {
            "device": self.runtime.device,
            "dtype": torch.float32,
            "non_blocking": True
        }
        return (
            condition.to(**move_options),
            target.to(**move_options),
            target_mask.to(**move_options)
        )

    def _training_target(self, condition, target):
        if self.config.model.target_mode == "absolute":
            return target
        last_day = condition[
            :,
            self.config.model.input_days - 1:
            self.config.model.input_days
        ]
        return target - last_day

    def _train_epoch(self, epoch):
        self.model.train()
        self.data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_batches = 0
        iterator = tqdm(
            self.data.loader,
            disable=not self.runtime.is_main_process,
            desc=(
                f"epoch {epoch + 1}/"
                f"{self.config.num_epochs}"
            )
        )
        for batch_index, batch in enumerate(iterator):
            condition, target, target_mask = (
                self._move_batch(batch)
            )
            if epoch == 0 and batch_index < 20:
                self.check_batch(
                    condition,
                    target,
                    target_mask
                )
            group_start = (
                batch_index
                // self.config.gradient_accumulation
                * self.config.gradient_accumulation
            )
            group_size = min(
                self.config.gradient_accumulation,
                len(self.data.loader) - group_start
            )
            should_update = (
                (batch_index + 1)
                % self.config.gradient_accumulation == 0
                or (batch_index + 1)
                == len(self.data.loader)
            )
            sync_context = nullcontext()
            if self.runtime.distributed and not should_update:
                sync_context = self.model.no_sync()
            with sync_context:
                with autocast("cuda", enabled=self.amp_enabled):
                    training_target = self._training_target(
                        condition,
                        target
                    )
                    loss = self.model(
                        training_target,
                        condition,
                        target_mask
                    )
                    scaled_loss = loss / group_size
                self.scaler.scale(scaled_loss).backward()
            if should_update:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.runtime.is_main_process:
                    self.history.record_gradient(
                        self.global_step,
                        grad_norm.detach().item()
                    )
            loss_value = loss.detach().item()
            epoch_loss += loss_value
            epoch_batches += 1
            if self.runtime.is_main_process:
                loss_step = (
                    epoch * len(self.data.loader)
                    + batch_index
                    + 1
                )
                self.history.record_loss(
                    loss_step,
                    loss_value
                )
                iterator.set_postfix(
                    loss=f"{loss_value:.6f}",
                    lr=(
                        f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    ),
                    step=self.global_step
                )
        return epoch_loss, epoch_batches

    def _mean_train_loss(
            self,
            epoch_loss,
            epoch_batches,
        ):
        statistics = torch.tensor(
            [epoch_loss, epoch_batches],
            dtype=torch.float64,
            device=self.runtime.device
        )
        if self.runtime.distributed:
            dist.all_reduce(
                statistics,
                op=dist.ReduceOp.SUM
            )
        return (
            statistics[0]
            / statistics[1].clamp_min(1)
        ).item()

    def _finish_epoch(self, epoch, mean_train_loss):
        random_state = (
            self.checkpoints.capture_random_state()
        )
        if self.runtime.distributed:
            random_states = [
                None
                for _ in range(self.runtime.world_size)
            ]
            dist.all_gather_object(
                random_states,
                random_state
            )
        else:
            random_states = [random_state]
        if self.runtime.is_main_process:
            peak_memory = 0.0
            if self.runtime.device.type == "cuda":
                peak_memory = (
                    torch.cuda.max_memory_allocated(
                        self.runtime.device
                    )
                    / 1024 ** 3
                )
            print(
                f"epoch={epoch + 1} "
                f"train_loss={mean_train_loss:.6f} "
                f"peak_memory={peak_memory:.2f}GB"
            )
            self.checkpoints.save(
                os.path.join(
                    self.config.output_dir,
                    "latest.pth"
                ),
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch + 1,
                self.global_step,
                mean_train_loss,
                self.data.dataset,
                random_states
            )
            if (
                    (epoch + 1)
                    % self.config.checkpoint_interval == 0
                ):
                self.checkpoints.save(
                    os.path.join(
                        self.config.output_dir,
                        f"epoch_{epoch + 1:03d}.pth"
                    ),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch + 1,
                    self.global_step,
                    mean_train_loss,
                    self.data.dataset,
                    random_states
                )
            self.history.save()
        self.runtime.barrier()

    def train(self):
        try:
            self.setup()
            for epoch in range(
                    self.start_epoch,
                    self.config.num_epochs
                ):
                epoch_loss, epoch_batches = self._train_epoch(
                    epoch
                )
                mean_train_loss = self._mean_train_loss(
                    epoch_loss,
                    epoch_batches
                )
                self._finish_epoch(
                    epoch,
                    mean_train_loss
                )
        finally:
            self.runtime.cleanup()
