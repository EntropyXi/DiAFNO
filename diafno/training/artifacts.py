from .normalization import NormalizationState
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from deterministic_iafno.checkpoint_semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    build_semantic_manifest,
    get_compatible_mismatches,
    validate_semantic_manifest,
)


class TrainingHistory:
    def __init__(self, output_dir, max_grad_norm):
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.loss_steps = []
        self.loss_values = []
        self.gradient_steps = []
        self.gradient_norms = []
        self.skipped_optimizer_steps = []

    def record_loss(self, step, value):
        self.loss_steps.append(step)
        self.loss_values.append(value)

    def record_gradient(self, step, value):
        self.gradient_steps.append(step)
        self.gradient_norms.append(value)

    def record_skipped_step(self, step):
        self.skipped_optimizer_steps.append(step)

    def load(self):
        path = os.path.join(
            self.output_dir,
            "training_curves.npz"
        )
        if not os.path.isfile(path):
            return
        with np.load(path) as history:
            self.loss_steps = history[
                "loss_steps"
            ].astype(np.int64).tolist()
            self.loss_values = history[
                "loss_values"
            ].astype(np.float32).tolist()
            self.gradient_steps = history[
                "gradient_steps"
            ].astype(np.int64).tolist()
            self.gradient_norms = history[
                "gradient_norms"
            ].astype(np.float32).tolist()
            if "skipped_optimizer_steps" in history:
                self.skipped_optimizer_steps = history[
                    "skipped_optimizer_steps"
                ].astype(np.int64).tolist()

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        np.savez(
            os.path.join(
                self.output_dir,
                "training_curves.npz"
            ),
            loss_steps=np.asarray(
                self.loss_steps,
                dtype=np.int64
            ),
            loss_values=np.asarray(
                self.loss_values,
                dtype=np.float32
            ),
            gradient_steps=np.asarray(
                self.gradient_steps,
                dtype=np.int64
            ),
            gradient_norms=np.asarray(
                self.gradient_norms,
                dtype=np.float32
            ),
            skipped_optimizer_steps=np.asarray(
                self.skipped_optimizer_steps,
                dtype=np.int64
            )
        )
        self._save_loss_curve()
        self._save_gradient_curve()

    def _save_loss_curve(self):
        if not self.loss_values:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.loss_steps,
            self.loss_values,
            linewidth=0.8
        )
        plt.xlabel("Training Batch")
        plt.ylabel("EDM Training Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "training_loss_curve.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    def _save_gradient_curve(self):
        if not self.gradient_norms:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.gradient_steps,
            self.gradient_norms,
            linewidth=0.8
        )
        plt.axhline(
            self.max_grad_norm,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"clip threshold = {self.max_grad_norm}"
        )
        plt.xlabel("Optimizer Step")
        plt.ylabel("Gradient Norm Before Clipping")
        plt.title("Training Gradient Norm")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "gradient_norm_curve.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


class CheckpointManager:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def unwrap_model(model):
        if isinstance(model, DDP):
            return model.module
        return model

    @staticmethod
    def capture_random_state():
        cuda_random_state = []
        if torch.cuda.is_available():
            cuda_random_state = torch.cuda.get_rng_state_all()
        return {
            "torch": torch.get_rng_state(),
            "cuda": cuda_random_state,
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }

    @staticmethod
    def restore_random_state(random_state):
        torch.set_rng_state(
            random_state["torch"].cpu()
        )
        if (
                torch.cuda.is_available()
                and random_state["cuda"]
            ):
            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state in random_state["cuda"]
                ]
            )
        np.random.set_state(random_state["numpy"])
        random.setstate(random_state["python"])

    def load(
            self,
            path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            rank,
            world_size,
        ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
        )
        compatible_mismatches = get_compatible_mismatches(
            checkpoint,
            self.config,
            world_size=world_size,
        )
        desired_scheduler = {
            "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min,
            "base_lrs": list(scheduler.base_lrs),
        }
        semantic_warnings = validate_semantic_manifest(
            checkpoint,
            self.config,
            world_size=world_size,
            allow_compatible_override=(
                self.config.allow_resume_override
            ),
        )
        if rank == 0:
            for warning in semantic_warnings:
                print(f"Resume warning: {warning}")
        checkpoint_config = checkpoint.get("config", {})
        daily_time_fields = (
            "input_days",
            "output_days"
        )
        legacy_time_fields = (
            "input_months",
            "output_months"
        )
        if all(
                field in checkpoint_config
                for field in daily_time_fields
            ):
            checkpoint_time_config = {
                field: checkpoint_config[field]
                for field in daily_time_fields
            }
        elif all(
                field in checkpoint_config
                for field in legacy_time_fields
            ):
            checkpoint_time_config = {
                "input_days": checkpoint_config["input_months"],
                "output_days": checkpoint_config["output_months"]
            }
        else:
            raise ValueError(
                "checkpoint uses the old weekly time indexing; "
                "daily OSTIA training must start without --resume"
            )
        for field in daily_time_fields:
            expected = getattr(self.config.model, field)
            actual = checkpoint_time_config[field]
            if actual != expected:
                raise ValueError(
                    f"checkpoint {field}={actual} does not match "
                    f"current {field}={expected}"
                )
        checkpoint_target_mode = checkpoint_config.get(
            "target_mode",
            "absolute"
        )
        if checkpoint_target_mode != self.config.model.target_mode:
            raise ValueError(
                "checkpoint target_mode="
                f"{checkpoint_target_mode} does not match current "
                f"target_mode={self.config.model.target_mode}; "
                "residual training must start without --resume"
            )
        self.unwrap_model(model).load_state_dict(
            checkpoint["model"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )
        if (
                compatible_mismatches
                and self.config.allow_resume_override
            ):
            self._apply_compatible_overrides(
                optimizer,
                scheduler,
                compatible_mismatches,
                desired_scheduler,
            )
        scaler.load_state_dict(checkpoint["scaler"])
        random_states = checkpoint.get("random_states")
        if random_states and rank < len(random_states):
            self.restore_random_state(
                random_states[rank]
            )
        elif rank == 0:
            legacy_random_state = {
                "torch": checkpoint["torch_random_state"],
                "cuda": checkpoint["cuda_random_state"],
                "numpy": checkpoint["numpy_random_state"],
                "python": checkpoint["python_random_state"]
            }
            self.restore_random_state(
                legacy_random_state
            )
        return checkpoint

    def _apply_compatible_overrides(
            self,
            optimizer,
            scheduler,
            mismatches,
            desired_scheduler,
        ):
        """Apply explicitly accepted CLI optimizer/schedule semantics.

        Optimizer and scheduler states are loaded first so moments and
        progress are preserved.  Only reviewed hyperparameters are then
        replaced, keeping the future checkpoint manifest aligned with
        the actual resumed objects.
        """
        if "learning_rate" in mismatches:
            for group in optimizer.param_groups:
                group["lr"] = self.config.learning_rate
                group["initial_lr"] = self.config.learning_rate
        if "weight_decay" in mismatches:
            for group in optimizer.param_groups:
                group["weight_decay"] = self.config.weight_decay

        schedule_fields = {
            "learning_rate",
            "min_learning_rate",
            "num_epochs",
            "samples_per_epoch",
            "effective_global_batch",
            "optimizer_steps_per_epoch",
        }
        if schedule_fields.intersection(mismatches):
            desired_t_max = int(desired_scheduler["T_max"])
            if desired_t_max <= int(scheduler.last_epoch):
                raise ValueError(
                    "reviewed resume schedule has no remaining steps: "
                    f"T_max={desired_t_max}, "
                    f"checkpoint last_epoch={scheduler.last_epoch}"
                )
            scheduler.T_max = desired_t_max
            scheduler.eta_min = float(
                desired_scheduler["eta_min"]
            )
            scheduler.base_lrs = list(
                desired_scheduler["base_lrs"]
            )
            if "learning_rate" in mismatches:
                scheduler.base_lrs = [
                    self.config.learning_rate
                    for _ in optimizer.param_groups
                ]
            scheduler._last_lr = [
                group["lr"]
                for group in optimizer.param_groups
            ]

    def save(
            self,
            path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            train_loss,
            dataset,
            random_states,
            skipped_optimizer_steps=0,
            skipped_optimizer_step_numbers=None,
        ):
        normalization = NormalizationState.from_dataset(dataset)
        main_random_state = random_states[0]
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": self.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "skipped_optimizer_steps": int(
                skipped_optimizer_steps
            ),
            "skipped_optimizer_step_numbers": [
                int(step)
                for step in (
                    skipped_optimizer_step_numbers or []
                )
            ],
            "normalization": normalization,
            "config": self.config.model.to_checkpoint(),
            "semantic_manifest": build_semantic_manifest(
                self.config,
                world_size=len(random_states),
            ),
            "random_states": random_states,
            "torch_random_state": main_random_state["torch"],
            "cuda_random_state": main_random_state["cuda"],
            "numpy_random_state": main_random_state["numpy"],
            "python_random_state": main_random_state["python"]
        }
        torch.save(checkpoint, path)
        semantics_path = path + ".semantics.json"
        with open(
                semantics_path,
                "w",
                encoding="utf-8"
            ) as file:
            json.dump(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "config": self.config.model.to_checkpoint(),
                    "semantic_manifest": checkpoint[
                        "semantic_manifest"
                    ],
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        NormalizationState.save(
            normalization,
            os.path.dirname(path)
        )
