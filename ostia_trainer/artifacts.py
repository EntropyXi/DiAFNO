from .normalization import NormalizationState
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel as DDP


class TrainingHistory:
    def __init__(self, output_dir, max_grad_norm):
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.loss_steps = []
        self.loss_values = []
        self.gradient_steps = []
        self.gradient_norms = []

    def record_loss(self, step, value):
        self.loss_steps.append(step)
        self.loss_values.append(value)

    def record_gradient(self, step, value):
        self.gradient_steps.append(step)
        self.gradient_norms.append(value)

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
        ):
        normalization = NormalizationState.from_dataset(dataset)
        checkpoint = {
            "model": self.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "normalization": normalization,
            "config": self.config.model.to_checkpoint(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate()
        }
        torch.save(checkpoint, path)
        NormalizationState.save(
            normalization,
            os.path.dirname(path)
        )
