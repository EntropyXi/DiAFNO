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
        ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
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
        ):
        normalization = NormalizationState.from_dataset(dataset)
        main_random_state = random_states[0]
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
            "random_states": random_states,
            "torch_random_state": main_random_state["torch"],
            "cuda_random_state": main_random_state["cuda"],
            "numpy_random_state": main_random_state["numpy"],
            "python_random_state": main_random_state["python"]
        }
        torch.save(checkpoint, path)
        NormalizationState.save(
            normalization,
            os.path.dirname(path)
        )
