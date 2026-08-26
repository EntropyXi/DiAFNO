import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Sampler

from ostia_dataset import OSTIAWeeklyDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RandomDistributedSubsetSampler(Sampler):
    def __init__(
            self,
            dataset,
            samples_per_epoch,
            num_replicas,
            rank,
            seed=123,
        ):
        self.dataset_size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        requested_samples = min(
            samples_per_epoch,
            self.dataset_size
        )
        self.total_size = (
            requested_samples
            // self.num_replicas
            * self.num_replicas
        )
        if self.total_size == 0:
            raise ValueError(
                "samples_per_epoch is smaller than world_size"
            )
        self.num_samples = (
            self.total_size
            // self.num_replicas
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size,
            generator=generator
        )[:self.total_size]
        indices = indices[
            self.rank:self.total_size:self.num_replicas
        ]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class OSTIATrainingData:
    def __init__(self, config, runtime):
        self.config = config
        self.runtime = runtime
        self.dataset = None
        self.sampler = None
        self.loader = None

    def setup(self):
        model_config = self.config.model
        self.dataset = OSTIAWeeklyDataset(
            h5_path=self.config.train_h5_path,
            split=self.config.split,
            input_weeks=model_config.input_weeks,
            output_weeks=model_config.output_weeks,
            condition_mode=self.config.condition_mode
        )
        self.sampler = RandomDistributedSubsetSampler(
            dataset=self.dataset,
            samples_per_epoch=self.config.samples_per_epoch,
            num_replicas=self.runtime.world_size,
            rank=self.runtime.rank,
            seed=self.config.seed
        )
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_per_gpu,
            sampler=self.sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=(
                self.config.num_workers > 0
            ),
            prefetch_factor=(
                self.config.prefetch_factor
                if self.config.num_workers > 0
                else None
            ),
            worker_init_fn=seed_worker,
            drop_last=False
        )
        return self
