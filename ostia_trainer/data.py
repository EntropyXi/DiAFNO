import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Sampler

from ostia_dataset import OSTIAMonthlyDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedSpatialBlockSampler(Sampler):
    def __init__(
            self,
            dataset,
            samples_per_epoch,
            batch_size,
            num_replicas,
            rank,
            seed=123,
        ):
        self.dataset_size = len(dataset)
        self.sequences_per_window = (
            dataset.sequences_per_window
        )
        self.samples_per_month = dataset.samples_per_month
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.blocks_per_sequence = (
            self.samples_per_month // self.batch_size
        )
        if self.blocks_per_sequence < 1:
            raise ValueError(
                "batch_size is larger than samples_per_month"
            )
        self.available_blocks = (
            self.sequences_per_window
            * self.blocks_per_sequence
        )
        requested_samples = min(
            samples_per_epoch,
            self.dataset_size
        )
        requested_blocks = min(
            requested_samples // self.batch_size,
            self.available_blocks
        )
        self.total_blocks = (
            requested_blocks // self.num_replicas
            * self.num_replicas
        )
        if self.total_blocks == 0:
            raise ValueError(
                "samples_per_epoch must contain at least "
                "one full batch per rank"
            )
        self.total_size = (
            self.total_blocks * self.batch_size
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
        block_indices = torch.randperm(
            self.available_blocks,
            generator=generator
        )[:self.total_blocks]
        block_indices = block_indices[
            self.rank:self.total_blocks:self.num_replicas
        ]
        spatial_offset = (
            self.epoch * self.batch_size
            % self.samples_per_month
        )
        indices = []
        for block_index in block_indices.tolist():
            sequence_index, spatial_block = divmod(
                block_index,
                self.blocks_per_sequence
            )
            spatial_start = (
                spatial_offset
                + spatial_block * self.batch_size
            ) % self.samples_per_month
            spatial_indices = (
                spatial_start
                + np.arange(self.batch_size)
            ) % self.samples_per_month
            indices.extend(
                (
                    sequence_index * self.samples_per_month
                    + spatial_indices
                ).tolist()
            )
        return iter(indices)

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
        self.dataset = OSTIAMonthlyDataset(
            h5_path=self.config.train_h5_path,
            split=self.config.split,
            input_months=model_config.input_months,
            output_months=model_config.output_months,
            condition_mode=self.config.condition_mode
        )
        self.sampler = DistributedSpatialBlockSampler(
            dataset=self.dataset,
            samples_per_epoch=self.config.samples_per_epoch,
            batch_size=self.config.batch_per_gpu,
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
