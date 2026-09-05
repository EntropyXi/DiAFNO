# 用途：构造训练采样器与 DataLoader，并组织批量 HDF5 读取。
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Sampler

from ..data.ostia import (
    PROVENANCE_FIELDS,
    OSTIADailyDataset,
    copy_dataset_provenance,
)


# 用途：dataloader worker 初始化钩子：按 worker 种子重设随机态，保证数据顺序可复现。
# 参数：输入 worker_id（worker 序号）；输出 无。
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedSpatialBlockSampler(Sampler):
    # 用途：初始化分布式空间块采样器：按 seed 确定整 epoch 的样本计划并按 rank 切分。
    # 参数：输入 dataset（数据集）、samples_per_epoch（每 epoch 样本数）、batch_size（每 rank 批大小）、num_replicas/rank（DDP 拓扑）、seed（随机种子）；输出 无。
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
        self.samples_per_day = dataset.samples_per_day
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.blocks_per_sequence = (
            self.samples_per_day // self.batch_size
        )
        if self.blocks_per_sequence < 1:
            raise ValueError(
                "batch_size is larger than samples_per_day"
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

    # 用途：设置 epoch 号，使每个 epoch 的采样计划不同且可复现。
    # 参数：输入 epoch（epoch 号）；输出 无。
    def set_epoch(self, epoch):
        self.epoch = epoch

    # 用途：产出本 rank 当前 epoch 的批索引：样本按空间块分组打乱后顺序成批。
    # 参数：无输入；输出 迭代器，逐批产出索引张量。
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
            % self.samples_per_day
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
            ) % self.samples_per_day
            spatial_indices = (
                spatial_start
                + np.arange(self.batch_size)
            ) % self.samples_per_day
            indices.extend(
                (
                    sequence_index * self.samples_per_day
                    + spatial_indices
                ).tolist()
            )
        return iter(indices)

    # 用途：返回本 rank 每个 epoch 的批数。
    # 参数：无输入；输出 int 批数。
    def __len__(self):
        return self.num_samples


class OSTIATrainingData:
    # 用途：保存配置与运行时引用，数据组件延后到 setup() 构建。
    # 参数：输入 config（训练配置）、runtime（分布式运行时）；输出 无。
    def __init__(self, config, runtime):
        self.config = config
        self.runtime = runtime
        self.dataset = None
        self.sampler = None
        self.loader = None

    # 用途：构建数据集、采样器与 DataLoader（含 seed_worker 与分布式切分）。
    # 参数：无输入；输出 无（组件写入实例属性）。
    def setup(self):
        model_config = self.config.model
        # The training-level condition switch is canonical for fresh
        # runs; adopt it (channel count/names/version) before the
        # dataset and the model are built so both sides always share
        # one construction logic.  Resume runs restore condition_mode
        # onto the training config first, so this sync is a no-op for
        # them.
        model_config.adopt_condition_mode(
            self.config.condition_mode
        )
        self.dataset = OSTIADailyDataset(
            h5_path=self.config.train_h5_path,
            split=self.config.split,
            input_days=model_config.input_days,
            output_days=model_config.output_days,
            condition_mode=model_config.condition_mode,
            data_manifest=self.config.data_manifest_path
        )
        if self.dataset.condition_chans != model_config.cond_chans:
            raise ValueError(
                "condition channel contract mismatch between the "
                "dataset and the model config: dataset condition_mode="
                f"{self.dataset.condition_mode!r} produces "
                f"{self.dataset.condition_chans} channels but the "
                f"model declares cond_chans={model_config.cond_chans}; "
                "the condition schema must agree before training"
            )
        # Persist the HDF5-proven date/lat-lon/time-axis facts on the
        # model config so the checkpoint and sidecar carry exactly
        # what this file (and its data manifest, when one is used)
        # proved; legacy modes keep None.  A resume already restored
        # the checkpoint provenance from its sidecar before the
        # dataset was built: it must match the current HDF5 exactly
        # and is never silently overwritten; only a fresh run writes
        # the provenance.
        if self.config.resume_path is not None:
            mismatches = {}
            for field in PROVENANCE_FIELDS:
                recorded = getattr(model_config, field, None)
                current = getattr(self.dataset, field)
                if recorded is None:
                    if current is not None:
                        raise ValueError(
                            "checkpoint has no recorded provenance "
                            f"({field}=None) but the current HDF5 "
                            "requires it; refusing to resume without "
                            "provable date, geospatial or time-axis "
                            "semantics"
                        )
                    # Legacy manifest with a legacy (non-geo) dataset:
                    # nothing was recorded and nothing to compare.
                    continue
                if recorded != current:
                    mismatches[field] = {
                        "checkpoint": recorded,
                        "current_hdf5": current,
                    }
            if mismatches:
                raise ValueError(
                    "checkpoint provenance does not match the current "
                    "HDF5; refusing to resume with different date, "
                    "geospatial or time-mapping semantics "
                    f"(checkpoint vs current HDF5): {mismatches}"
                )
        else:
            copy_dataset_provenance(model_config, self.dataset)
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
