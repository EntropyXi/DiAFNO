# 用途：从切块 HDF5 构造同区域的 7 日输入与 15 日目标样本。
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class OSTIADailyDataset(Dataset):
    split_ranges = {
        "train": (0.0, 0.7),
        "val": (0.7, 0.9),
        "test": (0.9, 1.0)
    }

    # 用途：构造数据集：校验参数、探测 HDF5 结构并加载/估计训练集标准化统计量。
    # 参数：输入 h5_path（预处理 HDF5 路径）、split（train/val/test 时间划分）、input_days（输入日数 7）、output_days（目标日数 15）、condition_mode（sst 或 sst_mask）；输出 无（写入实例属性）。
    def __init__(
            self,
            h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask",
        ):
        if split not in self.split_ranges:
            raise ValueError(
                f"split must be one of {tuple(self.split_ranges)}, "
                f"but got {split}"
            )
        if input_days < 1 or output_days < 1:
            raise ValueError(
                "input_days and output_days must be positive"
            )
        if condition_mode not in ("sst", "sst_mask"):
            raise ValueError(
                "condition_mode must be 'sst' or 'sst_mask'"
            )
        self.h5_path = os.path.abspath(h5_path)
        self.split = split
        self.input_days = input_days
        self.output_days = output_days
        self.sequence_days = input_days + output_days
        self.condition_mode = condition_mode
        self.day_offsets = (
            np.arange(self.sequence_days, dtype=np.int64)
        )
        self._h5_file = None
        self._h5_pid = None
        self._inspect_file()
        self.sst_mean, self.sst_std = (
            self._load_or_estimate_normalization()
        )
        self.normalization = {
            "sst_mean": self.sst_mean,
            "sst_std": self.sst_std,
            "temporal_stride_days": 1,
            "source": "training_split_sample"
        }

    # 用途：只读校验 HDF5 数据集形状与时间轴，推断 samples_per_day、总天数和本 split 的日窗口。
    # 参数：无输入；输出 无（结果写入 self.num_rows/samples_per_day/split_start_day 等属性，异常时 fail-fast）。
    def _inspect_file(self):
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(self.h5_path)
        required = ("sst", "mask", "lat", "lon", "time")
        with h5py.File(self.h5_path, "r") as h5_file:
            missing = [
                name for name in required
                if name not in h5_file
            ]
            if missing:
                raise KeyError(
                    f"Missing HDF5 datasets: {missing}"
                )
            sst = h5_file["sst"]
            mask = h5_file["mask"]
            time = h5_file["time"]
            if sst.ndim != 4 or sst.shape[1] != 1:
                raise ValueError(
                    "sst must have shape [N,1,H,W], "
                    f"but got {sst.shape}"
                )
            if mask.shape != (
                sst.shape[0],
                sst.shape[2],
                sst.shape[3]
            ):
                raise ValueError(
                    "mask shape does not match sst: "
                    f"{mask.shape} versus {sst.shape}"
                )
            self.num_rows = sst.shape[0]
            self.image_shape = tuple(sst.shape[2:])
            self.first_time = int(time[0])
            left = 1
            right = self.num_rows
            while left < right:
                middle = (left + right) // 2
                if int(time[middle]) == self.first_time:
                    left = middle + 1
                else:
                    right = middle
            self.samples_per_day = left
            if self.num_rows % self.samples_per_day != 0:
                raise ValueError(
                    "HDF5 rows do not contain complete daily windows"
                )
            self.num_days = (
                self.num_rows // self.samples_per_day
            )
            if int(time[-1]) != (
                self.first_time + self.num_days - 1
            ):
                raise ValueError(
                    "time values must be consecutive daily indices"
                )
            self.chunk_rows = (
                sst.chunks[0] if sst.chunks else 1
            )
            attrs = dict(h5_file.attrs)
        self.total_days = self.num_days
        split_start, split_end = self.split_ranges[self.split]
        self.split_start_day = int(
            self.total_days * split_start
        )
        self.split_end_day = int(
            self.total_days * split_end
        )
        self.sequences_per_window = (
            self.split_end_day
            - self.split_start_day
            - self.sequence_days
            + 1
        )
        if self.sequences_per_window < 1:
            raise ValueError(
                f"{self.split} split is shorter than "
                f"{self.sequence_days} days"
            )
        self._file_sst_mean = attrs.get("sst_mean")
        self._file_sst_std = attrs.get("sst_std")

    @staticmethod
    # 用途：合成有效海洋像素布尔阵（mask 位 2 为 0、数值有限且在 -5~350 内）。
    # 参数：输入 sst（原始 SST 数组）、mask（原始 mask 数组）；输出 同形状 bool 数组（True=有效海洋）。
    def _valid_ocean(sst, mask):
        return (
            ((mask.astype(np.uint8) & 2) == 0)
            & np.isfinite(sst)
            & (sst > -5.0)
            & (sst < 350.0)
        )

    # 用途：确定标准化统计量：优先读 HDF5 属性，否则在 train 段按 chunk 抽样估计。
    # 参数：无输入；输出 (sst_mean, sst_std) 二元组。
    def _load_or_estimate_normalization(self):
        if (
            self._file_sst_mean is not None
            and self._file_sst_std is not None
            and float(self._file_sst_std) > 0
        ):
            return (
                float(self._file_sst_mean),
                float(self._file_sst_std)
            )
        train_end_day = int(
            self.total_days
            * self.split_ranges["train"][1]
        )
        train_end_row = min(
            train_end_day
            * self.samples_per_day,
            self.num_rows
        )
        block_rows = min(
            self.chunk_rows,
            self.samples_per_day
        )
        block_count = 8
        max_start = max(0, train_end_row - block_rows)
        starts = np.linspace(
            0,
            max_start,
            block_count,
            dtype=np.int64
        )
        starts = np.unique(
            (starts // self.chunk_rows) * self.chunk_rows
        )
        value_sum = 0.0
        squared_sum = 0.0
        value_count = 0
        with h5py.File(self.h5_path, "r") as h5_file:
            for start in starts:
                end = min(
                    int(start) + block_rows,
                    train_end_row
                )
                sst = np.asarray(
                    h5_file["sst"][int(start):end, 0],
                    dtype=np.float32
                )
                mask = np.asarray(
                    h5_file["mask"][int(start):end],
                    dtype=np.uint8
                )
                valid = self._valid_ocean(sst, mask)
                values = sst[valid].astype(
                    np.float64,
                    copy=False
                )
                value_sum += values.sum()
                squared_sum += np.square(values).sum()
                value_count += values.size
        if value_count < 2:
            raise ValueError(
                "Could not find valid ocean SST values"
            )
        mean = value_sum / value_count
        variance = max(
            squared_sum / value_count - mean * mean,
            1e-12
        )
        return float(mean), float(np.sqrt(variance))

    # 用途：按进程惰性打开 HDF5 文件句柄（512MB 块缓存，支持 dataloader worker）。
    # 参数：无输入；输出 h5py.File 句柄。
    def _get_file(self):
        pid = os.getpid()
        if (
            self._h5_file is None
            or self._h5_pid != pid
        ):
            self.close()
            self._h5_file = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=512 * 1024 ** 2,
                rdcc_nslots=1000003
            )
            self._h5_pid = pid
        return self._h5_file

    # 用途：返回本 split 的样本总数 = 序列数 × 每日空间块数。
    # 参数：无输入；输出 int 样本数。
    def __len__(self):
        return (
            self.sequences_per_window
            * self.samples_per_day
        )

    # 用途：把负索引折算为正索引并做越界检查。
    # 参数：输入 index（任意整数索引）；输出 规范化后的索引（越界抛 IndexError）。
    def _normalize_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    @staticmethod
    # 用途：把升序索引数组切分为连续段，用于按段批量读 HDF5。
    # 参数：输入 indices（升序 int 数组）；输出 [(start, end), ...] 半开区间列表。
    def _contiguous_runs(indices):
        if indices.size == 0:
            return []
        starts = [0]
        ends = []
        for index in range(1, indices.size):
            if indices[index] != indices[index - 1] + 1:
                ends.append(index)
                starts.append(index)
        ends.append(indices.size)
        return [
            (
                int(indices[start]),
                int(indices[end - 1]) + 1
            )
            for start, end in zip(starts, ends)
        ]

    # 用途：读取同一时间窗口的多个空间块，构造 22 个连续日的条件/目标/mask 样本（含标准化与均值填充）。
    # 参数：输入 sequence_index（时间序列号，0 起）、spatial_indices（空间块号列表）；输出 样本 dict 列表（condition/target/target_mask/metadata）。
    def _load_sequence_batch(
            self,
            sequence_index,
            spatial_indices,
        ):
        start_day = (
            self.split_start_day + sequence_index
        )
        days = (
            start_day + self.day_offsets
        )
        unique_spatial, restore = np.unique(
            spatial_indices,
            return_inverse=True
        )
        runs = self._contiguous_runs(unique_spatial)
        h5_file = self._get_file()
        sst_days = []
        for day in days:
            base = int(day) * self.samples_per_day
            sst_parts = [
                np.asarray(
                    h5_file["sst"][
                        base + start:base + end,
                        0
                    ],
                    dtype=np.float32
                )
                for start, end in runs
            ]
            sst_days.append(
                sst_parts[0] if len(sst_parts) == 1
                else np.concatenate(sst_parts, axis=0)
            )
        sst = np.stack(
            sst_days,
            axis=0
        ).transpose(1, 0, 2, 3)
        del sst_days
        mask_days = []
        for day in days:
            base = int(day) * self.samples_per_day
            mask_parts = [
                np.asarray(
                    h5_file["mask"][
                        base + start:base + end
                    ],
                    dtype=np.uint8
                )
                for start, end in runs
            ]
            mask_days.append(
                mask_parts[0] if len(mask_parts) == 1
                else np.concatenate(mask_parts, axis=0)
            )
        mask = np.stack(
            mask_days,
            axis=0
        ).transpose(1, 0, 2, 3)
        del mask_days
        if not np.array_equal(
                unique_spatial,
                spatial_indices
            ):
            sst = sst[restore]
            mask = mask[restore]
        times = (
            self.first_time + days
        ).astype(
            np.int64,
            copy=False
        )
        valid = self._valid_ocean(sst, mask)
        sst = np.where(
            valid,
            sst,
            self.sst_mean
        )
        sst = (
            (sst - self.sst_mean) / self.sst_std
        ).astype(np.float32, copy=False)
        samples = []
        for batch_index, spatial_index in enumerate(
                spatial_indices
            ):
            sample_sst = sst[batch_index]
            sample_valid = valid[batch_index]
            input_sst = sample_sst[:self.input_days]
            target = sample_sst[self.input_days:]
            target_mask = sample_valid[
                self.input_days:
            ].astype(
                np.float32,
                copy=False
            )
            if self.condition_mode == "sst_mask":
                condition = np.concatenate(
                    (
                        input_sst,
                        sample_valid[
                            self.input_days - 1
                        ][None].astype(
                            np.float32,
                            copy=False
                        )
                    ),
                    axis=0
                )
            else:
                condition = input_sst
            condition = np.ascontiguousarray(
                condition[..., None]
            )
            target = np.ascontiguousarray(
                target[..., None]
            )
            target_mask = np.ascontiguousarray(
                target_mask[..., None]
            )
            metadata = {
                "sequence_index": np.int64(sequence_index),
                "spatial_index": np.int64(spatial_index),
                "input_start_time": np.int64(times[0]),
                "target_start_time": np.int64(
                    times[self.input_days]
                ),
                "target_end_time": np.int64(times[-1])
            }
            samples.append(
                {
                    "condition": torch.from_numpy(condition),
                    "target": torch.from_numpy(target),
                    "target_mask": torch.from_numpy(target_mask),
                    "metadata": metadata
                }
            )
        return samples

    # 用途：批量取样入口：按序列号分组合并读取，减少 HDF5 随机访问。
    # 参数：输入 indices（样本索引数组/列表）；输出 与索引顺序一致的样本 dict 列表。
    def __getitems__(self, indices):
        indices = np.asarray(
            [
                self._normalize_index(int(index))
                for index in indices
            ],
            dtype=np.int64
        )
        if indices.size == 0:
            return []
        sequence_indices = (
            indices // self.samples_per_day
        )
        spatial_indices = (
            indices % self.samples_per_day
        )
        samples = [None] * indices.size
        for sequence_index in np.unique(sequence_indices):
            positions = np.flatnonzero(
                sequence_indices == sequence_index
            )
            sequence_samples = self._load_sequence_batch(
                int(sequence_index),
                spatial_indices[positions]
            )
            for position, sample in zip(
                    positions,
                    sequence_samples
                ):
                samples[int(position)] = sample
        return samples

    # 用途：单样本入口，内部委托 __getitems__。
    # 参数：输入 index（样本索引）；输出 样本 dict（condition/target/target_mask/metadata）。
    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    # 用途：把模型空间 SST 反标准化回开尔文。
    # 参数：输入 value（标准化值）；输出 value*std+mean。
    def inverse_transform_sst(self, value):
        return value * self.sst_std + self.sst_mean

    # 用途：关闭 HDF5 句柄并清空缓存引用。
    # 参数：无输入；输出 无。
    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
        self._h5_file = None
        self._h5_pid = None

    # 用途：pickle 前剔除文件句柄，保证 dataloader worker 可序列化。
    # 参数：无输入；输出 去掉 _h5_file/_h5_pid 的状态字典。
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        return state

    # 用途：对象销毁时确保关闭 HDF5 句柄。
    # 参数：无输入；输出 无。
    def __del__(self):
        self.close()
