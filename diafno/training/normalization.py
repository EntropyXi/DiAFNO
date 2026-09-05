# 用途：管理训练集标准化统计量及张量变换。
import json
import os

import numpy as np
import torch


class NormalizationState:
    @staticmethod
    # 用途：把数值转换为 JSON 可序列化的 float（static 方法）。
    # 参数：输入 value（标量/数组）；输出 float。
    def _serializable(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size == 1:
            return float(array.reshape(-1)[0])
        return array.tolist()

    @classmethod
    # 用途：从数据集实例提取标准化统计量并构造状态对象（类方法）。
    # 参数：输入 dataset（OSTIADailyDataset）；输出 NormalizationState 实例。
    def from_dataset(cls, dataset):
        state = getattr(dataset, "normalization", None)
        if hasattr(state, "to_dict"):
            state = state.to_dict()
        if isinstance(state, dict):
            return {
                key: cls._serializable(value)
                for key, value in state.items()
            }
        mean = getattr(dataset, "sst_mean", None)
        std = getattr(dataset, "sst_std", None)
        if mean is None or std is None:
            return None
        return {
            "sst_mean": cls._serializable(mean),
            "sst_std": cls._serializable(std)
        }

    @staticmethod
    # 用途：把标准化统计量写入 normalization.json（含来源与划分语义）。
    # 参数：输入 state（NormalizationState）、output_dir（输出目录）；输出 无。
    def save(state, output_dir):
        if state is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "normalization.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                state,
                file,
                ensure_ascii=False,
                indent=2
            )
