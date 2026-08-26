import json
import os

import numpy as np
import torch


class NormalizationState:
    @staticmethod
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
