import json
import os

import numpy as np
import torch


class InferenceSampleWriter:
    def __init__(
            self,
            output_dir,
            checkpoint_path,
            sampling_steps,
            save_members,
            compress,
        ):
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.sampling_steps = sampling_steps
        self.save_members = save_members
        self.compress = compress
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def metadata_item(metadata, index):
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            result = {}
            for key, value in metadata.items():
                if torch.is_tensor(value):
                    selected = value[index]
                    if selected.ndim == 0:
                        selected = selected.item()
                    else:
                        selected = (
                            selected.detach()
                            .cpu()
                            .tolist()
                        )
                elif isinstance(value, np.ndarray):
                    selected = value[index].tolist()
                elif isinstance(value, (list, tuple)):
                    selected = value[index]
                else:
                    selected = value
                result[key] = selected
            return result
        if isinstance(metadata, (list, tuple)):
            return metadata[index]
        return {"metadata": str(metadata)}

    @staticmethod
    def to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return (
                value.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return np.asarray(value, dtype=np.float32)

    def save(
            self,
            index,
            prediction,
            target,
            target_mask,
            ensemble_members,
            metadata,
        ):
        prediction = self.to_numpy(prediction)
        target = self.to_numpy(target)
        target_mask = self.to_numpy(target_mask)
        ensemble_members = self.to_numpy(
            ensemble_members
        )
        if target_mask is not None:
            valid = target_mask > 0
            prediction = np.where(
                valid,
                prediction,
                np.nan
            )
            if target is not None:
                target = np.where(
                    valid,
                    target,
                    np.nan
                )
        payload = {
            "prediction": prediction,
            "metadata_json": np.asarray(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
            ),
            "checkpoint": np.asarray(
                self.checkpoint_path
            ),
            "sampling_steps": np.asarray(
                self.sampling_steps,
                dtype=np.int32
            )
        }
        if target is not None:
            payload["target"] = target
        if target_mask is not None:
            payload["target_mask"] = target_mask
        if ensemble_members is not None:
            payload["ensemble_model_space"] = (
                ensemble_members
            )
        output_path = os.path.join(
            self.output_dir,
            f"sample_{index:08d}.npz"
        )
        if self.compress:
            np.savez_compressed(
                output_path,
                **payload
            )
        else:
            np.savez(
                output_path,
                **payload
            )
