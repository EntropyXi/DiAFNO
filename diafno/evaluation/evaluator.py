import glob
import json
import os

import numpy as np

from .metrics import RunningSSTMetrics


class OSTIAEvaluator:
    def __init__(self, prediction_dir, output_path):
        self.prediction_dir = prediction_dir
        self.output_path = output_path

    @staticmethod
    def ensure_batch_axis(value):
        value = np.asarray(value)
        if value.ndim == 3:
            return value[None, ...]
        if value.ndim != 4:
            raise ValueError(
                f"Expected [lead, H, W] or [batch, lead, H, W], "
                f"got {value.shape}"
            )
        return value

    def run(self):
        paths = sorted(glob.glob(
            os.path.join(self.prediction_dir, "sample_*.npz")
        ))
        if not paths:
            raise FileNotFoundError(
                f"No sample_*.npz found in {self.prediction_dir}"
            )
        overall = RunningSSTMetrics()
        by_lead = None
        num_samples = 0
        for path in paths:
            with np.load(path) as data:
                prediction = self.ensure_batch_axis(
                    data["prediction"]
                )
                target = self.ensure_batch_axis(data["target"])
                mask = self.ensure_batch_axis(data["target_mask"])
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target mismatch in {path}: "
                    f"{prediction.shape} vs {target.shape}"
                )
            if mask.shape != target.shape:
                mask = np.broadcast_to(mask, target.shape)
            if by_lead is None:
                by_lead = [
                    RunningSSTMetrics()
                    for _ in range(prediction.shape[1])
                ]
            if len(by_lead) != prediction.shape[1]:
                raise ValueError(
                    f"Inconsistent lead count in {path}"
                )
            overall.update(prediction, target, mask)
            for lead_index, metrics in enumerate(by_lead):
                metrics.update(
                    prediction[:, lead_index],
                    target[:, lead_index],
                    mask[:, lead_index]
                )
            num_samples += prediction.shape[0]
        result = {
            "num_samples": num_samples,
            "overall": overall.compute(),
            "by_lead_day": {
                str(index + 1): metrics.compute()
                for index, metrics in enumerate(by_lead)
            }
        }
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )
        return result
