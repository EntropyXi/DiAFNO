# 用途：读取推理文件，累计并输出离线 SST 指标。
import glob
import json
import os

import numpy as np

from .metrics import RunningSSTMetrics


class OSTIAEvaluator:
    # 用途：初始化离线评估器：登记预测目录与输出路径。
    # 参数：输入 prediction_dir（已保存预测的目录）、output_path（指标输出 JSON 路径）；输出 无。
    def __init__(self, prediction_dir, output_path):
        self.prediction_dir = prediction_dir
        self.output_path = output_path

    @staticmethod
    # 用途：确保数组带 batch 轴（单样本时补一维，static 方法）。
    # 参数：输入 value（数组）；输出 至少二维的数组。
    def ensure_batch_axis(value):
        value = np.asarray(value)
        if value.ndim in (4, 5) and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim == 3:
            return value[None, ...]
        if value.ndim != 4:
            raise ValueError(
                "Expected [lead,H,W], [lead,H,W,1], "
                "[batch,lead,H,W] or [batch,lead,H,W,1], "
                f"got {value.shape}"
            )
        return value

    # 用途：遍历已保存的预测文件，累计逐 lead 指标并写评估 JSON。
    # 参数：无输入（读实例配置）；输出 无（结果落盘）。
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
