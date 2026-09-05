# 用途：累计有效像素的 SST 误差、相关系数及 persistence skill。
import math

import numpy as np


class RunningSSTMetrics:
    def __init__(self):
        self.count = 0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_error = 0.0
        self.sum_prediction = 0.0
        self.sum_target = 0.0
        self.sum_prediction_squared = 0.0
        self.sum_target_squared = 0.0
        self.sum_product = 0.0
        self.prediction_min = math.inf
        self.prediction_max = -math.inf
        self.target_min = math.inf
        self.target_max = -math.inf

    def update(self, prediction, target, mask):
        valid = (
            np.isfinite(prediction)
            & np.isfinite(target)
            & (mask > 0)
        )
        prediction = prediction[valid].astype(np.float64)
        target = target[valid].astype(np.float64)
        if prediction.size == 0:
            return
        error = prediction - target
        self.count += int(prediction.size)
        self.sum_abs_error += float(np.abs(error).sum())
        self.sum_squared_error += float(np.square(error).sum())
        self.sum_error += float(error.sum())
        self.sum_prediction += float(prediction.sum())
        self.sum_target += float(target.sum())
        self.sum_prediction_squared += float(np.square(prediction).sum())
        self.sum_target_squared += float(np.square(target).sum())
        self.sum_product += float((prediction * target).sum())
        self.prediction_min = min(
            self.prediction_min,
            float(prediction.min())
        )
        self.prediction_max = max(
            self.prediction_max,
            float(prediction.max())
        )
        self.target_min = min(
            self.target_min,
            float(target.min())
        )
        self.target_max = max(
            self.target_max,
            float(target.max())
        )

    def compute(self):
        if self.count == 0:
            return {
                "mae": None,
                "mse": None,
                "rmse": None,
                "bias": None,
                "correlation": None,
                "acc": None,
                "prediction_mean": None,
                "prediction_std": None,
                "prediction_min": None,
                "prediction_max": None,
                "target_mean": None,
                "target_std": None,
                "std_ratio": None,
                "target_min": None,
                "target_max": None,
                "valid_pixels": 0
            }
        covariance = (
            self.sum_product
            - self.sum_prediction * self.sum_target / self.count
        )
        prediction_variance = (
            self.sum_prediction_squared
            - self.sum_prediction ** 2 / self.count
        )
        target_variance = (
            self.sum_target_squared
            - self.sum_target ** 2 / self.count
        )
        denominator = math.sqrt(
            max(prediction_variance, 0.0)
            * max(target_variance, 0.0)
        )
        correlation = (
            covariance / denominator
            if denominator > 0.0
            else None
        )
        prediction_std = math.sqrt(
            max(prediction_variance / self.count, 0.0)
        )
        target_std = math.sqrt(
            max(target_variance / self.count, 0.0)
        )
        return {
            "mae": self.sum_abs_error / self.count,
            "mse": self.sum_squared_error / self.count,
            "rmse": math.sqrt(
                self.sum_squared_error / self.count
            ),
            "bias": self.sum_error / self.count,
            "correlation": correlation,
            "acc": correlation,
            "prediction_mean": (
                self.sum_prediction / self.count
            ),
            "prediction_std": prediction_std,
            "prediction_min": self.prediction_min,
            "prediction_max": self.prediction_max,
            "target_mean": self.sum_target / self.count,
            "target_std": target_std,
            "std_ratio": (
                prediction_std / target_std
                if target_std > 0.0
                else None
            ),
            "target_min": self.target_min,
            "target_max": self.target_max,
            "valid_pixels": self.count
        }


def persistence_skill(model_metrics, persistence_metrics):
    model_mse = model_metrics.get("mse")
    persistence_mse = persistence_metrics.get("mse")
    if (
            model_mse is None
            or persistence_mse is None
            or persistence_mse <= 0.0
        ):
        return None
    return 1.0 - model_mse / persistence_mse
