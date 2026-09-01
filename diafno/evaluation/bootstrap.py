import math

import numpy as np


def _validate_inputs(
        model_sse,
        persistence_sse,
        valid_counts,
        initialization_times,
        block_days,
        replicates,
        confidence_level,
    ):
    model_sse = np.asarray(model_sse, dtype=np.float64)
    persistence_sse = np.asarray(
        persistence_sse,
        dtype=np.float64
    )
    valid_counts = np.asarray(valid_counts, dtype=np.float64)
    initialization_times = np.asarray(
        initialization_times,
        dtype=np.int64
    )
    if model_sse.ndim != 2:
        raise ValueError("paired SSE arrays must have shape [sample, lead]")
    if persistence_sse.shape != model_sse.shape:
        raise ValueError("model and persistence SSE shapes must match")
    if valid_counts.shape != model_sse.shape:
        raise ValueError("valid-count shape must match SSE shape")
    if initialization_times.shape != (model_sse.shape[0],):
        raise ValueError("initialization_times must have shape [sample]")
    if model_sse.shape[0] < 1 or model_sse.shape[1] < 1:
        raise ValueError("paired bootstrap requires at least one sample and lead")
    if not np.all(np.isfinite(model_sse)):
        raise ValueError("model SSE contains non-finite values")
    if not np.all(np.isfinite(persistence_sse)):
        raise ValueError("persistence SSE contains non-finite values")
    if not np.all(np.isfinite(valid_counts)):
        raise ValueError("valid counts contain non-finite values")
    if np.any(model_sse < 0) or np.any(persistence_sse < 0):
        raise ValueError("SSE values must be non-negative")
    if np.any(valid_counts < 0):
        raise ValueError("valid counts must be non-negative")
    if block_days < 1:
        raise ValueError("block_days must be positive")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return (
        model_sse,
        persistence_sse,
        valid_counts,
        initialization_times,
    )


def _point_metrics(model_sse, persistence_sse, valid_counts):
    count = valid_counts.sum(axis=0)
    if np.any(count <= 0):
        raise ValueError("every lead must contain at least one valid pixel")
    model_mse = model_sse.sum(axis=0) / count
    persistence_mse = persistence_sse.sum(axis=0) / count
    if np.any(persistence_mse <= 0):
        raise ValueError("persistence MSE must be positive for skill")
    overall_count = count.sum()
    overall_model_mse = model_sse.sum() / overall_count
    overall_persistence_mse = persistence_sse.sum() / overall_count
    return {
        "model_rmse": math.sqrt(overall_model_mse),
        "persistence_rmse": math.sqrt(overall_persistence_mse),
        "rmse_difference": (
            math.sqrt(overall_model_mse)
            - math.sqrt(overall_persistence_mse)
        ),
        "mse_skill": 1.0 - (
            overall_model_mse / overall_persistence_mse
        ),
        "by_lead": {
            "model_rmse": np.sqrt(model_mse),
            "persistence_rmse": np.sqrt(persistence_mse),
            "rmse_difference": (
                np.sqrt(model_mse) - np.sqrt(persistence_mse)
            ),
            "mse_skill": 1.0 - model_mse / persistence_mse,
        },
    }


def _interval(values, confidence_level):
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        values,
        (alpha, 1.0 - alpha),
        axis=0
    )
    return lower, upper


def paired_temporal_block_bootstrap(
        model_sse,
        persistence_sse,
        valid_counts,
        initialization_times,
        *,
        block_days=22,
        replicates=10000,
        confidence_level=0.95,
        seed=123,
        block_origin_time=None,
    ):
    """Paired temporal-block bootstrap for model-vs-persistence errors.

    Every selected sample contributes model and persistence errors over
    exactly the same pixels.  Initializations in the same non-overlapping
    temporal block are resampled together, preserving both the pairing and
    within-block spatial/temporal dependence.
    """
    (
        model_sse,
        persistence_sse,
        valid_counts,
        initialization_times,
    ) = _validate_inputs(
        model_sse,
        persistence_sse,
        valid_counts,
        initialization_times,
        block_days,
        replicates,
        confidence_level,
    )
    if block_origin_time is None:
        block_origin_time = int(initialization_times.min())
    block_ids = np.floor_divide(
        initialization_times - int(block_origin_time),
        int(block_days)
    )
    unique_blocks, inverse = np.unique(
        block_ids,
        return_inverse=True
    )
    num_blocks = int(unique_blocks.size)
    if num_blocks < 2:
        raise ValueError("paired bootstrap requires at least two time blocks")
    num_leads = model_sse.shape[1]
    block_model_sse = np.zeros((num_blocks, num_leads), dtype=np.float64)
    block_persistence_sse = np.zeros_like(block_model_sse)
    block_valid_counts = np.zeros_like(block_model_sse)
    block_sample_counts = np.zeros(num_blocks, dtype=np.int64)
    np.add.at(block_model_sse, inverse, model_sse)
    np.add.at(block_persistence_sse, inverse, persistence_sse)
    np.add.at(block_valid_counts, inverse, valid_counts)
    np.add.at(block_sample_counts, inverse, 1)

    point = _point_metrics(
        block_model_sse,
        block_persistence_sse,
        block_valid_counts
    )
    overall_difference = np.empty(replicates, dtype=np.float64)
    overall_skill = np.empty(replicates, dtype=np.float64)
    lead_difference = np.empty(
        (replicates, num_leads),
        dtype=np.float64
    )
    lead_skill = np.empty_like(lead_difference)
    generator = np.random.default_rng(seed)
    chunk_size = min(512, replicates)
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        sampled = generator.integers(
            0,
            num_blocks,
            size=(end - start, num_blocks)
        )
        model_sum = block_model_sse[sampled].sum(axis=1)
        persistence_sum = block_persistence_sse[sampled].sum(axis=1)
        count_sum = block_valid_counts[sampled].sum(axis=1)
        if np.any(count_sum <= 0):
            raise ValueError("a bootstrap replicate has no valid pixels")
        model_mse = model_sum / count_sum
        persistence_mse = persistence_sum / count_sum
        if np.any(persistence_mse <= 0):
            raise ValueError("a bootstrap replicate has zero persistence MSE")
        model_overall_mse = model_sum.sum(axis=1) / count_sum.sum(axis=1)
        persistence_overall_mse = (
            persistence_sum.sum(axis=1) / count_sum.sum(axis=1)
        )
        overall_difference[start:end] = (
            np.sqrt(model_overall_mse)
            - np.sqrt(persistence_overall_mse)
        )
        overall_skill[start:end] = (
            1.0 - model_overall_mse / persistence_overall_mse
        )
        lead_difference[start:end] = (
            np.sqrt(model_mse) - np.sqrt(persistence_mse)
        )
        lead_skill[start:end] = 1.0 - model_mse / persistence_mse

    difference_interval = _interval(
        overall_difference,
        confidence_level
    )
    skill_interval = _interval(overall_skill, confidence_level)
    lead_difference_interval = _interval(
        lead_difference,
        confidence_level
    )
    lead_skill_interval = _interval(
        lead_skill,
        confidence_level
    )
    by_lead_day = {}
    for lead_index in range(num_leads):
        by_lead_day[str(lead_index + 1)] = {
            "model_rmse": float(point["by_lead"]["model_rmse"][lead_index]),
            "persistence_rmse": float(
                point["by_lead"]["persistence_rmse"][lead_index]
            ),
            "rmse_difference": float(
                point["by_lead"]["rmse_difference"][lead_index]
            ),
            "rmse_difference_ci": [
                float(lead_difference_interval[0][lead_index]),
                float(lead_difference_interval[1][lead_index]),
            ],
            "mse_skill": float(point["by_lead"]["mse_skill"][lead_index]),
            "mse_skill_ci": [
                float(lead_skill_interval[0][lead_index]),
                float(lead_skill_interval[1][lead_index]),
            ],
            "bootstrap_fraction_model_better": float(
                np.mean(lead_difference[:, lead_index] < 0.0)
            ),
        }
    return {
        "method": "paired_nonoverlapping_temporal_block_bootstrap",
        "pairing": "same samples, leads, masks, and valid pixels",
        "block_unit": "forecast_initialization_time",
        "block_days": int(block_days),
        "block_origin_time": int(block_origin_time),
        "num_blocks": num_blocks,
        "num_samples": int(model_sse.shape[0]),
        "samples_per_block_min": int(block_sample_counts.min()),
        "samples_per_block_max": int(block_sample_counts.max()),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "overall": {
            "model_rmse": float(point["model_rmse"]),
            "persistence_rmse": float(point["persistence_rmse"]),
            "rmse_difference": float(point["rmse_difference"]),
            "rmse_difference_ci": [
                float(difference_interval[0]),
                float(difference_interval[1]),
            ],
            "mse_skill": float(point["mse_skill"]),
            "mse_skill_ci": [
                float(skill_interval[0]),
                float(skill_interval[1]),
            ],
            "bootstrap_fraction_model_better": float(
                np.mean(overall_difference < 0.0)
            ),
        },
        "by_lead_day": by_lead_day,
    }
