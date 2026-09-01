"""Centered innovation statistics validation and provenance helpers.

These functions are the single authoritative validation surface for
train-only centered innovation stats JSON files and for the frozen mean
checkpoint identity.  Every consumer (stats tool, training config, and
the per-rank trainer fresh-run check) goes through these helpers so the
fail-closed rules from PHASE2_MAIN_TRAINING_PLAN section 3.3/4 are
enforced identically everywhere.
"""

import hashlib
import json
import math
import os

# Frozen deterministic mean identity from PHASE2_MAIN_TRAINING_PLAN 0.2
# (experiments/det_lead_standardized/epoch_015.pth).
LOCKED_MEAN_CHECKPOINT_SHA256 = (
    "cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6"
)

CENTERED_TARGET_SPACE = "normalized_centered_residual"

# Keys that would betray val/test contamination; the payload must not
# contain them under any name.
_FORBIDDEN_SPLIT_KEYS = (
    "val",
    "validation",
    "test",
    "val_indices",
    "test_indices",
    "validation_indices",
    "val_metadata",
    "test_metadata",
)

MEAN_IMMUTABLE_EXPECTATIONS = {
    "model_type": "deterministic",
    "target_mode": "residual",
    "target_scaling": "lead_standardized",
    "input_days": 7,
    "output_days": 15,
}

MEAN_ARCH_FIELDS = (
    "input_days",
    "output_days",
    "cond_chans",
    "target_chans",
    "image_size",
    "patch_size",
    "embed_dim",
    "num_blocks",
    "explicit_layer",
    "implicit_layer",
    "hidden_size_factor",
)


def sha256_hex_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value):
    """Deterministic JSON serialization used for every semantics hash."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_of_normalized(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def indices_sha256(indices):
    import numpy as np
    array = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_positive_stats(values, name, expected_length):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be an array of floats")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} has {len(values)} entries; expected "
            f"{expected_length}"
        )
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in converted):
        raise ValueError(f"all {name} values must be finite")
    return converted


def _load_sidecar(mean_checkpoint_path):
    from deterministic_iafno.checkpoint_semantics import (
        load_semantic_sidecar,
    )
    sidecar = load_semantic_sidecar(mean_checkpoint_path)
    if sidecar is None:
        raise ValueError(
            "frozen mean checkpoint has no semantic sidecar: "
            f"{mean_checkpoint_path}; centered runs fail closed "
            "without the mean identity manifest"
        )
    return sidecar


def mean_sidecar_immutable(mean_checkpoint_path):
    sidecar = _load_sidecar(mean_checkpoint_path)
    manifest = sidecar.get("semantic_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "mean checkpoint sidecar has no semantic_manifest"
        )
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise ValueError(
            "mean checkpoint sidecar manifest has no immutable block"
        )
    return immutable


def cross_check_mean_sidecar(stats, mean_checkpoint_path):
    """Cross-check the frozen-mean sidecar against a centered stats JSON.

    Returns the sidecar's immutable block.  Fails closed when the
    sidecar is missing, its declared semantics disagree with the frozen
    deterministic mean contract, its own lead stats disagree with the
    stats JSON, or its recorded semantics hash does not match.
    """
    immutable = mean_sidecar_immutable(mean_checkpoint_path)
    for field, expected in MEAN_IMMUTABLE_EXPECTATIONS.items():
        actual = immutable.get(field)
        if _plain(actual) != _plain(expected):
            raise ValueError(
                "frozen mean sidecar immutable "
                f"{field}={actual!r} does not match the locked "
                f"deterministic mean contract ({expected!r})"
            )
    for field in ("lead_mean", "lead_std"):
        if field not in immutable:
            raise ValueError(
                f"frozen mean sidecar immutable lacks {field}"
            )
        stats_field = f"mean_{field}"
        if _plain(immutable[field]) != _plain(stats[stats_field]):
            raise ValueError(
                "frozen mean sidecar "
                f"{field}={immutable[field]} does not match centered "
                f"stats {stats_field}={stats[stats_field]}"
            )
    recorded_semantics_hash = stats.get("mean_semantics_sha256")
    computed = sha256_of_normalized(immutable)
    if recorded_semantics_hash != computed:
        raise ValueError(
            "centered stats mean_semantics_sha256="
            f"{recorded_semantics_hash} does not match the frozen mean "
            f"sidecar immutable hash {computed}"
        )
    return immutable


def _plain(value):
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def validate_centered_stats_payload(
        stats,
        target_chans=15,
        input_days=7,
        output_days=15,
        expected_mean_checkpoint_sha256=None,
    ):
    """Validate a centered innovation stats JSON payload.

    Enforces: train split provenance, the centered target space, day
    counts, per-lead counts, finite/positive stds, the locked frozen
    mean SHA identity, mean residual stats, index provenance, and the
    absence of any val/test metadata.  Returns a normalized dict of the
    validated values.
    """
    if expected_mean_checkpoint_sha256 is None:
        # Resolved at call time (not bound at import) so tests and
        # alternate locks can override the identity.
        expected_mean_checkpoint_sha256 = LOCKED_MEAN_CHECKPOINT_SHA256
    if not isinstance(stats, dict):
        raise ValueError(
            "centered stats payload must be a JSON object"
        )
    for forbidden in _FORBIDDEN_SPLIT_KEYS:
        if forbidden in stats:
            raise ValueError(
                f"centered stats must not contain {forbidden!r}; "
                "validation/test data must never enter training "
                "statistics"
            )
    split = stats.get("split")
    if split != "train":
        raise ValueError(
            "centered stats must come from the train split "
            f"(got {split!r}); validation/test data must never "
            "enter training statistics"
        )
    target_space = stats.get("target_space")
    if target_space != CENTERED_TARGET_SPACE:
        raise ValueError(
            "centered stats must declare target_space="
            f"'{CENTERED_TARGET_SPACE}' (got {target_space!r})"
        )
    if stats.get("input_days") != input_days:
        raise ValueError(
            f"centered stats input_days={stats.get('input_days')} "
            f"does not match {input_days}"
        )
    if stats.get("output_days") != output_days:
        raise ValueError(
            f"centered stats output_days={stats.get('output_days')} "
            f"does not match {output_days}"
        )
    condition_mode = stats.get("condition_mode")
    if condition_mode != "sst_mask":
        raise ValueError(
            "centered stats must declare condition_mode="
            f"'sst_mask' (got {condition_mode!r})"
        )
    lead_mean = _finite_positive_stats(
        stats.get("lead_mean"),
        "innovation lead_mean",
        target_chans,
    )
    lead_std_raw = _finite_positive_stats(
        stats.get("lead_std"),
        "innovation lead_std",
        target_chans,
    )
    if any(value <= 0.0 for value in lead_std_raw):
        raise ValueError(
            "all innovation lead_std values must be positive"
        )
    mean_lead_mean = _finite_positive_stats(
        stats.get("mean_lead_mean"),
        "mean_lead_mean",
        target_chans,
    )
    mean_lead_std_raw = _finite_positive_stats(
        stats.get("mean_lead_std"),
        "mean_lead_std",
        target_chans,
    )
    if any(value <= 0.0 for value in mean_lead_std_raw):
        raise ValueError(
            "all mean_lead_std values must be positive"
        )
    mean_sha = stats.get("mean_checkpoint_sha256")
    if not isinstance(mean_sha, str) or len(mean_sha) != 64:
        raise ValueError(
            "centered stats must contain a 64-char "
            "mean_checkpoint_sha256"
        )
    if mean_sha.lower() != expected_mean_checkpoint_sha256.lower():
        raise ValueError(
            "centered stats mean_checkpoint_sha256="
            f"{mean_sha} does not match the locked frozen mean "
            f"identity {expected_mean_checkpoint_sha256}"
        )
    mean_semantics = stats.get("mean_semantics_sha256")
    if not isinstance(mean_semantics, str) or len(mean_semantics) != 64:
        raise ValueError(
            "centered stats must contain a 64-char "
            "mean_semantics_sha256"
        )
    indices_hash = stats.get("indices_sha256")
    if not isinstance(indices_hash, str) or len(indices_hash) != 64:
        raise ValueError(
            "centered stats must contain a 64-char indices_sha256"
        )
    num_samples = stats.get("num_samples")
    dataset_size = stats.get("dataset_size")
    if (
            not isinstance(num_samples, int)
            or not isinstance(dataset_size, int)
            or num_samples < 1
            or dataset_size < num_samples
        ):
        raise ValueError(
            "centered stats must declare positive num_samples and "
            "dataset_size >= num_samples"
        )
    sst_mean = stats.get("sst_mean")
    sst_std = stats.get("sst_std")
    if (
            not isinstance(sst_mean, (int, float))
            or not isinstance(sst_std, (int, float))
            or not math.isfinite(float(sst_mean))
            or not math.isfinite(float(sst_std))
            or float(sst_std) <= 0.0
        ):
        raise ValueError(
            "centered stats must declare finite sst_mean and "
            "positive finite sst_std"
        )
    selection = stats.get("selection")
    if not isinstance(selection, str) or not selection:
        raise ValueError(
            "centered stats must declare the deterministic index "
            "selection method"
        )
    return {
        "lead_mean": lead_mean,
        "lead_std": lead_std_raw,
        "mean_lead_mean": mean_lead_mean,
        "mean_lead_std": mean_lead_std_raw,
        "mean_checkpoint_sha256": mean_sha.lower(),
        "mean_semantics_sha256": mean_semantics.lower(),
        "indices_sha256": indices_hash.lower(),
        "num_samples": int(num_samples),
        "dataset_size": int(dataset_size),
        "sst_mean": float(sst_mean),
        "sst_std": float(sst_std),
        "selection": selection,
        "mean_checkpoint": stats.get("mean_checkpoint"),
    }


def validate_centered_fresh_inputs(
        mean_checkpoint_path,
        centered_stats_path,
        model_config,
    ):
    """Per-rank fail-closed validation of a fresh centered run.

    Verifies the frozen mean checkpoint file SHA against both the
    locked identity and the stats JSON, cross-checks the mean sidecar,
    and compares the mean architecture with the centered model config.
    Raises ValueError on the first violation so every rank exits.
    """
    if not mean_checkpoint_path or not centered_stats_path:
        raise ValueError(
            "fresh centered run requires --mean-checkpoint and "
            "--centered-stats"
        )
    mean_checkpoint_path = os.path.abspath(mean_checkpoint_path)
    centered_stats_path = os.path.abspath(centered_stats_path)
    if not os.path.isfile(mean_checkpoint_path):
        raise FileNotFoundError(
            f"mean checkpoint not found: {mean_checkpoint_path}"
        )
    if not os.path.isfile(centered_stats_path):
        raise FileNotFoundError(
            f"centered stats not found: {centered_stats_path}"
        )
    with open(centered_stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)
    validated = validate_centered_stats_payload(
        stats,
        target_chans=model_config.target_chans,
        input_days=model_config.input_days,
        output_days=model_config.output_days,
    )
    file_sha = sha256_hex_file(mean_checkpoint_path)
    if file_sha != LOCKED_MEAN_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen mean checkpoint SHA-256 mismatch: file "
            f"{file_sha} vs locked {LOCKED_MEAN_CHECKPOINT_SHA256}"
        )
    if validated["mean_checkpoint_sha256"] != file_sha:
        raise ValueError(
            "centered stats mean_checkpoint_sha256="
            f"{validated['mean_checkpoint_sha256']} does not match "
            f"the mean checkpoint file SHA {file_sha}"
        )
    immutable = cross_check_mean_sidecar(
        stats,
        mean_checkpoint_path,
    )
    for field in MEAN_ARCH_FIELDS:
        sidecar_value = immutable.get(field)
        if sidecar_value is None:
            raise ValueError(
                f"frozen mean sidecar immutable lacks {field}"
            )
        config_value = _plain(getattr(model_config, field))
        if _plain(sidecar_value) != config_value:
            raise ValueError(
                "frozen mean architecture mismatch for "
                f"{field}: mean sidecar={sidecar_value} vs centered "
                f"config={config_value}"
            )
    return validated, immutable
