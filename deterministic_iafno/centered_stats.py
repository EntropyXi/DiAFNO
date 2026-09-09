# 用途：校验创新统计量、冻结均值身份及其来源一致性。
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

# v2 (A5-geo-season) frozen deterministic mean identity
# (experiments/a5_longtrain_v1_20260908/validation/best_val_rmse.pth,
# A5 long-run best epoch_030).  The v1 lock above is never changed:
# every stats payload resolves to exactly one protocol and its own
# locked mean identity (A5_CENTERED_DIAFNO_MAINTRAIN plan 5).
A5_LOCKED_MEAN_CHECKPOINT_SHA256 = (
    "4ce4984fe4b2e11748ca9bcacdedf0accc174a2721cf43b49167833f47c609bc"
)

CENTERED_STATS_SCHEMA_VERSION = 2

V1_MEAN_CONDITION_MODE = "sst_mask"
V2_MEAN_CONDITION_MODE = "sst_mask_geo_season"
V2_MEAN_COND_CHANS = 14

# v2 requires the frozen-mean sidecar to prove the geo-season data
# contract (condition schema, calendar, time axis, manifest identity);
# their exact values are bound by mean_semantics_sha256, presence is
# checked explicitly so an 8-channel legacy mean can never pass as the
# A5 mean.
V2_SIDECAR_PRESENCE_FIELDS = (
    "condition_mode",
    "condition_schema_version",
    "condition_channel_names",
    "calendar_encoding",
    "time_units_reference",
    "geospatial_summary",
    "time_axis_summary",
    "data_manifest_sha256",
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


# 用途：按载荷声明的 schema 版本与条件模式解析 centered 协议（v1 旧 / v2 A5-geo）。
# 参数：输入 stats（统计载荷）；输出 协议 spec dict（不匹配抛 ValueError）。
def centered_protocol_spec(stats):
    """Resolve the centered protocol (v1 legacy or v2 A5-geo-season).

    Version/condition pairs are exact and fail closed: v1 payloads
    declare schema_version 1 (or omit it) with condition_mode
    'sst_mask' and stay locked to the original frozen mean identity;
    v2 payloads must declare schema_version 2 with
    'sst_mask_geo_season' and lock to the A5 frozen mean identity.
    Anything else is refused instead of being silently interpreted.
    """
    schema_version = int(stats.get("schema_version", 1))
    condition_mode = stats.get("condition_mode")
    if schema_version == 1 and condition_mode == V1_MEAN_CONDITION_MODE:
        return {
            "schema_version": 1,
            "condition_mode": V1_MEAN_CONDITION_MODE,
            "mean_sha256": LOCKED_MEAN_CHECKPOINT_SHA256,
            "sidecar_presence_fields": (),
        }
    if (
            schema_version == CENTERED_STATS_SCHEMA_VERSION
            and condition_mode == V2_MEAN_CONDITION_MODE
        ):
        return {
            "schema_version": CENTERED_STATS_SCHEMA_VERSION,
            "condition_mode": V2_MEAN_CONDITION_MODE,
            "mean_sha256": A5_LOCKED_MEAN_CHECKPOINT_SHA256,
            "sidecar_presence_fields": V2_SIDECAR_PRESENCE_FIELDS,
        }
    raise ValueError(
        "unsupported centered stats protocol: schema_version="
        f"{stats.get('schema_version')!r} condition_mode="
        f"{condition_mode!r}; v1 requires '{V1_MEAN_CONDITION_MODE}', "
        f"v2 requires '{V2_MEAN_CONDITION_MODE}' with schema_version "
        f"{CENTERED_STATS_SCHEMA_VERSION}"
    )


# 用途：计算文件内容的 SHA256 十六进制摘要（用于冻结均值身份校验）。
# 参数：输入 path（文件路径）；输出 64 位十六进制字符串。
def sha256_hex_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 用途：把对象序列化为排序键、紧凑分隔的规范 JSON 字节。
# 参数：输入 value（可 JSON 化对象）；输出 规范化 UTF-8 字节。
def canonical_json_bytes(value):
    """Deterministic JSON serialization used for every semantics hash."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# 用途：计算对象规范 JSON 的 SHA256 摘要。
# 参数：输入 value（可 JSON 化对象）；输出 十六进制摘要。
def sha256_of_normalized(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# 用途：计算样本索引数组的 SHA256 摘要（绑定抽样计划）。
# 参数：输入 indices（int 数组/列表）；输出 十六进制摘要。
def indices_sha256(indices):
    import numpy as np
    array = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


# 用途：校验统计量数组长度、有限性且 std 为正。
# 参数：输入 values（[mean,std] 数组对）、name（名称，用于报错）、expected_length（期望长度）；输出 规范化后的 float 列表（违规抛 ValueError）。
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


# 用途：读取冻结均值 checkpoint 的语义 sidecar JSON。
# 参数：输入 mean_checkpoint_path（checkpoint 路径）；输出 sidecar dict（缺失或损坏抛异常）。
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


# 用途：校验冻结均值 sidecar 的语义字段与锁定期望完全一致。
# 参数：输入 mean_checkpoint_path（checkpoint 路径）；输出 无（不一致抛 ValueError）。
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


# 用途：交叉核对 centered 统计载荷与冻结均值 sidecar 的身份（模型语义、架构、SHA）。
# 参数：输入 stats（centered 统计载荷）、mean_checkpoint_path（冻结均值路径）；输出 无（不一致抛 ValueError）。
def cross_check_mean_sidecar(stats, mean_checkpoint_path):
    """Cross-check the frozen-mean sidecar against a centered stats JSON.

    Returns the sidecar's immutable block.  Fails closed when the
    sidecar is missing, its declared semantics disagree with the frozen
    deterministic mean contract, its own lead stats disagree with the
    stats JSON, or its recorded semantics hash does not match.
    """
    immutable = mean_sidecar_immutable(mean_checkpoint_path)
    # Resolve the protocol from the stats payload when it declares one;
    # a partial draft (compute_centered_stats cross-checks the mean file
    # before the full payload exists) falls back to the mean sidecar's
    # own condition mode so an A5-geo-season mean still gets the v2
    # checks and a legacy mean keeps the legacy behavior.
    if (
            stats.get("schema_version") is not None
            or stats.get("condition_mode") is not None
        ):
        spec = centered_protocol_spec(stats)
    elif immutable.get("condition_mode") == V2_MEAN_CONDITION_MODE:
        spec = {
            "schema_version": CENTERED_STATS_SCHEMA_VERSION,
            "condition_mode": V2_MEAN_CONDITION_MODE,
            "mean_sha256": A5_LOCKED_MEAN_CHECKPOINT_SHA256,
            "sidecar_presence_fields": V2_SIDECAR_PRESENCE_FIELDS,
        }
    else:
        spec = {
            "schema_version": 1,
            "condition_mode": V1_MEAN_CONDITION_MODE,
            "mean_sha256": LOCKED_MEAN_CHECKPOINT_SHA256,
            "sidecar_presence_fields": (),
        }
    if spec["schema_version"] == CENTERED_STATS_SCHEMA_VERSION:
        # v2: the mean must itself be an A5-geo-season deterministic
        # model -- 14 channels, condition mode and the geo/time/manifest
        # provenance present (values are bound by the semantics hash).
        if immutable.get("condition_mode") != V2_MEAN_CONDITION_MODE:
            raise ValueError(
                "v2 frozen mean sidecar condition_mode="
                f"{immutable.get('condition_mode')!r} does not match "
                f"'{V2_MEAN_CONDITION_MODE}'"
            )
        if int(immutable.get("cond_chans", -1)) != V2_MEAN_COND_CHANS:
            raise ValueError(
                "v2 frozen mean sidecar cond_chans="
                f"{immutable.get('cond_chans')!r} does not match "
                f"{V2_MEAN_COND_CHANS} (sst_mask_geo_season)"
            )
        for field in spec["sidecar_presence_fields"]:
            if immutable.get(field) is None:
                raise ValueError(
                    "v2 frozen mean sidecar immutable lacks "
                    f"{field!r}; a geo-season mean must prove its "
                    "calendar/time/manifest provenance"
                )
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


# 用途：把 numpy 标量/数组递归转为纯 python 类型以便 JSON 化。
# 参数：输入 value（任意对象）；输出 JSON 可序列化对象。
def _plain(value):
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


# 用途：centered innovation 统计载荷的权威校验：来源、划分、维度、统计量与冻结均值身份。
# 参数：输入 payload（统计 JSON）、mean_checkpoint_path（冻结均值路径）及期望的目标语义字段；输出 规范化载荷（违规抛 ValueError）。
def validate_centered_stats_payload(
        stats,
        target_chans=15,
        input_days=7,
        output_days=15,
        expected_mean_checkpoint_sha256=None,
    ):
    """Validate a centered innovation stats JSON payload.

    Enforces: train split provenance, the centered target space, day
    counts, per-lead counts, finite/positive stds, the per-protocol
    frozen mean SHA identity (v1 legacy vs v2 A5-geo-season), mean
    residual stats, index provenance, and the absence of any val/test
    metadata.  Returns a normalized dict of the validated values.
    """
    if not isinstance(stats, dict):
        raise ValueError(
            "centered stats payload must be a JSON object"
        )
    spec = centered_protocol_spec(stats)
    if expected_mean_checkpoint_sha256 is None:
        # Resolved at call time from the payload's own protocol (not
        # bound at import) so tests and alternate locks can override.
        expected_mean_checkpoint_sha256 = spec["mean_sha256"]
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
    if condition_mode != spec["condition_mode"]:
        raise ValueError(
            "centered stats condition_mode="
            f"{condition_mode!r} does not match its resolved protocol "
            f"({spec['condition_mode']!r})"
        )
    if spec["schema_version"] == CENTERED_STATS_SCHEMA_VERSION:
        manifest_sha = stats.get("data_manifest_sha256")
        if (
                not isinstance(manifest_sha, str)
                or len(manifest_sha) != 64
            ):
            raise ValueError(
                "v2 centered stats must declare a 64-char "
                "data_manifest_sha256 binding the gap-filtered train "
                "universe"
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


# 用途：全新 centered 训练启动前的输入校验（统计文件、冻结均值与配置三方一致性）。
# 参数：输入 统计路径/冻结均值路径/期望语义与架构字段；输出 无（违规抛异常）。
def validate_centered_fresh_inputs(
        mean_checkpoint_path,
        centered_stats_path,
        model_config,
    ):
    """Per-rank fail-closed validation of a fresh centered run.

    Verifies the frozen mean checkpoint file SHA against the
    stats-declared identity (which the payload validator locked to the
    per-protocol frozen-mean SHA), cross-checks the mean sidecar, and
    compares the mean architecture with the centered model config.
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
    if file_sha != validated["mean_checkpoint_sha256"]:
        raise ValueError(
            "frozen mean checkpoint SHA-256 mismatch: file "
            f"{file_sha} vs the stats-declared identity "
            f"{validated['mean_checkpoint_sha256']} (already locked "
            "per the payload's centered protocol)"
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
