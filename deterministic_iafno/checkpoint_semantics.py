# 用途：定义和校验 checkpoint 的模型、数据及续训语义。
import json
import os


CHECKPOINT_SCHEMA_VERSION = 5

MODEL_IMMUTABLE_FIELDS = (
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
    "sigma_data",
    "p_mean",
    "p_std",
    "target_mode",
    "model_type",
    "target_scaling",
    "lead_mean",
    "lead_std",
    # Phase 2 centered identity: frozen-mean residual lead stats and
    # the two provenance hashes.  None for legacy models.
    "mean_lead_mean",
    "mean_lead_std",
    "mean_checkpoint_sha256",
    "mean_semantics_sha256",
    # Condition-schema contract (version 5): the fixed channel layout,
    # decoded calendar semantics and the proven geospatial summary of
    # the training HDF5.  Legacy checkpoints (schema < 5) simply lack
    # these keys and are compared on the fields they actually store.
    # condition_mode itself lives on the training config and is
    # recorded separately in the manifest immutable block below.
    "condition_schema_version",
    "condition_channel_names",
    "calendar_encoding",
    "time_units_reference",
    "geospatial_summary",
    # Real-day time-axis provenance (geo-season mode): per-day offset
    # digest / gaps plus the data-manifest identity.  Two files with
    # the same shape but a different time mapping fail closed.
    "time_axis_summary",
    "data_manifest_sha256",
)

SAMPLER_FIELDS = (
    "sampling_steps",
    "sigma_min",
    "sigma_max",
    "rho",
)

COMPATIBLE_STRICT_FIELDS = (
    # Optimizer/scheduler/data-exposure semantics.  State is restored
    # first; reviewed CLI changes are applied only with an explicit
    # override so the live objects and the next manifest stay aligned.
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "num_epochs",
    "samples_per_epoch",
    "optimizer_steps_per_epoch",
)

COMPATIBLE_RECORDED_FIELDS = ()


def _plain(value):
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in value.items()
        }
    return value


def _model_values(config):
    if hasattr(config.model, "to_checkpoint"):
        return _plain(config.model.to_checkpoint())
    return _plain(vars(config.model))


def build_semantic_manifest(config, world_size=1):
    model_values = _model_values(config)
    immutable = {
        field: model_values[field]
        for field in MODEL_IMMUTABLE_FIELDS
        if field in model_values
    }
    immutable.update(
        {
            "split": config.split,
            "condition_mode": config.condition_mode,
        }
    )
    effective_global_batch = (
        int(config.batch_per_gpu)
        * int(config.gradient_accumulation)
        * int(world_size)
    )
    compatible = {
        field: _plain(getattr(config, field))
        for field in COMPATIBLE_STRICT_FIELDS
    }
    compatible["effective_global_batch"] = effective_global_batch
    recorded = {
        field: _plain(getattr(config, field))
        for field in COMPATIBLE_RECORDED_FIELDS
    }
    sampler = {
        field: model_values[field]
        for field in SAMPLER_FIELDS
        if field in model_values
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "immutable": immutable,
        "compatible": compatible,
        "recorded": recorded,
        "sampler": sampler,
        "runtime": {
            "world_size": int(world_size),
            "batch_per_gpu": int(config.batch_per_gpu),
            "gradient_accumulation": int(
                config.gradient_accumulation
            ),
            "num_workers": int(config.num_workers),
            "train_h5_path": os.path.abspath(
                config.train_h5_path
            ),
        },
    }


def _diff(expected, actual):
    keys = sorted(set(expected) | set(actual))
    return {
        key: {
            "checkpoint": actual.get(key),
            "current": expected.get(key),
        }
        for key in keys
        if _plain(expected.get(key)) != _plain(actual.get(key))
    }


def _legacy_manifest(checkpoint, config):
    checkpoint_model = _plain(checkpoint.get("config", {}))
    if (
            "input_days" not in checkpoint_model
            and "input_months" in checkpoint_model
        ):
        checkpoint_model["input_days"] = checkpoint_model[
            "input_months"
        ]
    if (
            "output_days" not in checkpoint_model
            and "output_months" in checkpoint_model
        ):
        checkpoint_model["output_days"] = checkpoint_model[
            "output_months"
        ]
    current = build_semantic_manifest(config)
    immutable = {
        field: checkpoint_model[field]
        for field in MODEL_IMMUTABLE_FIELDS
        if field in checkpoint_model
    }
    return {
        "schema_version": 1,
        "immutable": immutable,
        "compatible": {},
        "recorded": {},
        "sampler": {
            field: checkpoint_model[field]
            for field in SAMPLER_FIELDS
            if field in checkpoint_model
        },
        "runtime": {},
        "legacy_current": current,
    }


def validate_semantic_manifest(
        checkpoint,
        config,
        world_size=1,
        allow_compatible_override=False,
    ):
    current = build_semantic_manifest(config, world_size)
    saved = checkpoint.get("semantic_manifest")
    warnings = []
    if saved is None:
        saved = _legacy_manifest(checkpoint, config)
        warnings.append(
            "legacy checkpoint has no semantic manifest; only fields "
            "stored in its model config can be validated"
        )

    saved_immutable = saved.get("immutable", {})
    current_immutable = current["immutable"]
    if saved.get("schema_version", 1) < CHECKPOINT_SCHEMA_VERSION:
        current_immutable = {
            key: value
            for key, value in current_immutable.items()
            if key in saved_immutable
        }
    immutable_mismatches = _diff(
        current_immutable,
        saved_immutable,
    )
    if immutable_mismatches:
        raise ValueError(
            "checkpoint immutable semantic mismatch "
            f"(checkpoint vs current): {immutable_mismatches}"
        )

    compatible_mismatches = get_compatible_mismatches(
        checkpoint,
        config,
        world_size,
    )
    if compatible_mismatches:
        if not allow_compatible_override:
            raise ValueError(
                "checkpoint training compatibility mismatch "
                f"(checkpoint vs current): {compatible_mismatches}; "
                "pass --allow-resume-override only after reviewing "
                "the changed optimizer/schedule/batch semantics"
            )
        warnings.append(
            "explicitly accepted training compatibility mismatch: "
            f"{compatible_mismatches}"
        )

    saved_recorded = saved.get("recorded", {})
    current_recorded = current["recorded"]
    if saved.get("schema_version", 1) < CHECKPOINT_SCHEMA_VERSION:
        current_recorded = {
            key: value
            for key, value in current_recorded.items()
            if key in saved_recorded
        }
    recorded_mismatches = _diff(
        current_recorded,
        saved_recorded,
    )
    if recorded_mismatches:
        warnings.append(
            "recorded training fields differ from checkpoint "
            f"(provenance only, not an error): {recorded_mismatches}"
        )

    sampler_mismatches = _diff(
        current["sampler"],
        saved.get("sampler", {}),
    )
    if sampler_mismatches:
        warnings.append(
            "evaluation sampler profile differs from checkpoint; "
            "do not compare the resulting validation curve directly "
            f"with the old protocol: {sampler_mismatches}"
        )
    return warnings


def get_compatible_mismatches(checkpoint, config, world_size=1):
    """Return saved-vs-current optimizer/schedule/batch differences.

    Legacy checkpoints without a semantic manifest have no reliable
    compatible-field record, so they return an empty mapping.
    """
    saved = checkpoint.get("semantic_manifest")
    if saved is None:
        return {}
    current = build_semantic_manifest(config, world_size)
    saved_compatible = saved.get("compatible", {})
    current_compatible = current["compatible"]
    if saved.get("schema_version", 1) < CHECKPOINT_SCHEMA_VERSION:
        current_compatible = {
            key: value
            for key, value in current_compatible.items()
            if key in saved_compatible
        }
    return _diff(current_compatible, saved_compatible)


def sidecar_path_for(checkpoint_path):
    """Return the per-checkpoint semantic sidecar path.

    New checkpoints write ``<checkpoint>.semantics.json`` next to the
    checkpoint so resume can restore immutable semantics without
    loading the (large) checkpoint just to read its config.
    """
    return checkpoint_path + ".semantics.json"


def resolve_sidecar_path(checkpoint_path):
    """Locate a semantic sidecar for a checkpoint.

    Prefers the per-checkpoint sidecar, then the legacy directory-level
    ``checkpoint_semantics.json`` written by early drafts.
    """
    candidates = (
        sidecar_path_for(checkpoint_path),
        os.path.join(
            os.path.dirname(checkpoint_path),
            "checkpoint_semantics.json"
        ),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_semantic_sidecar(checkpoint_path):
    """Read the semantic sidecar JSON, or None for legacy checkpoints."""
    path = resolve_sidecar_path(checkpoint_path)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def restore_resume_semantics(
        sidecar,
        config,
        default_values,
        explicit_fields=None,
    ):
    """Restore immutable model/data/training-noise semantics from a
    semantic sidecar before the model is built.

    Rule per immutable field: if the current CLI/default value equals
    the checkpoint value, nothing changes; if it equals the factory
    default, the checkpoint value is restored (so a bare ``--resume``
    continues with exactly the checkpoint semantics instead of silently
    adopting newer CLI defaults); any other explicit difference is an
    error (fail closed).

    Sampler profile fields are restored under the same default rule, but
    an explicit difference is only a warning (the CLI value wins and
    will be recorded into future checkpoints).

    Returns a list of human-readable notices (never raises for the
    default-rule cases).  Raises ValueError on explicit immutable
    conflicts.
    """
    notices = []
    explicit_fields = (
        None
        if explicit_fields is None
        else set(explicit_fields)
    )
    manifest = sidecar.get("semantic_manifest", sidecar)

    tuple_fields = (
        "lead_mean",
        "lead_std",
        "mean_lead_mean",
        "mean_lead_std",
        "condition_channel_names",
    )

    def _locate(container, field):
        """Return the object that owns ``field``: the training config
        or its nested model config."""
        if hasattr(container, field):
            return container
        model = getattr(container, "model", None)
        if model is not None and hasattr(model, field):
            return model
        return None

    saved_immutable = manifest.get("immutable", {})
    for field, checkpoint_value in saved_immutable.items():
        owner = _locate(config, field)
        if owner is None:
            continue
        current_value = getattr(owner, field)
        if _plain(current_value) == _plain(checkpoint_value):
            continue
        if explicit_fields is not None and field in explicit_fields:
            raise ValueError(
                "resume immutable semantic conflict for explicitly "
                f"set {field}: checkpoint={checkpoint_value}, "
                f"current={current_value}; start fresh or fix the "
                "CLI instead of overriding immutable semantics"
            )
        default_value = default_values.get(field, None)
        if (
                explicit_fields is not None
                or _plain(current_value) == _plain(default_value)
            ):
            if (
                    field in tuple_fields
                    and isinstance(checkpoint_value, list)
                ):
                checkpoint_value = tuple(checkpoint_value)
            setattr(owner, field, checkpoint_value)
            notices.append(
                f"restored immutable semantics from checkpoint: "
                f"{field}={checkpoint_value}"
            )
        else:
            raise ValueError(
                "resume immutable semantic conflict for "
                f"{field}: checkpoint={checkpoint_value}, "
                f"current={current_value}; start fresh or fix the "
                "CLI instead of overriding immutable semantics"
            )
    saved_sampler = manifest.get("sampler", {})
    for field, checkpoint_value in saved_sampler.items():
        owner = _locate(config, field)
        if owner is None:
            continue
        current_value = getattr(owner, field)
        if _plain(current_value) == _plain(checkpoint_value):
            continue
        if explicit_fields is not None and field in explicit_fields:
            notices.append(
                f"sampler profile {field}={current_value} from "
                f"explicit CLI differs from checkpoint "
                f"{checkpoint_value}; the CLI value wins and will be "
                "recorded into future checkpoints (do not compare "
                "validation curves across profiles)"
            )
            continue
        default_value = default_values.get(field, None)
        if (
                explicit_fields is not None
                or _plain(current_value) == _plain(default_value)
            ):
            setattr(owner, field, checkpoint_value)
            notices.append(
                f"restored sampler profile from checkpoint: "
                f"{field}={checkpoint_value}"
            )
        else:
            notices.append(
                f"sampler profile {field}={current_value} from CLI "
                f"differs from checkpoint {checkpoint_value}; the CLI "
                "value wins and will be recorded into future "
                "checkpoints (do not compare validation curves "
                "across profiles)"
            )
    return notices
