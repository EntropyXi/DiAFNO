# 用途：统一 SST、mask、位置和季节条件通道的顺序与版本。
"""Canonical OSTIA condition-mode channel contract.

Single source of truth for every condition mode, its fixed channel
order, its channel count and its schema version:

- ``sst``: 7 normalized input SST fields (legacy mode, no mask);
- ``sst_mask``: the 7 SST fields plus the last-input-day valid-ocean
  mask (legacy mode, 8 channels);
- ``sst_mask_geo_season``: ``sst_mask`` plus four static latitude /
  longitude sine-cosine grids and the two seasonal (day-of-year)
  sine-cosine channels of the initialization day ``t0`` (14 channels).

Training, validation and inference must all build conditions through
``OSTIADailyDataset`` so that the fixed order below is the only layout
any model ever sees.  ``diafno.models.config`` validates model channel
counts against these tables instead of trusting a hand-written
``cond_chans``.
"""

CONDITION_MODES = (
    "sst",
    "sst_mask",
    "sst_mask_geo_season",
)

# Channel names that follow the SST history.
VALID_MASK_CHANNEL_NAME = "valid_mask_t0"

# Static spatial / seasonal channels appended by the geo-season mode,
# in exactly this order.
GEO_STATIC_CHANNEL_NAMES = (
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "sin_doy",
    "cos_doy",
)

# Schema version 1: plain per-day SST (optionally with the t0 mask),
# no decoded calendar semantics.  Version 2: geo-season layout whose
# seasonal phase requires provable Gregorian date semantics.
CONDITION_SCHEMA_VERSIONS = {
    "sst": 1,
    "sst_mask": 1,
    "sst_mask_geo_season": 2,
}

# Checkpoint files older than this branch do not carry the new fields;
# when they are missing the config is interpreted as
# ``sst_mask``/version 1 and its stored ``cond_chans``.
LEGACY_MODE = "sst_mask"
LEGACY_SCHEMA_VERSION = 1


def sst_channel_names(input_days):
    """Names of the ``input_days`` normalized SST history channels.

    Channel ``k`` holds the SST ``input_days - 1 - k`` days before the
    initialization day, so the last history channel is always the
    initialization-day SST ``sst_t0`` (the residual anchor).
    """
    if int(input_days) < 1:
        raise ValueError("input_days must be positive")
    names = []
    for offset in range(int(input_days) - 1, -1, -1):
        if offset == 0:
            names.append("sst_t0")
        else:
            names.append(f"sst_tminus{offset}")
    return tuple(names)


def condition_channel_names(condition_mode, input_days):
    """Return the fixed condition channel order for a mode."""
    if condition_mode not in CONDITION_MODES:
        raise ValueError(
            f"condition_mode must be one of {CONDITION_MODES}, "
            f"but got {condition_mode!r}"
        )
    names = list(sst_channel_names(input_days))
    if condition_mode in ("sst_mask", "sst_mask_geo_season"):
        names.append(VALID_MASK_CHANNEL_NAME)
    if condition_mode == "sst_mask_geo_season":
        names.extend(GEO_STATIC_CHANNEL_NAMES)
    return tuple(names)


def condition_chans(condition_mode, input_days):
    """Return the condition channel count required by a mode."""
    return len(condition_channel_names(condition_mode, input_days))


def condition_schema_version_for(condition_mode):
    """Return the persisted schema version for a condition mode."""
    if condition_mode not in CONDITION_MODES:
        raise ValueError(
            f"condition_mode must be one of {CONDITION_MODES}, "
            f"but got {condition_mode!r}"
        )
    return CONDITION_SCHEMA_VERSIONS[condition_mode]


def resolve_condition_mode(cli_condition_mode, model_condition_mode,
                           purpose):
    """Resolve the data contract for validation/inference.

    ``cli_condition_mode`` is the explicit user override (None means
    "restore from the checkpoint").  The checkpoint's model config is
    authoritative: an explicit override that conflicts with it fails
    closed before any data is read.
    """
    if cli_condition_mode is None:
        return model_condition_mode
    if cli_condition_mode != model_condition_mode:
        raise ValueError(
            f"--condition-mode {cli_condition_mode!r} conflicts with "
            f"the checkpoint condition_mode={model_condition_mode!r}; "
            f"{purpose} must use the checkpoint's condition contract"
        )
    return cli_condition_mode
