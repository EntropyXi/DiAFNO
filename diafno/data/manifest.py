# 用途：校验真实日期 manifest，约束缺日过滤与数据来源。
"""OSTIA real-day data-manifest contract.

The patched training HDF5 stores one row per (compact day, spatial
patch) and -- on the real server file -- carries no calendar or
coordinate metadata at all, so geo-season training cannot decode the
time axis from the HDF5 alone.  The upstream NetCDF proves the true
daily offsets (including the two 31-day gaps in the real 1991-2021
series); ``scripts/audit_ostia_h5.py --source-netcdf`` turns that
evidence into a read-only *data manifest* consumed here.

Manifest contract (schema version 1):

- ``n_days``: number of compact days stored in the HDF5;
- ``day_offsets``: one true daily offset per compact day (ordinal
  alignment: compact day ``d`` holds the field of upstream day offset
  ``day_offsets[d]``);
- ``day_offset_sha256``: canonical sha256 of the offsets (little-
  endian int64, raw order);
- ``units`` / ``calendar``: the upstream time axis semantics;
- ``gaps``: list of ``{"after_ordinal", "missing_days"}`` entries;
- optional ``h5``: structural facts proven about the HDF5 the
  manifest was generated for (rows, samples_per_day, n_days, compact
  time-axis sha256, coordinate layout/units summaries), which the
  dataset verifies against the actual file.
"""

import hashlib
import json
import re

import numpy as np

DATA_MANIFEST_SCHEMA_VERSION = 1

ALLOWED_CALENDARS = (
    "standard",
    "gregorian",
    "proleptic_gregorian",
)

_DAYS_SINCE_PATTERN = re.compile(
    r"^\s*days\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})"
    r"(?:[T ].*)?$",
    flags=re.IGNORECASE,
)


def day_offset_sha256(day_offsets):
    """Canonical sha256 of the true daily offsets."""
    offsets = np.asarray(day_offsets, dtype=np.int64)
    canonical = np.ascontiguousarray(offsets, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def canonical_manifest_sha256(payload):
    """Deterministic identity of a whole manifest payload.

    Covers every field including the offsets, so two manifests that
    agree in shape but differ anywhere (a shifted mapping, a changed
    unit string, a different gap) produce different identities.
    """
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def parse_days_since_units(units_text):
    """Return (year, month, day) for 'days since YYYY-MM-DD...'."""
    if not isinstance(units_text, str):
        raise ValueError(
            "manifest units must be a string, got "
            f"{units_text!r}"
        )
    match = _DAYS_SINCE_PATTERN.match(units_text.strip())
    if match is None:
        raise ValueError(
            "manifest units must have the form "
            "'days since YYYY-MM-DD' (optional time part), but got "
            f"{units_text!r}"
        )
    return tuple(int(part) for part in match.groups())


def load_data_manifest(path):
    """Load and validate a data manifest (fail closed)."""
    if not isinstance(path, str) or not path:
        raise ValueError("data_manifest path must be a non-empty str")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("data manifest must contain a JSON object")
    version = payload.get("schema_version")
    if version != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported data-manifest schema_version "
            f"{version!r} (expected "
            f"{DATA_MANIFEST_SCHEMA_VERSION})"
        )
    n_days = payload.get("n_days")
    offsets = payload.get("day_offsets")
    if not isinstance(n_days, int) or n_days < 1:
        raise ValueError(
            "data manifest must declare a positive integer n_days"
        )
    if not isinstance(offsets, list) or len(offsets) != n_days:
        raise ValueError(
            f"data manifest day_offsets must be a list of exactly "
            f"n_days={n_days} entries, got "
            f"{type(offsets).__name__} of length "
            f"{len(offsets) if isinstance(offsets, list) else '?'}"
        )
    try:
        offsets = [int(value) for value in offsets]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "data manifest day_offsets must contain only integers"
        ) from error
    if any(offset < 0 for offset in offsets):
        raise ValueError(
            "data manifest day_offsets must be non-negative daily "
            "offsets"
        )
    units = payload.get("units")
    if units is None:
        raise ValueError(
            "data manifest must declare 'units' of the form "
            "'days since YYYY-MM-DD'"
        )
    parse_days_since_units(units)
    calendar = payload.get("calendar")
    if calendar is None:
        calendar = "standard"
    if calendar not in ALLOWED_CALENDARS:
        raise ValueError(
            "data manifest declares unsupported calendar "
            f"{calendar!r}; allowed: {ALLOWED_CALENDARS}"
        )
    recorded = payload.get("day_offset_sha256")
    recomputed = day_offset_sha256(offsets)
    if recorded != recomputed:
        raise ValueError(
            "data manifest day_offset_sha256 does not match its "
            f"day_offsets: recorded {recorded!r} vs recomputed "
            f"{recomputed!r}; the manifest is corrupt or hand-edited"
        )
    h5 = payload.get("h5")
    if h5 is not None and not isinstance(h5, dict):
        raise ValueError("data manifest 'h5' section must be an object")
    gaps = payload.get("gaps")
    if gaps is not None:
        if not isinstance(gaps, list):
            raise ValueError("data manifest 'gaps' must be a list")
        for gap in gaps:
            if not isinstance(gap, dict) or not isinstance(
                    gap.get("after_ordinal"), int
                ) or not isinstance(gap.get("missing_days"), int):
                raise ValueError(
                    "each manifest gap needs integer "
                    "'after_ordinal' and 'missing_days'"
                )
    coordinates = payload.get("coordinates")
    if coordinates is not None and not isinstance(coordinates, dict):
        raise ValueError(
            "data manifest 'coordinates' section must be an object"
        )
    payload = dict(payload)
    payload["day_offsets"] = offsets
    payload["calendar"] = calendar
    return payload
