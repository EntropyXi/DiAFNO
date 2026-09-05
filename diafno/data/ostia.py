# 用途：从切块 HDF5 构造同区域的 7 日输入与 15 日目标样本。
import hashlib
import os
import re
from datetime import date, timedelta

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .condition_schema import (
    CONDITION_MODES,
    GEO_STATIC_CHANNEL_NAMES,
    condition_channel_names,
    condition_chans,
    condition_schema_version_for,
)
from .manifest import (
    canonical_manifest_sha256,
    load_data_manifest,
    parse_days_since_units,
)

# Fields that prove the geo-season training contract (date semantics,
# geospatial facts, real-day time axis) and are persisted in the model
# config / checkpoint sidecar.  Training data setup and the
# validation/inference contract checks compare the same set.
PROVENANCE_FIELDS = (
    "calendar_encoding",
    "time_units_reference",
    "geospatial_summary",
    "time_axis_summary",
    "data_manifest_sha256",
)


def coordinate_sha256(values):
    """Deterministic fingerprint of a coordinate vector.

    Canonical representation: the raw stored order as little-endian
    float64 bytes (matching the HDF5 double layout), so the digest is
    stable across machines and changes whenever any coordinate value
    changes, regardless of shape or unit metadata.
    """
    array = np.asarray(values, dtype=np.float64)
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


_coordinate_sha256 = coordinate_sha256


def copy_dataset_provenance(model_config, dataset):
    """Persist the dataset-proven dataset facts onto a model config.

    Used by fresh training runs and by tests that build geo-season
    checkpoints; the stored facts are exactly what the dataset proved
    about its HDF5 (and its data manifest, when one was used).
    """
    for field in PROVENANCE_FIELDS:
        setattr(model_config, field, getattr(dataset, field))
    return model_config


def _h5_text(value):
    """Best-effort decode of an h5py attribute scalar into str or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _h5_text(value.reshape(-1)[0])
    if hasattr(value, "decode"):
        return _h5_text(value.decode("utf-8", errors="replace"))
    if hasattr(value, "item"):
        return _h5_text(value.item())
    return str(value)


def verify_checkpoint_data_contract(dataset, model_config):
    """Reject validation/inference under a different data contract.

    Geo-season checkpoints bind calendar, geometry and the real-day
    mapping.  Legacy-channel checkpoints may also bind a data manifest
    so every ablation uses the same gap-filtered sample universe; those
    checkpoints bind the calendar/time fields but not geometry.
    """
    if dataset.condition_mode != model_config.condition_mode:
        raise ValueError(
            "dataset condition_mode="
            f"{dataset.condition_mode!r} does not match the "
            "checkpoint condition_mode="
            f"{model_config.condition_mode!r}; validation/inference "
            "must use the checkpoint's condition contract"
        )
    if model_config.condition_mode == "sst_mask_geo_season":
        fields = PROVENANCE_FIELDS
    else:
        checkpoint_manifest = getattr(
            model_config, "data_manifest_sha256", None
        )
        dataset_manifest = getattr(
            dataset, "data_manifest_sha256", None
        )
        if checkpoint_manifest is None and dataset_manifest is None:
            return
        if checkpoint_manifest is None or dataset_manifest is None:
            raise ValueError(
                "checkpoint and validation/inference dataset disagree "
                "on whether a real-day data manifest is bound; refusing "
                "to change the gap-filtered sample universe"
            )
        fields = (
            "calendar_encoding",
            "time_units_reference",
            "time_axis_summary",
            "data_manifest_sha256",
        )
    for field in fields:
        dataset_value = getattr(dataset, field, None)
        checkpoint_value = getattr(model_config, field, None)
        if dataset_value is None:
            continue
        if checkpoint_value is None:
            raise ValueError(
                f"checkpoint model config is missing {field}, which "
                "is required to prove the geo-season data contract; "
                "refusing to validate/infer without provable date, "
                "geospatial and time-axis semantics"
            )
        if checkpoint_value != dataset_value:
            raise ValueError(
                f"checkpoint {field}={checkpoint_value!r} does not "
                f"match the current HDF5 {field}={dataset_value!r}; "
                "the checkpoint was trained on different date, "
                "geospatial or time-mapping semantics"
            )


class OSTIADailyDataset(Dataset):
    split_ranges = {
        "train": (0.0, 0.7),
        "val": (0.7, 0.9),
        "test": (0.9, 1.0)
    }

    # Supported static-geometry layouts.  Only full-grid, row-aligned
    # 1-D lat/lon vectors are implemented (lat over the image height,
    # lon over the image width, shared by every row of every day).
    # A real-data audit must confirm this layout before geo-season
    # training runs on the server HDF5; anything else fails closed.
    _supported_latlon_ndim = 1

    def __init__(
            self,
            h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask",
            data_manifest=None,
        ):
        if split not in self.split_ranges:
            raise ValueError(
                f"split must be one of {tuple(self.split_ranges)}, "
                f"but got {split}"
            )
        if input_days < 1 or output_days < 1:
            raise ValueError(
                "input_days and output_days must be positive"
            )
        if condition_mode not in CONDITION_MODES:
            raise ValueError(
                f"condition_mode must be one of {CONDITION_MODES}, "
                f"but got {condition_mode!r}"
            )
        self.h5_path = os.path.abspath(h5_path)
        self.split = split
        self.input_days = input_days
        self.output_days = output_days
        self.sequence_days = input_days + output_days
        self.condition_mode = condition_mode
        # Persisted condition-schema facts (round-tripped through the
        # model config and the checkpoint sidecar).
        self.condition_channel_names = condition_channel_names(
            condition_mode,
            input_days,
        )
        self.condition_chans = len(self.condition_channel_names)
        self.condition_schema_version = (
            condition_schema_version_for(condition_mode)
        )
        # Optional real-day data manifest (upstream-proven mapping of
        # each compact HDF5 day to its true daily offset).  Every
        # ablation configuration uses it to share one gap-filtered
        # sample universe; geo-season additionally consumes its dates.
        self.data_manifest_path = (
            os.path.abspath(data_manifest)
            if data_manifest
            else None
        )
        self._manifest = None
        if self.data_manifest_path is not None:
            if not os.path.isfile(self.data_manifest_path):
                raise FileNotFoundError(self.data_manifest_path)
            self._manifest = load_data_manifest(
                self.data_manifest_path
            )
        # Decoded date semantics: None for legacy modes that never
        # needed calendar information; filled fail-closed for the
        # geo-season mode only.
        self.calendar_encoding = None
        self.time_units_reference = None
        self._ref_date = None
        # Real-day time axis facts (geo-season mode only).
        self.time_axis_summary = None
        self.data_manifest_sha256 = None
        # Static geospatial facts: None for legacy modes.
        self.geospatial_summary = None
        self._coordinate_layout = None
        self._longitude_convention = None
        self._geo_lat_lon_channels = None
        self._lat_axes = None
        self._lon_axes = None
        self.day_offsets = (
            np.arange(self.sequence_days, dtype=np.int64)
        )
        self._h5_file = None
        self._h5_pid = None
        self._inspect_file()
        if self._manifest is not None:
            self._validate_manifest_identity()
            if self.condition_mode != "sst_mask_geo_season":
                self._resolve_time_semantics()
        self._resolve_static_geometry()
        self._build_valid_windows()
        self.sst_mean, self.sst_std = (
            self._load_or_estimate_normalization()
        )
        self.normalization = {
            "sst_mean": self.sst_mean,
            "sst_std": self.sst_std,
            "temporal_stride_days": 1,
            "source": "training_split_sample"
        }

    def _inspect_file(self):
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(self.h5_path)
        required = ("sst", "mask", "lat", "lon", "time")
        with h5py.File(self.h5_path, "r") as h5_file:
            missing = [
                name for name in required
                if name not in h5_file
            ]
            if missing:
                raise KeyError(
                    f"Missing HDF5 datasets: {missing}"
                )
            sst = h5_file["sst"]
            mask = h5_file["mask"]
            time = h5_file["time"]
            if sst.ndim != 4 or sst.shape[1] != 1:
                raise ValueError(
                    "sst must have shape [N,1,H,W], "
                    f"but got {sst.shape}"
                )
            if mask.shape != (
                sst.shape[0],
                sst.shape[2],
                sst.shape[3]
            ):
                raise ValueError(
                    "mask shape does not match sst: "
                    f"{mask.shape} versus {sst.shape}"
                )
            self.num_rows = sst.shape[0]
            self.image_shape = tuple(sst.shape[2:])
            self.first_time = int(time[0])
            left = 1
            right = self.num_rows
            while left < right:
                middle = (left + right) // 2
                if int(time[middle]) == self.first_time:
                    left = middle + 1
                else:
                    right = middle
            self.samples_per_day = left
            if self.num_rows % self.samples_per_day != 0:
                raise ValueError(
                    "HDF5 rows do not contain complete daily windows"
                )
            self.num_days = (
                self.num_rows // self.samples_per_day
            )
            if int(time[-1]) != (
                self.first_time + self.num_days - 1
            ):
                raise ValueError(
                    "time values must be consecutive daily indices"
                )
            self.chunk_rows = (
                sst.chunks[0] if sst.chunks else 1
            )
            attrs = dict(h5_file.attrs)
            self._file_attrs = attrs
            self._time_dataset_attrs = dict(time.attrs)
            self._lat_dataset_attrs = dict(
                h5_file["lat"].attrs
            )
            self._lon_dataset_attrs = dict(
                h5_file["lon"].attrs
            )
        self.total_days = self.num_days
        split_start, split_end = self.split_ranges[self.split]
        self.split_start_day = int(
            self.total_days * split_start
        )
        self.split_end_day = int(
            self.total_days * split_end
        )
        self.sequences_per_window = (
            self.split_end_day
            - self.split_start_day
            - self.sequence_days
            + 1
        )
        if self.sequences_per_window < 1:
            raise ValueError(
                f"{self.split} split is shorter than "
                f"{self.sequence_days} days"
            )
        self._file_sst_mean = attrs.get("sst_mean")
        self._file_sst_std = attrs.get("sst_std")

    # ------------------------------------------------------------------
    # Data-manifest identity and real-day window mapping
    # ------------------------------------------------------------------

    def _compact_time_sha256(self):
        """Canonical sha256 of the compact HDF5 time axis.

        The compact time axis holds one value per row (repeated
        samples_per_day times per day); its canonical form is the raw
        little-endian int64 row order, which is what the audit hashes.
        """
        digest = hashlib.sha256()
        h5_file = self._get_file()
        time_dataset = h5_file["time"]
        step = 1_000_000
        for start in range(0, self.num_rows, step):
            stop = min(start + step, self.num_rows)
            block = np.asarray(time_dataset[start:stop])
            digest.update(
                np.ascontiguousarray(block, dtype="<i8").tobytes()
            )
        return digest.hexdigest()

    def _validate_manifest_identity(self):
        """Fail-closed check of the manifest against this HDF5."""
        manifest = self._manifest
        n_days = int(manifest["n_days"])
        if n_days != self.num_days:
            raise ValueError(
                "data manifest n_days="
                f"{n_days} does not match the HDF5 num_days="
                f"{self.num_days}; the manifest belongs to a "
                "different file or a different compacting"
            )
        offsets = manifest["day_offsets"]
        if len(offsets) != n_days:
            raise ValueError(
                "data manifest day_offsets length does not match "
                "its n_days"
            )
        h5_facts = manifest.get("h5") or {}
        expected_samples_per_day = h5_facts.get(
            "samples_per_day"
        )
        if (
                expected_samples_per_day is not None
                and int(expected_samples_per_day)
                != self.samples_per_day
            ):
            raise ValueError(
                "data manifest samples_per_day="
                f"{expected_samples_per_day} does not match the HDF5 "
                f"samples_per_day={self.samples_per_day}"
            )
        recorded_time_sha = h5_facts.get("time_sha256")
        if recorded_time_sha is not None:
            actual_time_sha = self._compact_time_sha256()
            if recorded_time_sha != actual_time_sha:
                raise ValueError(
                    "data manifest compact time-axis sha256 does not "
                    "match this HDF5; the manifest belongs to a "
                    "different file: recorded "
                    f"{recorded_time_sha} vs actual {actual_time_sha}"
                )

    def _build_valid_windows(self):
        """Sequence-start ordinals usable as 22-day forecast windows.

        Without a manifest every compact day is a daily step, so the
        legacy window arithmetic is kept bit-for-bit.  With a
        manifest, a window is only valid when its 22 compact days map
        to 22 *consecutive real* day offsets -- windows that would
        cross a missing-day gap (31 days each in the real series) are
        excluded, so the seasonal phase can never drift over a gap.
        """
        if self._manifest is None:
            self.valid_start_days = list(range(
                self.split_start_day,
                self.split_end_day - self.sequence_days + 1,
            ))
            if len(self.valid_start_days) != self.sequences_per_window:
                raise RuntimeError(
                    "internal error: legacy window count mismatch"
                )
            return
        offsets = self._manifest["day_offsets"]
        valid = []
        for start in range(
                self.split_start_day,
                self.split_end_day - self.sequence_days + 1,
            ):
            base = int(offsets[start])
            consecutive = True
            for step in range(1, self.sequence_days):
                if int(offsets[start + step]) != base + step:
                    consecutive = False
                    break
            if consecutive:
                valid.append(start)
        if not valid:
            raise ValueError(
                f"{self.split} split contains no 22-day window with "
                "consecutive real day offsets under the data manifest "
                "(all windows cross a missing-day gap); cannot build "
                "geo-season samples"
            )
        self.valid_start_days = valid
        self.sequences_per_window = len(valid)

    def _valid_start_day(self, sequence_index):
        return self.valid_start_days[int(sequence_index)]

    @staticmethod
    def _valid_ocean(sst, mask):
        return (
            ((mask.astype(np.uint8) & 2) == 0)
            & np.isfinite(sst)
            & (sst > -5.0)
            & (sst < 350.0)
        )

    def _load_or_estimate_normalization(self):
        if (
            self._file_sst_mean is not None
            and self._file_sst_std is not None
            and float(self._file_sst_std) > 0
        ):
            return (
                float(self._file_sst_mean),
                float(self._file_sst_std)
            )
        train_end_day = int(
            self.total_days
            * self.split_ranges["train"][1]
        )
        train_end_row = min(
            train_end_day
            * self.samples_per_day,
            self.num_rows
        )
        block_rows = min(
            self.chunk_rows,
            self.samples_per_day
        )
        block_count = 8
        max_start = max(0, train_end_row - block_rows)
        starts = np.linspace(
            0,
            max_start,
            block_count,
            dtype=np.int64
        )
        starts = np.unique(
            (starts // self.chunk_rows) * self.chunk_rows
        )
        value_sum = 0.0
        squared_sum = 0.0
        value_count = 0
        with h5py.File(self.h5_path, "r") as h5_file:
            for start in starts:
                end = min(
                    int(start) + block_rows,
                    train_end_row
                )
                sst = np.asarray(
                    h5_file["sst"][int(start):end, 0],
                    dtype=np.float32
                )
                mask = np.asarray(
                    h5_file["mask"][int(start):end],
                    dtype=np.uint8
                )
                valid = self._valid_ocean(sst, mask)
                values = sst[valid].astype(
                    np.float64,
                    copy=False
                )
                value_sum += values.sum()
                squared_sum += np.square(values).sum()
                value_count += values.size
        if value_count < 2:
            raise ValueError(
                "Could not find valid ocean SST values"
            )
        mean = value_sum / value_count
        variance = max(
            squared_sum / value_count - mean * mean,
            1e-12
        )
        return float(mean), float(np.sqrt(variance))

    # ------------------------------------------------------------------
    # Geo-season static geometry (only for condition_mode
    # 'sst_mask_geo_season'; legacy modes never call these paths).
    # ------------------------------------------------------------------

    def _attr_text(self, container, key, fallback_to_file=True):
        """Read one attribute (dataset attrs win over file attrs)."""
        value = container.get(key)
        if (
                value is None
                and fallback_to_file
                and container is not self._file_attrs
            ):
            value = self._file_attrs.get(key)
        return _h5_text(value)

    def _resolve_time_semantics(self):
        """Fail-closed Gregorian date decoding for the seasonal phase.

        The seasonal channels encode the day of year of the last input
        day (t0).  Provenance order:

        1. a data manifest (upstream NetCDF) carries the true daily
           offset of every compact day plus units/calendar;
        2. otherwise the HDF5 itself must prove date semantics:
           ``units`` = ``days since YYYY-MM-DD`` (with optional
           ``calendar``; CF default standard) or ``first_date``.

        Raw integer time is never interpreted as a Unix timestamp.
        """
        if self._manifest is not None:
            units = self._manifest["units"]
            year, month, day = parse_days_since_units(units)
            self._ref_date = date(year, month, day)
            self._reference_is_time_zero = True
            self.time_units_reference = (
                f"days since {self._ref_date.isoformat()}"
            )
            resolved_calendar = self._manifest["calendar"]
            self.calendar_encoding = resolved_calendar
            self._build_time_axis_summary(source="data_manifest")
            return
        units_text = self._attr_text(self._time_dataset_attrs, "units")
        match = None
        if units_text is not None:
            match = re.match(
                r"^\s*days\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})"
                r"(?:[T ].*)?$",
                units_text.strip(),
                flags=re.IGNORECASE,
            )
        if match is not None:
            year, month, day = (int(part) for part in match.groups())
            self._ref_date = date(year, month, day)
            self._reference_is_time_zero = True
            self.time_units_reference = (
                f"days since {self._ref_date.isoformat()}"
            )
        else:
            first_date_text = self._attr_text(
                self._time_dataset_attrs,
                "first_date",
            )
            if first_date_text is None:
                raise ValueError(
                    "condition_mode='sst_mask_geo_season' requires "
                    "provable Gregorian date semantics, but neither a "
                    "data manifest (--data-manifest) nor HDF5 "
                    "metadata was provided: no 'units' attribute of "
                    "the form 'days since YYYY-MM-DD' and no "
                    "'first_date' attribute of the form 'YYYY-MM-DD' "
                    "were found on the time dataset or the file.  "
                    "Integer time values are never interpreted as "
                    "Unix timestamps; add the metadata, pass the "
                    "manifest, or use a legacy condition mode."
                )
            match = re.match(
                r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?$",
                first_date_text.strip(),
            )
            if match is None:
                raise ValueError(
                    "the HDF5 'first_date' attribute must have the "
                    f"form 'YYYY-MM-DD', but got {first_date_text!r}"
                )
            year, month, day = (int(part) for part in match.groups())
            self._ref_date = date(year, month, day)
            self._reference_is_time_zero = False
            self.time_units_reference = (
                f"first_date={self._ref_date.isoformat()}"
            )
        calendar_text = self._attr_text(
            self._time_dataset_attrs,
            "calendar",
        )
        resolved_calendar = (
            "standard"
            if calendar_text is None
            else calendar_text.strip().lower()
        )
        if resolved_calendar not in (
                "standard",
                "gregorian",
                "proleptic_gregorian",
            ):
            raise ValueError(
                "unsupported time calendar "
                f"{calendar_text!r}: only standard/gregorian/"
                "proleptic_gregorian daily time is implemented "
                "(360-day and no-leap calendars fail closed)"
            )
        self.calendar_encoding = resolved_calendar
        self._build_time_axis_summary(source="h5_attrs")

    def _build_time_axis_summary(self, source):
        """Persisted, comparable summary of the real-day time axis.

        ``day_offset_sha256`` pins the exact mapping: manifest mode
        hashes the upstream offsets, attrs mode hashes the absolute
        day offsets implied by the HDF5 time values.  Two files with
        the same shape but a different mapping can never pass the
        checkpoint contract.
        """
        if self._manifest is not None:
            offsets = [int(value)
                       for value in self._manifest["day_offsets"]]
            gaps = self._manifest.get("gaps") or []
            self.data_manifest_sha256 = canonical_manifest_sha256(
                self._manifest
            )
        else:
            if self._reference_is_time_zero:
                offsets = [
                    self.first_time + ordinal
                    for ordinal in range(self.num_days)
                ]
            else:
                offsets = list(range(self.num_days))
            gaps = []
            self.data_manifest_sha256 = None
        from .manifest import day_offset_sha256
        self.time_axis_summary = {
            "source": source,
            "n_days": int(self.num_days),
            "first_time": int(self.first_time),
            "units": str(self.time_units_reference),
            "calendar": str(self.calendar_encoding),
            "day_offset_sha256": day_offset_sha256(offsets),
            "gaps": gaps,
        }

    def _date_for_time(self, time_value):
        """Calendar date of an absolute daily time value."""
        if self._ref_date is None:
            raise ValueError(
                "no reference date is available for this dataset; "
                "seasonal channels cannot be built"
            )
        if self._reference_is_time_zero:
            return (
                self._ref_date
                + timedelta(days=int(time_value))
            )
        return (
            self._ref_date
            + timedelta(days=int(time_value) - int(self.first_time))
        )

    def _date_for_ordinal(self, ordinal):
        """Calendar date of a compact day ordinal.

        Manifest mode resolves the ordinal through the upstream true
        day offsets (so windows after the 31-day gaps keep their real
        dates); attrs mode keeps the historical HDF5 decode.
        """
        ordinal = int(ordinal)
        if self._manifest is not None:
            offsets = self._manifest["day_offsets"]
            return self._ref_date + timedelta(
                days=int(offsets[ordinal])
            )
        if self._reference_is_time_zero:
            return self._ref_date + timedelta(
                days=self.first_time + ordinal
            )
        return self._ref_date + timedelta(days=ordinal)

    def _manifest_coordinate_units(self, name):
        """Coordinate units proven by the data manifest, if any."""
        if self._manifest is None:
            return None
        coordinates = self._manifest.get("coordinates") or {}
        units = coordinates.get("units")
        if isinstance(units, dict):
            units = units.get(name)
        return _h5_text(units)

    @staticmethod
    def _parse_angular_units(text, name):
        """Resolve an angular unit string to 'degrees'/'radians'."""
        if text is None:
            return None
        lowered = str(text).strip().lower()
        if "degree" in lowered or lowered in ("deg",):
            return "degrees"
        if "radian" in lowered:
            return "radians"
        raise ValueError(
            f"unsupported {name} units {text!r}: geo-season mode "
            "accepts degree or radian units only"
        )

    def _resolve_geospatial(self):
        """Dispatch to the proven coordinate layouts.

        Supported layouts (checked on dataset metadata only, before
        any coordinate data is loaded):

        1. ``full_grid_1d_row_aligned``: 1-D ``lat[img_h]`` and 1-D
           ``lon[img_w]`` shared by every row (synthetic/tests);
        2. ``per_spatial_index_patch_axes``: per-row 2-D grids
           ``[num_rows, img_h, img_w]`` where each of the
           ``samples_per_day`` spatial patches carries its own grid
           and the grids repeat identically every day (the real server
           file).  Only the first day's patches are cached (as compact
           per-patch axes); first/middle/last days are sampled to
           prove the repeat.
        """
        h5_file = self._get_file()
        lat_ds = h5_file["lat"]
        lon_ds = h5_file["lon"]
        height, width = self.image_shape
        if (
                lat_ds.ndim == 1
                and lon_ds.ndim == 1
            ):
            if (
                    lat_ds.shape[0] != height
                    or lon_ds.shape[0] != width
                ):
                raise ValueError(
                    "geo-season mode supports only full-grid 1-D "
                    f"lat[{height}] and lon[{width}] datasets, but "
                    f"got lat shape {lat_ds.shape} and lon shape "
                    f"{lon_ds.shape}"
                )
            return self._resolve_geospatial_1d()
        if (
                lat_ds.ndim == 3
                and lon_ds.ndim == 3
                and lat_ds.shape[0] == self.num_rows
                and lon_ds.shape[0] == self.num_rows
                and tuple(lat_ds.shape[1:]) == (height, width)
                and tuple(lon_ds.shape[1:]) == (height, width)
            ):
            return self._resolve_geospatial_per_row()
        raise ValueError(
            "unsupported lat/lon layout: got lat shape "
            f"{list(lat_ds.shape)} and lon shape {list(lon_ds.shape)} "
            f"for image {self.image_shape} with "
            f"samples_per_day={self.samples_per_day}; supported "
            "layouts are full-grid 1-D lat[img_h]/lon[img_w] or "
            "per-row patch grids [num_rows, img_h, img_w]"
        )

    def _resolve_angular_domain(
            self,
            lat_values,
            lon_values,
            units,
        ):
        """Range-check angular values and detect the lon convention.

        Latitude is strictly [-90, 90] degrees / [-pi/2, pi/2]
        radians.  Longitude accepts both common conventions of the
        source grids: [-180, 180] or [0, 360] degrees (and [-pi, pi]
        or [0, 2*pi] radians) -- the real server file uses 0..360
        (first lon patch 80.025..199.975).  sin/cos are 2*pi-periodic
        so both conventions encode identically after conversion.

        Returns (radians_factor, longitude_convention).
        """
        if units == "radians":
            if np.abs(lat_values).max() > np.pi / 2 + 1e-5:
                raise ValueError(
                    "radian latitude outside [-pi/2, pi/2] "
                    "indicates a units mismatch (got "
                    f"max |lat|={float(np.abs(lat_values).max())})"
                )
            convention = self._detect_longitude_convention(
                lon_values,
                half_range=np.pi,
                full_range=2.0 * np.pi,
                units="radians",
            )
            return 1.0, convention
        if np.abs(lat_values).max() > 90.0 + 1e-5:
            raise ValueError(
                "degree latitude outside [-90, 90] indicates a units "
                f"mismatch (got max |lat|="
                f"{float(np.abs(lat_values).max())})"
            )
        convention = self._detect_longitude_convention(
            lon_values,
            half_range=180.0,
            full_range=360.0,
            units="degrees",
        )
        return float(np.pi / 180.0), convention

    def _detect_longitude_convention(self, lon_values, half_range,
                                     full_range, units):
        """Return the longitude convention or fail closed.

        Accepts [-half_range, half_range] (the -180/180 style) or
        [0, full_range] (the 0..360 style); anything else -- values
        below 0 in half style, above full_range, or a mix of the two
        conventions -- is a units mismatch.
        """
        maximum = float(np.abs(lon_values).max())
        minimum = float(lon_values.min())
        # Prefer the full-range reading ([0, 360] style) whenever the
        # values are non-negative: source grids in that convention are
        # ambiguous only when entirely inside [-180, 180], and the
        # sin/cos encoding is identical either way.
        if minimum >= -1e-5 and float(lon_values.max()) <= (
                full_range + 1e-5
            ):
            return f"[0, {full_range:g}]"
        if maximum <= half_range + 1e-5:
            return f"[-{half_range:g}, {half_range:g}]"
        raise ValueError(
            f"{units} longitude values are outside both supported "
            f"conventions [-{half_range:g}, {half_range:g}] and "
            f"[0, {full_range:g}] (got min={minimum:g}, "
            f"max={float(lon_values.max()):g}); this indicates a "
            "units mismatch"
        )

    def _resolve_geospatial_1d(self):
        """Full-grid 1-D lat/lon (shared by every row of every day)."""
        h5_file = self._get_file()
        height, width = self.image_shape
        lat = np.asarray(h5_file["lat"], dtype=np.float64)
        lon = np.asarray(h5_file["lon"], dtype=np.float64)
        if (
                not np.isfinite(lat).all()
                or not np.isfinite(lon).all()
            ):
            raise ValueError(
                "geo-season mode rejects non-finite lat/lon values "
                "(NaN/Inf must not silently propagate)"
            )
        lat_units = self._parse_angular_units(
            self._attr_text(
                self._lat_dataset_attrs,
                "units",
                fallback_to_file=False,
            )
            or self._manifest_coordinate_units("lat"),
            "lat",
        )
        lon_units = self._parse_angular_units(
            self._attr_text(
                self._lon_dataset_attrs,
                "units",
                fallback_to_file=False,
            )
            or self._manifest_coordinate_units("lon"),
            "lon",
        )
        if lat_units is None:
            lat_units = "degrees"
        if lon_units is None:
            lon_units = "degrees"
        if lat_units != lon_units:
            raise ValueError(
                f"lat units {lat_units!r} and lon units {lon_units!r} "
                "do not agree; refusing to guess angular semantics"
            )
        self._angular_factor, self._longitude_convention = (
            self._resolve_angular_domain(
                lat,
                lon,
                lat_units,
            )
        )
        lat_rad = lat * self._angular_factor
        lon_rad = lon * self._angular_factor
        sin_lat = np.broadcast_to(
            np.sin(lat_rad)[:, None],
            (height, width),
        )
        cos_lat = np.broadcast_to(
            np.cos(lat_rad)[:, None],
            (height, width),
        )
        sin_lon = np.broadcast_to(
            np.sin(lon_rad)[None, :],
            (height, width),
        )
        cos_lon = np.broadcast_to(
            np.cos(lon_rad)[None, :],
            (height, width),
        )
        self._coordinate_layout = "full_grid_1d_row_aligned"
        self._geo_lat_lon_channels = np.stack(
            (sin_lat, cos_lat, sin_lon, cos_lon),
            axis=0,
        ).astype(np.float32, copy=False)
        self.geospatial_summary = {
            "encoding": "sin_cos_radians",
            "resolved_units": lat_units,
            "longitude_convention": self._longitude_convention,
            "lat_units_attr": _h5_text(
                self._lat_dataset_attrs.get("units")
            ),
            "lon_units_attr": _h5_text(
                self._lon_dataset_attrs.get("units")
            ),
            "lat_shape": [int(lat.size)],
            "lon_shape": [int(lon.size)],
            "nonfinite": 0,
            "layout": "full_grid_1d_row_aligned",
            # Deterministic coordinate fingerprint: unit-agnostic raw
            # values.  min/max catch coarse swaps cheaply; the SHA-256
            # is computed on the canonical little-endian float64 byte
            # representation of the stored order, so any coordinate
            # change (same shape, same units) changes the digest.
            # NaN/Inf were already rejected above.
            "lat_min": float(lat.min()),
            "lat_max": float(lat.max()),
            "lon_min": float(lon.min()),
            "lon_max": float(lon.max()),
            "lat_sha256": _coordinate_sha256(lat),
            "lon_sha256": _coordinate_sha256(lon),
            "digest_spec": "sha256_le_f8_raw_order",
        }

    def _coord_row(self, name, row_index):
        """Read one coordinate grid row (bounded, ~1.6 MB on 448^2)."""
        return np.asarray(
            self._get_file()[name][int(row_index)],
            dtype=np.float64,
        )

    def _resolve_geospatial_per_row(self):
        """Per-spatial-index patch grids repeated across days.

        Reads only the first day's ``samples_per_day`` patches into
        compact per-patch axes (lat varies along the height, lon along
        the width), then samples the middle and last days row by row
        to prove the grids repeat identically.  The full 1126100-row
        coordinate datasets are never loaded.
        """
        h5_file = self._get_file()
        height, width = self.image_shape
        samples_per_day = self.samples_per_day
        sampled_ordinals = (
            0,
            self.num_days // 2,
            self.num_days - 1,
        )
        lat_axes = []
        lon_axes = []
        max_deviation = 0.0
        nonfinite = 0
        lat_min = lon_min = float("inf")
        lat_max = lon_max = float("-inf")
        for spatial in range(samples_per_day):
            lat_row = self._coord_row("lat", spatial)
            lon_row = self._coord_row("lon", spatial)
            if lat_row.shape != (height, width) or lon_row.shape != (
                    height, width
                ):
                raise ValueError(
                    "per-row coordinate grid shape mismatch: "
                    f"lat {lat_row.shape} / lon {lon_row.shape} "
                    f"versus image {self.image_shape}"
                )
            if not np.isfinite(lat_row).all() or not np.isfinite(
                    lon_row
                ).all():
                raise ValueError(
                    "per-row geo-season mode rejects non-finite "
                    "lat/lon values (NaN/Inf must not silently "
                    "propagate)"
                )
            # Latitudes must be constant along the width, longitudes
            # along the height (proven server layout).
            lat_dev = float(np.abs(
                lat_row - lat_row[:, :1]
            ).max())
            lon_dev = float(np.abs(
                lon_row - lon_row[:1, :]
            ).max())
            max_deviation = max(max_deviation, lat_dev, lon_dev)
            if lat_dev > 1e-6 or lon_dev > 1e-6:
                raise ValueError(
                    "per-row geo-season layout violated: latitude is "
                    f"not constant along the width (max {lat_dev:.3e}) "
                    f"or longitude along the height (max {lon_dev:.3e}) "
                    f"for spatial index {spatial}"
                )
            lat_axis = lat_row[:, 0]
            lon_axis = lon_row[0, :]
            lat_axes.append(lat_axis)
            lon_axes.append(lon_axis)
            lat_min = min(lat_min, float(lat_axis.min()))
            lat_max = max(lat_max, float(lat_axis.max()))
            lon_min = min(lon_min, float(lon_axis.min()))
            lon_max = max(lon_max, float(lon_axis.max()))
        # Cross-day repeat proof on sampled days (per-row reads only).
        identity_exact = True
        identity_max_dev = 0.0
        for ordinal in sampled_ordinals[1:]:
            base = ordinal * samples_per_day
            for spatial in range(samples_per_day):
                lat_row = self._coord_row(
                    "lat", base + spatial
                )
                lon_row = self._coord_row(
                    "lon", base + spatial
                )
                if not np.isfinite(lat_row).all() or not np.isfinite(
                        lon_row
                    ).all():
                    raise ValueError(
                        "per-row coordinates contain non-finite "
                        f"values on sampled day ordinal {ordinal}"
                    )
                lat_expected = np.broadcast_to(
                    lat_axes[spatial][:, None],
                    (height, width),
                )
                lon_expected = np.broadcast_to(
                    lon_axes[spatial][None, :],
                    (height, width),
                )
                if not np.array_equal(lat_row, lat_expected) or not (
                        np.array_equal(lon_row, lon_expected)
                    ):
                    identity_exact = False
                    dev = max(
                        float(np.abs(lat_row - lat_expected).max()),
                        float(np.abs(lon_row - lon_expected).max()),
                    )
                    identity_max_dev = max(identity_max_dev, dev)
                    if dev > 1e-6:
                        raise ValueError(
                            "per-row coordinates are NOT static across "
                            f"days: sampled day ordinal {ordinal} "
                            f"spatial index {spatial} deviates by "
                            f"{dev:.3e} from day 0; refusing to build "
                            "static seasonal channels"
                        )
        lat_units = self._parse_angular_units(
            self._attr_text(
                self._lat_dataset_attrs,
                "units",
                fallback_to_file=False,
            )
            or self._manifest_coordinate_units("lat"),
            "lat",
        )
        lon_units = self._parse_angular_units(
            self._attr_text(
                self._lon_dataset_attrs,
                "units",
                fallback_to_file=False,
            )
            or self._manifest_coordinate_units("lon"),
            "lon",
        )
        if lat_units is None:
            lat_units = "degrees"
        if lon_units is None:
            lon_units = "degrees"
        if lat_units != lon_units:
            raise ValueError(
                f"lat units {lat_units!r} and lon units {lon_units!r} "
                "do not agree; refusing to guess angular semantics"
            )
        lat_stack = np.stack(lat_axes, axis=0)
        lon_stack = np.stack(lon_axes, axis=0)
        self._angular_factor, self._longitude_convention = (
            self._resolve_angular_domain(
                lat_stack,
                lon_stack,
                lat_units,
            )
        )
        self._lat_axes = lat_stack
        self._lon_axes = lon_stack
        self._coordinate_layout = "per_spatial_index_patch_axes"
        self._geo_lat_lon_channels = None
        digest = hashlib.sha256()
        for spatial in range(samples_per_day):
            digest.update(
                np.ascontiguousarray(
                    lat_axes[spatial], dtype="<f8"
                ).tobytes()
            )
            digest.update(
                np.ascontiguousarray(
                    lon_axes[spatial], dtype="<f8"
                ).tobytes()
            )
        self.geospatial_summary = {
            "encoding": "sin_cos_radians",
            "resolved_units": lat_units,
            "longitude_convention": self._longitude_convention,
            "lat_units_attr": _h5_text(
                self._lat_dataset_attrs.get("units")
            ),
            "lon_units_attr": _h5_text(
                self._lon_dataset_attrs.get("units")
            ),
            "layout": "per_spatial_index_patch_axes",
            "samples_per_day": int(samples_per_day),
            "lat_patch_axis": [int(height)],
            "lon_patch_axis": [int(width)],
            "nonfinite": nonfinite,
            "within_patch_constancy": {
                "max_abs_deviation": max_deviation,
                "exact": max_deviation == 0.0,
            },
            "cross_day_identity": {
                "sampled_ordinals": list(sampled_ordinals),
                "exact": identity_exact,
                "max_abs_deviation": identity_max_dev,
            },
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "patch_axes_sha256": digest.hexdigest(),
            "digest_spec": "sha256_le_f8_per_patch_axes",
        }
        del h5_file

    def _geo_grid_channels(self, spatial_index):
        """Static [4,H,W] sin/cos channels of one sample's grid."""
        if self._coordinate_layout == "full_grid_1d_row_aligned":
            return self._geo_lat_lon_channels
        if self._coordinate_layout != "per_spatial_index_patch_axes":
            raise RuntimeError(
                "geo grid channels requested before coordinate layout "
                "resolution"
            )
        height, width = self.image_shape
        spatial_index = int(spatial_index)
        lat_rad = self._lat_axes[spatial_index] * self._angular_factor
        lon_rad = self._lon_axes[spatial_index] * self._angular_factor
        sin_lat = np.broadcast_to(
            np.sin(lat_rad)[:, None],
            (height, width),
        )
        cos_lat = np.broadcast_to(
            np.cos(lat_rad)[:, None],
            (height, width),
        )
        sin_lon = np.broadcast_to(
            np.sin(lon_rad)[None, :],
            (height, width),
        )
        cos_lon = np.broadcast_to(
            np.cos(lon_rad)[None, :],
            (height, width),
        )
        return np.stack(
            (sin_lat, cos_lat, sin_lon, cos_lon),
            axis=0,
        ).astype(np.float32, copy=False)

    def _seasonal_pair_for_ordinal(self, ordinal):
        """(sin_doy, cos_doy) of the calendar date of a compact day.

        The phase uses the real Gregorian year length of the decoded
        date (365 or 366 days) and the standard day-of-year with 1
        being January 1.  It is computed from the last input day (t0)
        only -- never from any future target date -- and resolves
        through the manifest's true day offsets when one is used.
        """
        decoded = self._date_for_ordinal(ordinal)
        year = decoded.year
        leap = (year % 4 == 0) and (
            year % 100 != 0 or year % 400 == 0
        )
        year_length = 366 if leap else 365
        day_of_year = decoded.timetuple().tm_yday
        angle = 2.0 * np.pi * (day_of_year - 1) / year_length
        return float(np.sin(angle)), float(np.cos(angle))

    def _resolve_static_geometry(self):
        if self.condition_mode != "sst_mask_geo_season":
            return
        self._resolve_time_semantics()
        self._resolve_geospatial()

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _get_file(self):
        pid = os.getpid()
        if (
            self._h5_file is None
            or self._h5_pid != pid
        ):
            self.close()
            self._h5_file = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=512 * 1024 ** 2,
                rdcc_nslots=1000003
            )
            self._h5_pid = pid
        return self._h5_file

    def __len__(self):
        return (
            self.sequences_per_window
            * self.samples_per_day
        )

    def _normalize_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    @staticmethod
    def _contiguous_runs(indices):
        if indices.size == 0:
            return []
        starts = [0]
        ends = []
        for index in range(1, indices.size):
            if indices[index] != indices[index - 1] + 1:
                ends.append(index)
                starts.append(index)
        ends.append(indices.size)
        return [
            (
                int(indices[start]),
                int(indices[end - 1]) + 1
            )
            for start, end in zip(starts, ends)
        ]

    def _condition_channels(
            self,
            input_sst,
            t0_mask,
            doy_channels,
            spatial_index,
        ):
        """Build one sample's condition channels [C,H,W] for the mode.

        Every mode is assembled from the same well-defined pieces so
        training, validation and inference share one construction path.
        The first ``input_days`` channels are the normalized SST
        history, the next channel (sst_mask modes) is the t0 valid
        mask; the geo-season mode appends the four static sin/cos
        grids of the sample's spatial patch and the two seasonal
        channels afterwards.
        """
        if self.condition_mode == "sst":
            return input_sst
        if self.condition_mode == "sst_mask":
            return np.concatenate(
                (
                    input_sst,
                    t0_mask[None].astype(
                        np.float32,
                        copy=False
                    )
                ),
                axis=0
            )
        return np.concatenate(
            (
                input_sst,
                t0_mask[None].astype(
                    np.float32,
                    copy=False
                ),
                self._geo_grid_channels(spatial_index),
                doy_channels,
            ),
            axis=0
        )

    def _load_sequence_batch(
            self,
            sequence_index,
            spatial_indices,
        ):
        start_day = self._valid_start_day(sequence_index)
        days = (
            start_day + self.day_offsets
        )
        unique_spatial, restore = np.unique(
            spatial_indices,
            return_inverse=True
        )
        runs = self._contiguous_runs(unique_spatial)
        h5_file = self._get_file()
        sst_days = []
        for day in days:
            base = int(day) * self.samples_per_day
            sst_parts = [
                np.asarray(
                    h5_file["sst"][
                        base + start:base + end,
                        0
                    ],
                    dtype=np.float32
                )
                for start, end in runs
            ]
            sst_days.append(
                sst_parts[0] if len(sst_parts) == 1
                else np.concatenate(sst_parts, axis=0)
            )
        sst = np.stack(
            sst_days,
            axis=0
        ).transpose(1, 0, 2, 3)
        del sst_days
        mask_days = []
        for day in days:
            base = int(day) * self.samples_per_day
            mask_parts = [
                np.asarray(
                    h5_file["mask"][
                        base + start:base + end
                    ],
                    dtype=np.uint8
                )
                for start, end in runs
            ]
            mask_days.append(
                mask_parts[0] if len(mask_parts) == 1
                else np.concatenate(mask_parts, axis=0)
            )
        mask = np.stack(
            mask_days,
            axis=0
        ).transpose(1, 0, 2, 3)
        del mask_days
        if not np.array_equal(
                unique_spatial,
                spatial_indices
            ):
            sst = sst[restore]
            mask = mask[restore]
        times = (
            self.first_time + days
        ).astype(
            np.int64,
            copy=False
        )
        valid = self._valid_ocean(sst, mask)
        sst = np.where(
            valid,
            sst,
            self.sst_mean
        )
        sst = (
            (sst - self.sst_mean) / self.sst_std
        ).astype(np.float32, copy=False)
        doy_channels = None
        if self.condition_mode == "sst_mask_geo_season":
            # One seasonal phase per sequence: the last input day is
            # shared by every spatial sample of this window.  The
            # phase must never read target days or target masks, and
            # it resolves through the real day offsets when a data
            # manifest is used.
            t0_ordinal = start_day + self.input_days - 1
            sin_doy, cos_doy = self._seasonal_pair_for_ordinal(
                t0_ordinal
            )
            doy_channels = np.broadcast_to(
                np.asarray(
                    [sin_doy, cos_doy],
                    dtype=np.float32,
                )[:, None, None],
                (2, self.image_shape[0], self.image_shape[1]),
            )
        samples = []
        for batch_index, spatial_index in enumerate(
                spatial_indices
            ):
            sample_sst = sst[batch_index]
            sample_valid = valid[batch_index]
            input_sst = sample_sst[:self.input_days]
            target = sample_sst[self.input_days:]
            target_mask = sample_valid[
                self.input_days:
            ].astype(
                np.float32,
                copy=False
            )
            condition = self._condition_channels(
                input_sst,
                sample_valid[self.input_days - 1],
                doy_channels,
                spatial_index,
            )
            condition = np.ascontiguousarray(
                condition[..., None]
            )
            target = np.ascontiguousarray(
                target[..., None]
            )
            target_mask = np.ascontiguousarray(
                target_mask[..., None]
            )
            metadata = {
                "sequence_index": np.int64(sequence_index),
                "spatial_index": np.int64(spatial_index),
                "input_start_time": np.int64(times[0]),
                "target_start_time": np.int64(
                    times[self.input_days]
                ),
                "target_end_time": np.int64(times[-1])
            }
            samples.append(
                {
                    "condition": torch.from_numpy(condition),
                    "target": torch.from_numpy(target),
                    "target_mask": torch.from_numpy(target_mask),
                    "metadata": metadata
                }
            )
        return samples

    def __getitems__(self, indices):
        indices = np.asarray(
            [
                self._normalize_index(int(index))
                for index in indices
            ],
            dtype=np.int64
        )
        if indices.size == 0:
            return []
        sequence_indices = (
            indices // self.samples_per_day
        )
        spatial_indices = (
            indices % self.samples_per_day
        )
        samples = [None] * indices.size
        for sequence_index in np.unique(sequence_indices):
            positions = np.flatnonzero(
                sequence_indices == sequence_index
            )
            sequence_samples = self._load_sequence_batch(
                int(sequence_index),
                spatial_indices[positions]
            )
            for position, sample in zip(
                    positions,
                    sequence_samples
                ):
                samples[int(position)] = sample
        return samples

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def inverse_transform_sst(self, value):
        return value * self.sst_std + self.sst_mean

    def close(self):
        h5_file = getattr(self, "_h5_file", None)
        if h5_file is not None:
            h5_file.close()
        self._h5_file = None
        self._h5_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        self.close()
