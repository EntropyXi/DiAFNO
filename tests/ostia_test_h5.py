# 用途：构造满足时间、坐标与 mask 协议的合成 HDF5 测试夹具。
"""Shared builders for the OSTIA spatiotemporal ablation tests.

The synthetic HDF5 files mirror the *proven* contract the data code
requires: daily int64 time values with consecutive daily indices,
``sst`` [N,1,H,W] float32, ``mask`` [N,H,W] uint8 (bit 1 = land),
row-aligned 1-D ``lat``/``lon`` and (for geo-season tests) provable
date metadata (``units`` = "days since YYYY-MM-DD", optional
``calendar``) plus file-level ``sst_mean``/``sst_std`` so the
normalization never needs the train-split estimator.
"""

import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta

import h5py
import numpy as np


def reference_date_from_units(units_text, calendar_text=None):
    match = __import__("re").match(
        r"^\s*days\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})",
        units_text.strip(),
    )
    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day)


def time_value_date(units_text, time_value):
    """Calendar date of an absolute daily time value."""
    return (
        reference_date_from_units(units_text)
        + timedelta(days=int(time_value))
    )


def synthetic_field_values(day, spatial, height, width, seed=0):
    """Deterministic smooth synthetic SST content (Kelvin-ish)."""
    rng = np.random.default_rng(seed + day)
    base = 280.0 + 8.0 * np.sin(day * 0.7 + spatial)
    rows = np.linspace(0.05, 0.95, height)[:, None]
    cols = np.linspace(0.05, 0.95, width)[None, :]
    values = (
        base
        + 6.0 * np.cos(2.0 * np.pi * cols + day * 0.13)
        + 3.0 * np.sin(2.0 * np.pi * rows * 2.0 + spatial)
        + rng.standard_normal((height, width)) * 0.05
    )
    return values


def default_grid(day, spatial, height, width, seed=0):
    """Land pattern: mask bit 1 set on the first column and NaN-free;
    additionally one NaN pixel and one >350 spike inside otherwise
    valid ocean to exercise the validity rules."""
    values = synthetic_field_values(
        day, spatial, height, width, seed=seed
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    # Invalid ocean pixels.
    mask[:, 0] = 2  # land
    mask[1, width // 2] = 2
    values[1, width // 2] = 300.0
    values[height // 2, width - 1] = np.nan  # NaN ocean
    values[height - 1, 1] = 400.0  # out of range ocean
    return values.astype(np.float32), mask


def per_spatial_axes(samples_per_day, height, width):
    """Realistic per-patch axes: lat varies along the height (rows),
    lon along the width (columns), distinct per spatial index.

    Longitudes follow the real server convention 0..360 (the first
    lon patch spans ~80..200 degrees) and always stay inside
    [0, 360) with ascending values (no wrapped patch)."""
    lat_axes = np.stack(
        [
            np.linspace(
                -70.0 + 0.5 * spatial,
                -70.0 + 0.5 * spatial + 60.0,
                height,
            )
            for spatial in range(samples_per_day)
        ]
    )
    lon_axes = np.stack(
        [
            np.linspace(
                15.0 + (5.0 * spatial) % 200.0,
                15.0 + (5.0 * spatial) % 200.0 + 100.0,
                width,
            )
            for spatial in range(samples_per_day)
        ]
    )
    return lat_axes, lon_axes


def make_synthetic_h5(
        path,
        total_days=240,
        samples_per_day=1,
        height=8,
        width=10,
        first_time=0,
        units="days since 2019-01-01",
        calendar="standard",
        with_time_metadata=True,
        sst_mean=280.0,
        sst_std=10.0,
        lat=None,
        lon=None,
        lat_units="degrees_north",
        lon_units="degrees_east",
        seed=0,
        coordinate_layout="1d",
    ):
    """Write the canonical synthetic OSTIA file used by the tests.

    ``coordinate_layout`` selects the coordinate datasets: '1d' stores
    full-grid row-aligned lat[height]/lon[width] shared by every row
    (the original synthetic layout); 'per_row' stores the real server
    layout [num_rows, height, width] with one grid per spatial patch,
    latitudes constant along the width, longitudes along the height,
    and every day repeating the day-0 patches exactly.
    """
    rows = total_days * samples_per_day
    if lat is None:
        lat = np.linspace(-80.0, 82.0, height)
    if lon is None:
        lon = np.linspace(-177.0, 177.0, width)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if coordinate_layout == "per_row":
        lat_axes, lon_axes = per_spatial_axes(
            samples_per_day, height, width
        )
        lat_dataset = np.zeros((rows, height, width), dtype=np.float64)
        lon_dataset = np.zeros((rows, height, width), dtype=np.float64)
        for day in range(total_days):
            for spatial in range(samples_per_day):
                row = day * samples_per_day + spatial
                lat_dataset[row] = np.broadcast_to(
                    lat_axes[spatial][:, None],
                    (height, width),
                )
                lon_dataset[row] = np.broadcast_to(
                    lon_axes[spatial][None, :],
                    (height, width),
                )
    elif coordinate_layout == "1d":
        # Original synthetic layout: full-grid row-aligned 1-D
        # lat[height]/lon[width] shared by every row.
        lat_dataset = np.asarray(lat, dtype=np.float64)
        lon_dataset = np.asarray(lon, dtype=np.float64)
    else:
        raise ValueError(
            f"unknown coordinate_layout {coordinate_layout!r}"
        )
    with h5py.File(path, "w") as file:
        sst = file.create_dataset(
            "sst",
            shape=(rows, 1, height, width),
            dtype=np.float32,
            chunks=(max(samples_per_day, 1), 1, height, width),
        )
        mask = file.create_dataset(
            "mask",
            shape=(rows, height, width),
            dtype=np.uint8,
            chunks=(max(samples_per_day, 1), height, width),
        )
        for day in range(total_days):
            for spatial in range(samples_per_day):
                row = day * samples_per_day + spatial
                values, mask_values = default_grid(
                    day,
                    spatial,
                    height,
                    width,
                    seed=seed,
                )
                sst[row, 0] = values
                mask[row] = mask_values
        file["time"] = np.concatenate(
            [
                np.full(
                    samples_per_day,
                    first_time + day,
                    dtype=np.int64,
                )
                for day in range(total_days)
            ]
        )
        time_ds = file["time"]
        if with_time_metadata:
            time_ds.attrs["units"] = units
            if calendar is not None:
                time_ds.attrs["calendar"] = calendar
        file.create_dataset("lat", data=lat_dataset)
        if lat_units is not None:
            file["lat"].attrs["units"] = lat_units
        file.create_dataset("lon", data=lon_dataset)
        if lon_units is not None:
            file["lon"].attrs["units"] = lon_units
        file.attrs["sst_mean"] = float(sst_mean)
        file.attrs["sst_std"] = float(sst_std)
    return path


def compact_time_sha256(h5_path):
    """Canonical sha256 of the full compact time axis (row order)."""
    import hashlib
    with h5py.File(h5_path, "r") as file:
        values = np.asarray(file["time"], dtype=np.int64)
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def write_synthetic_data_manifest(path, h5_path, offsets=None,
                                  units="days since 2019-01-01",
                                  calendar="standard",
                                  coordinate_units=None):
    """Write a data manifest aligned with a synthetic HDF5.

    Default offsets map every compact day to itself plus the file's
    first_time value (the identity mapping the HDF5 time axis
    implies); tests that need real gaps pass custom ``offsets``.
    """
    import json
    from diafno.data.manifest import day_offset_sha256
    with h5py.File(h5_path, "r") as file:
        first_time = int(np.asarray(file["time"][0]))
    num_rows = int(h5py.File(h5_path, "r")["sst"].shape[0])
    samples_per_day = _samples_per_day(h5_path)
    n_days = num_rows // samples_per_day
    if offsets is None:
        offsets = [first_time + day for day in range(n_days)]
    offsets = [int(value) for value in offsets]
    if len(offsets) != n_days:
        raise ValueError(
            f"offsets length {len(offsets)} != n_days {n_days}"
        )
    gaps = []
    for index in range(1, len(offsets)):
        delta = offsets[index] - offsets[index - 1]
        if delta > 1:
            gaps.append({
                "after_ordinal": index - 1,
                "missing_days": delta - 1,
                "next_offset": offsets[index],
            })
    payload = {
        "schema_version": 1,
        "n_days": n_days,
        "day_offsets": offsets,
        "day_offset_sha256": day_offset_sha256(offsets),
        "units": units,
        "calendar": calendar,
        "gaps": gaps,
        "h5": {
            "path": os.path.abspath(h5_path),
            "num_rows": num_rows,
            "samples_per_day": samples_per_day,
            "n_days": n_days,
            "time_sha256": compact_time_sha256(h5_path),
            "coordinate_layout": "synthetic_test",
        },
    }
    if coordinate_units is not None:
        payload["coordinates"] = {"units": coordinate_units}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return payload


def _samples_per_day(h5_path):
    with h5py.File(h5_path, "r") as file:
        time = np.asarray(file["time"], dtype=np.int64)
        first_time = int(time[0])
        num_rows = int(file["sst"].shape[0])
    left, right = 1, num_rows
    while left < right:
        middle = (left + right) // 2
        if int(time[middle]) == first_time:
            left = middle + 1
        else:
            right = middle
    return left


class OSTIATestCase(unittest.TestCase):
    """Temporary-directory lifecycle shared by the OSTIA tests."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ostia_test_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def tmp_path(self, name):
        return os.path.join(self._tmp, name)
