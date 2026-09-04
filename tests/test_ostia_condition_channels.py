"""Geo-season condition channels: values, order, dateline, calendars,
leap/cross-year windows, fail-closed date/geometry rules, fast-path
parity and legacy-mode regression (tests for plan section 6.1)."""

import unittest
from datetime import timedelta

import h5py
import numpy as np
import torch

from diafno.data.condition_schema import (
    GEO_STATIC_CHANNEL_NAMES,
    VALID_MASK_CHANNEL_NAME,
    condition_channel_names,
)
from diafno.data.ostia import (
    OSTIADailyDataset,
    coordinate_sha256,
)

from .ostia_test_h5 import (
    OSTIATestCase,
    make_synthetic_h5,
    time_value_date,
)


def valid_ocean_reference(sst, mask):
    return (
        ((mask.astype(np.uint8) & 2) == 0)
        & np.isfinite(sst)
        & (sst > -5.0)
        & (sst < 350.0)
    )


def _h5_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def expected_sample(h5_path, dataset, sequence_index, spatial_index):
    """Recompute one sample's condition/target/target_mask from the
    raw HDF5 rows with the documented formulas (independent of the
    dataset code path)."""
    input_days = dataset.input_days
    with h5py.File(h5_path, "r") as file:
        days = (
            dataset.split_start_day
            + sequence_index
            + np.arange(dataset.sequence_days, dtype=np.int64)
        )
        sst_rows = np.stack(
            [
                np.asarray(
                    file["sst"][
                        int(day) * dataset.samples_per_day
                        + spatial_index,
                    0,
                ],
                dtype=np.float32,
            )
            for day in days
        ])
        mask_rows = np.stack(
            [
                np.asarray(
                    file["mask"][
                        int(day) * dataset.samples_per_day
                        + spatial_index
                    ],
                    dtype=np.uint8,
                )
                for day in days
            ])
        raw_lat = np.asarray(file["lat"], dtype=np.float64)
        raw_lon = np.asarray(file["lon"], dtype=np.float64)
        time_attrs = dict(file["time"].attrs)
        units = time_attrs.get("units")
        if units is None:
            units = file.attrs.get("units")
        units_text = _h5_text(units)
    valid = valid_ocean_reference(sst_rows, mask_rows)
    mean = float(dataset.sst_mean)
    std = float(dataset.sst_std)
    filled = np.where(valid, sst_rows, mean)
    normalized = ((filled - mean) / std).astype(
        np.float32, copy=False
    )
    input_sst = normalized[:input_days]
    t0_mask = valid[input_days - 1].astype(np.float32, copy=False)
    parts = [input_sst, t0_mask[None]]
    if dataset.condition_mode == "sst_mask_geo_season":
        height, width = dataset.image_shape
        lat_rad = np.deg2rad(raw_lat)
        lon_rad = np.deg2rad(raw_lon)
        latlon = np.stack(
            (
                np.broadcast_to(
                    np.sin(lat_rad)[:, None], (height, width)
                ),
                np.broadcast_to(
                    np.cos(lat_rad)[:, None], (height, width)
                ),
                np.broadcast_to(
                    np.sin(lon_rad)[None, :], (height, width)
                ),
                np.broadcast_to(
                    np.cos(lon_rad)[None, :], (height, width)
                ),
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        parts.append(latlon)
        times = dataset.first_time + days
        anchor_date = time_value_date(
            units_text,
            int(times[input_days - 1]),
        )
        year = anchor_date.year
        leap = (year % 4 == 0) and (
            year % 100 != 0 or year % 400 == 0
        )
        year_length = 366 if leap else 365
        angle = 2.0 * np.pi * (
            anchor_date.timetuple().tm_yday - 1
        ) / year_length
        doy = np.broadcast_to(
            np.asarray(
                [np.sin(angle), np.cos(angle)],
                dtype=np.float32,
            )[:, None, None],
            (2, height, width),
        )
        parts.append(doy)
    if dataset.condition_mode == "sst":
        condition = input_sst
    else:
        condition = np.concatenate(parts, axis=0)
    target = normalized[input_days:]
    target_mask = valid[input_days:].astype(
        np.float32, copy=False
    )
    return {
        "condition": np.ascontiguousarray(condition[..., None]),
        "target": np.ascontiguousarray(target[..., None]),
        "target_mask": np.ascontiguousarray(target_mask[..., None]),
    }


class GeoSeasonConditionTests(OSTIATestCase):
    def setUp(self):
        super().setUp()
        self.h5_path = make_synthetic_h5(
            self.tmp_path("synthetic_geo.h5"),
            total_days=240,
            samples_per_day=1,
            height=8,
            width=10,
            first_time=30,
        )

    def geo_dataset(self, split="train", index=None):
        return OSTIADailyDataset(
            h5_path=self.h5_path,
            split=split,
            input_days=7,
            output_days=15,
            condition_mode="sst_mask_geo_season",
        )

    def test_condition_shape_channels_and_order(self):
        dataset = self.geo_dataset()
        self.assertEqual(
            dataset.condition_channel_names,
            condition_channel_names("sst_mask_geo_season", 7),
        )
        sample = dataset[0]
        self.assertEqual(
            tuple(sample["condition"].shape),
            (14, 8, 10, 1),
        )
        self.assertEqual(
            tuple(sample["target"].shape),
            (15, 8, 10, 1),
        )
        self.assertEqual(
            tuple(sample["target_mask"].shape),
            (15, 8, 10, 1),
        )
        self.assertEqual(sample["condition"].dtype, torch.float32)
        # SST history channels first, then the t0 mask, then the six
        # static channels in the fixed order.
        names = dataset.condition_channel_names
        self.assertEqual(names[:7][0], "sst_tminus6")
        self.assertEqual(names[6], "sst_t0")
        self.assertEqual(names[7], VALID_MASK_CHANNEL_NAME)
        self.assertEqual(
            names[8:],
            GEO_STATIC_CHANNEL_NAMES,
        )

    def test_condition_values_match_documented_formula(self):
        dataset = self.geo_dataset()
        for index in (0, 13, 77, 140):
            sample = dataset[index]
            expected = expected_sample(
                self.h5_path,
                dataset,
                index // dataset.samples_per_day,
                index % dataset.samples_per_day,
            )
            for key in ("condition", "target", "target_mask"):
                self.assertTrue(
                    np.array_equal(
                        sample[key].numpy(),
                        expected[key],
                    ),
                    f"index {index} key {key} differs",
                )

    def test_latitude_channels_vary_along_height_only(self):
        dataset = self.geo_dataset()
        condition = dataset[0]["condition"].numpy()
        sin_lat = condition[8, :, :, 0]
        cos_lat = condition[9, :, :, 0]
        with h5py.File(self.h5_path, "r") as file:
            lat = np.asarray(file["lat"], dtype=np.float64)
        self.assertTrue(np.allclose(
            sin_lat,
            np.sin(np.deg2rad(lat))[:, None],
            atol=1e-6,
        ))
        self.assertTrue(np.allclose(
            cos_lat,
            np.cos(np.deg2rad(lat))[:, None],
            atol=1e-6,
        ))
        # Constant along the width axis.
        self.assertTrue(np.allclose(
            sin_lat,
            np.broadcast_to(
                sin_lat[:, :1], sin_lat.shape
            ),
        ))

    def test_longitude_channels_vary_along_width_only(self):
        dataset = self.geo_dataset()
        condition = dataset[0]["condition"].numpy()
        sin_lon = condition[10, :, :, 0]
        cos_lon = condition[11, :, :, 0]
        with h5py.File(self.h5_path, "r") as file:
            lon = np.asarray(file["lon"], dtype=np.float64)
        self.assertTrue(np.allclose(
            sin_lon,
            np.sin(np.deg2rad(lon))[None, :],
            atol=1e-6,
        ))
        self.assertTrue(np.allclose(
            cos_lon,
            np.cos(np.deg2rad(lon))[None, :],
            atol=1e-6,
        ))
        self.assertTrue(np.allclose(
            sin_lon,
            np.broadcast_to(sin_lon[:1, :], sin_lon.shape),
        ))

    def test_dateline_encoding_is_continuous(self):
        # lon spans [-177, 177] with a true cyclic distance of 6
        # degrees between the endpoints; sin/cos encoding must not
        # introduce any wrap discontinuity at +/-180.
        dataset = self.geo_dataset()
        condition = dataset[0]["condition"].numpy()
        with h5py.File(self.h5_path, "r") as file:
            lon = np.asarray(file["lon"], dtype=np.float64)
        cyclic_delta = 360.0 - (lon[-1] - lon[0])
        self.assertLess(cyclic_delta, 7.0)
        for channel, trig in ((10, np.sin), (11, np.cos)):
            values = condition[channel, :, :, 0]
            self.assertTrue(np.allclose(
                values,
                trig(np.deg2rad(lon))[None, :],
                atol=1e-6,
            ))
            first = values[:, 0]
            last = values[:, -1]
            # Chord distance of the endpoint pair is bounded by the
            # chord of the cyclic gap: no larger-than-physics jump.
            bound = 2.0 * np.sin(np.deg2rad(cyclic_delta) / 2.0)
            self.assertTrue(np.all(
                np.abs(first - last) <= bound + 1e-5
            ))

    def test_seasonal_phase_uses_decoded_gregorian_date(self):
        dataset = self.geo_dataset()
        # first_time=30, units days since 2019-01-01 -> time[0] is
        # 2019-01-31; sample 0's t0 is 6 days later: 2019-02-06
        # (doy 37, year length 365).
        sample = dataset[0]["condition"].numpy()
        sin_doy = sample[12, 0, 0, 0]
        cos_doy = sample[13, 0, 0, 0]
        anchor = time_value_date(
            "days since 2019-01-01",
            dataset.first_time + 6,
        )
        self.assertEqual(anchor.isoformat(), "2019-02-06")
        angle = 2.0 * np.pi * (37 - 1) / 365.0
        self.assertAlmostEqual(float(sin_doy), float(np.sin(angle)),
                               places=6)
        self.assertAlmostEqual(float(cos_doy), float(np.cos(angle)),
                               places=6)
        # The seasonal channels are broadcast over the whole grid.
        height, width = dataset.image_shape
        self.assertTrue(np.allclose(
            sample[12, :, :, 0],
            np.full((height, width), sample[12, 0, 0, 0]),
            atol=0.0,
        ))

    def test_leap_year_february_29_decodes_to_doy_60(self):
        # first_time=418 -> time[418] is 2020-02-23 (2019-01-01 plus
        # 418 days); t0 of sequence 0 is 2020-02-29: doy 60 in a
        # 366-day year.
        h5_path = make_synthetic_h5(
            self.tmp_path("leap.h5"),
            total_days=60,
            first_time=418,
            height=8,
            width=10,
        )
        dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        sample = dataset[0]["condition"].numpy()
        anchor = time_value_date(
            "days since 2019-01-01",
            dataset.first_time + 6,
        )
        self.assertEqual(anchor.isoformat(), "2020-02-29")
        angle = 2.0 * np.pi * (60 - 1) / 366.0
        self.assertAlmostEqual(
            float(sample[12, 0, 0, 0]),
            float(np.sin(angle)),
            places=6,
        )
        self.assertAlmostEqual(
            float(sample[13, 0, 0, 0]),
            float(np.cos(angle)),
            places=6,
        )

    def test_cross_year_window_uses_t0_not_target_dates(self):
        # first_time=719 with units days since 2019-01-01: sequence 0's
        # t0 lands in late December 2020 and the 15 target days run
        # into January 2021.
        h5_path = make_synthetic_h5(
            self.tmp_path("cross_year.h5"),
            total_days=60,
            first_time=719,
            height=8,
            width=10,
        )
        dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        sample = dataset[0]
        sin_doy = sample["condition"][12, 0, 0, 0].item()
        cos_doy = sample["condition"][13, 0, 0, 0].item()
        t0_time = dataset.first_time + 6
        anchor = time_value_date(
            "days since 2019-01-01",
            t0_time,
        )
        # The window really crosses into the next year.
        self.assertEqual(anchor.year, 2020)
        self.assertEqual(anchor.month, 12)
        self.assertGreaterEqual(anchor.day, 25)
        year_length = 366 if (
            anchor.year % 4 == 0
        ) else 365
        angle = 2.0 * np.pi * (
            anchor.timetuple().tm_yday - 1
        ) / year_length
        self.assertAlmostEqual(
            sin_doy, float(np.sin(angle)), places=6
        )
        self.assertAlmostEqual(
            cos_doy, float(np.cos(angle)), places=6
        )
        # The last target date (in January 2021, a 365-day year) must
        # NOT drive the seasonal phase.
        target_anchor = anchor + timedelta(days=14)
        self.assertEqual(target_anchor.year, 2021)
        wrong_angle = 2.0 * np.pi * (
            target_anchor.timetuple().tm_yday - 1
        ) / 365.0
        self.assertNotAlmostEqual(
            sin_doy,
            float(np.sin(wrong_angle)),
            places=3,
        )
        # Metadata still proves the window crosses the year boundary
        # while the target times stay strictly after t0.
        metadata = sample["metadata"]
        self.assertGreater(
            int(metadata["target_start_time"]), t0_time
        )
        self.assertEqual(
            int(metadata["target_end_time"]),
            dataset.first_time + 6 + 15,
        )

    def test_invalid_sst_fills_mean_and_becomes_zero(self):
        dataset = self.geo_dataset()
        sample = dataset[0]
        condition = sample["condition"].numpy()
        target_mask = sample["target_mask"].numpy()
        # Land column (mask bit 2) and NaN/out-of-range pixels: SST
        # and mask channels must be exactly zero after mean filling
        # (the static sin/cos channels legitimately sit over land).
        self.assertTrue(np.all(condition[:8, :, 0, 0] == 0.0))
        # The NaN pixel is in the middle row, last column.
        self.assertTrue(np.all(condition[:8, 4, 9, 0] == 0.0))
        self.assertEqual(condition[7, 4, 9, 0], 0.0)
        # Mask channel: binary 0/1 with both classes represented.
        t0_valid = condition[7, :, :, 0]
        self.assertTrue(np.all(
            (t0_valid == 0.0) | (t0_valid == 1.0)
        ))
        self.assertTrue(np.any(t0_valid == 1.0))
        self.assertTrue(np.any(t0_valid == 0.0))
        # The first target-day mask is the same validity view.
        self.assertTrue(np.array_equal(
            target_mask[0, :, :, 0],
            t0_valid,
        ))

    def test_geo_prefix_is_identical_to_sst_mask_mode(self):
        geo = self.geo_dataset()
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
        )
        for index in (0, 42, 146):
            geo_condition = geo[index]["condition"].numpy()
            legacy_condition = legacy[index]["condition"].numpy()
            self.assertTrue(np.array_equal(
                geo_condition[:8],
                legacy_condition,
            ))
            self.assertTrue(np.array_equal(
                geo[index]["target"].numpy(),
                legacy[index]["target"].numpy(),
            ))
            self.assertTrue(np.array_equal(
                geo[index]["target_mask"].numpy(),
                legacy[index]["target_mask"].numpy(),
            ))

    def test_legacy_sst_mode_is_history_only(self):
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst",
        )
        geo = self.geo_dataset()
        for index in (0, 100):
            self.assertTrue(np.array_equal(
                legacy[index]["condition"].numpy(),
                geo[index]["condition"].numpy()[:7],
            ))
        self.assertEqual(
            legacy[index]["condition"].shape,
            (7, 8, 10, 1),
        )

    def test_target_has_no_coupling_with_static_channels(self):
        # Changing only lat/lon content must not change target or
        # target_mask at all (the forecast targets come from the same
        # raw windows).
        rotated = make_synthetic_h5(
            self.tmp_path("rotated.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lat=np.linspace(-10.0, 12.0, 8),
        )
        dataset = self.geo_dataset()
        other = OSTIADailyDataset(
            h5_path=rotated,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        for index in (0, 50, 120):
            self.assertTrue(np.array_equal(
                dataset[index]["target"].numpy(),
                other[index]["target"].numpy(),
            ))
            self.assertTrue(np.array_equal(
                dataset[index]["target_mask"].numpy(),
                other[index]["target_mask"].numpy(),
            ))
            self.assertFalse(np.array_equal(
                dataset[index]["condition"].numpy(),
                other[index]["condition"].numpy(),
            ))

    def test_getitems_fast_path_matches_getitem(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("multi_spatial.h5"),
            total_days=300,
            samples_per_day=3,
            height=8,
            width=10,
            first_time=12,
        )
        for mode in (
                "sst",
                "sst_mask",
                "sst_mask_geo_season",
            ):
            dataset = OSTIADailyDataset(
                h5_path=h5_path,
                split="train",
                condition_mode=mode,
            )
            indices = [0, 5, 17, 5, 120, 399, 398, 0]
            batch = dataset.__getitems__(indices)
            self.assertEqual(len(batch), len(indices))
            for position, index in enumerate(indices):
                single = dataset[index]
                for key in (
                        "condition",
                        "target",
                        "target_mask",
                    ):
                    self.assertTrue(np.array_equal(
                        batch[position][key].numpy(),
                        single[key].numpy(),
                    ), f"{mode} {key} index {index}")
                self.assertEqual(
                    int(batch[position]["metadata"][
                        "spatial_index"
                    ]),
                    index % 3,
                )

    def test_split_boundaries_unchanged(self):
        dataset = self.geo_dataset()
        self.assertEqual(dataset.split_ranges["train"], (0.0, 0.7))
        self.assertEqual(dataset.split_ranges["val"], (0.7, 0.9))
        self.assertEqual(dataset.split_ranges["test"], (0.9, 1.0))
        self.assertEqual(dataset.total_days, 240)
        self.assertEqual(dataset.split_start_day, 0)
        self.assertEqual(dataset.split_end_day, 168)
        self.assertEqual(
            len(dataset),
            (168 - 22 + 1) * dataset.samples_per_day,
        )
        val = self.geo_dataset(split="val")
        self.assertEqual(
            (val.split_start_day, val.split_end_day),
            (168, 216),
        )
        test_split = self.geo_dataset(split="test")
        self.assertEqual(
            (test_split.split_start_day, test_split.split_end_day),
            (216, 240),
        )
        # Sample windows never reach outside their split window.
        metadata = test_split[0]["metadata"]
        self.assertGreaterEqual(
            int(metadata["input_start_time"]),
            dataset.first_time + 216,
        )
        self.assertLess(
            int(metadata["target_end_time"]),
            dataset.first_time + 240,
        )
        # Train/val/test boundaries of a geo run equal the legacy run.
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            condition_mode="sst_mask",
        )
        self.assertEqual(
            (legacy.split_start_day, legacy.split_end_day),
            (val.split_start_day, val.split_end_day),
        )
        self.assertEqual(
            len(legacy),
            len(val),
        )

    def test_normalization_is_train_only_and_never_sees_static(self):
        dataset = self.geo_dataset()
        self.assertEqual(dataset.sst_mean, 280.0)
        self.assertEqual(dataset.sst_std, 10.0)
        self.assertEqual(
            dataset.normalization["source"],
            "training_split_sample",
        )
        # Same raw content but a different SST normalization leaves the
        # static lat/lon/seasonal channels untouched.
        other_file = make_synthetic_h5(
            self.tmp_path("other_norm.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            sst_mean=290.0,
            sst_std=5.0,
        )
        other = OSTIADailyDataset(
            h5_path=other_file,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        reference = dataset[0]["condition"].numpy()
        normalized_other = other[0]["condition"].numpy()
        self.assertTrue(np.allclose(
            reference[8:],
            normalized_other[8:],
            atol=0.0,
        ))
        self.assertFalse(np.allclose(
            reference[:7],
            normalized_other[:7],
            atol=1e-5,
        ))

    def test_fail_closed_missing_date_metadata(self):
        bare = make_synthetic_h5(
            self.tmp_path("bare.h5"),
            total_days=240,
            first_time=30,
            with_time_metadata=False,
        )
        with self.assertRaisesRegex(ValueError, "units"):
            OSTIADailyDataset(
                h5_path=bare,
                split="train",
                condition_mode="sst_mask_geo_season",
            )
        # The same file still works for the legacy mode: date semantics
        # are only required by the seasonal channels.
        legacy = OSTIADailyDataset(
            h5_path=bare,
            split="train",
            condition_mode="sst_mask",
        )
        self.assertEqual(
            legacy[0]["condition"].shape,
            (8, 8, 10, 1),
        )

    def test_fail_closed_unsupported_calendar(self):
        weird = make_synthetic_h5(
            self.tmp_path("calendar.h5"),
            total_days=240,
            first_time=30,
            calendar="360_day",
        )
        with self.assertRaisesRegex(ValueError, "calendar"):
            OSTIADailyDataset(
                h5_path=weird,
                split="train",
                condition_mode="sst_mask_geo_season",
            )

    def test_fail_closed_nonfinite_latlon(self):
        bad = make_synthetic_h5(
            self.tmp_path("nanlat.h5"),
            total_days=240,
            first_time=30,
            lat=np.array([-80.0, np.nan, 0.0, 10.0, 20.0, 30.0,
                           40.0, 50.0]),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            OSTIADailyDataset(
                h5_path=bad,
                split="train",
                condition_mode="sst_mask_geo_season",
            )

    def test_fail_closed_wrong_latlon_shape(self):
        bad = make_synthetic_h5(
            self.tmp_path("twod_lat.h5"),
            total_days=240,
            first_time=30,
            lat=np.ones((8, 10)),
        )
        with self.assertRaisesRegex(ValueError, "1-D"):
            OSTIADailyDataset(
                h5_path=bad,
                split="train",
                condition_mode="sst_mask_geo_season",
            )

    def test_fail_closed_unknown_angular_units(self):
        bad = make_synthetic_h5(
            self.tmp_path("furlong.h5"),
            total_days=240,
            first_time=30,
            lat_units="furlongs",
            lon_units="furlongs",
        )
        with self.assertRaisesRegex(ValueError, "units"):
            OSTIADailyDataset(
                h5_path=bad,
                split="train",
                condition_mode="sst_mask_geo_season",
            )

    def test_fail_closed_out_of_range_degrees(self):
        bad = make_synthetic_h5(
            self.tmp_path("range.h5"),
            total_days=240,
            first_time=30,
            lat=np.linspace(-90.0, 95.0, 8),
        )
        with self.assertRaisesRegex(ValueError, "units mismatch"):
            OSTIADailyDataset(
                h5_path=bad,
                split="train",
                condition_mode="sst_mask_geo_season",
            )

    def test_geospatial_digest_pins_coordinate_values(self):
        dataset = self.geo_dataset()
        summary = dataset.geospatial_summary
        with h5py.File(self.h5_path, "r") as file:
            lat = np.asarray(file["lat"], dtype=np.float64)
            lon = np.asarray(file["lon"], dtype=np.float64)
        self.assertEqual(float(summary["lat_min"]),
                         float(lat.min()))
        self.assertEqual(float(summary["lat_max"]),
                         float(lat.max()))
        self.assertEqual(float(summary["lon_min"]),
                         float(lon.min()))
        self.assertEqual(float(summary["lon_max"]),
                         float(lon.max()))
        self.assertEqual(
            summary["lat_sha256"],
            coordinate_sha256(lat),
        )
        self.assertEqual(
            summary["lon_sha256"],
            coordinate_sha256(lon),
        )
        self.assertEqual(
            summary["digest_spec"],
            "sha256_le_f8_raw_order",
        )
        # Any single coordinate change flips the digest even when the
        # shape and the units stay identical.
        shifted = make_synthetic_h5(
            self.tmp_path("shifted.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lat=np.linspace(-80.0, 82.0, 8) + 0.001,
        )
        other = OSTIADailyDataset(
            h5_path=shifted,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertNotEqual(
            summary["lat_sha256"],
            other.geospatial_summary["lat_sha256"],
        )
        self.assertEqual(
            summary["lon_sha256"],
            other.geospatial_summary["lon_sha256"],
        )

    def test_longitude_0_to_360_degrees_supported(self):
        # Real server fact: the source grid uses the 0..360 degrees
        # convention (first lon patch 80.025..199.975), which the old
        # |lon|<=180 check would wrongly reject.
        east = make_synthetic_h5(
            self.tmp_path("lon0_360.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lon=np.linspace(80.025, 199.975, 10),
        )
        dataset = OSTIADailyDataset(
            h5_path=east,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            dataset.geospatial_summary["longitude_convention"],
            "[0, 360]",
        )
        condition = dataset[0]["condition"].numpy()
        with h5py.File(east, "r") as file:
            lon = np.asarray(file["lon"], dtype=np.float64)
        self.assertTrue(np.allclose(
            condition[10, :, :, 0],
            np.sin(np.deg2rad(lon))[None, :],
            atol=1e-6,
        ))
        self.assertTrue(np.allclose(
            condition[11, :, :, 0],
            np.cos(np.deg2rad(lon))[None, :],
            atol=1e-6,
        ))
        # Period equivalence: the 0..360 values encode exactly like
        # their -180..180 counterparts.
        negative_form = np.mod(lon + 180.0, 360.0) - 180.0
        self.assertTrue(np.allclose(
            np.cos(np.deg2rad(lon)),
            np.cos(np.deg2rad(negative_form)),
            atol=0.0,
        ))

    def test_longitude_0_to_360_wrap_is_continuous(self):
        wrapped = make_synthetic_h5(
            self.tmp_path("lon_wrap.h5"),
            total_days=240,
            height=8,
            width=12,
            first_time=30,
            lon=np.linspace(0.2, 359.8, 12),
        )
        dataset = OSTIADailyDataset(
            h5_path=wrapped,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        condition = dataset[0]["condition"].numpy()
        cos_lon = condition[11, :, :, 0]
        sin_lon = condition[10, :, :, 0]
        with h5py.File(wrapped, "r") as file:
            lon = np.asarray(file["lon"], dtype=np.float64)
        # Cyclic gap between the last and first column: 0.4 degrees.
        cyclic_delta = 360.0 - (lon[-1] - lon[0])
        self.assertAlmostEqual(cyclic_delta, 0.4, places=12)
        bound = 2.0 * np.sin(np.deg2rad(cyclic_delta) / 2.0)
        self.assertTrue(np.all(
            np.abs(cos_lon[:, 0] - cos_lon[:, -1]) <= bound + 1e-6
        ))
        self.assertTrue(np.all(
            np.abs(sin_lon[:, 0] - sin_lon[:, -1]) <= bound + 1e-6
        ))

    def test_longitude_outside_supported_conventions_rejected(self):
        # A mix/overflow of the [-180,180] and [0,360] conventions is
        # a units bug, not a third convention.
        below = make_synthetic_h5(
            self.tmp_path("lon_below.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lon=np.linspace(-200.0, -100.0, 10),
        )
        with self.assertRaisesRegex(ValueError, "outside both"):
            OSTIADailyDataset(
                h5_path=below,
                split="val",
                condition_mode="sst_mask_geo_season",
            )
        above = make_synthetic_h5(
            self.tmp_path("lon_above.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lon=np.linspace(100.0, 400.0, 10),
        )
        with self.assertRaisesRegex(ValueError, "outside both"):
            OSTIADailyDataset(
                h5_path=above,
                split="val",
                condition_mode="sst_mask_geo_season",
            )

    def test_radian_longitude_supports_both_conventions(self):
        # [-pi, pi] radians.
        minus_pi = make_synthetic_h5(
            self.tmp_path("rad_minus_pi.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lat=np.linspace(-1.3, 1.3, 8),
            lon=np.linspace(-3.0, 3.0, 10),
            lat_units="radians",
            lon_units="radians",
        )
        dataset = OSTIADailyDataset(
            h5_path=minus_pi,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            dataset.geospatial_summary["longitude_convention"],
            "[-3.14159, 3.14159]",
        )
        # [0, 2*pi] radians (real source grid style).
        two_pi = make_synthetic_h5(
            self.tmp_path("rad_0_2pi.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lat=np.linspace(-1.3, 1.3, 8),
            lon=np.linspace(1.4, 5.8, 10),
            lat_units="radians",
            lon_units="radians",
        )
        dataset = OSTIADailyDataset(
            h5_path=two_pi,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            dataset.geospatial_summary["longitude_convention"],
            "[0, 6.28319]",
        )
        condition = dataset[0]["condition"].numpy()
        with h5py.File(two_pi, "r") as file:
            lon_rad = np.asarray(file["lon"], dtype=np.float64)
        self.assertTrue(np.allclose(
            condition[11, :, :, 0],
            np.cos(lon_rad)[None, :],
            atol=1e-6,
        ))
        # Latitude stays strictly bounded in radians too.
        bad_lat = make_synthetic_h5(
            self.tmp_path("rad_bad_lat.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
            lat=np.linspace(-1.8, 1.8, 8),
            lon=np.linspace(0.5, 5.5, 10),
            lat_units="radians",
            lon_units="radians",
        )
        with self.assertRaisesRegex(ValueError, "latitude"):
            OSTIADailyDataset(
                h5_path=bad_lat,
                split="val",
                condition_mode="sst_mask_geo_season",
            )

    def test_invalid_condition_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "condition_mode"):
            OSTIADailyDataset(
                h5_path=self.h5_path,
                split="train",
                condition_mode="sst_mask_geo",
            )


if __name__ == "__main__":
    unittest.main()
