# 用途：验证真实日期、缺日过滤与空间坐标 manifest。
"""Real-day data manifest + per-row patch coordinate layout tests.

Covers the server facts: manifest-driven true day offsets with two
31-day gaps, window filtering (no window crosses a gap, no seasonal
drift), per-spatial-index patch grids read only from the first day
(never the full 1126100-row coordinate datasets), provenance digests
and fail-closed identity checks.
"""

import json
import os
import sys
import unittest
from datetime import timedelta

import h5py
import numpy as np

from diafno.data.manifest import (
    canonical_manifest_sha256,
    day_offset_sha256,
    load_data_manifest,
)
from diafno.data.ostia import (
    OSTIADailyDataset,
    copy_dataset_provenance,
    verify_checkpoint_data_contract,
)

from .ostia_test_h5 import (
    OSTIATestCase,
    compact_time_sha256,
    make_synthetic_h5,
    write_synthetic_data_manifest,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def gapped_offsets():
    """240 compact days with two 31-day gaps (real-series shape)."""
    return (
        list(range(0, 100))
        + list(range(131, 231))
        + list(range(262, 302))
    )


def _seasonal_expected(dataset, ordinal):
    from datetime import date
    decoded = dataset._date_for_ordinal(ordinal)
    year = decoded.year
    leap = (year % 4 == 0) and (
        year % 100 != 0 or year % 400 == 0
    )
    angle = 2.0 * np.pi * (
        decoded.timetuple().tm_yday - 1
    ) / (366 if leap else 365)
    return float(np.sin(angle)), float(np.cos(angle))


class ManifestGapWindowTests(OSTIATestCase):
    def setUp(self):
        super().setUp()
        self.h5_path = make_synthetic_h5(
            self.tmp_path("gaps.h5"),
            total_days=240,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=0,
            coordinate_layout="per_row",
        )
        self.offsets = gapped_offsets()
        self.manifest_path = self.tmp_path("manifest.json")
        write_synthetic_data_manifest(
            self.manifest_path,
            self.h5_path,
            offsets=self.offsets,
        )
        self.dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
            data_manifest=self.manifest_path,
        )

    def test_windows_filtered_around_gaps(self):
        offsets = self.offsets
        valid = self.dataset.valid_start_days
        # 0..78 valid (before the first gap) + 100..146 valid (inside
        # the second 100-day segment) = 126 starts.
        self.assertEqual(valid[0], 0)
        self.assertEqual(valid[78], 78)
        self.assertEqual(valid[79], 100)
        self.assertEqual(valid[-1], 146)
        self.assertEqual(len(valid), 126)
        for d0 in range(0, 147):
            if d0 in valid:
                self.assertTrue(all(
                    offsets[d0 + step] == offsets[d0] + step
                    for step in range(22)
                ))
            else:
                self.assertFalse(all(
                    offsets[d0 + step] == offsets[d0] + step
                    for step in range(22)
                ))
        # Windows that would cross either gap boundary are excluded.
        for crossing in range(79, 100):
            self.assertNotIn(crossing, valid)
        self.assertEqual(
            len(self.dataset),
            len(valid) * self.dataset.samples_per_day,
        )

    def test_season_uses_true_day_offsets_no_drift(self):
        # Sequence 79 starts at compact day 100 whose true offset is
        # 131; t0 = compact 106 -> true 137.  Using the compact index
        # (106) instead would drift by 31 days.
        dataset = self.dataset
        sequence = 79
        d0 = dataset.valid_start_days[sequence]
        self.assertEqual(d0, 100)
        sample = dataset[sequence * dataset.samples_per_day]
        sin_doy = sample["condition"][12, 0, 0, 0].item()
        cos_doy = sample["condition"][13, 0, 0, 0].item()
        true_expected = _seasonal_expected(dataset, d0 + 6)
        self.assertAlmostEqual(
            sin_doy, true_expected[0], places=6
        )
        self.assertAlmostEqual(
            cos_doy, true_expected[1], places=6
        )
        # The decoded date is really 31 days later than the compact
        # index suggests: prove the mapping is not the compact index.
        decoded = dataset._date_for_ordinal(d0 + 6)
        compact_wrong = dataset._ref_date + timedelta(days=d0 + 6)
        self.assertEqual(
            decoded, dataset._ref_date + timedelta(days=137)
        )
        self.assertNotEqual(decoded, compact_wrong)

    def test_all_windows_have_consecutive_true_offsets(self):
        offsets = np.asarray(self.offsets, dtype=np.int64)
        dataset = self.dataset
        for sequence in range(len(dataset.valid_start_days)):
            d0 = dataset.valid_start_days[sequence]
            window = offsets[d0:d0 + 22]
            self.assertTrue(np.all(
                window == window[0] + np.arange(22)
            ), f"sequence {sequence} (d0={d0}) crosses a gap")

    def test_manifest_identity_facts(self):
        dataset = self.dataset
        self.assertEqual(dataset.time_axis_summary["source"],
                         "data_manifest")
        self.assertEqual(dataset.time_axis_summary["n_days"], 240)
        self.assertEqual(
            dataset.time_axis_summary["day_offset_sha256"],
            day_offset_sha256(self.offsets),
        )
        self.assertEqual(
            dataset.data_manifest_sha256,
            canonical_manifest_sha256(
                load_data_manifest(self.manifest_path)
            ),
        )
        gaps = dataset.time_axis_summary["gaps"]
        self.assertEqual(
            [(gap["after_ordinal"], gap["missing_days"])
             for gap in gaps],
            [(99, 31), (199, 31)],
        )
        self.assertEqual(dataset.calendar_encoding, "standard")
        self.assertEqual(dataset.time_units_reference,
                         "days since 2019-01-01")
        manifest = load_data_manifest(self.manifest_path)
        self.assertEqual(manifest["h5"]["time_sha256"],
                         compact_time_sha256(self.h5_path))

    def test_manifest_n_days_mismatch_fails(self):
        manifest = load_data_manifest(self.manifest_path)
        manifest["n_days"] = 239
        manifest["day_offsets"] = manifest["day_offsets"][:239]
        manifest["day_offset_sha256"] = day_offset_sha256(
            manifest["day_offsets"]
        )
        bad_path = self.tmp_path("bad_ndays.json")
        with open(bad_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)
        with self.assertRaisesRegex(ValueError, "n_days"):
            OSTIADailyDataset(
                h5_path=self.h5_path,
                split="train",
                condition_mode="sst_mask_geo_season",
                data_manifest=bad_path,
            )

    def test_manifest_time_sha_mismatch_fails(self):
        manifest = load_data_manifest(self.manifest_path)
        manifest["h5"]["time_sha256"] = "0" * 64
        bad_path = self.tmp_path("bad_sha.json")
        with open(bad_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)
        with self.assertRaisesRegex(ValueError, "time-axis sha256"):
            OSTIADailyDataset(
                h5_path=self.h5_path,
                split="train",
                condition_mode="sst_mask_geo_season",
                data_manifest=bad_path,
            )

    def test_same_shape_shifted_mapping_fails_contract(self):
        shifted_path = self.tmp_path("shifted.json")
        shifted = [value + 1 for value in self.offsets]
        write_synthetic_data_manifest(
            shifted_path,
            self.h5_path,
            offsets=shifted,
        )
        other = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
            data_manifest=shifted_path,
        )
        model_config = copy_dataset_provenance(
            type(
                "Cfg",
                (),
                {"condition_mode": "sst_mask_geo_season"},
            )(),
            self.dataset,
        )
        # Same HDF5, same shape, different time mapping: fail closed.
        with self.assertRaisesRegex(ValueError, "time_axis_summary"):
            verify_checkpoint_data_contract(
                other, model_config
            )

    def test_manifest_aligns_legacy_mode_to_same_windows(self):
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
            data_manifest=self.manifest_path,
        )
        self.assertEqual(
            legacy.valid_start_days, self.dataset.valid_start_days
        )
        self.assertEqual(legacy.condition_chans, 8)
        self.assertEqual(
            legacy.data_manifest_sha256,
            self.dataset.data_manifest_sha256,
        )
        self.assertEqual(
            legacy.time_axis_summary, self.dataset.time_axis_summary
        )
        index = 79 * legacy.samples_per_day
        self.assertTrue(np.array_equal(
            legacy[index]["condition"].numpy(),
            self.dataset[index]["condition"][:8].numpy(),
        ))
        self.assertTrue(np.array_equal(
            legacy[index]["target"].numpy(),
            self.dataset[index]["target"].numpy(),
        ))

    def test_legacy_checkpoint_binds_manifest_identity(self):
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
            data_manifest=self.manifest_path,
        )
        model_config = copy_dataset_provenance(
            type("Cfg", (), {"condition_mode": "sst_mask"})(),
            legacy,
        )
        without_manifest = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            verify_checkpoint_data_contract(
                without_manifest, model_config
            )

        shifted_path = self.tmp_path("legacy_shifted.json")
        write_synthetic_data_manifest(
            shifted_path,
            self.h5_path,
            offsets=[value + 1 for value in self.offsets],
        )
        shifted = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
            data_manifest=shifted_path,
        )
        with self.assertRaisesRegex(ValueError, "time_axis_summary"):
            verify_checkpoint_data_contract(shifted, model_config)

    def test_manifest_mode_season_matches_attrs_mode_when_identity(self):
        # On a file with provable HDF5 time metadata and an identity
        # manifest (no gaps) both construction paths must produce
        # identical samples.
        identity_path = self.tmp_path("identity.json")
        write_synthetic_data_manifest(
            identity_path,
            self.h5_path,
            offsets=list(range(240)),
        )
        manifest_dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
            data_manifest=identity_path,
        )
        attrs_dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        # Gap-free identity mapping: window lists must agree.
        self.assertEqual(
            manifest_dataset.valid_start_days,
            attrs_dataset.valid_start_days,
        )
        for index in (0, 50, 400):
            self.assertTrue(np.array_equal(
                manifest_dataset[index]["condition"].numpy(),
                attrs_dataset[index]["condition"].numpy(),
            ))
            self.assertTrue(np.array_equal(
                manifest_dataset[index]["target"].numpy(),
                attrs_dataset[index]["target"].numpy(),
            ))
        self.assertEqual(
            manifest_dataset.time_axis_summary["day_offset_sha256"],
            attrs_dataset.time_axis_summary["day_offset_sha256"],
        )


class PerRowCoordinateTests(OSTIATestCase):
    def setUp(self):
        super().setUp()
        self.h5_path = make_synthetic_h5(
            self.tmp_path("per_row.h5"),
            total_days=240,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=0,
            coordinate_layout="per_row",
        )
        self.dataset = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
        )

    def _day0_axes(self):
        with h5py.File(self.h5_path, "r") as file:
            lat_axes = np.stack([
                np.asarray(file["lat"][spatial],
                           dtype=np.float64)[:, 0]
                for spatial in range(4)
            ])
            lon_axes = np.stack([
                np.asarray(file["lon"][spatial],
                           dtype=np.float64)[0, :]
                for spatial in range(4)
            ])
        return lat_axes, lon_axes

    def test_per_row_layout_detected_and_summarised(self):
        summary = self.dataset.geospatial_summary
        self.assertEqual(
            summary["layout"], "per_spatial_index_patch_axes"
        )
        self.assertEqual(summary["samples_per_day"], 4)
        self.assertEqual(summary["lat_patch_axis"], [8])
        self.assertEqual(summary["lon_patch_axis"], [10])
        self.assertEqual(summary["nonfinite"], 0)
        self.assertTrue(summary["within_patch_constancy"]["exact"])
        self.assertTrue(summary["cross_day_identity"]["exact"])
        self.assertEqual(
            summary["cross_day_identity"]["sampled_ordinals"],
            [0, 120, 239],
        )
        lat_axes, lon_axes = self._day0_axes()
        self.assertAlmostEqual(
            summary["lat_min"], float(lat_axes.min()), places=12
        )
        self.assertAlmostEqual(
            summary["lat_max"], float(lat_axes.max()), places=12
        )
        self.assertAlmostEqual(
            summary["lon_min"], float(lon_axes.min()), places=12
        )
        self.assertAlmostEqual(
            summary["lon_max"], float(lon_axes.max()), places=12
        )
        # The fixture longitudes follow the real 0..360 convention.
        self.assertEqual(
            summary["longitude_convention"], "[0, 360]"
        )
        # Digest is deterministic across instances.
        again = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="val",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            again.geospatial_summary["patch_axes_sha256"],
            summary["patch_axes_sha256"],
        )
        # 1-D synthetic layout keeps its own summary contract.
        legacy_path = make_synthetic_h5(
            self.tmp_path("legacy_1d.h5"),
            total_days=60,
            height=8,
            width=10,
            coordinate_layout="1d",
        )
        legacy = OSTIADailyDataset(
            h5_path=legacy_path,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(
            legacy.geospatial_summary["layout"],
            "full_grid_1d_row_aligned",
        )

    def test_static_channels_match_per_patch_axes(self):
        lat_axes, lon_axes = self._day0_axes()
        dataset = self.dataset
        height, width = dataset.image_shape
        lat_rad = np.deg2rad(lat_axes)
        lon_rad = np.deg2rad(lon_axes)
        for spatial in range(4):
            sample = dataset[spatial]
            condition = sample["condition"].numpy()
            sin_lat = np.broadcast_to(
                np.sin(lat_rad[spatial])[:, None],
                (height, width),
            )
            cos_lat = np.broadcast_to(
                np.cos(lat_rad[spatial])[:, None],
                (height, width),
            )
            sin_lon = np.broadcast_to(
                np.sin(lon_rad[spatial])[None, :],
                (height, width),
            )
            cos_lon = np.broadcast_to(
                np.cos(lon_rad[spatial])[None, :],
                (height, width),
            )
            self.assertTrue(np.allclose(
                condition[8, :, :, 0], sin_lat, atol=1e-6
            ))
            self.assertTrue(np.allclose(
                condition[9, :, :, 0], cos_lat, atol=1e-6
            ))
            self.assertTrue(np.allclose(
                condition[10, :, :, 0], sin_lon, atol=1e-6
            ))
            self.assertTrue(np.allclose(
                condition[11, :, :, 0], cos_lon, atol=1e-6
            ))
        # Different spatial patches carry different static grids.
        self.assertFalse(np.allclose(
            dataset[0]["condition"].numpy()[8],
            dataset[1]["condition"].numpy()[8],
        ))

    def test_sst_prefix_matches_legacy_mode(self):
        legacy = OSTIADailyDataset(
            h5_path=self.h5_path,
            split="train",
            condition_mode="sst_mask",
        )
        for index in (0, 7, 100):
            self.assertTrue(np.array_equal(
                self.dataset[index]["condition"].numpy()[:8],
                legacy[index]["condition"].numpy(),
            ))
            self.assertTrue(np.array_equal(
                self.dataset[index]["target"].numpy(),
                legacy[index]["target"].numpy(),
            ))

    def test_coordinate_change_same_shape_fails_contract(self):
        other_path = make_synthetic_h5(
            self.tmp_path("shifted_patches.h5"),
            total_days=240,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=0,
            coordinate_layout="per_row",
        )
        # Perturb the other file's day-0 axes so the patch digest
        # provably changes.
        import shutil
        from tests.ostia_test_h5 import per_spatial_axes
        lat_axes, lon_axes = per_spatial_axes(4, 8, 10)
        with h5py.File(other_path, "r+") as file:
            for spatial in range(4):
                shifted = lat_axes[spatial] + 0.5
                file["lat"][spatial] = np.broadcast_to(
                    shifted[:, None], (8, 10)
                )
                # keep other days untouched (day-repeat still holds
                # for the perturbed first day? no -- day0 differs from
                # mid/last days now, which must fail closed too).
        with self.assertRaisesRegex(ValueError, "NOT static"):
            OSTIADailyDataset(
                h5_path=other_path,
                split="train",
                condition_mode="sst_mask_geo_season",
            )
        # For a clean same-shape change propagate the new axes to
        # every day: rebuild the file entirely instead.
        other_clean = make_synthetic_h5(
            self.tmp_path("other_clean.h5"),
            total_days=240,
            samples_per_day=4,
            height=8,
            width=10,
            first_time=0,
            coordinate_layout="per_row",
        )
        from tests.ostia_test_h5 import per_spatial_axes
        with h5py.File(other_clean, "r+") as file:
            lat_ds = file["lat"]
            lon_ds = file["lon"]
            for day in range(240):
                base = day * 4
                for spatial in range(4):
                    lat_ds[base + spatial] = np.broadcast_to(
                        (per_spatial_axes(4, 8, 10)[0][spatial]
                         + 0.5)[:, None],
                        (8, 10),
                    )
                    lon_ds[base + spatial] = np.broadcast_to(
                        per_spatial_axes(4, 8, 10)[1][spatial][
                            None, :
                        ],
                        (8, 10),
                    )
        other = OSTIADailyDataset(
            h5_path=other_clean,
            split="train",
            condition_mode="sst_mask_geo_season",
        )
        self.assertNotEqual(
            other.geospatial_summary["patch_axes_sha256"],
            self.dataset.geospatial_summary["patch_axes_sha256"],
        )
        model_config = copy_dataset_provenance(
            type(
                "Cfg",
                (),
                {"condition_mode": "sst_mask_geo_season"},
            )(),
            self.dataset,
        )
        with self.assertRaisesRegex(
                ValueError, "geospatial_summary"
            ):
            verify_checkpoint_data_contract(other, model_config)


class NetcdfAuditManifestRoundtripTests(OSTIATestCase):
    def _write_upstream_netcdf(self, path, n_days, offsets,
                               units="days since 2019-01-01"):
        import netCDF4
        dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
        try:
            time_variable = dataset.createDimension("time", n_days)
            time = dataset.createVariable(
                "time", "i4", ("time",)
            )
            time.units = units
            time.calendar = "proleptic_gregorian"
            time[:] = offsets
            latitude = dataset.createDimension(
                "latitude", 3
            )
            lat = dataset.createVariable(
                "latitude", "f8", ("latitude",)
            )
            lat.units = "degrees_north"
            lat[:] = [-10.0, 0.0, 10.0]
            longitude = dataset.createDimension(
                "longitude", 3
            )
            lon = dataset.createVariable(
                "longitude", "f8", ("longitude",)
            )
            lon.units = "degrees_east"
            lon[:] = [170.0, 180.0, -170.0]
        finally:
            dataset.close()

    def test_audit_builds_manifest_consumed_by_dataset(self):
        import audit_ostia_h5 as audit
        offsets = gapped_offsets()
        n_days = len(offsets)
        h5_path = make_synthetic_h5(
            self.tmp_path("netcdf_flow.h5"),
            total_days=n_days,
            samples_per_day=2,
            height=8,
            width=10,
            first_time=0,
            coordinate_layout="per_row",
            # Real server file: no coordinate unit attributes; the
            # units must come from the upstream manifest.
            lat_units=None,
            lon_units=None,
        )
        netcdf_path = self.tmp_path("upstream.nc")
        self._write_upstream_netcdf(
            netcdf_path, n_days, offsets
        )
        payload = audit.audit_h5_to_json(h5_path)
        data_manifest = audit.build_data_manifest_payload(
            netcdf_path, h5_path, payload
        )
        self.assertEqual(data_manifest["n_days"], n_days)
        self.assertEqual(
            data_manifest["day_offsets"][99:102],
            [99, 131, 132],
        )
        self.assertEqual(
            data_manifest["gaps"][0]["missing_days"], 31
        )
        self.assertEqual(
            data_manifest["units"],
            "days since 2019-01-01",
        )
        self.assertEqual(
            data_manifest["coordinates"]["units"]["lat"],
            "degrees_north",
        )
        self.assertEqual(
            data_manifest["h5"]["time_sha256"],
            payload["time_axis_sha256"],
        )
        manifest_path = self.tmp_path("manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(data_manifest, file)
        dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="train",
            condition_mode="sst_mask_geo_season",
            data_manifest=manifest_path,
        )
        self.assertEqual(dataset.time_axis_summary["source"],
                         "data_manifest")
        self.assertEqual(len(dataset.valid_start_days), 126)
        # The upstream coordinate units flow into the dataset summary.
        self.assertEqual(
            dataset.geospatial_summary["resolved_units"],
            "degrees",
        )
        self.assertIsNone(
            dataset.geospatial_summary["lat_units_attr"]
        )
        # Refusal semantics for the manifest output.
        audit.write_audit_report(
            self.tmp_path("report.json"), payload
        )
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            audit.write_audit_report(
                self.tmp_path("report.json"), payload
            )


if __name__ == "__main__":
    unittest.main()
