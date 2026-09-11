# 用途：统一协议 field dump 与共享色标绘图（裁剪、case 组装、图片写出）单测。
"""Unit tests for the protocol field dump figures.

``build_case`` is the pure part (fixed sample + fixed relative crop);
the plotting path itself is exercised once on a tiny synthetic case to
prove the panels, shared colour scale and both file formats work.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from diafno.evaluation.method_comparison import draw_forecasts
from scripts.plot_ostia_protocol_figures import (
    LEADS,
    build_case,
    parse_crop,
)


def make_fields(samples=2, leads=15, height=32, width=40):
    rng = np.random.default_rng(5)
    fields = {
        "target_kelvin": rng.normal(
            290.0, 3.0, size=(samples, leads, height, width)
        ).astype(np.float16),
        "target_mask": np.ones(
            (samples, leads, height, width), dtype=np.uint8
        ),
        "spatial_index": np.arange(samples),
        "input_date_last": np.asarray(
            [f"2020-01-{position + 1:02d}" for position in range(samples)]
        ),
    }
    for slug in ("a5_centered", "a5", "persistence"):
        fields[f"prediction_{slug}"] = rng.normal(
            290.0, 3.0, size=(samples, leads, height, width)
        ).astype(np.float16)
    return fields


class CropParsingTests(unittest.TestCase):
    def test_parses_a_window(self):
        rows, columns = parse_crop("4:8,1:3")
        self.assertEqual((rows.start, rows.stop), (4, 8))
        self.assertEqual((columns.start, columns.stop), (1, 3))

    def test_rejects_bad_windows(self):
        for text in ("4-8", "8:4,1:3", "0:1,0:0", "a:b,c:d"):
            with self.assertRaises(ValueError):
                parse_crop(text)


class BuildCaseTests(unittest.TestCase):
    def setUp(self):
        self.fields = make_fields()

    def test_crop_is_applied_to_every_panel(self):
        case = build_case(
            self.fields, 1, parse_crop("0:16,10:30"),
            ("a5_centered", "a5"),
        )
        for key in ("target", "a5_centered", "a5"):
            self.assertEqual(case[key].shape, (15, 16, 20))
        self.assertEqual(case["target_mask"].shape, (15, 16, 20))
        self.assertEqual(case["metadata"]["spatial_index"], 1)
        self.assertEqual(case["metadata"]["input_start_time"], "2020-01-02")
        self.assertTrue(np.array_equal(
            case["target"],
            np.asarray(self.fields["target_kelvin"][1],
                       dtype=np.float32)[:, 0:16, 10:30],
        ))

    def test_missing_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prediction_old_iafno"):
            build_case(
                self.fields, 0, parse_crop("0:16,0:16"),
                ("old_iafno",),
            )

    def test_all_land_crop_is_rejected(self):
        fields = dict(self.fields)
        fields["target_mask"] = np.zeros_like(fields["target_mask"])
        with self.assertRaisesRegex(ValueError, "no valid ocean pixel"):
            build_case(fields, 0, parse_crop("0:16,0:16"), ("a5",))


class DrawForecastsTests(unittest.TestCase):
    def test_writes_png_and_pdf_with_shared_scale(self):
        fields = make_fields(samples=1, height=24, width=24)
        case = build_case(
            fields, 0, parse_crop("0:24,0:24"),
            ("a5_centered", "a5", "persistence"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            images = draw_forecasts(
                [case], tmp, unit="K", dpi=60,
                panel_keys=("target", "a5_centered", "a5", "persistence"),
                panel_labels=(
                    "Ground Truth", "A5-centered", "A5", "Persistence",
                ),
                leads=LEADS,
            )
            self.assertEqual(images, ["forecast_region_000.png"])
            for name in ("forecast_region_000.png", "forecast_region_000.pdf"):
                path = os.path.join(tmp, name)
                self.assertTrue(os.path.isfile(path))
                self.assertGreater(os.path.getsize(path), 1000)

    def test_panel_label_length_must_match(self):
        fields = make_fields(samples=1, height=12, width=12)
        case = build_case(fields, 0, parse_crop("0:12,0:12"), ("a5",))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "panel_labels"):
                draw_forecasts(
                    [case], tmp,
                    panel_keys=("target", "a5"),
                    panel_labels=("Ground Truth",),
                    leads=(1,),
                )


if __name__ == "__main__":
    unittest.main()
