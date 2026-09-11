# 用途：统一协议 Markdown 报告渲染（表格、技能分、bootstrap 区间、概率辅助项）单测。
"""Unit tests for the unified-protocol Markdown report."""

import json
import os
import tempfile
import unittest

from scripts.report_ostia_protocol import build_report, skill


def make_metrics():
    def row(rmse, mae, bias, correlation, crps):
        return {
            "rmse": rmse, "mse": rmse ** 2, "mae": mae, "bias": bias,
            "correlation": correlation, "crps": crps,
        }

    methods = {
        "A5_centered_DiAFNO": {
            "overall": row(0.55, 0.36, -0.02, 0.998, 0.30),
            "by_lead_day": {
                lead: row(0.2, 0.15, 0.0, 0.999, 0.12)
                for lead in ("1", "5", "10", "15")
            },
            "probability_auxiliary": {
                "members": 16, "spread": 0.31, "skill": 0.55,
                "spread_skill_ratio": 0.5636, "coverage_50": 0.41,
                "coverage_90": 0.83, "spread_8members": 0.30,
                "coverage_50_8members": 0.39, "coverage_90_8members": 0.79,
            },
        },
        "A5": {
            "overall": row(0.57, 0.38, -0.04, 0.997, 0.38),
            "by_lead_day": {
                lead: row(0.24, 0.16, 0.0, 0.998, 0.16)
                for lead in ("1", "5", "10", "15")
            },
        },
        "persistence": {
            "overall": row(0.86, 0.60, -0.10, 0.99, 0.60),
            "by_lead_day": {
                lead: row(0.5, 0.4, 0.0, 0.99, 0.4)
                for lead in ("1", "5", "10", "15")
            },
        },
    }
    return {
        "provenance": {
            "split": "test",
            "sample_manifest_sha256": "a" * 64,
            "ensemble_members": 16,
        },
        "num_samples": 200,
        "methods": methods,
        **{
            f"delta_A5_centered_DiAFNO_minus_{other}": {
                "num_blocks": 27,
                "rmse_difference": -0.02,
                "rmse_difference_ci": [-0.05, 0.01],
                "crps_difference": -0.08,
                "crps_difference_ci": [-0.12, -0.04],
            }
            for other in ("A5", "persistence")
        },
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.metrics = make_metrics()
        self.bootstrap = {
            "block_rule": "real t0 rule",
            "block_days": 22,
            "replicates": 2000,
            "seed": 123,
            "deltas": {
                key: value
                for key, value in self.metrics.items()
                if key.startswith("delta_")
            },
        }

    def test_skill_definition(self):
        self.assertAlmostEqual(skill(0.5, 1.0), 0.5)
        self.assertIsNone(skill(0.5, 0))
        self.assertIsNone(skill(0.5, None))

    def test_report_contains_every_required_section(self):
        text = build_report(self.metrics, self.bootstrap)
        for heading in (
                "## 1. 总体指标",
                "## 2. 代表 lead",
                "## 3. 配对真实日块 bootstrap",
                "## 4. 概率辅助项",
                "## 5. 预报图",
                "## 6. 来源与复现",
                "## 7. 限制与不宣称"):
            self.assertIn(heading, text)
        self.assertIn("A5_centered_DiAFNO", text)
        self.assertIn("| persistence |", text)
        self.assertIn("**Day 5**", text)
        # Skill against persistence: MSE 0.3025 vs 0.7396 -> +59%.
        self.assertIn("+59.10%", text)
        self.assertIn("A5_centered_DiAFNO − A5", text)
        self.assertIn("[-0.0500, +0.0100]", text)
        self.assertIn("0.4100", text)

    def test_missing_bootstrap_and_figures_are_stated(self):
        text = build_report(self.metrics)
        self.assertIn("未提供 bootstrap.json", text)
        self.assertIn("未提供 figure_fields.npz", text)

    def test_figures_are_referenced_with_relative_paths(self):
        figures = {
            "samples": 4,
            "crop": "112:336,112:336",
            "images": ["forecast_region_000.png"],
        }
        text = build_report(self.metrics, self.bootstrap, figures=figures)
        self.assertIn("![forecast_region_000.png](figures/forecast_region_000.png)", text)

    def test_ci_absent_is_marked_and_noted(self):
        bootstrap = json.loads(json.dumps(self.bootstrap))
        for entry in bootstrap["deltas"].values():
            entry["rmse_difference_ci"] = None
            entry["crps_difference_ci"] = None
        text = build_report(self.metrics, bootstrap)
        self.assertIn("bootstrap 不适用", text)

    def test_deterministic_rows_have_no_auxiliary_section_row(self):
        text = build_report(self.metrics, self.bootstrap)
        auxiliary_block = text.split("## 4.")[1].split("## 5.")[0]
        self.assertIn("A5_centered_DiAFNO", auxiliary_block)
        self.assertNotIn("| A5 |", auxiliary_block)


class ReportCliTests(unittest.TestCase):
    def test_cli_writes_the_report(self):
        from scripts.report_ostia_protocol import main
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = os.path.join(tmp, "metrics.json")
            bootstrap_path = os.path.join(tmp, "bootstrap.json")
            output_path = os.path.join(tmp, "REPORT.md")
            with open(metrics_path, "w", encoding="utf-8") as file:
                json.dump(make_metrics(), file)
            with open(bootstrap_path, "w", encoding="utf-8") as file:
                json.dump({"deltas": {}}, file)
            argv = sys.argv
            sys.argv = [
                "report", "--metrics", metrics_path,
                "--bootstrap", bootstrap_path, "--output", output_path,
            ]
            try:
                self.assertEqual(main(), 0)
            finally:
                sys.argv = argv
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as file:
                self.assertIn("# ", file.read())


if __name__ == "__main__":
    unittest.main()
