# 用途：overall coverage 聚合修正（代码路径与修正脚本）单测。
"""Tests for the pooled coverage aggregation and its repair script.

The bug these cover: coverage accumulates a per-sample mean per lead, so
the pooled value must divide by the number of (sample, lead)
contributions.  Dividing by pixels produced ~3e-06 where ~0.82 was
correct, and the earlier test only asserted "within [0, 1]", which the
broken value satisfied.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from scripts.compare_ostia_protocol import (
    ensemble_probabilistic_stats,
    evaluate_protocol,
)
from scripts.recompute_overall_coverage import (
    COVERAGE_FIELDS,
    recompute_overall_coverage,
)
from tests.test_compare_protocol_centered import make_test_fixture


class CoverageAggregationTests(unittest.TestCase):
    """Pooled coverage must be a coverage, not a tiny pixel ratio."""

    def setUp(self):
        self.fixture = make_test_fixture()

    def test_pooled_coverage_matches_the_per_lead_mean(self):
        report = self.fixture.run()
        centered = report["methods"]["A5_centered_DiAFNO"]
        auxiliary = centered["probability_auxiliary"]
        per_lead = [
            centered["by_lead_day"][lead]["coverage_90"]
            for lead in centered["by_lead_day"]
        ]
        self.assertAlmostEqual(
            auxiliary["coverage_90"],
            sum(per_lead) / len(per_lead),
            places=9,
        )
        # An overdispersed ensemble brackets its target almost always;
        # a pixel-count denominator would report ~1e-05 here.
        self.assertGreater(auxiliary["coverage_90"], 0.5)
        self.assertGreater(auxiliary["coverage_50"], 0.5)
        self.assertLessEqual(
            auxiliary["coverage_50"], auxiliary["coverage_90"]
        )

    def test_recompute_repairs_a_pixel_divided_row(self):
        report = self.fixture.report_payload()
        centered = report["methods"]["A5_centered_DiAFNO"]
        auxiliary = centered["probability_auxiliary"]
        pixel_count = 15 * 16 * 16 * 6      # leads x pixels x samples
        for field in COVERAGE_FIELDS:
            auxiliary[field] = auxiliary[field] / pixel_count
        corrected, changes = recompute_overall_coverage(report)
        fixed = corrected["methods"]["A5_centered_DiAFNO"][
            "probability_auxiliary"
        ]
        self.assertTrue(changes)
        for field in COVERAGE_FIELDS:
            self.assertGreater(fixed[field], 0.0)
        self.assertIn(
            "coverage_aggregation_correction",
            corrected["provenance"],
        )
        # Deterministic rows are untouched.
        self.assertNotIn(
            "probability_auxiliary", corrected["methods"]["A5"]
        )


if __name__ == "__main__":
    unittest.main()
