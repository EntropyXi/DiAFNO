# 用途：val-200 扫描汇总脚本（候选读取、排序、两种 best 与 Markdown 表）单测。
"""Unit tests for the val-200 sweep summary writer."""

import json
import os
import tempfile
import unittest

from scripts.summarize_centered_val_sweep import (
    collect_candidates,
    write_summary,
)


def write_candidate(validation_dir, label, *, epoch, global_step, rmse,
                    crps, mae=None, bias=None, with_protocol=False):
    candidate_dir = os.path.join(validation_dir, label)
    os.makedirs(candidate_dir, exist_ok=True)
    with open(
            os.path.join(candidate_dir, "validation.json"),
            "w", encoding="utf-8",
        ) as file:
        json.dump({
            "overall": {
                "rmse": rmse,
                "crps": crps,
                "mae": mae if mae is not None else crps,
                "bias": bias if bias is not None else 0.0,
                "valid_pixels": 1000,
            }
        }, file)
    with open(
            os.path.join(candidate_dir, "checkpoint.json"),
            "w", encoding="utf-8",
        ) as file:
        json.dump({
            "path": f"/tmp/{label}.pth",
            "sha256": f"{label}-sha",
            "epoch": epoch,
            "global_step": global_step,
            "successful_updates": global_step - 2,
            "skipped_optimizer_steps": 2,
        }, file)
    if with_protocol:
        with open(
                os.path.join(candidate_dir, "protocol.json"),
                "w", encoding="utf-8",
            ) as file:
            json.dump({"split": "val", "members": 16}, file)


class SweepSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="val_summary_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_collects_candidates_in_epoch_order(self):
        write_candidate(
            self.tmp, "epoch_010", epoch=10, global_step=2500,
            rmse=0.62, crps=0.41, with_protocol=True,
        )
        write_candidate(
            self.tmp, "epoch_005", epoch=5, global_step=1250,
            rmse=0.63, crps=0.42,
        )
        candidates, protocol = collect_candidates(self.tmp)
        self.assertEqual(
            [entry["label"] for entry in candidates],
            ["epoch_005", "epoch_010"],
        )
        self.assertEqual(protocol, {"split": "val", "members": 16})

    def test_selection_uses_unrounded_values_and_reports_both(self):
        write_candidate(
            self.tmp, "epoch_005", epoch=5, global_step=1250,
            rmse=0.63004, crps=0.40001,
        )
        write_candidate(
            self.tmp, "epoch_015", epoch=15, global_step=3750,
            rmse=0.63001, crps=0.42,
        )
        write_candidate(
            self.tmp, "epoch_030", epoch=30, global_step=7500,
            rmse=0.65, crps=0.39,
        )
        output = os.path.join(self.tmp, "VAL_SWEEP.md")
        summary = write_summary(
            self.tmp, output,
            baseline={"label": "A5 frozen", "rmse": 0.5856, "crps": 0.30},
        )
        self.assertEqual(summary["best_rmse"]["label"], "epoch_015")
        self.assertEqual(summary["best_crps"]["label"], "epoch_030")
        with open(output, "r", encoding="utf-8") as file:
            text = file.read()
        self.assertIn("Frozen best by RMSE: **epoch_015**", text)
        self.assertIn("Frozen best by CRPS: **epoch_030**", text)
        self.assertIn("| A5 frozen |", text)
        self.assertIn("| epoch_005 | 5 | 1250 | 1248 |", text)

    def test_ties_resolve_to_earlier_steps(self):
        write_candidate(
            self.tmp, "epoch_020", epoch=20, global_step=5000,
            rmse=0.6, crps=0.4,
        )
        write_candidate(
            self.tmp, "epoch_010", epoch=10, global_step=2500,
            rmse=0.6, crps=0.4,
        )
        summary = write_summary(
            self.tmp, os.path.join(self.tmp, "out.md")
        )
        self.assertEqual(summary["best_rmse"]["label"], "epoch_010")
        self.assertEqual(summary["best_crps"]["label"], "epoch_010")

    def test_empty_directory_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no validated candidates"):
            write_summary(
                self.tmp, os.path.join(self.tmp, "out.md")
            )


if __name__ == "__main__":
    unittest.main()
