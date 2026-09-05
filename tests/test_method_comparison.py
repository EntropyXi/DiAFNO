import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diafno.evaluation.method_comparison import (
    METHODS, ComparisonScores, empirical_crps, paired_skill_ci,
    draw_forecasts, write_markdown,
)
from diafno.evaluation.validator import OSTIAValidator
from scripts.compare_ostia_methods import predict_physical_members, assert_paired
from scripts.finalize_ostia_comparison import choose_region


def create_layout_demo(output):
    """Synthetic illustration only; not real checkpoint/test performance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[:64, :64]
    scores, cases = ComparisonScores(), []
    for sample in range(3):
        target = np.stack([18 + x / 12 + np.sin(y / 11 + sample) + day * 0.035 for day in range(15)])
        mask = np.broadcast_to((x + y > 25) & (x > 3), target.shape).copy()
        persistence = np.repeat(target[:1] - 0.04, 15, axis=0)
        det = target + np.cos(x / 10) * 0.2
        members = np.stack([target + np.sin(y / 10) * 0.17 + (member - 3.5) * 0.09 for member in range(8)])
        forecasts = dict(DiAFNO=members, IAFNO=det[None], persistence=persistence[None])
        scores.update(forecasts, target, mask, sample * 30)
        case = {method: forecasts[method].mean(axis=0) for method in METHODS}
        case.update(target_mask=mask, metadata=dict(spatial_index=sample, input_start_time=sample * 30))
        cases.append(case)
    report = dict(num_samples=3, provenance=dict(split="SYNTHETIC DEMO — not experiment results", ensemble_members=8, sst_unit="degC", block_days=22, bootstrap_replicates=100), metrics=scores.compute(origin=0, replicates=100))
    report["figures"] = draw_forecasts(cases[:1], output, "degC", dpi=100, title_prefix="SYNTHETIC LAYOUT DEMO | ")
    write_markdown(report, output / "REPORT.md")
    return report


class ComparisonTests(unittest.TestCase):
    def test_selection_respects_ocean_fraction_and_positive_skill(self):
        counts = np.array([[10, 10], [60, 60], [70, 70], [90, 90]])
        main = np.array([[0., 0.], [3., 3.], [8., 8.], [1., 1.]])
        baseline = np.array([[1., 1.], [8., 8.], [10., 10.], [0.5, 0.5]])
        result = choose_region([10, 20, 30, 40], counts, main, baseline, 100)
        self.assertEqual(result['selected']['dataset_index'], 20)
        self.assertFalse(result['candidates'][0]['eligible'])
        with self.assertRaisesRegex(ValueError, 'No candidate'):
            choose_region([10], counts[:1], main[:1], baseline[:1], 100)

    def test_crps_matches_pairwise_formula_and_single_member(self):
        rng = np.random.default_rng(72)
        members, target = rng.normal(size=(8, 11)), rng.normal(size=11)
        reference = np.abs(members - target).mean(axis=0) - 0.5 * np.abs(members[:, None] - members[None, :]).mean(axis=(0, 1))
        np.testing.assert_allclose(empirical_crps(members, target), reference)
        np.testing.assert_allclose(empirical_crps(members[:1], target), np.abs(members[0] - target))
        self.assertAlmostEqual(empirical_crps(np.array([[0.], [2.]]), np.array([1.]))[0], 0.5)

    def test_ensemble_mean_and_all_lead_overall(self):
        target = np.zeros((15, 2, 2))
        mask = np.ones_like(target)
        main = np.stack([target - 1, target + 1])
        deterministic = target.copy()
        deterministic[1] = 3  # Day 2 is outside displayed four days, included in overall.
        mask[1, 0, 0] = 0
        scores = ComparisonScores()
        scores.update(dict(DiAFNO=main, IAFNO=deterministic[None], persistence=(target + 2)[None]), target, mask, 0)
        result = scores.compute(origin=0, replicates=0)
        self.assertEqual(result["overall"]["DiAFNO"]["mse"], 0)
        self.assertAlmostEqual(result["overall"]["DiAFNO"]["crps"], 0.5)
        self.assertAlmostEqual(result["overall"]["IAFNO"]["mse"], 27 / 59)
        self.assertEqual(result["1"]["IAFNO"]["mse"], 0)
        self.assertAlmostEqual(result["overall"]["persistence"]["mse_skill"], 0)
        self.assertAlmostEqual(result["overall"]["persistence"]["crps"], result["overall"]["persistence"]["mae"])

    def test_ci_preserves_ratio_and_zero_baseline(self):
        result = paired_skill_ci([1, 2, 3], [2, 4, 6], [0, 30, 60], origin=0, replicates=100)
        np.testing.assert_allclose(result["interval"], [0.5, 0.5])
        self.assertIsNone(paired_skill_ci([1, 1], [0, 0], [0, 30], origin=0, replicates=10)["interval"])
        self.assertIsNone(paired_skill_ci([1, 1], [2, 2], [0, 1], origin=0, replicates=10)["interval"])

    def test_nonfinite_forecast_rejected_before_accumulation(self):
        target = np.zeros((15, 2, 2))
        forecasts = {m: target[None].copy() for m in METHODS}
        forecasts["IAFNO"][0, 0, 0, 0] = np.nan
        scores = ComparisonScores()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            scores.update(forecasts, target, np.ones_like(target), 0)
        self.assertFalse(scores.times)

    def test_member_prediction_reanchors_once_and_restores_config(self):
        validator = OSTIAValidator.__new__(OSTIAValidator)
        validator.config = SimpleNamespace(seed=123, ensemble_members=16, prediction_mode="model", condition_ablation="none")
        validator.model_config = SimpleNamespace(model_type="centered_diffusion", target_mode="residual", input_days=7, output_days=15)
        validator.dataset = SimpleNamespace(sst_mean=280, sst_std=2)
        validator.amp_enabled = False
        validator.sampling_steps = 16
        seeds = []
        def sample(condition, num_sample_steps, seed):
            seeds.append(seed)
            return torch.ones(1, 15, 2, 2, 1) * 0.5
        validator.model = SimpleNamespace(sample=sample)
        condition = torch.zeros(1, 8, 2, 2, 1)
        condition[:, 6] = 3
        members = predict_physical_members(validator, condition, 12, 2)
        np.testing.assert_allclose(members, 287)  # (0.5 + 3) * 2 + 280
        self.assertEqual(seeds, [12123, 12124])
        self.assertEqual(validator.config.ensemble_members, 16)
        self.assertEqual(validator.config.seed, 123)

    def test_paired_metadata_rejected(self):
        metadata = {key: torch.tensor([1]) for key in ("input_start_time", "target_start_time", "target_end_time", "spatial_index")}
        left = dict(metadata=metadata, target_mask=torch.ones(1))
        right = dict(metadata={**metadata, "spatial_index": torch.tensor([2])}, target_mask=torch.ones(1))
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            assert_paired(left, right)

    def test_figure_and_five_markdown_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            report = create_layout_demo(out)
            self.assertEqual(len(report["metrics"]), 5)
            text = (out / "REPORT.md").read_text(encoding="utf-8")
            self.assertEqual(text.count("| forecast |"), 5)
            self.assertIn("CRPS skill 95% CI", text)
            self.assertTrue((out / "forecast_region_000.png").exists())
            self.assertTrue((out / "forecast_region_000.pdf").exists())

    def test_end_to_end_real_tiny_checkpoint_and_h5(self):
        from diafno.models.config import OSTIAModelConfig
        from scripts.compare_ostia_methods import main
        from tests.ostia_test_h5 import make_synthetic_h5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            h5 = root / "data.h5"
            make_synthetic_h5(str(h5), total_days=260, height=8, width=8)
            for name, model_type in (("main", "diffusion"), ("det", "deterministic")):
                config = OSTIAModelConfig(model_type=model_type, target_mode="residual", image_size=(8, 8, 1), patch_size=(2, 2, 1), embed_dim=8, num_blocks=2, explicit_layer=1, implicit_layer=1, hidden_size_factor=2)
                model = config.build_model(torch.device("cpu"), 2)
                torch.save(dict(config=config.to_checkpoint(), model=model.state_dict(), normalization=dict(sst_mean=280., sst_std=10.)), root / f"{name}.pth")
            argv = ["compare", "--diafno-checkpoint", str(root / "main.pth"), "--iafno-checkpoint", str(root / "det.pth"), "--h5-path", str(h5), "--output-dir", str(root / "output"), "--max-samples", "1", "--plot-samples", "1", "--ensemble-members", "2", "--sampling-steps", "2", "--bootstrap-replicates", "0", "--device", "cpu", "--num-workers", "0", "--dpi", "72"]
            with patch("sys.argv", argv):
                main()
            report = json.loads((root / "output" / "comparison.json").read_text())
            self.assertEqual(report["num_samples"], 1)
            self.assertEqual(len(report["metrics"]), 5)
            self.assertIsNotNone(report["metrics"]["overall"]["DiAFNO"]["crps"])
            selected = argv.copy()
            selected[selected.index("--output-dir") + 1] = str(root / "selected")
            selected.extend(["--sample-index", "0"])
            with patch("sys.argv", selected):
                main()
            chosen = json.loads((root / "selected" / "comparison.json").read_text())
            self.assertEqual(chosen['provenance']['sample_indices'], [0])


if __name__ == "__main__":
    unittest.main()
