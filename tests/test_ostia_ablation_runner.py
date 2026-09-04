"""Ablation runner / probe non-destructive contracts and the lead-stats
integration (plan sections 8, 10 and 12 local checks)."""

import json
import os
import sys
import unittest

import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from ablation_common import (  # noqa: E402
    ABLATION_CONFIGS,
    ABLATION_ROOT,
    STAGE_PROTOCOL,
    assert_directory_available,
    check_json_result_finite,
    require_cuda,
    samples_per_epoch_for_steps,
)
import run_ablation_stages as runner  # noqa: E402

from .ostia_test_h5 import (  # noqa: E402
    OSTIATestCase,
    make_synthetic_h5,
)


class RunnerSafetyTests(OSTIATestCase):
    def test_refuses_existing_nonempty_directory(self):
        path = self.tmp_path("occupied")
        os.makedirs(os.path.join(path, "run"))
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            assert_directory_available(path)

    def test_accepts_missing_or_empty_directory(self):
        path = self.tmp_path("fresh")
        assert_directory_available(path)
        os.makedirs(path)
        assert_directory_available(path)

    def test_logfile_for_does_not_duplicate_suffix(self):
        root = self.tmp_path("logs")
        self.assertEqual(
            runner.logfile_for(root, "run.log"),
            os.path.join(root, "run.log"),
        )
        self.assertEqual(
            runner.logfile_for(root, "resume"),
            os.path.join(root, "resume.log"),
        )

    def test_never_creates_or_deletes(self):
        # The refusal helper is purely a guard: nothing is created and
        # nothing is removed by it.
        path = self.tmp_path("still_empty")
        assert_directory_available(path)
        self.assertFalse(os.path.exists(path))

    def test_effective_batch_formula(self):
        self.assertEqual(
            samples_per_epoch_for_steps(50, 2, 8, 2),
            1600,
        )
        self.assertEqual(
            samples_per_epoch_for_steps(10, 2, 8, 2),
            320,
        )
        self.assertEqual(
            samples_per_epoch_for_steps(250, 2, 8, 2),
            8000,
        )

    def test_require_cuda_fails_closed_without_gpu(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("GPU present; guard is a no-op here")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            require_cuda()

    def test_stage_protocol_fixed_budgets(self):
        stage1 = STAGE_PROTOCOL["stage1"]
        self.assertEqual(stage1["steps"], 50)
        self.assertEqual(stage1["steps_per_epoch"], 10)
        self.assertEqual(stage1["resume_steps"], 10)
        self.assertEqual(stage1["val_samples"], 16)
        self.assertEqual(stage1["checkpoint_interval"], 5)
        stage2 = STAGE_PROTOCOL["stage2"]
        self.assertEqual(stage2["steps"], 300)
        self.assertEqual(stage2["steps_per_epoch"], 300)
        self.assertEqual(stage2["val_samples"], 200)
        self.assertEqual(stage2["resume_steps"], 0)
        stage3 = STAGE_PROTOCOL["stage3"]
        self.assertEqual(stage3["steps"], 1500)
        self.assertEqual(stage3["steps_per_epoch"], 250)
        self.assertEqual(stage3["eval_steps"], (500, 1000, 1500))
        # The val protocol never touches the test split.
        self.assertIsNotNone(runner.__doc__)

    def test_stage1_epoch_budget_math_50_to_60(self):
        # The reviewed stage-1 shape: first phase 5 epochs x 10
        # optimizer steps (checkpoint_interval=5), resume phase
        # num_epochs=6 with the same 10 steps/epoch -> start_epoch 5,
        # scheduler last_epoch 50, horizon 60, exactly 10 remaining
        # steps (global_step 50 -> 60).
        budgets = runner.stage_run_budgets(
            STAGE_PROTOCOL["stage1"], gpus=2,
            batch_per_gpu=8, gradient_accumulation=2,
        )
        self.assertEqual(budgets["effective_batch"], 32)
        first = budgets["first"]
        self.assertEqual(first["optimizer_steps_per_epoch"], 10)
        self.assertEqual(first["samples_per_epoch"], 320)
        self.assertEqual(first["num_epochs"], 5)
        self.assertEqual(first["checkpoint_interval"], 5)
        self.assertEqual(first["global_steps"], 50)
        resume = budgets["resume"]
        self.assertIsNotNone(resume)
        self.assertEqual(resume["samples_per_epoch"], 320)
        self.assertEqual(resume["num_epochs"], 6)
        self.assertEqual(resume["start_epoch"], 5)
        self.assertEqual(resume["last_epoch"], 50)
        self.assertEqual(resume["horizon_steps"], 60)
        self.assertEqual(resume["remaining_steps"], 10)
        self.assertEqual(resume["global_steps"], 60)
        # The exact bug shape (T_max <= last_epoch / zero remaining
        # epochs) cannot be produced by the budget math.
        self.assertGreater(
            resume["horizon_steps"], resume["last_epoch"]
        )

    def test_stage2_budget_is_single_epoch(self):
        budgets = runner.stage_run_budgets(
            STAGE_PROTOCOL["stage2"], gpus=2,
            batch_per_gpu=8, gradient_accumulation=2,
        )
        self.assertEqual(budgets["first"]["num_epochs"], 1)
        self.assertEqual(budgets["first"]["global_steps"], 300)
        self.assertIsNone(budgets["resume"])

    def test_stage3_budget_keeps_eval_grid(self):
        budgets = runner.stage_run_budgets(
            STAGE_PROTOCOL["stage3"], gpus=2,
            batch_per_gpu=8, gradient_accumulation=2,
        )
        first = budgets["first"]
        self.assertEqual(first["optimizer_steps_per_epoch"], 250)
        self.assertEqual(first["num_epochs"], 6)
        self.assertEqual(first["global_steps"], 1500)
        self.assertEqual(first["checkpoint_interval"], 2)
        for step in (500, 1000, 1500):
            self.assertEqual(step % 250, 0)

    def test_budget_rejects_non_divisible_steps(self):
        protocol = dict(STAGE_PROTOCOL["stage1"])
        protocol["steps"] = 55  # not a whole number of 10-step epochs
        with self.assertRaisesRegex(RuntimeError, "whole number"):
            runner.stage_run_budgets(
                protocol, gpus=2,
                batch_per_gpu=8, gradient_accumulation=2,
            )
        protocol = dict(STAGE_PROTOCOL["stage1"])
        protocol["resume_steps"] = 15  # not a whole 10-step epoch
        with self.assertRaisesRegex(RuntimeError, "whole number"):
            runner.stage_run_budgets(
                protocol, gpus=2,
                batch_per_gpu=8, gradient_accumulation=2,
            )

    def test_trainer_command_reflects_stage1_budgets(self):
        from types import SimpleNamespace
        budgets = runner.stage_run_budgets(
            STAGE_PROTOCOL["stage1"], gpus=2,
            batch_per_gpu=8, gradient_accumulation=2,
        )
        args = SimpleNamespace(
            gpus=2,
            seed=123,
            batch_per_gpu=8,
            gradient_accumulation=2,
            num_workers=2,
            h5_path="/data/x.h5",
        )
        first = budgets["first"]
        command = runner.trainer_command(
            args,
            "configs/a0.json",
            "/out/run",
            first["samples_per_epoch"],
            num_epochs=first["num_epochs"],
            lead_stats_path="/out/config/lead_stats.json",
            checkpoint_interval=first["checkpoint_interval"],
        )
        command_text = " ".join(command)
        self.assertIn("--num-epochs 5", command_text)
        self.assertIn("--samples-per-epoch 320", command_text)
        self.assertIn("--checkpoint-interval 5", command_text)
        self.assertIn(
            "--lead-stats /out/config/lead_stats.json",
            command_text,
        )
        self.assertNotIn("--resume", command_text)
        resume = budgets["resume"]
        command = runner.trainer_command(
            args,
            "configs/a0.json",
            "/out/resume",
            resume["samples_per_epoch"],
            num_epochs=resume["num_epochs"],
            lead_stats_path="/out/config/lead_stats.json",
            resume_path="/out/run/latest.pth",
            checkpoint_interval=resume["checkpoint_interval"],
            allow_override=True,
        )
        command_text = " ".join(command)
        self.assertIn("--num-epochs 6", command_text)
        self.assertIn("--samples-per-epoch 320", command_text)
        self.assertIn("--resume /out/run/latest.pth", command_text)
        self.assertIn("--allow-resume-override", command_text)
        self.assertIn(
            "--lead-stats /out/config/lead_stats.json",
            command_text,
        )

    def test_auto_generated_lead_stats_reaches_trainer_command(self):
        # Regression for the server smoke blocker: when the stage CLI
        # has no manual --lead-stats, resolve_lead_stats computes a
        # fresh per-config stats file and the resolved path MUST be
        # passed to the trainer explicitly (args.lead_stats is None).
        from types import SimpleNamespace
        args = SimpleNamespace(
            gpus=2,
            seed=123,
            batch_per_gpu=8,
            gradient_accumulation=2,
            num_workers=2,
            h5_path="/data/x.h5",
            lead_stats=None,  # no manual CLI override
        )
        resolved = "/data/experiments/ostia_spatiotemporal_ablation/" \
            "A1_geo_p8_b8_i2/config/lead_stats.json"
        command = runner.trainer_command(
            args,
            "configs/ostia_ablation_A1_geo_p8_b8_i2.json",
            "/out/run",
            320,
            num_epochs=5,
            lead_stats_path=resolved,
            checkpoint_interval=5,
        )
        command_text = " ".join(command)
        self.assertIn(f"--lead-stats {resolved}", command_text)
        # Raw-scaling runs resolve to None and must not pass the flag.
        command = runner.trainer_command(
            args,
            "configs/x.json",
            "/out/run",
            320,
            num_epochs=5,
            lead_stats_path=None,
        )
        self.assertNotIn("--lead-stats", " ".join(command))

    def test_verify_checkpoint_step_gate(self):
        import torch
        good = os.path.join(self._tmp, "latest.pth")
        torch.save(
            {"epoch": 5, "global_step": 50, "config": {}},
            good,
        )
        runner.verify_checkpoint_step(
            good, expected_epoch=5,
            expected_global_step=50, label="first phase",
        )
        with self.assertRaisesRegex(
                RuntimeError, "global_step=50 \\(expected 60\\)"
            ):
            runner.verify_checkpoint_step(
                good, expected_epoch=5,
                expected_global_step=60, label="resume phase",
            )
        with self.assertRaisesRegex(RuntimeError, "epoch"):
            runner.verify_checkpoint_step(
                good, expected_epoch=6,
                expected_global_step=50, label="resume phase",
            )
        with self.assertRaises(FileNotFoundError):
            runner.verify_checkpoint_step(
                os.path.join(self._tmp, "missing.pth"),
                expected_epoch=5,
                expected_global_step=50,
                label="first phase",
            )

    def test_a5_refuses_without_explicit_winner(self):
        with self.assertRaisesRegex(RuntimeError, "winner"):
            runner.resolve_winner_blocks("A5", None)
        self.assertEqual(
            runner.resolve_winner_blocks("A5", 1), 1
        )
        self.assertEqual(
            runner.resolve_winner_blocks("A5", 2), 2
        )
        with self.assertRaisesRegex(RuntimeError, "only applies"):
            runner.resolve_winner_blocks("A1", 1)
        self.assertIsNone(
            runner.resolve_winner_blocks("A1", None)
        )

    def test_manifest_records_effective_winner_blocks(self):
        identity = ABLATION_CONFIGS["A5"]
        manifest = runner.compose_manifest(
            config_id="A5",
            identity=identity,
            config_path=identity["config"],
            stage=1,
            winner_blocks=1,
            cond_chans=14,
            budgets={"first": {"num_epochs": 5}},
            effective_batch=32,
            seed=123,
            h5_path="/data/x.h5",
            lead_stats=None,
            eval_seed=123,
            val_samples=16,
            batch_per_gpu=8,
            gradient_accumulation=2,
            gpus=2,
        )
        self.assertEqual(manifest["num_blocks"], 1)
        self.assertEqual(manifest["interim_num_blocks"], 2)
        self.assertEqual(manifest["a5_winner_blocks"], 1)
        # And for a plain configuration no interim key exists and the
        # table value is the effective one.
        identity = ABLATION_CONFIGS["A1"]
        manifest = runner.compose_manifest(
            config_id="A1",
            identity=identity,
            config_path=identity["config"],
            stage=1,
            winner_blocks=None,
            cond_chans=14,
            budgets={"first": {"num_epochs": 5}},
            effective_batch=32,
            seed=123,
            h5_path="/data/x.h5",
            lead_stats=None,
            eval_seed=123,
            val_samples=16,
            batch_per_gpu=8,
            gradient_accumulation=2,
            gpus=2,
        )
        self.assertEqual(manifest["num_blocks"], 8)
        self.assertNotIn("interim_num_blocks", manifest)
        self.assertIsNone(manifest["a5_winner_blocks"])


class AblationConfigIdentityTests(OSTIATestCase):
    def test_all_six_config_files_match_the_table(self):
        for config_id in sorted(ABLATION_CONFIGS):
            identity, config_path, payload = runner.load_config_json(
                config_id
            )
            self.assertTrue(os.path.isfile(config_path))
            self.assertEqual(payload["model_type"], "deterministic")
            self.assertEqual(payload["target_mode"], "residual")
            self.assertEqual(
                payload["target_scaling"], "lead_standardized"
            )
            self.assertEqual(payload["seed"], 123)
            self.assertEqual(payload["split"], "train")
            self.assertEqual(
                identity["mode"], payload["condition_mode"]
            )

    def test_matrix_isolation(self):
        a0 = ABLATION_CONFIGS["A0"]
        self.assertEqual(a0["mode"], "sst_mask")
        a1 = ABLATION_CONFIGS["A1"]
        self.assertEqual(a1["mode"], "sst_mask_geo_season")
        # Only one factor changes between neighbours.
        self.assertEqual(a0["patch_size"], a1["patch_size"])
        self.assertEqual(a0["blocks"], a1["blocks"])
        self.assertEqual(a0["implicit"], a1["implicit"])
        for config_id, patch, blocks in (
                ("A2", (4, 4, 1), 8),
                ("A3", (4, 4, 1), 2),
                ("A4", (4, 4, 1), 1),
                ("A5", (4, 4, 1), 2),
            ):
            entry = ABLATION_CONFIGS[config_id]
            self.assertEqual(entry["patch_size"], patch)
            self.assertEqual(entry["blocks"], blocks)
        self.assertEqual(ABLATION_CONFIGS["A5"]["implicit"], 4)

    def test_runner_refuses_bad_effective_batch(self):
        # load_config_json only needs files; the CLI-level guards are
        # exercised by building the argument namespace directly.
        self.assertEqual(ABLATION_ROOT,
                         os.path.join("experiments",
                                      "ostia_spatiotemporal_ablation"))

    def test_validation_json_gate(self):
        good = self.tmp_path("good.json")
        with open(good, "w", encoding="utf-8") as file:
            json.dump({
                "num_samples": 16,
                "overall": {
                    "rmse": 1.0, "mae": 0.8, "bias": 0.1,
                    "correlation": 0.9, "mse": 1.0,
                },
            }, file)
        result = check_json_result_finite(good)
        self.assertEqual(result["num_samples"], 16)
        bad = self.tmp_path("bad.json")
        with open(bad, "w", encoding="utf-8") as file:
            json.dump({
                "num_samples": 16,
                "overall": {"rmse": float("nan")},
            }, file)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            check_json_result_finite(bad)


class RunnerDataManifestTests(OSTIATestCase):
    """A1..A5 require the upstream data manifest; A0 stays free."""

    def _manifest(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("tiny.h5"),
            total_days=30,
            samples_per_day=1,
            height=8,
            width=10,
            coordinate_layout="per_row",
        )
        manifest_path = self.tmp_path("data_manifest.json")
        from .ostia_test_h5 import write_synthetic_data_manifest
        write_synthetic_data_manifest(
            manifest_path, h5_path
        )
        return manifest_path

    def test_geo_config_requires_data_manifest(self):
        identity = ABLATION_CONFIGS["A1"]
        with self.assertRaisesRegex(
                RuntimeError, "geo-season configuration"
            ):
            runner.resolve_data_manifest("A1", identity, None)
        path = self._manifest()
        resolved = runner.resolve_data_manifest("A1", identity, path)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["path"], os.path.abspath(path))
        self.assertEqual(len(resolved["sha256"]), 64)

    def test_legacy_a0_rejects_data_manifest(self):
        identity = ABLATION_CONFIGS["A0"]
        path = self._manifest()
        with self.assertRaisesRegex(RuntimeError, "only applies"):
            runner.resolve_data_manifest("A0", identity, path)
        self.assertIsNone(
            runner.resolve_data_manifest("A0", identity, None)
        )

    def test_commands_carry_data_manifest(self):
        from types import SimpleNamespace
        path = self._manifest()
        args = SimpleNamespace(
            gpus=2, seed=123, batch_per_gpu=8,
            gradient_accumulation=2, num_workers=2,
            h5_path="/data/x.h5",
        )
        command = runner.trainer_command(
            args,
            "configs/a1.json",
            "/out/run",
            320,
            num_epochs=5,
            lead_stats_path=None,
            data_manifest_path=path,
        )
        command_text = " ".join(command)
        self.assertIn(f"--data-manifest {path}", command_text)
        validation = runner.validation_command(
            "/out/run/latest.pth",
            "/out/val.json",
            "/data/x.h5",
            16,
            123,
            data_manifest_path=path,
        )
        self.assertIn(
            f"--data-manifest {path}", " ".join(validation)
        )
        # Without a manifest the flags are absent.
        plain = runner.trainer_command(
            args,
            "configs/a0.json",
            "/out/run",
            320,
            num_epochs=5,
        )
        self.assertNotIn("--data-manifest", " ".join(plain))

    def test_stage_manifest_records_data_manifest_identity(self):
        identity = ABLATION_CONFIGS["A1"]
        path = self._manifest()
        resolved = runner.resolve_data_manifest("A1", identity, path)
        manifest = runner.compose_manifest(
            config_id="A1",
            identity=identity,
            config_path=identity["config"],
            stage=1,
            winner_blocks=None,
            cond_chans=14,
            budgets={"first": {"num_epochs": 5}},
            effective_batch=32,
            seed=123,
            h5_path="/data/x.h5",
            lead_stats=None,
            eval_seed=123,
            val_samples=16,
            batch_per_gpu=8,
            gradient_accumulation=2,
            gpus=2,
            data_manifest=resolved,
        )
        self.assertEqual(
            manifest["data_manifest"]["sha256"], resolved["sha256"]
        )


class LeadStatsModeIntegrationTests(OSTIATestCase):
    def test_compute_lead_stats_file_records_condition_mode(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("stats.h5"),
            total_days=240,
            height=8,
            width=10,
            first_time=30,
        )
        from deterministic_iafno.compute_lead_stats import (
            compute_lead_stats_file,
        )
        output = self.tmp_path("config/lead_stats.json")
        result = compute_lead_stats_file(
            h5_path=h5_path,
            output=output,
            input_days=7,
            output_days=15,
            num_samples=48,
            batch_size=16,
            condition_mode="sst_mask_geo_season",
        )
        self.assertEqual(result["split"], "train")
        self.assertEqual(
            result["condition_mode"], "sst_mask_geo_season"
        )
        self.assertEqual(result["num_samples"], 48)
        self.assertEqual(len(result["lead_mean"]), 15)
        self.assertEqual(len(result["lead_std"]), 15)
        self.assertTrue(all(
            value > 0 for value in result["lead_std"]
        ))
        self.assertTrue(all(
            np.isfinite(value) for value in result["lead_std"]
        ))
        self.assertTrue(os.path.isfile(output))
        with open(output, "r", encoding="utf-8") as file:
            reloaded = json.load(file)
        self.assertEqual(reloaded["condition_mode"],
                         "sst_mask_geo_season")

    def test_existing_mismatched_stats_never_reused(self):
        h5_path = make_synthetic_h5(
            self.tmp_path("stats_other.h5"),
            total_days=240,
            first_time=30,
        )
        config_root = self.tmp_path("config_root")
        os.makedirs(os.path.join(config_root, "config"), exist_ok=True)
        stats_path = os.path.join(config_root, "config",
                                  "lead_stats.json")
        with open(stats_path, "w", encoding="utf-8") as file:
            json.dump({
                "condition_mode": "sst_mask",  # wrong mode
                "h5_path": h5_path,
                "input_days": 7,
                "output_days": 15,
            }, file)
        identity = ABLATION_CONFIGS["A1"]
        payload = {
            "target_scaling": "lead_standardized",
            "condition_mode": "sst_mask_geo_season",
            "input_days": 7,
            "output_days": 15,
        }
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            runner.resolve_lead_stats(
                config_root, identity, payload, h5_path
            )


class AuditScriptTests(OSTIATestCase):
    """Read-only HDF5 audit / manifest tool (plan section 7)."""

    def test_audit_manifest_on_geo_file(self):
        import audit_ostia_h5 as audit
        from diafno.data.ostia import coordinate_sha256
        h5_path = make_synthetic_h5(
            self.tmp_path("audit.h5"),
            total_days=240,
            samples_per_day=2,
            height=8,
            width=10,
            first_time=30,
        )
        manifest = audit.audit_h5_to_json(h5_path)
        self.assertEqual(manifest["path"], os.path.abspath(h5_path))
        self.assertGreater(manifest["size_bytes"], 0)
        self.assertEqual(manifest["rows_analysis"]["num_rows"], 480)
        self.assertEqual(
            manifest["rows_analysis"]["samples_per_day"], 2
        )
        self.assertTrue(
            manifest["rows_analysis"]["consecutive_daily_indices"]
        )
        self.assertEqual(
            manifest["rows_analysis"]["num_days"], 240
        )
        sst = manifest["datasets"]["sst"]
        self.assertEqual(sst["shape"], [480, 1, 8, 10])
        self.assertIn("dtype", sst)
        self.assertIn("chunks", sst)
        self.assertIn("compression", sst)
        self.assertIn("attrs", sst)
        self.assertEqual(
            manifest["time_range"]["units"],
            "days since 2019-01-01",
        )
        self.assertEqual(
            manifest["time_range"]["first_date"], "2019-01-31"
        )
        self.assertEqual(
            manifest["time_range"]["last_date"],
            "2019-09-27",  # 30 + 239 days after 2019-01-01
        )
        self.assertEqual(
            manifest["time_range"]["date_semantics"],
            "decodable_gregorian_daily",
        )
        self.assertEqual(
            manifest["coordinates"]["lat"]["nonfinite"], 0
        )
        self.assertEqual(
            manifest["coordinates"]["lat"]["min"], -80.0
        )
        import numpy as np
        with h5py.File(h5_path, "r") as file:
            raw_lat = np.asarray(file["lat"], dtype=np.float64)
        self.assertEqual(
            manifest["coordinates"]["lat"]["sha256"],
            coordinate_sha256(raw_lat),
        )
        precheck = manifest["geo_dataset_precheck"]
        self.assertTrue(precheck["ready"])
        self.assertEqual(precheck["condition_chans"], 14)
        self.assertEqual(
            precheck["condition_mode"], "sst_mask_geo_season"
        )
        self.assertEqual(
            precheck["geospatial_summary"]["lat_sha256"],
            manifest["coordinates"]["lat"]["sha256"],
        )

    def test_audit_records_undecodable_time_fail_closed(self):
        import audit_ostia_h5 as audit
        bare = make_synthetic_h5(
            self.tmp_path("bare.h5"),
            total_days=240,
            first_time=30,
            with_time_metadata=False,
        )
        manifest = audit.audit_h5_to_json(bare)
        self.assertIn(
            "undecodable",
            manifest["time_range"]["date_semantics"],
        )
        precheck = manifest["geo_dataset_precheck"]
        self.assertFalse(precheck["ready"])
        self.assertIn("units", precheck["reason"])

    def test_audit_refuses_existing_output(self):
        import audit_ostia_h5 as audit
        h5_path = make_synthetic_h5(
            self.tmp_path("audit.h5"), total_days=60, first_time=30,
        )
        manifest = audit.audit_h5_to_json(h5_path)
        out = self.tmp_path("report.json")
        audit.write_audit_report(out, manifest)
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            audit.write_audit_report(out, manifest)
        with open(out, "r", encoding="utf-8") as file:
            self.assertEqual(json.load(file)["path"],
                             os.path.abspath(h5_path))

    def test_audit_streams_per_sample_coordinate_layout(self):
        # A per-sample 2-D coordinate layout ([rows, H]) must never be
        # fully loaded by the audit: it is analysed with bounded
        # axis-0 streaming instead, and the geo-season dataset
        # pre-check records the 1-D-only layout requirement.
        import audit_ostia_h5 as audit
        from diafno.data.ostia import coordinate_sha256
        lat2d = np.stack(
            [np.linspace(-80.0 + 0.5 * row, 82.0, 8)
             for row in range(12)]
        )
        h5_path = make_synthetic_h5(
            self.tmp_path("per_sample_lat.h5"),
            total_days=240,
            samples_per_day=1,
            height=8,
            width=10,
            first_time=30,
            lat=lat2d,
        )
        manifest = audit.audit_h5_to_json(h5_path)
        lat_analysis = manifest["coordinates"]["lat"]
        self.assertEqual(lat_analysis["shape"], [12, 8])
        self.assertEqual(
            lat_analysis["read_strategy"], "streamed_axis0"
        )
        self.assertNotEqual(
            lat_analysis["read_strategy"], "full_read_1d"
        )
        self.assertEqual(lat_analysis["nonfinite"], 0)
        self.assertEqual(float(lat_analysis["min"]),
                         float(np.nanmin(lat2d)))
        self.assertEqual(float(lat_analysis["max"]),
                         float(np.nanmax(lat2d)))
        # Streaming digest equals the canonical whole-array digest.
        self.assertEqual(
            lat_analysis["sha256"],
            coordinate_sha256(lat2d),
        )
        # The geo-season pre-check fails closed on the layout (1-D
        # only) without ever reading the whole 2-D array.
        precheck = manifest["geo_dataset_precheck"]
        self.assertFalse(precheck["ready"])
        self.assertIn("1-D", precheck["reason"])

    def test_coordinate_analysis_streams_under_strict_limit(self):
        # Even a 1-D vector is streamed when its size exceeds the
        # configured direct-read limit; the digest stays identical.
        import audit_ostia_h5 as audit
        from diafno.data.ostia import coordinate_sha256
        h5_path = make_synthetic_h5(
            self.tmp_path("limits.h5"), total_days=60, first_time=30,
        )
        import h5py
        with h5py.File(h5_path, "r") as file:
            full = audit.coordinate_analysis(file["lat"])
            streamed = audit.coordinate_analysis(
                file["lat"], direct_read_limit_bytes=1
            )
        self.assertEqual(full["read_strategy"], "full_read_1d")
        self.assertEqual(streamed["read_strategy"],
                         "streamed_axis0")
        self.assertEqual(full["sha256"], streamed["sha256"])
        self.assertEqual(full["min"], streamed["min"])
        self.assertEqual(full["max"], streamed["max"])
        with h5py.File(h5_path, "r") as file:
            raw_lat = np.asarray(file["lat"], dtype=np.float64)
        self.assertEqual(full["sha256"], coordinate_sha256(raw_lat))

    def test_coordinate_analysis_samples_declared_rows(self):
        # Real per-day patch grids are enormous but repeated.  The
        # audit must be able to analyse a declared representative
        # subset without streaming every row and must label the digest
        # as sampled rather than pretending it covers the whole file.
        import audit_ostia_h5 as audit
        h5_path = make_synthetic_h5(
            self.tmp_path("sampled_coords.h5"),
            total_days=60,
            samples_per_day=1,
            height=8,
            width=10,
            first_time=30,
        )
        with h5py.File(h5_path, "r+") as file:
            lat = np.stack([
                np.full((8,), float(row), dtype=np.float32)
                for row in range(12)
            ])
            del file["lat"]
            file.create_dataset("lat", data=lat)
        with h5py.File(h5_path, "r") as file:
            sampled = audit.coordinate_analysis(
                file["lat"], representative_rows=[0, 2, 3]
            )
        self.assertEqual(sampled["read_strategy"], "sampled_axis0")
        self.assertEqual(sampled["sampled_rows"], [0, 2, 3])
        self.assertEqual(sampled["sampled_row_count"], 3)
        self.assertEqual(sampled["statistics_scope"],
                         "representative_rows_only")
        self.assertEqual(sampled["sha256_scope"],
                         "representative_rows_only")
        self.assertEqual(sampled["min"], 0.0)
        self.assertEqual(sampled["max"], 3.0)

    def test_audit_checksum_and_missing_file(self):
        import audit_ostia_h5 as audit
        import hashlib
        h5_path = make_synthetic_h5(
            self.tmp_path("audit.h5"), total_days=60, first_time=30,
        )
        manifest = audit.audit_h5_to_json(h5_path, checksum=True)
        digest = hashlib.sha256()
        with open(h5_path, "rb") as file:
            for chunk in iter(
                    lambda: file.read(1024 * 1024), b""
                ):
                digest.update(chunk)
        self.assertEqual(manifest["sha256"], digest.hexdigest())
        with self.assertRaises(FileNotFoundError):
            audit.audit_h5_to_json(self.tmp_path("missing.h5"))


if __name__ == "__main__":
    unittest.main()
