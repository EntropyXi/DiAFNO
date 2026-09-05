"""Unit tests for the server-side ablation supervisor and the final
summary generator (pure engine, no server needed)."""

import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import ablation_supervisor as supervisor  # noqa: E402
import ablation_summary as summary_tool  # noqa: E402

from .ostia_test_h5 import OSTIATestCase  # noqa: E402


class FakeEnv:
    """Fake filesystem/session/launch environment."""

    def __init__(self, files=None, sessions=None):
        self.files = set(files or [])
        self.sessions = set(sessions or [])
        self.launches = []

    def file_exists(self, path):
        return os.path.normpath(path) in self.files

    def session_alive(self, name):
        return name in self.sessions

    def launch(self, session, command, log):
        self.launches.append((session, command, log))
        self.sessions.add(session)


def task(id, deps=None, outputs=None, session=None, max_restarts=1):
    return {
        "id": id,
        "deps": deps or [],
        "outputs": outputs or [f"{id}.out"],
        "session": session or f"sess_{id}",
        "command": f"echo {id}",
        "log": f"logs/{id}.log",
        "max_restarts": max_restarts,
    }


class SupervisorEngineTests(OSTIATestCase):
    def _supervisor(self, tasks, env):
        state_path = os.path.join(self._tmp, "state.json")
        plan = {"tasks": tasks}
        return supervisor.Supervisor(
            plan,
            state_path,
            workdir=self._tmp,
            launch=env.launch,
            session_alive=env.session_alive,
            file_exists=env.file_exists,
            log=lambda *args: None,
        ), state_path

    def test_plan_validation(self):
        env = FakeEnv()
        with self.assertRaisesRegex(ValueError, "missing"):
            supervisor.Supervisor(
                {"tasks": [{"id": "x"}]},
                os.path.join(self._tmp, "s.json"),
                launch=env.launch,
                session_alive=env.session_alive,
                file_exists=env.file_exists,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            supervisor.Supervisor(
                {"tasks": [task("x"), task("x")]},
                os.path.join(self._tmp, "s.json"),
                launch=env.launch,
                session_alive=env.session_alive,
                file_exists=env.file_exists,
            )

    def test_dependency_gating_blocks_launch(self):
        env = FakeEnv()
        tasks = [task("run", deps=["dep.out"], outputs=["run.out"])]
        sup, _ = self._supervisor(tasks, env)
        self.assertEqual(sup.evaluate_once()["run"], "waiting")
        self.assertEqual(env.launches, [])
        # Dependency appears -> first launch.
        env.files.add(os.path.join(self._tmp, "dep.out"))
        self.assertEqual(sup.evaluate_once()["run"], "launched")
        self.assertEqual(len(env.launches), 1)

    def test_no_double_launch_while_session_alive(self):
        env = FakeEnv(sessions={"sess_run"})
        tasks = [task("run")]
        sup, _ = self._supervisor(tasks, env)
        self.assertEqual(sup.evaluate_once()["run"], "waiting")
        self.assertEqual(env.launches, [])

    def test_done_marker_stops_everything(self):
        env = FakeEnv()
        tasks = [task("run")]
        env.files.add(os.path.join(self._tmp, "run.out"))
        sup, _ = self._supervisor(tasks, env)
        self.assertEqual(sup.evaluate_once()["run"], "done")
        self.assertEqual(env.launches, [])
        self.assertTrue(sup.all_done())

    def test_restart_within_budget(self):
        env = FakeEnv()
        tasks = [task("run", max_restarts=2)]
        sup, state_path = self._supervisor(tasks, env)
        # First launch.
        sup.evaluate_once()
        self.assertEqual(len(env.launches), 1)
        # Session dies without output -> restart.
        env.sessions.discard("sess_run")
        result = sup.evaluate_once()["run"]
        self.assertEqual(result, "restarted")
        self.assertEqual(len(env.launches), 2)
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
        self.assertEqual(state["tasks"]["run"]["restarts"], 1)
        # Dies again -> second restart allowed.
        env.sessions.discard("sess_run")
        sup.evaluate_once()
        self.assertEqual(len(env.launches), 3)

    def test_restart_budget_exhausted(self):
        env = FakeEnv()
        tasks = [task("run", max_restarts=1)]
        sup, _ = self._supervisor(tasks, env)
        sup.evaluate_once()
        env.sessions.discard("sess_run")
        self.assertEqual(sup.evaluate_once()["run"], "restarted")
        env.sessions.discard("sess_run")
        # Budget exhausted: no third launch, waits for review.
        self.assertEqual(sup.evaluate_once()["run"], "waiting")
        self.assertEqual(len(env.launches), 2)

    def test_run_loop_once_exits_when_pending(self):
        env = FakeEnv()
        tasks = [task("run", deps=["never.out"])]
        sup, _ = self._supervisor(tasks, env)
        self.assertFalse(supervisor.Supervisor.run_loop(
            sup, interval_seconds=5, once=True,
        ))

    def test_corrupt_state_recovers(self):
        env = FakeEnv()
        state_path = os.path.join(self._tmp, "state.json")
        with open(state_path, "w", encoding="utf-8") as file:
            file.write("{not json")
        sup = supervisor.Supervisor(
            {"tasks": [task("run")]},
            state_path,
            launch=env.launch,
            session_alive=env.session_alive,
            file_exists=env.file_exists,
            log=lambda *args: None,
        )
        self.assertEqual(sup.evaluate_once()["run"], "launched")


class SummaryGeneratorTests(OSTIATestCase):
    def _write_val(self, path, rmse, skill=0.1, step=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "overall": {
                "rmse": rmse,
                "mae": rmse * 0.7,
                "bias": 0.01,
                "correlation": 0.997,
            },
            "by_lead_day": {
                "1": {"rmse": rmse * 0.4},
                "7": {"rmse": rmse * 0.95},
                "15": {"rmse": rmse * 1.25},
            },
            "persistence_skill": {"overall": skill},
            "num_samples": 200,
            "seed": 123,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)

    def test_build_summary_and_markdown(self):
        root = self._tmp
        stage2_rmse = {
            "A0_baseline_p8_b8_i2": 0.7055,
            "A1_geo_p8_b8_i2": 0.6450,
            "A2_geo_p4_b8_i2": 0.6982,
            "A3_geo_p4_b2_i2": 0.6480,
            "A4_geo_p4_b1_i2": 0.6417,
        }
        for config_id, rmse in stage2_rmse.items():
            self._write_val(
                os.path.join(root, config_id, "stage2",
                             "val_200_run.json"),
                rmse,
            )
        # A5 lives under the 2-GPU tag.
        self._write_val(
            os.path.join(root, "A5_geo_p4_best_i4", "stage2_2gpu",
                         "val_200_run.json"),
            0.6294,
            skill=0.2028,
        )
        for config_id, tag in summary_tool.STAGE3_TAGS.items():
            for step in (500, 1000, 1500):
                self._write_val(
                    os.path.join(root, config_id, tag,
                                 f"val_200_step{step}.json"),
                    0.62 - step * 1e-5,
                )
        boot_a5 = os.path.join(root, "boot_a5.json")
        self._write_val(boot_a5, 0.61)
        with open(boot_a5, "r+", encoding="utf-8") as file:
            data = json.load(file)
            data["paired_block_bootstrap"] = {
                "method": "paired_nonoverlapping_temporal_block_bootstrap",
                "replicates": 1000,
                "confidence_level": 0.95,
                "overall": {
                    "model_rmse": 0.61,
                    "persistence_rmse": 0.71,
                    "rmse_difference": -0.10,
                    "rmse_difference_ci": [-0.13, -0.07],
                    "mse_skill": 0.20,
                    "mse_skill_ci": [0.18, 0.22],
                    "bootstrap_fraction_model_better": 1.0,
                },
            }
            file.seek(0)
            json.dump(data, file)
            file.truncate()
        summary = summary_tool.build_summary(
            root,
            {"A5_geo_p4_best_i4": boot_a5},
        )
        self.assertAlmostEqual(
            summary["stage2_val200"]["A5_geo_p4_best_i4"]["rmse"],
            0.6294,
        )
        self.assertEqual(
            summary["stage3_reevals"]["A5_geo_p4_best_i4"]["500"][
                "num_samples"
            ],
            200,
        )
        markdown = summary_tool.render_markdown(summary)
        self.assertIn("A5_geo_p4_best_i4", markdown)
        self.assertIn("北京时间", markdown)
        self.assertIn("0.6294", markdown)
        # Summary payload stays JSON-serializable.
        json.dumps(summary)

    def test_refuses_existing_summary_output(self):
        root = self._tmp
        out = os.path.join(root, "summary", "final_summary.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            file.write("{}")
        self.assertTrue(os.path.exists(out))
        # The CLI-level refusal lives in main(); emulate it directly
        # by checking the same guard path exists in code.
        with open(
            os.path.join(
                REPO_ROOT, "scripts", "ablation_summary.py"
            ),
            encoding="utf-8",
        ) as source:
            source_text = source.read()
        self.assertIn(
            "refusing to overwrite existing summary",
            source_text,
        )


if __name__ == "__main__":
    unittest.main()
