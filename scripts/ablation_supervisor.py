# 用途：按声明式任务依赖调度消融并记录状态，不作为训练入口自动运行。
"""Server-side non-destructive ablation supervisor.

Watches declarative pipeline tasks, launches each one exactly when its
dependencies are satisfied, restarts crashed sessions (bounded), and
stops when everything is done.  All pipeline state lives on disk:

- ``done`` of a task = its declared output file exists;
- ``started`` = the supervisor itself launched (or adopted) the tmux
  session (recorded in the supervisor state file);
- the runner remains the authority that refuses non-empty output
  directories, so the supervisor never needs to delete anything.

Loop is evaluate-only: launching is the only side effect, everything
else is read-only.  Used from tmux:

    python scripts/ablation_supervisor.py \
        --plan artifacts/supervisor_plan.json \
        --state artifacts/supervisor_state.json \
        [--interval 60] [--once]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --- injectable environment seams (monkeypatched in tests) ----------

# 用途：判断产物文件是否已存在。
# 参数：输入 path；输出 布尔值。
def _file_exists(path):
    return os.path.isfile(path)


# 用途：判断 tmux 会话是否存活。
# 参数：输入 session_name；输出 布尔值。
def _session_alive(session_name):
    completed = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    return completed.returncode == 0


# 用途：在 tmux 会话中启动任务命令。
# 参数：输入 session_name、command、log_path；输出 无。
def _launch(session_name, command, log_path):
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    wrapped = f"{command} > {log_path} 2>&1"
    completed = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            wrapped,
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"tmux launch failed for {session_name}: "
            f"{completed.stderr.decode(errors='replace')}"
        )


# 用途：当前 UTC 时间（可注入测试）。
# 参数：无输入；输出 datetime。
def _now_utc():
    return datetime.now(timezone.utc)


# 用途：可注入的休眠。
# 参数：输入 seconds（秒）；输出 无。
def _sleep(seconds):
    time.sleep(seconds)


# --- core engine -----------------------------------------------------

class Supervisor:
    """Dependency-gated, restart-bounded pipeline driver."""

    # 用途：初始化监督器：计划、状态文件、日志与可注入环境钩子。
    # 参数：输入 plan（任务计划）、state_path（状态文件）、log（日志）、launch/session_alive/file_exists/now/sleep（可注入钩子）、workdir（工作目录）；输出 无。
    def __init__(self, plan, state_path, log=None,
                 launch=_launch,
                 session_alive=_session_alive,
                 file_exists=_file_exists,
                 now=_now_utc,
                 sleep=_sleep,
                 workdir=None):
        if not isinstance(plan, dict) or not isinstance(
                plan.get("tasks"), list
            ):
            raise ValueError(
                "plan must be an object with a 'tasks' list"
            )
        self.tasks = plan["tasks"]
        self.state_path = os.path.abspath(state_path)
        self.launch = launch
        self.session_alive = session_alive
        self.file_exists = file_exists
        self.now = now
        self.sleep = sleep
        self.workdir = (
            workdir
            if workdir is not None
            else REPO_ROOT
        )
        self.log = log or print
        self.state = self._load_state()
        self._validate_tasks()

    # 用途：加载持久化任务状态（损坏时回退空状态）。
    # 参数：无输入；输出 状态 dict。
    def _load_state(self):
        if not os.path.isfile(self.state_path):
            return {"version": 1, "tasks": {}}
        try:
            with open(self.state_path, "r", encoding="utf-8") as file:
                state = json.load(file)
        except (json.JSONDecodeError, OSError) as error:
            self.log(f"[supervisor] unreadable state "
                     f"{self.state_path}: {error}; starting fresh")
            return {"version": 1, "tasks": {}}
        if not isinstance(state, dict) or not isinstance(
                state.get("tasks"), dict
            ):
            return {"version": 1, "tasks": {}}
        return state

    # 用途：持久化任务状态。
    # 参数：无输入；输出 无。
    def _save_state(self):
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = self.state_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(self.state, file, ensure_ascii=False, indent=2)
        os.replace(temporary, self.state_path)

    # 用途：校验任务计划的 id 唯一性与依赖闭合。
    # 参数：无输入；输出 无（违规抛异常）。
    def _validate_tasks(self):
        for task in self.tasks:
            for key in ("id", "session", "command", "log", "outputs",
                        "max_restarts"):
                if key not in task:
                    raise ValueError(
                        f"task {task.get('id')!r} is missing {key!r}"
                    )
            outputs = task["outputs"]
            if not isinstance(outputs, list) or not outputs:
                raise ValueError(
                    f"task {task['id']!r} must declare a non-empty "
                    "'outputs' list (its done markers)"
                )
            if "deps" not in task:
                task["deps"] = []
            ids = [item["id"] for item in self.tasks]
            if len(ids) != len(set(ids)):
                raise ValueError("task ids must be unique")

    # 用途：判断任务的输出产物是否齐备。
    # 参数：输入 task（任务 dict）；输出 布尔值。
    def _task_done(self, task):
        return all(
            self.file_exists(os.path.join(self.workdir, path))
            for path in task["outputs"]
        )

    # 用途：判断任务依赖是否全部完成。
    # 参数：输入 task；输出 布尔值。
    def _deps_satisfied(self, task):
        return all(
            self.file_exists(os.path.join(self.workdir, path))
            for path in task.get("deps", [])
        )

    # 用途：取任务状态记录（缺省新建）。
    # 参数：输入 task_id；输出 状态 dict。
    def _task_state(self, task_id):
        entry = self.state["tasks"].get(task_id)
        if not isinstance(entry, dict):
            entry = {"started": False, "restarts": 0}
            self.state["tasks"][task_id] = entry
        return entry

    # 用途：评估单个任务：返回 done/waiting/launched/restarted 之一并执行相应动作。
    # 参数：输入 task（任务 dict）；输出 状态字符串。
    def evaluate_task(self, task):
        """Return 'done' | 'waiting' | 'launched' | 'restarted'."""
        task_id = task["id"]
        entry = self._task_state(task_id)
        if self._task_done(task):
            entry["started"] = True
            return "done"
        if not self._deps_satisfied(task):
            return "waiting"
        alive = self.session_alive(task["session"])
        if alive:
            entry["started"] = True
            return "waiting"
        if entry.get("started"):
            # Session died without producing the outputs: bounded
            # restart.  The runner itself still refuses non-empty
            # output dirs, so a restart never overwrites anything.
            restarts = int(entry.get("restarts", 0)) + 1
            if restarts > int(task["max_restarts"]):
                self.log(
                    f"[supervisor] {task_id}: restart budget "
                    f"exhausted ({task['max_restarts']}); waiting for "
                    "manual review"
                )
                return "waiting"
            entry["restarts"] = restarts
            self.launch(
                task["session"],
                task["command"],
                task["log"],
            )
            self._save_state()
            self.log(
                f"[supervisor] {task_id}: restarted session "
                f"{task['session']} (attempt {restarts})"
            )
            return "restarted"
        # First launch.
        self.launch(task["session"], task["command"], task["log"])
        entry["started"] = True
        entry["restarts"] = 0
        self._save_state()
        self.log(
            f"[supervisor] {task_id}: launched session "
            f"{task['session']}"
        )
        return "launched"

    # 用途：完整评估一轮全部任务。
    # 参数：无输入；输出 无。
    def evaluate_once(self):
        results = {}
        for task in self.tasks:
            results[task["id"]] = self.evaluate_task(task)
        self._save_state()
        return results

    # 用途：判断是否全部任务完成。
    # 参数：无输入；输出 布尔值。
    def all_done(self):
        return all(
            self._task_done(task)
            for task in self.tasks
        )

    # 用途：监督主循环：周期评估直至全部完成或单次模式。
    # 参数：输入 interval_seconds（轮询间隔）、once（只跑一轮）；输出 无。
    def run_loop(self, interval_seconds, once=False):
        timestamp = self.now()
        beijing = timestamp + timedelta(hours=8)
        self.log(
            f"[supervisor] start "
            f"(UTC {timestamp.strftime('%Y-%m-%d %H:%M:%S')} / "
            f"Beijing {beijing.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        while True:
            self.evaluate_once()
            if self.all_done():
                timestamp = self.now()
                self.log(
                    f"[supervisor] all tasks done at UTC "
                    f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                return True
            if once:
                return False
            self.sleep(int(interval_seconds))


# 用途：读取并解析任务计划 JSON。
# 参数：输入 path；输出 计划 dict。
def load_plan(path):
    with open(path, "r", encoding="utf-8") as file:
        plan = json.load(file)
    return plan


# 用途：入口：解析参数并启动监督循环。
# 参数：无输入（读命令行）；输出 无。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    if args.interval < 5:
        raise ValueError("--interval must be at least 5 seconds")
    plan = load_plan(args.plan)
    supervisor = Supervisor(
        plan,
        args.state,
        workdir=args.workdir,
    )
    supervisor.run_loop(args.interval, once=args.once)


if __name__ == "__main__":
    main()
