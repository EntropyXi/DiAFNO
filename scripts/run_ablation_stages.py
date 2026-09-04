"""OSTIA spatiotemporal ablation stage runner (server-side).

Executes one fixed stage (1/2/3 from the ablation plan) for one
configuration id (A0..A5) with a fixed seed, sample plan, effective
batch 32 and validation protocol.  Never runs without a CUDA GPU,
never deletes or overwrites anything, and refuses to start into a
non-empty stage directory.

Usage (from the repository root, server GPU host):

    python scripts/run_ablation_stages.py --config-id A0 --stage 1 \
        --h5-path /data2/.../ocean_temperature_data_patched.h5

Runs are meant to be launched after a VRAM probe selected the
micro-batch; adjust with --batch-per-gpu/--gradient-accumulation
(their product x gpus must keep the effective batch at 32).
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ablation_common import (
    ABLATION_CONFIGS,
    ABLATION_ROOT,
    STAGE_PROTOCOL,
    assert_directory_available,
    check_json_result_finite,
    git_revision,
    require_cuda,
    run_command,
    samples_per_epoch_for_steps,
)


def stage_run_budgets(protocol, gpus, batch_per_gpu,
                      gradient_accumulation):
    """Whole-epoch stage budgets with a runnable resume horizon.

    Every stage runs whole epochs of exactly ``steps_per_epoch``
    optimizer steps.  The first phase needs
    ``num_epochs = steps / steps_per_epoch`` epochs; the resume phase
    continues with ``num_epochs = first_epochs + resume_epochs`` so
    the restored scheduler (whose last epoch equals the first-phase
    step count) always keeps ``resume_epochs * steps_per_epoch``
    remaining optimizer steps -- the exact bug shape
    "T_max <= last_epoch / zero remaining epochs" is rejected here.
    """
    steps = int(protocol["steps"])
    steps_per_epoch = int(protocol["steps_per_epoch"])
    resume_steps = int(protocol.get("resume_steps", 0))
    if steps_per_epoch < 1:
        raise RuntimeError(
            "protocol steps_per_epoch must be positive"
        )
    if steps % steps_per_epoch != 0:
        raise RuntimeError(
            f"protocol steps={steps} is not a whole number of "
            f"epochs of {steps_per_epoch} steps"
        )
    if resume_steps % steps_per_epoch != 0:
        raise RuntimeError(
            f"protocol resume_steps={resume_steps} is not a whole "
            f"number of epochs of {steps_per_epoch} steps"
        )
    effective_batch = (
        int(gpus) * int(batch_per_gpu)
        * int(gradient_accumulation)
    )
    samples_per_epoch = samples_per_epoch_for_steps(
        steps_per_epoch,
        gpus,
        batch_per_gpu,
        gradient_accumulation,
    )
    first_epochs = steps // steps_per_epoch
    first = {
        "optimizer_steps_per_epoch": steps_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "num_epochs": first_epochs,
        "checkpoint_interval": int(
            protocol.get("checkpoint_interval", 1)
        ),
        "global_steps": first_epochs * steps_per_epoch,
    }
    if first["global_steps"] != steps:
        raise RuntimeError(
            "internal budget error: first phase does not reach the "
            f"protocol steps ({first['global_steps']} vs {steps})"
        )
    resume = None
    if resume_steps > 0:
        resume_epochs = resume_steps // steps_per_epoch
        num_epochs = first_epochs + resume_epochs
        horizon = num_epochs * steps_per_epoch
        resume = {
            "optimizer_steps_per_epoch": steps_per_epoch,
            "samples_per_epoch": samples_per_epoch,
            "num_epochs": num_epochs,
            "checkpoint_interval": 1,
            "start_epoch": first_epochs,
            "last_epoch": first["global_steps"],
            "horizon_steps": horizon,
            "remaining_steps": horizon - first["global_steps"],
            "global_steps": horizon,
        }
        if resume["remaining_steps"] != resume_steps:
            raise RuntimeError(
                "internal budget error: resume horizon does not leave "
                f"exactly {resume_steps} steps "
                f"({resume['remaining_steps']})"
            )
    return {
        "effective_batch": effective_batch,
        "first": first,
        "resume": resume,
    }


def resolve_winner_blocks(config_id, a5_winner_blocks):
    """A5 never silently runs with the interim num_blocks.

    A5 (implicit_layer=4) only makes sense with the Stage-2 winner
    between A3 (num_blocks=2) and A4 (num_blocks=1).  Without an
    explicit choice the runner refuses; the choice is recorded in the
    stage manifest.
    """
    if config_id != "A5":
        if a5_winner_blocks is not None:
            raise RuntimeError(
                "--a5-winner-blocks only applies to A5"
            )
        return None
    if a5_winner_blocks is None:
        raise RuntimeError(
            "A5 (implicit_layer=4) must not run with the interim "
            "num_blocks=2: decide the Stage-2 winner between A3 "
            "(num_blocks=2) and A4 (num_blocks=1), re-run the VRAM "
            "probe for the chosen winner, then pass "
            "--a5-winner-blocks {1,2}"
        )
    return int(a5_winner_blocks)


def effective_num_blocks(identity, winner_blocks):
    """The num_blocks the stage actually trains with.

    A5's table/JSON value is only an interim placeholder: once a
    winner is chosen the manifest must record the actual winner
    blocks, not the interim value.  The interim value is returned
    separately so it can be recorded for provenance.
    """
    effective = (
        int(winner_blocks)
        if winner_blocks is not None
        else int(identity["blocks"])
    )
    interim = (
        int(identity["blocks"])
        if winner_blocks is not None
        else None
    )
    return effective, interim


def compose_manifest(config_id, identity, config_path, stage,
                     winner_blocks, cond_chans, budgets,
                     effective_batch, seed, h5_path, lead_stats,
                     eval_seed, val_samples, batch_per_gpu,
                     gradient_accumulation, gpus, data_manifest=None):
    """Immutable stage manifest with the *effective* architecture."""
    num_blocks, interim_num_blocks = effective_num_blocks(
        identity, winner_blocks
    )
    manifest = {
        "configuration": config_id,
        "dir": identity["dir"],
        "config_json": config_path,
        "stage": stage,
        "condition_mode": identity["mode"],
        "cond_chans": cond_chans,
        "patch_size": list(identity["patch_size"]),
        "num_blocks": num_blocks,
        "implicit_layer": identity["implicit"],
        "a5_winner_blocks": winner_blocks,
        "seed": seed,
        "effective_batch": effective_batch,
        "batch_per_gpu": batch_per_gpu,
        "gradient_accumulation": gradient_accumulation,
        "gpus": gpus,
        "budgets": budgets,
        "h5_path": os.path.abspath(h5_path),
        "data_manifest": data_manifest,
        "lead_stats": lead_stats,
        "validation_protocol": {
            "split": "val",
            "seed": eval_seed,
            "num_samples": val_samples,
            "test_split_used": False,
        },
        "git": git_revision(REPO_ROOT),
    }
    if interim_num_blocks is not None:
        manifest["interim_num_blocks"] = interim_num_blocks
    return manifest


def load_config_json(config_id):
    identity = ABLATION_CONFIGS[config_id]
    config_path = os.path.join(REPO_ROOT, identity["config"])
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"ablation config JSON not found: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    # Guard against drift between the table and the JSON files.
    expected = {
        "condition_mode": identity["mode"],
        "num_blocks": identity["blocks"],
        "implicit_layer": identity["implicit"],
    }
    got = {
        "condition_mode": payload.get("condition_mode"),
        "num_blocks": payload.get("num_blocks"),
        "implicit_layer": payload.get("implicit_layer"),
    }
    if expected != got:
        raise ValueError(
            f"config {config_path} disagrees with the ablation table: "
            f"expected {expected}, got {got}"
        )
    patch_size = tuple(payload.get("patch_size"))
    if patch_size != identity["patch_size"]:
        raise ValueError(
            f"config {config_path} patch_size {patch_size} disagrees "
            f"with the ablation table {identity['patch_size']}"
        )
    return identity, config_path, payload


def resolve_lead_stats(config_root, identity, payload, h5_path,
                       data_manifest=None):
    """Reuse or recompute the train-only lead statistics.

    Deterministic lead-standardized runs require per-mode stats; they
    are computed once per configuration root and never overwritten.
    A stats file that does not match the current mode/HDF5/data
    manifest is an error, never a silent recompute.
    """
    if payload.get("target_scaling") != "lead_standardized":
        return None
    config_dir = os.path.join(config_root, "config")
    os.makedirs(config_dir, exist_ok=True)
    stats_path = os.path.join(config_dir, "lead_stats.json")
    if os.path.isfile(stats_path):
        validate_lead_stats_identity(
            stats_path, identity, payload, h5_path, data_manifest
        )
        print(f"[ablation] reusing lead stats {stats_path}")
        return stats_path
    from deterministic_iafno.compute_lead_stats import (
        compute_lead_stats_file,
    )
    print(
        f"[ablation] computing train-only lead stats under "
        f"condition_mode={identity['mode']!r}"
    )
    compute_lead_stats_file(
        h5_path=h5_path,
        output=stats_path,
        input_days=payload.get("input_days", 7),
        output_days=payload.get("output_days", 15),
        num_samples=4096,
        batch_size=32,
        condition_mode=identity["mode"],
        data_manifest=data_manifest,
    )
    return stats_path


def validate_lead_stats_identity(stats_path, identity, payload,
                                 h5_path, data_manifest=None):
    """Prove an automatic or explicit stats file belongs to the run."""
    with open(stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)
    expected = {
        "condition_mode": identity["mode"],
        "h5_path": os.path.abspath(h5_path),
        "input_days": payload.get("input_days", 7),
        "output_days": payload.get("output_days", 15),
    }
    if data_manifest is not None:
        expected["data_manifest_sha256"] = _manifest_sha256(
            data_manifest
        )
    actual = {
        "condition_mode": stats.get("condition_mode"),
        "h5_path": stats.get("h5_path"),
        "input_days": stats.get("input_days"),
        "output_days": stats.get("output_days"),
    }
    if data_manifest is not None:
        actual["data_manifest_sha256"] = stats.get(
            "data_manifest_sha256"
        )
    if actual != expected:
        raise RuntimeError(
            f"lead stats {stats_path} do not match the current run "
            f"({actual} versus {expected}); use a compatible artifact "
            "instead of relabeling statistics"
        )
    return stats


def _manifest_sha256(path):
    from diafno.data.manifest import (
        canonical_manifest_sha256,
        load_data_manifest,
    )
    return canonical_manifest_sha256(load_data_manifest(path))


def resolve_data_manifest(config_id, identity, data_manifest_arg):
    """Validate the --data-manifest contract of a configuration.

    Every A0..A5 run must use the same read-only upstream manifest so
    chronological split windows and fixed-seed validation samples are
    identical.  A1..A5 additionally derive seasonal channels from it.
    """
    if not data_manifest_arg:
        raise RuntimeError(
            f"{config_id} requires the shared upstream data manifest "
            "so all ablations use identical gap-filtered windows; "
            "generate it with scripts/audit_ostia_h5.py "
            "--source-netcdf <upstream.nc> --manifest-out "
            "<manifest.json> and pass --data-manifest <manifest.json>"
        )
    path = os.path.abspath(data_manifest_arg)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sha256 = _manifest_sha256(path)
    print(
        f"[ablation] data manifest {path} "
        f"(identity sha256 {sha256[:16]}...)"
    )
    return {"path": path, "sha256": sha256}


def trainer_command(args, config_path, output_dir, samples_per_epoch,
                    num_epochs, lead_stats_path=None, resume_path=None,
                    checkpoint_interval=1, allow_override=False,
                    extra_args=None, data_manifest_path=None):
    """Build one torchrun trainer command.

    ``lead_stats_path`` is the *resolved* per-config stats file: it is
    passed explicitly by the caller (the freshly computed/reused path
    from ``resolve_lead_stats``), never derived from ``args.lead_stats``
    again, so an automatically generated stats file reaches the
    trainer even when the user did not pass ``--lead-stats`` on the
    stage CLI.
    """
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        os.path.join(REPO_ROOT, "trainer_ostia.py"),
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--samples-per-epoch",
        str(samples_per_epoch),
        "--num-epochs",
        str(num_epochs),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--seed",
        str(args.seed),
        "--batch-per-gpu",
        str(args.batch_per_gpu),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--num-workers",
        str(args.num_workers),
        "--train-h5-path",
        args.h5_path,
    ]
    if lead_stats_path is not None:
        command.extend(["--lead-stats", lead_stats_path])
    if data_manifest_path is not None:
        command.extend(["--data-manifest", data_manifest_path])
    if extra_args:
        command.extend(extra_args)
    if resume_path is not None:
        command.extend(["--resume", resume_path])
        if allow_override:
            command.append("--allow-resume-override")
    return command


def verify_checkpoint_step(checkpoint_path, expected_epoch,
                           expected_global_step, label=""):
    """Read latest.pth (CPU, read-only) and verify the resume
    bookkeeping matches the stage budget.

    Every trainer phase must end exactly at its budgeted epoch and
    global step (stage 1: epoch 5 / step 50, then the resume phase
    epoch 6 / step 60); any drift fails the stage instead of being
    silently validated with the wrong checkpoint.
    """
    import torch
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"{label}: expected checkpoint missing after the "
            f"trainer phase: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    epoch = int(checkpoint.get("epoch", -1))
    global_step = int(checkpoint.get("global_step", -1))
    problems = []
    if epoch != int(expected_epoch):
        problems.append(
            f"epoch={epoch} (expected {int(expected_epoch)})"
        )
    if global_step != int(expected_global_step):
        problems.append(
            f"global_step={global_step} "
            f"(expected {int(expected_global_step)})"
        )
    if problems:
        raise RuntimeError(
            f"{label}: checkpoint step bookkeeping mismatch for "
            f"{checkpoint_path}: " + "; ".join(problems)
        )
    print(
        f"[ablation] {label}: verified {checkpoint_path} "
        f"epoch={epoch} global_step={global_step}"
    )
    return checkpoint


def validation_command(checkpoint_path, output_path, h5_path,
                       num_samples, seed, data_manifest_path=None):
    command = [
        sys.executable,
        os.path.join(REPO_ROOT, "validate_ostia.py"),
        "--checkpoint",
        checkpoint_path,
        "--h5-path",
        h5_path,
        "--output-path",
        output_path,
        "--split",
        "val",
        "--max-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
    ]
    if data_manifest_path is not None:
        command.extend(["--data-manifest", data_manifest_path])
    return command


def logfile_for(stage_root, name):
    filename = name if name.endswith(".log") else f"{name}.log"
    path = os.path.join(stage_root, filename)
    os.makedirs(stage_root, exist_ok=True)
    return path


def run_trainer_phase(args, command, log_path):
    env = os.environ.copy()
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    with open(log_path, "wb") as log:
        import subprocess
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        tail = []
        with open(log_path, "rb") as log:
            tail = log.read().decode(
                "utf-8", errors="replace"
            ).splitlines()[-40:]
        print("--- trainer log tail ---", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        raise RuntimeError(
            f"trainer phase failed (exit code "
            f"{completed.returncode}); log: {log_path}"
        )


def write_manifest(stage_root, manifest):
    path = os.path.join(stage_root, "manifest.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"[ablation] manifest: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-id", required=True,
                        choices=sorted(ABLATION_CONFIGS))
    parser.add_argument("--stage", required=True, type=int,
                        choices=(1, 2, 3))
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--root", default=ABLATION_ROOT)
    parser.add_argument("--tag", default=None,
                        help="stage tag (default: stage<N>)")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu-ids", default=None,
                        help="comma list for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-per-gpu", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int,
                        default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lead-stats", default=None,
                        help="override the per-config lead stats path")
    parser.add_argument("--eval-seed", type=int, default=123,
                        help="fixed validation subset seed")
    parser.add_argument(
        "--data-manifest",
        default=None,
        help=(
            "upstream data manifest (audit --source-netcdf "
            "--manifest-out); required for geo-season configs A1..A5"
        ),
    )
    parser.add_argument(
        "--a5-winner-blocks",
        type=int,
        choices=(1, 2),
        default=None,
        help=(
            "required for A5: the Stage-2 winner between A3 "
            "(num_blocks=2) and A4 (num_blocks=1); A5 refuses to run "
            "with the interim num_blocks without this explicit choice"
        ),
    )
    args = parser.parse_args()

    require_cuda()
    import torch
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(
            f"stage needs {args.gpus} GPUs but only "
            f"{torch.cuda.device_count()} are visible"
        )
    effective_batch = (
        args.batch_per_gpu
        * args.gradient_accumulation
        * args.gpus
    )
    if effective_batch != 32:
        raise RuntimeError(
            "the ablation protocol fixes the effective batch at 32; "
            f"got batch_per_gpu={args.batch_per_gpu} x accumulation="
            f"{args.gradient_accumulation} x gpus={args.gpus} = "
            f"{effective_batch}"
        )
    if not os.path.isfile(args.h5_path):
        raise FileNotFoundError(args.h5_path)

    identity, config_path, payload = load_config_json(args.config_id)
    protocol = STAGE_PROTOCOL[f"stage{args.stage}"]
    winner_blocks = resolve_winner_blocks(
        args.config_id, args.a5_winner_blocks
    )
    data_manifest = resolve_data_manifest(
        args.config_id, identity, args.data_manifest
    )
    config_root = os.path.join(args.root, identity["dir"])
    tag = args.tag or f"stage{args.stage}"
    stage_root = os.path.join(config_root, tag)
    assert_directory_available(stage_root)
    os.makedirs(stage_root, exist_ok=True)

    if args.lead_stats is None:
        lead_stats = resolve_lead_stats(
            config_root,
            identity,
            payload,
            args.h5_path,
            data_manifest=(
                data_manifest["path"]
                if data_manifest is not None
                else None
            ),
        )
    else:
        lead_stats = os.path.abspath(args.lead_stats)
        if not os.path.isfile(lead_stats):
            raise FileNotFoundError(lead_stats)
        validate_lead_stats_identity(
            lead_stats,
            identity,
            payload,
            args.h5_path,
            data_manifest=(
                data_manifest["path"]
                if data_manifest is not None
                else None
            ),
        )

    from diafno.data.condition_schema import condition_chans
    cond_chans = condition_chans(
        identity["mode"],
        payload.get("input_days", 7),
    )
    budgets = stage_run_budgets(
        protocol,
        args.gpus,
        args.batch_per_gpu,
        args.gradient_accumulation,
    )
    extra_args = []
    if winner_blocks is not None:
        extra_args.extend(["--num-blocks", str(winner_blocks)])
    manifest = compose_manifest(
        config_id=args.config_id,
        identity=identity,
        config_path=identity["config"],
        stage=args.stage,
        winner_blocks=winner_blocks,
        cond_chans=cond_chans,
        budgets=budgets,
        effective_batch=effective_batch,
        seed=args.seed,
        h5_path=args.h5_path,
        lead_stats=lead_stats,
        eval_seed=args.eval_seed,
        val_samples=protocol["val_samples"],
        batch_per_gpu=args.batch_per_gpu,
        gradient_accumulation=args.gradient_accumulation,
        gpus=args.gpus,
        data_manifest=data_manifest,
    )
    write_manifest(stage_root, manifest)
    data_manifest_path = (
        data_manifest["path"]
        if data_manifest is not None
        else None
    )

    first = budgets["first"]
    resume = budgets["resume"]
    print(
        f"[ablation] stage {args.stage}: {args.config_id} -> "
        f"{stage_root}: {protocol['steps']} optimizer steps in "
        f"{first['num_epochs']} epochs of "
        f"{first['optimizer_steps_per_epoch']} "
        f"(samples_per_epoch={first['samples_per_epoch']})"
    )

    if args.stage in (1, 2):
        run_dir = os.path.join(stage_root, "run")
        assert_directory_available(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        run_trainer_phase(
            args,
            trainer_command(
                args,
                config_path,
                run_dir,
                first["samples_per_epoch"],
                num_epochs=first["num_epochs"],
                lead_stats_path=lead_stats,
                data_manifest_path=data_manifest_path,
                checkpoint_interval=first["checkpoint_interval"],
                extra_args=extra_args,
            ),
            logfile_for(stage_root, "run.log"),
        )
        run_checkpoint = os.path.join(run_dir, "latest.pth")
        verify_checkpoint_step(
            run_checkpoint,
            expected_epoch=first["num_epochs"],
            expected_global_step=first["global_steps"],
            label="first phase",
        )
        checkpoints = [run_checkpoint]
        if resume is not None:
            resume_dir = os.path.join(stage_root, "resume")
            assert_directory_available(resume_dir)
            os.makedirs(resume_dir, exist_ok=True)
            run_trainer_phase(
                args,
                trainer_command(
                    args,
                    config_path,
                    resume_dir,
                    resume["samples_per_epoch"],
                    num_epochs=resume["num_epochs"],
                    lead_stats_path=lead_stats,
                    data_manifest_path=data_manifest_path,
                    resume_path=checkpoints[0],
                    checkpoint_interval=resume[
                        "checkpoint_interval"
                    ],
                    allow_override=True,
                    extra_args=extra_args,
                ),
                logfile_for(stage_root, "resume.log"),
            )
            resume_checkpoint = os.path.join(
                resume_dir, "latest.pth"
            )
            verify_checkpoint_step(
                resume_checkpoint,
                expected_epoch=resume["num_epochs"],
                expected_global_step=resume["global_steps"],
                label="resume phase",
            )
            checkpoints.append(resume_checkpoint)
        for checkpoint_path in checkpoints:
            name = os.path.basename(
                os.path.dirname(checkpoint_path)
            )
            val_path = os.path.join(
                stage_root,
                f"val_{protocol['val_samples']}_{name}.json",
            )
            run_command(
                validation_command(
                    checkpoint_path,
                    val_path,
                    args.h5_path,
                    protocol["val_samples"],
                    args.eval_seed,
                    data_manifest_path=data_manifest_path,
                ),
                f"validation of {checkpoint_path}",
            )
            payload_result = check_json_result_finite(val_path)
            print(
                f"[ablation] {name}: overall rmse "
                f"{payload_result['overall']['rmse']:.6f}, "
                f"num_samples={payload_result['num_samples']}"
            )
    else:
        # Stage 3: 6 epochs x 250 steps, re-evaluating the same fixed
        # val-200 set at steps 500/1000/1500 (epoch checkpoints 2/4/6).
        run_dir = os.path.join(stage_root, "run")
        assert_directory_available(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        run_trainer_phase(
            args,
            trainer_command(
                args,
                config_path,
                run_dir,
                first["samples_per_epoch"],
                num_epochs=first["num_epochs"],
                lead_stats_path=lead_stats,
                data_manifest_path=data_manifest_path,
                checkpoint_interval=first["checkpoint_interval"],
                extra_args=extra_args,
            ),
            logfile_for(stage_root, "run.log"),
        )
        verify_checkpoint_step(
            os.path.join(run_dir, "latest.pth"),
            expected_epoch=first["num_epochs"],
            expected_global_step=first["global_steps"],
            label="stage 3 run",
        )
        for step in protocol["eval_steps"]:
            epoch = step // first["optimizer_steps_per_epoch"]
            checkpoint_path = os.path.join(
                run_dir,
                f"epoch_{epoch:03d}.pth",
            )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"expected stage-3 checkpoint at step {step}: "
                    f"{checkpoint_path}"
                )
            val_path = os.path.join(
                stage_root,
                f"val_{protocol['val_samples']}_step{step}.json",
            )
            run_command(
                validation_command(
                    checkpoint_path,
                    val_path,
                    args.h5_path,
                    protocol["val_samples"],
                    args.eval_seed,
                    data_manifest_path=data_manifest_path,
                ),
                f"validation of {checkpoint_path}",
            )
            payload_result = check_json_result_finite(val_path)
            print(
                f"[ablation] step {step}: overall rmse "
                f"{payload_result['overall']['rmse']:.6f}"
            )
    print(f"[ablation] stage {args.stage} of {args.config_id} done")
    print(f"[ablation] results in {stage_root}")


if __name__ == "__main__":
    main()
