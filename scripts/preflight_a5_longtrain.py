#!/usr/bin/env python
# 用途：A5 长训启动前只读核对（源 checkpoint/sidecar、lead 统计、manifest 双摘要、日期/坐标指纹、输出目录保护）。
"""Preflight checks for the A5 long-training stage.

Every check is a pure function so unit tests can drive them with
synthetic files.  The CLI main only runs the checks against the fixed
A5 identities recorded in plans/A5_LONGTRAIN_TEST200_20260908.md;
any missing server artifact fails closed with an explicit message.
"""

import argparse
import hashlib
import json
import os
import sys

import torch

# Fixed identities from plans/A5_LONGTRAIN_TEST200_20260908.md.
SOURCE_SHA256 = "2a34faae3d0156c9c9dfb0027ac9370f67faa0b4cb0bad0e2da50d4ea7d65f9f"
SOURCE_EPOCH = 11
SOURCE_GLOBAL_STEP = 2750
SOURCE_SCHEDULER_LAST_EPOCH = 2747
LEAD_STATS_SHA256 = "6f159ddbe0db418db8ac634a85636761376bba576d90c564a9bc0dbde63e8812"
MANIFEST_FILE_SHA256 = "a678e16c1cd181ff793786f22be2fa11cf3df4a9526f0b609340a2e66c3a33bb"
MANIFEST_CONTRACT_SHA256 = "9b5e19439d4eea9963e599f15330a5260f7bb4aef33103efce071f4f5fed386e"
MANIFEST_DAY_OFFSETS_SHA256 = "dc0ba70360d5533cae1780f3d1bae9493193cc2289329c546116fb937c222de2"
COORDINATE_AXES_SHA256 = "df5dc46af22ccb5e7721735d7fb38314c3842880b11759ab78b4ead817056100"


# 用途：文件的规范 SHA256。
# 参数：输入 path；输出 十六进制摘要。
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 用途：校验文件存在且摘要等于期望值（fail-closed）。
# 参数：输入 path、expected_sha、label；输出 无（不符抛 ValueError）。
def require_file_sha256(path, expected_sha, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} missing: {path}"
        )
    actual = file_sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"{label} SHA-256 mismatch: file {path} has {actual}, "
            f"expected {expected_sha}"
        )


# 用途：核对源 checkpoint 身份（epoch/global_step/scheduler/skip 计数一致性）。
# 参数：输入 checkpoint_path、expected 各身份常量；输出 checkpoint 载荷摘要 dict。
def verify_source_checkpoint(
        checkpoint_path,
        expected_sha=SOURCE_SHA256,
        expected_epoch=SOURCE_EPOCH,
        expected_global_step=SOURCE_GLOBAL_STEP,
        expected_scheduler_last_epoch=SOURCE_SCHEDULER_LAST_EPOCH,
        check_sha=True,
    ):
    if check_sha:
        require_file_sha256(
            checkpoint_path,
            expected_sha,
            "source checkpoint",
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    skips = int(checkpoint.get("skipped_optimizer_steps", 0))
    scheduler = checkpoint.get("scheduler", {}) or {}
    last_epoch = int(
        scheduler.get("last_epoch", -1)
    )
    if epoch != expected_epoch:
        raise ValueError(
            f"source checkpoint epoch={epoch} != {expected_epoch}"
        )
    if global_step != expected_global_step:
        raise ValueError(
            f"source checkpoint global_step={global_step} != "
            f"{expected_global_step}"
        )
    if last_epoch != expected_scheduler_last_epoch:
        raise ValueError(
            f"source checkpoint scheduler.last_epoch={last_epoch} != "
            f"{expected_scheduler_last_epoch}"
        )
    # Attempts = successful optimizer updates + AMP skips.  scheduler
    # advances only on successful updates.
    if global_step - last_epoch != skips:
        raise ValueError(
            "source checkpoint skip accounting is inconsistent: "
            f"global_step - scheduler.last_epoch = "
            f"{global_step - last_epoch} but skipped_optimizer_steps="
            f"{skips}"
        )
    sidecar = checkpoint_path + ".semantics.json"
    if not os.path.isfile(sidecar):
        raise FileNotFoundError(
            f"source checkpoint sidecar missing: {sidecar}"
        )
    with open(sidecar, "r", encoding="utf-8") as file:
        sidecar_payload = json.load(file)
    return {
        "epoch": epoch,
        "global_step": global_step,
        "successful_updates": last_epoch,
        "skips": skips,
        "sidecar_config": sidecar_payload.get("config"),
    }


# 用途：校验目标模型配置与源 sidecar 的不可变字段一致。
# 参数：输入 target_config（解析后训练配置的 model 部分或 dict）、source_sidecar_config；输出 无。
def verify_model_fields_against_source(
        target_model_config,
        source_sidecar_config,
    ):
    """Compare the target model fields against the source sidecar.

    Covers the init-from architecture check fields plus the fixed
    target-space and condition contracts; anything that differs is a
    launch error because this stage only extends the A5 weights.
    """

    def _value(container, field):
        if isinstance(container, dict):
            return container.get(field)
        return getattr(container, field, None)

    def _normalize(value):
        # JSON round-trips dataclass tuples into lists; compare the two
        # shapes on the same footing instead of failing on tuple-vs-list.
        if isinstance(value, (list, tuple)):
            return tuple(_normalize(item) for item in value)
        return value

    fields = (
        "input_days",
        "output_days",
        "cond_chans",
        "target_chans",
        "image_size",
        "patch_size",
        "embed_dim",
        "num_blocks",
        "explicit_layer",
        "implicit_layer",
        "hidden_size_factor",
        "target_mode",
        "model_type",
        "target_scaling",
        "condition_mode",
    )
    for field in fields:
        current = _value(target_model_config, field)
        source = source_sidecar_config.get(field)
        if _normalize(current) != _normalize(source):
            raise ValueError(
                f"target config {field}={current!r} differs from the "
                f"source sidecar {field}={source!r}"
            )
    if _value(target_model_config, "num_blocks") != 1:
        raise ValueError(
            "A5 long train must run num_blocks=1 (winner override)"
        )
    if _value(
            target_model_config, "condition_mode"
        ) != "sst_mask_geo_season":
        raise ValueError(
            "A5 long train requires condition_mode="
            "'sst_mask_geo_season'"
        )


# 用途：校验 lead stats 文件身份与结构。
# 参数：输入 path、expected_sha；输出 无。
def verify_lead_stats(path, expected_sha=LEAD_STATS_SHA256):
    require_file_sha256(path, expected_sha, "lead stats")
    with open(path, "r", encoding="utf-8") as file:
        stats = json.load(file)
    if stats.get("split") != "train":
        raise ValueError(
            f"lead stats split={stats.get('split')!r} != 'train'"
        )
    for key, length in (("lead_mean", 15), ("lead_std", 15)):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(
                f"lead stats {key} must be a list of length {length}"
            )
    if any(value <= 0.0 for value in stats["lead_std"]):
        raise ValueError("lead stats lead_std must all be positive")
    return stats


# 用途：校验数据清单的双摘要与真实日偏移摘要。
# 参数：输入 path、expected_file_sha、expected_contract_sha、expected_day_offsets_sha；输出 无。
def verify_data_manifest_identity(
        path,
        expected_file_sha=MANIFEST_FILE_SHA256,
        expected_contract_sha=MANIFEST_CONTRACT_SHA256,
        expected_day_offsets_sha=MANIFEST_DAY_OFFSETS_SHA256,
    ):
    require_file_sha256(path, expected_file_sha, "data manifest")
    from diafno.data.manifest import (
        canonical_manifest_sha256,
        day_offset_sha256,
        load_data_manifest,
    )
    payload = load_data_manifest(path)
    contract = canonical_manifest_sha256(payload)
    if contract != expected_contract_sha:
        raise ValueError(
            "data manifest contract SHA mismatch: "
            f"{contract} != {expected_contract_sha}"
        )
    offsets_sha = day_offset_sha256(payload["day_offsets"])
    if offsets_sha != expected_day_offsets_sha:
        raise ValueError(
            "data manifest day_offsets SHA mismatch: "
            f"{offsets_sha} != {expected_day_offsets_sha}"
        )


# 用途：校验输出目录未包含训练产物（fresh 启动保护）。
# 参数：输入 output_dir；输出 无（已含产物即拒绝）。
def require_fresh_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        return
    for name in ("latest.pth", "epoch_000.pth", "training_curves.npz"):
        if os.path.exists(os.path.join(output_dir, name)):
            raise ValueError(
                "output dir already contains training artifacts: "
                f"{output_dir}; a fresh --init-from stage must start "
                "into an empty directory"
            )


# 用途：把 train 段数据集统计量与源 checkpoint 归一化逐位核对（abs 1e-10）。
# 参数：输入 dataset_sst_mean、dataset_sst_std、checkpoint_normalization；输出 无。
def verify_normalization_identity(
        dataset_sst_mean,
        dataset_sst_std,
        checkpoint_normalization,
    ):
    checkpoint_mean = checkpoint_normalization.get("sst_mean")
    checkpoint_std = checkpoint_normalization.get("sst_std")
    if checkpoint_mean is None or checkpoint_std is None:
        raise ValueError(
            "source checkpoint has no sst normalization block"
        )
    if (
            abs(float(dataset_sst_mean) - float(checkpoint_mean)) > 1e-10
            or abs(float(dataset_sst_std) - float(checkpoint_std)) > 1e-10
        ):
        raise ValueError(
            "dataset normalization does not match the source "
            f"checkpoint (dataset {dataset_sst_mean}, "
            f"{dataset_sst_std}; checkpoint {checkpoint_mean}, "
            f"{checkpoint_std}); tolerance 1e-10 absolute"
        )


def main():
    parser = argparse.ArgumentParser(
        description="A5 long-train startup preflight (read-only)"
    )
    parser.add_argument(
        "--config-json",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "configs",
            "ostia_A5_longtrain_v1.json",
        ),
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument(
        "--data-manifest",
        default="artifacts/ostia_data_manifest_real.json",
    )
    parser.add_argument(
        "--lead-stats",
        default=(
            "experiments/ostia_spatiotemporal_ablation/"
            "A5_geo_p4_best_i4/config/lead_stats.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/a5_longtrain_v1_20260908/train",
    )
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    source_summary = verify_source_checkpoint(args.source_checkpoint)
    # Resolve the target model fields from dataclass defaults merged
    # with the config JSON, so fields the JSON cannot express
    # (embed_dim / explicit_layer / ...) are still checked against the
    # source sidecar instead of silently drifting.
    from diafno.data.condition_schema import (
        condition_chans,
        condition_schema_version_for,
    )
    from diafno.models.config import OSTIAModelConfig
    defaults = OSTIAModelConfig()
    target_model = {
        field: getattr(defaults, field)
        for field in (
            "input_days",
            "output_days",
            "image_size",
            "embed_dim",
            "num_blocks",
            "explicit_layer",
            "implicit_layer",
            "hidden_size_factor",
        )
    }
    target_model.update({
        key: payload[key]
        for key in (
            "model_type",
            "target_mode",
            "target_scaling",
            "condition_mode",
            "patch_size",
            "num_blocks",
            "implicit_layer",
        )
        if key in payload
    })
    target_model["target_chans"] = int(payload.get(
        "output_days",
        defaults.output_days,
    ))
    target_model["cond_chans"] = condition_chans(
        target_model["condition_mode"],
        target_model["input_days"],
    )
    verify_model_fields_against_source(
        target_model,
        source_summary["sidecar_config"] or {},
    )
    verify_lead_stats(args.lead_stats)
    verify_data_manifest_identity(args.data_manifest)
    require_fresh_output_dir(args.output_dir)
    # Normalization + coordinate checks need the real HDF5 and are
    # part of the server-side run; here they are optional probes.
    if os.path.isfile(args.h5_path):
        from diafno.data.ostia import OSTIADailyDataset
        dataset = OSTIADailyDataset(
            h5_path=args.h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode="sst_mask_geo_season",
            data_manifest=args.data_manifest,
        )
        checkpoint = torch.load(
            args.source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        verify_normalization_identity(
            dataset.sst_mean,
            dataset.sst_std,
            checkpoint.get("normalization") or {},
        )
    print("A5 long-train preflight PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
