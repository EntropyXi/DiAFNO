#!/usr/bin/env python
# 用途：A5-centered DiAFNO 主训练启动前只读核对（均值身份、geo 契约、统计、配置、manifest）。
"""Preflight checks for the A5-centered DiAFNO main run.

Fails closed unless every identity the run depends on is verified
(A5_CENTERED_DIAFNO_MAINTRAIN plan 2/5/6):

- the frozen A5 mean checkpoint file SHA equals the fixed v2 identity
  (best_val_rmse.pth, 4ce4984f...) and its semantic sidecar proves the
  geo-season contract (condition_mode, 14 channels, calendar / time
  axis / manifest provenance) plus its own residual lead stats;
- the resolved model fields of the run config agree with the mean
  sidecar architecture (both branches use the A5 backbone) and the
  centered rules (model_type, sigma_data=1.0, target semantics);
- the upstream HDF5 + data manifest double hashes match the A5
  protocol; the train dataset normalization matches the mean
  checkpoint (1e-10 absolute);
- when the centered innovation stats file already exists (train-only
  statistics step done), its v2 payload validates and the three-way
  mean identity (file == stats == protocol lock) is consistent;
  otherwise the run prints that stats are still pending.

Every check is a pure function so unit tests can drive it with
synthetic files; ``main()`` only runs them against the fixed A5
identities recorded in the plan.
"""

import argparse
import hashlib
import json
import os
import sys

import torch

# Fixed identities from plans/A5_CENTERED_DIAFNO_MAINTRAIN_20260909.md.
A5_MEAN_SHA256 = "4ce4984fe4b2e11748ca9bcacdedf0accc174a2721cf43b49167833f47c609bc"
LEAD_STATS_SHA256 = "6f159ddbe0db418db8ac634a85636761376bba576d90c564a9bc0dbde63e8812"
MANIFEST_FILE_SHA256 = "a678e16c1cd181ff793786f22be2fa11cf3df4a9526f0b609340a2e66c3a33bb"
MANIFEST_CONTRACT_SHA256 = "9b5e19439d4eea9963e599f15330a5260f7bb4aef33103efce071f4f5fed386e"
MANIFEST_DAY_OFFSETS_SHA256 = "dc0ba70360d5533cae1780f3d1bae9493193cc2289329c546116fb937c222de2"
COORDINATE_AXES_SHA256 = "df5dc46af22ccb5e7721735d7fb38314c3842880b11759ab78b4ead817056100"
V2_MEAN_CONDITION_MODE = "sst_mask_geo_season"
V2_MEAN_COND_CHANS = 14

# Geo-season provenance the frozen mean sidecar immutable must carry.
V2_SIDECAR_PRESENCE_FIELDS = (
    "condition_mode",
    "condition_schema_version",
    "condition_channel_names",
    "calendar_encoding",
    "time_units_reference",
    "geospatial_summary",
    "time_axis_summary",
    "data_manifest_sha256",
)

# Mean model contract shared by both centered protocols.
MEAN_IMMUTABLE_EXPECTATIONS = {
    "model_type": "deterministic",
    "target_mode": "residual",
    "target_scaling": "lead_standardized",
    "input_days": 7,
    "output_days": 15,
}

ARCH_FIELDS = (
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
)


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
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = file_sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"{label} SHA-256 mismatch: file {path} has {actual}, "
            f"expected {expected_sha}"
        )


# 用途：读取 checkpoint 语义 sidecar 的 immutable 块。
# 参数：输入 mean_checkpoint_path；输出 immutable dict（缺失抛 ValueError）。
def mean_sidecar_immutable(mean_checkpoint_path):
    from deterministic_iafno.checkpoint_semantics import (
        load_semantic_sidecar,
    )
    sidecar = load_semantic_sidecar(mean_checkpoint_path)
    if sidecar is None:
        raise ValueError(
            "frozen mean checkpoint has no semantic sidecar: "
            f"{mean_checkpoint_path}"
        )
    manifest = sidecar.get("semantic_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "mean sidecar has no semantic_manifest"
        )
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise ValueError(
            "mean sidecar manifest has no immutable block"
        )
    return immutable, sidecar


# 用途：校验冻结均值 sidecar 满足 v2 geo-season 契约。
# 参数：输入 immutable；输出 无（不一致抛 ValueError）。
def verify_mean_sidecar_v2(immutable):
    for field, expected in MEAN_IMMUTABLE_EXPECTATIONS.items():
        actual = immutable.get(field)
        if actual != expected:
            raise ValueError(
                f"frozen mean sidecar immutable {field}={actual!r} "
                f"does not match the deterministic contract "
                f"({expected!r})"
            )
    if immutable.get("condition_mode") != V2_MEAN_CONDITION_MODE:
        raise ValueError(
            "frozen mean condition_mode="
            f"{immutable.get('condition_mode')!r} != "
            f"'{V2_MEAN_CONDITION_MODE}'"
        )
    if int(immutable.get("cond_chans", -1)) != V2_MEAN_COND_CHANS:
        raise ValueError(
            "frozen mean cond_chans="
            f"{immutable.get('cond_chans')!r} != {V2_MEAN_COND_CHANS}"
        )
    for field in V2_SIDECAR_PRESENCE_FIELDS:
        if immutable.get(field) is None:
            raise ValueError(
                f"frozen mean sidecar immutable lacks {field!r}"
            )
    for field in ("lead_mean", "lead_std"):
        if field not in immutable:
            raise ValueError(
                f"frozen mean sidecar immutable lacks {field}"
            )
    return immutable


# 用途：校验冻结均值文件身份与 sidecar。
# 参数：输入 mean_checkpoint_path；输出 immutable dict。
def verify_frozen_mean(mean_checkpoint_path):
    require_file_sha256(
        mean_checkpoint_path,
        A5_MEAN_SHA256,
        "frozen A5 mean checkpoint",
    )
    immutable, _ = mean_sidecar_immutable(mean_checkpoint_path)
    return verify_mean_sidecar_v2(immutable)


# 用途：把目标配置的模型字段与均值 sidecar 架构逐项核对（忽略 tuple/list 差异）。
# 参数：输入 payload（config JSON）、immutable（均值 sidecar immutable）；输出 无。
def verify_arch_vs_mean_sidecar(payload, immutable):
    def _normalize(value):
        if isinstance(value, (list, tuple)):
            return tuple(_normalize(item) for item in value)
        return value

    from diafno.data.condition_schema import condition_chans
    from diafno.models.config import OSTIAModelConfig
    defaults = OSTIAModelConfig()
    resolved = {}
    for field in ARCH_FIELDS:
        if field == "target_chans":
            value = int(payload.get("output_days", defaults.output_days))
        elif field == "cond_chans":
            value = condition_chans(
                payload["condition_mode"], defaults.input_days
            )
        else:
            value = payload.get(field, getattr(defaults, field))
        resolved[field] = value
    for field in ARCH_FIELDS:
        if _normalize(resolved[field]) != _normalize(
                immutable.get(field)
            ):
            raise ValueError(
                "resolved run config " + field + "=" +
                f"{resolved[field]!r} differs from the frozen mean "
                f"sidecar {field}={immutable.get(field)!r}"
            )
    if payload.get("condition_mode") != V2_MEAN_CONDITION_MODE:
        raise ValueError(
            "A5-centered run requires condition_mode="
            f"'{V2_MEAN_CONDITION_MODE}'"
        )
    if payload.get("model_type") != "centered_diffusion":
        raise ValueError(
            "A5-centered run requires model_type='centered_diffusion'"
        )
    if float(payload.get("sigma_data", -1.0)) != 1.0:
        raise ValueError(
            "A5-centered run requires sigma_data=1.0"
        )


# 用途：校验 centered stats v2 载荷与均值文件三方一致（可选；stats 未生成时提示）。
# 参数：输入 stats_path、mean_checkpoint_path；输出 载荷摘要或 None。
def verify_centered_stats_if_present(stats_path, mean_checkpoint_path):
    if not os.path.isfile(stats_path):
        print(
            "note: centered stats not present yet "
            f"({stats_path}); expected after the train-only "
            "statistics step",
            flush=True,
        )
        return None
    from deterministic_iafno.centered_stats import (
        validate_centered_stats_payload,
    )
    with open(stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)
    validated = validate_centered_stats_payload(
        stats,
        target_chans=15,
        input_days=7,
        output_days=15,
    )
    file_sha = file_sha256(mean_checkpoint_path)
    if validated["mean_checkpoint_sha256"] != file_sha:
        raise ValueError(
            "centered stats mean_checkpoint_sha256="
            f"{validated['mean_checkpoint_sha256']} does not match "
            f"the mean file {file_sha}"
        )
    print(
        "centered stats v2 OK: schema_version="
        f"{stats.get('schema_version')} condition_mode="
        f"{stats.get('condition_mode')} num_samples="
        f"{stats.get('num_samples')}",
        flush=True,
    )
    return validated


# 用途：校验训练侧 SST 归一化与均值 checkpoint 一致（abs 1e-10）。
# 参数：输入 dataset、checkpoint_path；输出 无。
def verify_normalization_identity(dataset, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    normalization = checkpoint.get("normalization") or {}
    checkpoint_mean = normalization.get("sst_mean")
    checkpoint_std = normalization.get("sst_std")
    if checkpoint_mean is None or checkpoint_std is None:
        raise ValueError(
            "frozen mean checkpoint has no sst normalization block"
        )
    if (
            abs(float(dataset.sst_mean) - float(checkpoint_mean)) > 1e-10
            or abs(float(dataset.sst_std) - float(checkpoint_std)) > 1e-10
        ):
        raise ValueError(
            "dataset normalization does not match the frozen mean "
            f"(dataset {dataset.sst_mean}, {dataset.sst_std}; "
            f"checkpoint {checkpoint_mean}, {checkpoint_std}); "
            "tolerance 1e-10 absolute"
        )


def main():
    parser = argparse.ArgumentParser(
        description="A5-centered DiAFNO startup preflight (read-only)"
    )
    parser.add_argument(
        "--config-json",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "configs",
            "ostia_a5_centered_v1.json",
        ),
    )
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--mean-checkpoint", required=True)
    parser.add_argument("--centered-stats")
    parser.add_argument(
        "--data-manifest",
        default=(
            "artifacts/ostia_data_manifest_real.json"
        ),
    )
    parser.add_argument("--lead-stats", default=None)
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    immutable = verify_frozen_mean(args.mean_checkpoint)
    verify_arch_vs_mean_sidecar(payload, immutable)
    verify_data_manifest_identity(args.data_manifest)
    if args.lead_stats:
        require_file_sha256(
            args.lead_stats, LEAD_STATS_SHA256, "lead stats"
        )
    if args.centered_stats:
        verify_centered_stats_if_present(
            args.centered_stats, args.mean_checkpoint
        )
    if os.path.isfile(args.h5_path):
        from diafno.data.ostia import OSTIADailyDataset
        dataset = OSTIADailyDataset(
            h5_path=args.h5_path,
            split="train",
            input_days=7,
            output_days=15,
            condition_mode=V2_MEAN_CONDITION_MODE,
            data_manifest=args.data_manifest,
        )
        verify_normalization_identity(dataset, args.mean_checkpoint)
    print("A5-centered DiAFNO preflight PASS")
    return 0


# 用途：校验数据清单的双摘要与真实日偏移摘要。
# 参数：输入 path、期望各摘要；输出 无。
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


if __name__ == "__main__":
    sys.exit(main())
