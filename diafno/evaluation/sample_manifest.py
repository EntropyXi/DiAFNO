# 用途：冻结的 val-200/test-200 物理样本清单（读取、校验与构建）。
"""Frozen physical sample manifests for the A5 long-run validation.

A sample manifest pins which physical (region, forecast window) samples
a split evaluation uses, independent of any loader's internal index.
Every entry records:

- the compact window start day and the spatial patch index (physical
  identity of the 7-day input window on the real server HDF5);
- the real calendar dates (Gregorian, from the data manifest's true
  day offsets) of the input window and of the 15 forecast days;
- a coordinate fingerprint of the patch's four static geo channels
  (re-derivable from any loaded sample condition);
- the ``dataset_index`` of the entry inside the *current*
  (gap-filtered, manifest-validated) dataset universe;
- the ``legacy_dataset_index = (compact_start - split_start) * 100 +
  spatial_index`` used by historical evaluation code and by the fixed
  legacy-DiAFNO member seeding rule.

Schema/identity rules follow plans/A5_LONGTRAIN_TEST200_20260908.md:
the 200 entries are drawn with the frozen rule
``np.sort(np.random.default_rng(123).choice(dataset_size, 200,
replace=False))`` and are never re-selected by error, ocean fraction or
any method's behaviour.
"""

import hashlib
import json
from datetime import date

import numpy as np

SAMPLE_MANIFEST_SCHEMA_VERSION = 1

SPLITS = ("val", "test")

# Keys that identify one physical sample across loaders.
PAIRING_KEYS = (
    "input_date_first",
    "input_date_last",
    "target_date_first",
    "target_date_last",
    "spatial_index",
)

_ENTRY_INT_FIELDS = (
    "compact_start",
    "spatial_index",
    "dataset_index",
    "legacy_dataset_index",
)

_ENTRY_STRING_FIELDS = (
    "input_date_first",
    "input_date_last",
    "target_date_first",
    "target_date_last",
    "coordinate_fingerprint",
)


# 用途：计算样本清单的确定性身份摘要（自指字段剔除后）。
# 参数：输入 payload（不含 self-sha 的载荷）；输出 十六进制摘要。
def sample_manifest_sha256(payload):
    """Deterministic identity of a sample-manifest payload.

    Computed over the whole payload with the self-referential
    ``manifest_sha256`` field removed, so the digest changes whenever
    any entry, split header or recorded version changes.
    """
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


# 用途：从样本条件数组提取四个静态 geo 通道并做坐标指纹。
# 参数：输入 condition（numpy [14,H,W] 或 [14,H,W,1] 的 sst_mask_geo_season 条件）；输出 指纹字符串。
def geo_fingerprint_from_condition(condition):
    """Deterministic fingerprint of the four static geo channels.

    Uses geo channels 8..12 of the fixed 14-channel order
    (sin_lat, cos_lat, sin_lon, cos_lon), cast to canonical little-
    endian float64 bytes (``coordinate_sha256`` convention).
    """
    array = np.asarray(condition)
    if array.ndim in (4, 5) and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[0] != 14:
        raise ValueError(
            "condition must have [14,H,W], [14,H,W,1] or "
            f"[14,H,W,1,1] layout, got {tuple(array.shape)}"
        )
    geo = np.asarray(array[8:12], dtype=np.float64)
    canonical = np.ascontiguousarray(geo, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


# 用途：对日期字段做基本校验并返回 (year, month, day)。
# 参数：输入 text（ISO 日期字符串）；输出 tuple。
def _parse_iso_date(text, field):
    try:
        parsed = date.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sample-manifest entry {field}={text!r} is not an "
            "ISO date"
        ) from error
    return parsed.year, parsed.month, parsed.day


# 用途：加载并校验样本清单（fail-closed）。
# 参数：输入 path、split（期望 split）、expected_count、dataset_size（当前数据集长度）；输出 校验后的 payload。
def load_sample_manifest(
        path,
        *,
        split=None,
        expected_count=None,
        dataset_size=None,
    ):
    """Load and validate a frozen physical sample manifest."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("sample manifest must contain an object")
    if payload.get("schema_version") != SAMPLE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported sample-manifest schema_version "
            f"{payload.get('schema_version')!r}"
        )
    recorded_split = payload.get("split")
    if recorded_split not in SPLITS:
        raise ValueError(
            f"sample manifest split must be one of {SPLITS}, "
            f"got {recorded_split!r}"
        )
    if split is not None and split != recorded_split:
        raise ValueError(
            f"sample manifest declares split={recorded_split!r} but "
            f"the requested evaluation split is {split!r}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sample manifest entries must be a non-empty list")
    recorded_count = payload.get("count")
    if recorded_count != len(entries):
        raise ValueError(
            "sample manifest count field does not match its entries: "
            f"{recorded_count} vs {len(entries)}"
        )
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"sample manifest has {len(entries)} entries but the "
            f"evaluation requests {expected_count}"
        )
    geo_size = payload.get("dataset_size_geo")
    if not isinstance(geo_size, int) or geo_size < 1:
        raise ValueError(
            "sample manifest must declare a positive "
            "dataset_size_geo"
        )
    if dataset_size is not None and dataset_size != geo_size:
        raise ValueError(
            f"sample manifest declares dataset_size_geo={geo_size} but "
            f"the current dataset has length {dataset_size}; the two "
            "loaders must share one gap-filtered sample universe"
        )
    seen_keys = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"sample manifest entry {index} is not an object")
        for field in _ENTRY_INT_FIELDS:
            value = entry.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"sample manifest entry {index} field {field} must "
                    "be a non-negative integer"
                )
        for field in _ENTRY_STRING_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"sample manifest entry {index} field {field} must "
                    "be a non-empty string"
                )
        digest = entry["coordinate_fingerprint"]
        if len(digest) != 64:
            raise ValueError(
                f"sample manifest entry {index} coordinate "
                "fingerprint must be a 64-char hex sha256"
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                f"sample manifest entry {index} coordinate "
                f"fingerprint {digest!r} is not hex"
            ) from error
        _parse_iso_date(entry["input_date_first"], "input_date_first")
        _parse_iso_date(entry["input_date_last"], "input_date_last")
        _parse_iso_date(entry["target_date_first"], "target_date_first")
        _parse_iso_date(entry["target_date_last"], "target_date_last")
        key = tuple(entry[field] for field in PAIRING_KEYS)
        if key in seen_keys:
            raise ValueError(
                f"sample manifest entry {index} duplicates a physical "
                "pairing key"
            )
        seen_keys.add(key)
    if len(seen_keys) != len(entries):
        raise ValueError("sample manifest pairing keys are not unique")
    recorded_sha = payload.get("manifest_sha256")
    recomputed_sha = sample_manifest_sha256(payload)
    if recorded_sha != recomputed_sha:
        raise ValueError(
            "sample manifest manifest_sha256 does not match its "
            f"payload: recorded {recorded_sha!r} vs recomputed "
            f"{recomputed_sha!r}"
        )
    return payload


# 用途：返回清单内全部条目的 dataset_index（排序稳定）。
# 参数：输入 payload；输出 int 列表。
def manifest_dataset_indices(payload):
    """The gap-filtered dataset indices of all manifest entries."""
    return sorted(
        int(entry["dataset_index"])
        for entry in payload["entries"]
    )


# 用途：校验"清单声明的宇宙"与"实际加载数据集"完全一致（fail-closed）。
# 参数：输入 payload、dataset_size（实际数据集长度）、split（实际 split）、label；输出 无。
def ensure_manifest_universe(
        payload,
        dataset_size,
        *,
        split=None,
        label="sample manifest",
    ):
    """Fail closed when a manifest is applied to the wrong universe.

    A frozen sample manifest addresses physical samples by
    ``dataset_index`` *inside the gap-filtered universe it was frozen
    from*.  Loading it against a different split (or a differently
    filtered dataset) silently shifts or overruns those indices -- the
    historical ``IndexError(110894)`` came from a val-universe
    manifest (length 218900) being read through a test-split dataset
    (length 110600).  This guard turns that whole class of bug into an
    immediate, explicit error naming both universes.
    """
    recorded_split = payload.get("split")
    if split is not None and recorded_split != split:
        raise ValueError(
            f"{label} declares split={recorded_split!r} but the "
            f"dataset was built with split={split!r}; a frozen sample "
            "manifest must be evaluated in the universe it was frozen "
            "from"
        )
    geo_size = payload.get("dataset_size_geo")
    if int(dataset_size) != int(geo_size):
        raise ValueError(
            f"{label} declares dataset_size_geo={geo_size} but the "
            f"loaded dataset has length {dataset_size} (split="
            f"{recorded_split!r}); the manifest and the loader must "
            "share one gap-filtered sample universe, otherwise every "
            "dataset_index addresses a different physical sample"
        )


# 用途：为指定数据集与物理条目构造清单负载。
# 参数：输入 dataset、indices、split、spatial_entries、geo_size、legacy_split_start；输出 payload。
def build_sample_manifest_payload(
        dataset,
        indices,
        *,
        split,
        spatial_entries,
        dataset_size_geo,
        legacy_split_start,
    ):
    """Assemble a validated sample-manifest payload.

    ``indices`` are the frozen sorted dataset indices;
    ``spatial_entries`` maps each dataset index to a dict with
    ``compact_start`` and ``spatial_index``; calendar dates are decoded
    through the dataset's real day axis; the legacy index uses
    ``legacy_split_start`` (compact split-start day).
    """
    if not getattr(dataset, "has_real_day_axis", False):
        raise ValueError(
            "sample manifests require a dataset bound to a data "
            "manifest (real day axis); refusing to freeze compact-only "
            "indices as physical identity"
        )
    entries = []
    for dataset_index in indices:
        physical = spatial_entries[int(dataset_index)]
        compact_start = int(physical["compact_start"])
        spatial_index = int(physical["spatial_index"])
        entry = {
            "compact_start": compact_start,
            "spatial_index": spatial_index,
            "dataset_index": int(dataset_index),
            "legacy_dataset_index": (
                (compact_start - int(legacy_split_start))
                * int(dataset.samples_per_day)
                + spatial_index
            ),
            "input_date_first": dataset._date_for_ordinal(
                compact_start
            ).isoformat(),
            "input_date_last": dataset._date_for_ordinal(
                compact_start + dataset.input_days - 1
            ).isoformat(),
            "target_date_first": dataset._date_for_ordinal(
                compact_start + dataset.input_days
            ).isoformat(),
            "target_date_last": dataset._date_for_ordinal(
                compact_start + dataset.input_days
                + dataset.output_days - 1
            ).isoformat(),
            "coordinate_fingerprint": geo_fingerprint_from_condition(
                dataset[int(dataset_index)]["condition"].numpy()
            ),
        }
        entries.append(entry)
    payload = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "split": split,
        "count": len(entries),
        "dataset_size_geo": int(dataset_size_geo),
        "legacy_split_start_day": int(legacy_split_start),
        "entries": entries,
    }
    payload["manifest_sha256"] = sample_manifest_sha256(payload)
    return payload
