"""Read-only OSTIA HDF5 audit / manifest tool (plan section 7).

Inspects an OSTIA daily HDF5 without writing to it and records a JSON
manifest: file facts, dataset shapes/dtypes/chunks/compression,
attributes, time coverage and decodable dates, samples-per-day
structure, lat/lon min/max/non-finite counts and deterministic
coordinate digests, plus a geo-season ``OSTIADailyDataset`` contract
pre-check (instantiating the exact dataset the runner would use).

Non-destructive by design: the manifest target must not exist (the
tool refuses to overwrite), and no data file is ever modified.

Usage (read-only server pre-check):

    python scripts/audit_ostia_h5.py --h5-path /data2/.../file.h5 \
        --out manifests/ostia_h5_audit.json
"""

import argparse
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _h5_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_h5_text(item) for item in value]
    if hasattr(value, "decode"):
        return _h5_text(value.decode("utf-8", errors="replace"))
    if hasattr(value, "item"):
        return _h5_text(value.item())
    return str(value)


def _attrs_snapshot(container):
    return {
        str(key): _h5_text(value)
        for key, value in dict(container.attrs).items()
    }


def _dataset_summary(dataset):
    summary = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": (
            list(dataset.chunks) if dataset.chunks else None
        ),
        "compression": dataset.compression,
        "compression_opts": (
            _h5_text(dataset.compression_opts)
            if dataset.compression_opts is not None
            else None
        ),
        "shuffle": dataset.shuffle,
        "attrs": _attrs_snapshot(dataset),
    }
    return summary


def _decode_time_range(time_values, attrs, file_attrs):
    """Best-effort decoding of the daily time values.

    Mirrors the fail-closed dataset semantics: 'days since
    YYYY-MM-DD' (+optional calendar) or 'first_date'; never treats
    the integers as Unix timestamps.
    """
    import numpy as np
    from datetime import date, timedelta

    def attr(container, key):
        value = container.get(key)
        if value is None and container is not file_attrs:
            value = file_attrs.get(key)
        return _h5_text(value)

    units = attr(attrs, "units")
    calendar = attr(attrs, "calendar")
    first_time = int(time_values[0])
    last_time = int(time_values[-1])
    result = {
        "first_time": first_time,
        "last_time": last_time,
        "num_values": int(time_values.size),
        "calendar_attr": calendar,
    }
    match = None
    if units is not None:
        match = re.match(
            r"^\s*days\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})"
            r"(?:[T ].*)?$",
            units.strip(),
            flags=re.IGNORECASE,
        )
    reference = None
    offset_from_time_zero = True
    if match is not None:
        year, month, day = (int(part) for part in match.groups())
        reference = date(year, month, day)
        result["units"] = f"days since {reference.isoformat()}"
    else:
        first_date = attr(attrs, "first_date")
        if first_date is not None:
            match = re.match(
                r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?$",
                first_date.strip(),
            )
            if match is not None:
                year, month, day = (
                    int(part) for part in match.groups()
                )
                reference = date(year, month, day)
                offset_from_time_zero = False
                result["first_date_attr"] = reference.isoformat()
    if reference is None:
        result["date_semantics"] = (
            "undecodable: no 'days since YYYY-MM-DD' units or "
            "'first_date' attribute; geo-season mode fails closed"
        )
        return result
    def decode(value):
        if offset_from_time_zero:
            return reference + timedelta(days=int(value))
        return reference + timedelta(
            days=int(value) - first_time
        )
    result["first_date"] = decode(first_time).isoformat()
    result["last_date"] = decode(last_time).isoformat()
    result["date_semantics"] = "decodable_gregorian_daily"
    return result


# Coordinates are only ever touched under a strict memory budget:
# 1-D vectors up to DIRECT_READ_LIMIT_BYTES are read whole; every
# other layout (per-sample grids, huge 1-D vectors, ...) is analysed
# with bounded streaming over axis 0.  Nothing is ever fully loaded
# just because a shape is unknown.
_DIRECT_READ_LIMIT_BYTES = 128 * 1024 * 1024
_STREAM_BLOCK_BYTES = 16 * 1024 * 1024


def coordinate_analysis(dataset, direct_read_limit_bytes=None,
                        representative_rows=None):
    """min/max/non-finite count and canonical-bytes sha256 of a
    coordinate dataset without unbounded host memory use.

    Numeric analysis first inspects the shape and dtype metadata.  A
    full read happens only for 1-D datasets whose total byte size is
    within ``direct_read_limit_bytes``.  Small multidimensional
    datasets stream axis-0 blocks.  Huge per-row coordinate grids use
    explicit representative rows (normally every patch from the
    first day), avoiding a terabyte-scale scan of values that the
    converter duplicates for every day.  The report labels sampled
    results honestly; the dataset contract performs separate
    first/middle/last-day repeat checks.  Non-numeric dtypes fail
    closed with a recorded reason instead of being read.
    """
    import hashlib

    import numpy as np

    if direct_read_limit_bytes is None:
        direct_read_limit_bytes = _DIRECT_READ_LIMIT_BYTES
    shape = list(dataset.shape)
    dtype = str(dataset.dtype)
    result = {
        "shape": shape,
        "dtype": dtype,
        "elements": int(
            np.prod(shape, dtype=np.int64)
        ) if shape else 0,
    }
    if not shape or 0 in shape:
        result.update({
            "empty": True,
            "min": None,
            "max": None,
            "nonfinite": 0,
            "sha256": None,
            "read_strategy": "none_empty",
        })
        return result
    kind = np.dtype(dataset.dtype).kind
    if kind not in ("f", "i", "u"):
        result.update({
            "unsupported_dtype": True,
            "min": None,
            "max": None,
            "nonfinite": None,
            "sha256": None,
            "read_strategy": "none_unsupported_dtype",
        })
        return result
    itemsize = np.dtype(dataset.dtype).itemsize
    total_bytes = result["elements"] * itemsize
    rows = int(shape[0])
    row_bytes = max(1, total_bytes // rows)
    step = max(1, _STREAM_BLOCK_BYTES // row_bytes)
    step = min(step, rows)
    direct = (
        dataset.ndim == 1
        and total_bytes <= int(direct_read_limit_bytes)
    )
    digest = hashlib.sha256()
    minimum = None
    maximum = None
    nonfinite = 0

    def _absorb(values):
        nonlocal minimum, maximum, nonfinite
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        nonfinite += int((~finite).sum())
        if finite.any():
            block_min = float(values[finite].min())
            block_max = float(values[finite].max())
            minimum = (
                block_min
                if minimum is None
                else min(minimum, block_min)
            )
            maximum = (
                block_max
                if maximum is None
                else max(maximum, block_max)
            )
        digest.update(
            np.ascontiguousarray(values, dtype="<f8").tobytes()
        )

    if direct:
        _absorb(dataset[()])
        strategy = "full_read_1d"
    elif representative_rows is not None:
        selected = sorted({
            int(index) for index in representative_rows
            if 0 <= int(index) < rows
        })
        if not selected:
            raise ValueError(
                "representative_rows did not select any coordinate "
                "rows"
            )
        for start, stop in _contiguous_ranges(selected):
            _absorb(dataset[start:stop])
        strategy = "sampled_axis0"
    else:
        strategy = "streamed_axis0"
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            _absorb(dataset[start:stop])
    result.update({
        "min": minimum,
        "max": maximum,
        "nonfinite": nonfinite,
        "sha256": digest.hexdigest(),
        "digest_spec": "sha256_le_f8_raw_order",
        "read_strategy": strategy,
    })
    if representative_rows is not None and not direct:
        result.update({
            "sampled_rows": selected,
            "sampled_row_count": len(selected),
            "statistics_scope": "representative_rows_only",
            "sha256_scope": "representative_rows_only",
        })
    else:
        result.update({
            "statistics_scope": "entire_dataset",
            "sha256_scope": "entire_dataset",
        })
    return result


def _contiguous_ranges(indices):
    """Yield half-open ranges covering sorted integer indices."""
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            yield start, previous + 1
            start = value
        previous = value
    yield start, previous + 1


def read_upstream_time_axis(netcdf_path):
    """Read the upstream NetCDF time axis (read-only, fail closed).

    Returns dict(units, calendar, day_offsets, gaps) with the true
    daily offsets of the upstream series (one entry per compact day
    the patched HDF5 was built from).  Supports netCDF4-classic and
    netCDF4/HDF5 files through netCDF4 when available, with a scipy
    fallback for classic files.
    """
    import numpy as np

    time_values = None
    units = None
    calendar = None
    lat_units = lon_units = None
    try:
        import netCDF4
        with netCDF4.Dataset(netcdf_path, "r") as dataset:
            variable = dataset.variables["time"]
            time_values = np.asarray(variable[:])
            units = _h5_text(
                getattr(variable, "units", None)
            )
            calendar = _h5_text(
                getattr(variable, "calendar", None)
            )
            for name in ("latitude", "lat"):
                if name in dataset.variables:
                    lat_units = _h5_text(getattr(
                        dataset.variables[name], "units", None
                    ))
                    break
            for name in ("longitude", "lon"):
                if name in dataset.variables:
                    lon_units = _h5_text(getattr(
                        dataset.variables[name], "units", None
                    ))
                    break
    except Exception as first_error:  # noqa: BLE001 - try scipy
        try:
            from scipy.io import netcdf_file
            with netcdf_file(netcdf_path, "r", mmap=True) as dataset:
                time_values = np.asarray(dataset.variables["time"][:])
                units = _h5_text(
                    getattr(
                        dataset.variables["time"], "units", None
                    )
                )
                calendar = _h5_text(
                    getattr(
                        dataset.variables["time"],
                        "calendar",
                        None,
                    )
                )
                for name in ("latitude", "lat"):
                    if name in dataset.variables:
                        lat_units = _h5_text(getattr(
                            dataset.variables[name],
                            "units",
                            None,
                        ))
                        break
                for name in ("longitude", "lon"):
                    if name in dataset.variables:
                        lon_units = _h5_text(getattr(
                            dataset.variables[name],
                            "units",
                            None,
                        ))
                        break
        except Exception as second_error:  # noqa: BLE001
            raise ValueError(
                "could not read upstream NetCDF time axis from "
                f"{netcdf_path}: {type(first_error).__name__}: "
                f"{first_error} (netCDF4/scipy fallback: "
                f"{second_error})"
            ) from second_error
    if time_values is None or time_values.size == 0:
        raise ValueError(
            "upstream NetCDF has no readable 'time' variable"
        )
    offsets = []
    for value in np.asarray(time_values):
        rounded = round(float(value))
        if abs(float(value) - rounded) > 1e-6:
            raise ValueError(
                "upstream time values are not integral daily "
                f"offsets (got {value!r})"
            )
        offsets.append(int(rounded))
    previous = offsets[0]
    if previous < 0:
        raise ValueError(
            "upstream day offsets must be non-negative, got "
            f"first={previous}"
        )
    gaps = []
    for index in range(1, len(offsets)):
        delta = offsets[index] - offsets[index - 1]
        if delta < 1:
            raise ValueError(
                "upstream day offsets are not strictly increasing "
                f"at index {index} ({offsets[index - 1]} -> "
                f"{offsets[index]})"
            )
        if delta > 1:
            gaps.append({
                "after_ordinal": index - 1,
                "missing_days": delta - 1,
                "next_offset": offsets[index],
            })
    if units is None:
        raise ValueError(
            "upstream NetCDF time variable has no 'units' attribute; "
            "geo-season day offsets cannot be decoded"
        )
    return {
        "units": units,
        "calendar": calendar,
        "day_offsets": offsets,
        "gaps": gaps,
        "coordinate_units": {"lat": lat_units, "lon": lon_units},
    }


def build_data_manifest_payload(netcdf_path, h5_path,
                                h5_audit_payload):
    """Compose the training-ready read-only data manifest.

    Merges the upstream-proven true day offsets with the structural
    facts proven about the patched HDF5 (rows, samples_per_day,
    compact time-axis sha256, coordinate layout).  Never copies or
    rewrites the HDF5; the manifest is pure JSON metadata.
    """
    upstream = read_upstream_time_axis(netcdf_path)
    rows = h5_audit_payload["rows_analysis"]
    n_days = rows["num_days"]
    offsets = upstream["day_offsets"]
    if len(offsets) != n_days:
        raise ValueError(
            f"upstream NetCDF has {len(offsets)} time entries but the "
            f"patched HDF5 stores {n_days} compact days; the two "
            "files do not align (different compaction or missing "
            "entries)"
        )
    coordinates = {}
    if upstream["coordinate_units"]["lat"] is not None:
        coordinates["units"] = {
            "lat": upstream["coordinate_units"]["lat"],
            "lon": upstream["coordinate_units"]["lon"],
        }
    return {
        "schema_version": 1,
        "n_days": int(n_days),
        "day_offsets": offsets,
        "day_offset_sha256": __import__(
            "diafno.data.manifest",
            fromlist=["day_offset_sha256"],
        ).day_offset_sha256(offsets),
        "units": upstream["units"],
        "calendar": (
            upstream["calendar"]
            if upstream["calendar"] is not None
            else "standard"
        ),
        "gaps": upstream["gaps"],
        "coordinates": coordinates,
        "source_netcdf": os.path.abspath(netcdf_path),
        "h5": {
            "path": os.path.abspath(h5_path),
            "num_rows": int(rows["num_rows"]),
            "samples_per_day": int(rows["samples_per_day"]),
            "n_days": int(n_days),
            "time_sha256": _compact_time_sha256_from_payload(
                h5_audit_payload
            ),
            "coordinate_layout": _coordinate_layout_name(
                h5_audit_payload
            ),
        },
    }


def _compact_time_sha256_from_payload(payload):
    """H5 compact time-axis sha256 recorded during the audit."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    recorded = payload.get("time_axis_sha256")
    if recorded:
        return recorded
    return None


def _coordinate_layout_name(payload):
    lat = payload.get("coordinates", {}).get("lat", {})
    shape = lat.get("shape") or []
    if len(shape) == 1:
        return "full_grid_1d_row_aligned"
    if len(shape) == 3:
        return "per_row_patch_grids"
    return "unknown"


def audit_h5_to_json(h5_path, input_days=7, output_days=15,
                     checksum=False):
    """Produce the read-only audit manifest for one HDF5 file."""
    import numpy as np
    import h5py

    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    manifest = {
        "path": h5_path,
        "size_bytes": os.path.getsize(h5_path),
    }
    if checksum:
        digest = hashlib.sha256()
        with open(h5_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["sha256"] = digest.hexdigest()
    with h5py.File(h5_path, "r") as file:
        manifest["file_attrs"] = _attrs_snapshot(file)
        dataset_names = [
            name for name in ("sst", "mask", "lat", "lon", "time")
            if name in file
        ]
        manifest["datasets"] = {}
        for name in dataset_names:
            manifest["datasets"][name] = _dataset_summary(
                file[name]
            )
        time_values = np.asarray(file["time"], dtype=np.int64)
        manifest["time_range"] = _decode_time_range(
            time_values,
            dict(file["time"].attrs),
            dict(file.attrs),
        )
        manifest["time_axis_sha256"] = hashlib.sha256(
            np.ascontiguousarray(
                time_values, dtype="<i8"
            ).tobytes()
        ).hexdigest()
        sst = file["sst"]
        num_rows = int(sst.shape[0])
        # samples_per_day detection mirrors OSTIADailyDataset.
        first_time = int(time_values[0])
        left, right = 1, num_rows
        while left < right:
            middle = (left + right) // 2
            if int(time_values[middle]) == first_time:
                left = middle + 1
            else:
                right = middle
        samples_per_day = left
        manifest["rows_analysis"] = {
            "num_rows": num_rows,
            "samples_per_day": samples_per_day,
            "num_days": (
                num_rows // samples_per_day
                if num_rows % samples_per_day == 0
                else None
            ),
            "consecutive_daily_indices": bool(
                num_rows % samples_per_day == 0
                and int(time_values[-1]) == (
                    first_time + num_rows // samples_per_day - 1
                )
            ),
            "spatial_index_count": samples_per_day,
        }
        # The real converter repeats the same 100 coordinate patches
        # on every day.  Hash/analyse the complete first-day patch set
        # for huge per-row grids; the dataset precheck separately
        # verifies those axes against sampled middle/last days.  Small
        # synthetic files remain fully analysed.
        representative_rows = None
        if (
                file["lat"].ndim > 1
                and int(file["lat"].size)
                * int(file["lat"].dtype.itemsize)
                > _DIRECT_READ_LIMIT_BYTES
            ):
            representative_rows = range(samples_per_day)
        manifest["coordinates"] = {
            "lat": coordinate_analysis(
                file["lat"],
                representative_rows=representative_rows,
            ),
            "lon": coordinate_analysis(
                file["lon"],
                representative_rows=representative_rows,
            ),
        }
    manifest["geo_dataset_precheck"] = _geo_dataset_precheck(
        h5_path,
        input_days=input_days,
        output_days=output_days,
    )
    return manifest


def _geo_dataset_precheck(h5_path, input_days=7, output_days=15,
                          data_manifest=None):
    """Instantiate the runner's exact geo dataset, read-only."""
    from diafno.data.ostia import OSTIADailyDataset

    dataset = None
    try:
        dataset = OSTIADailyDataset(
            h5_path=h5_path,
            split="val",
            input_days=input_days,
            output_days=output_days,
            condition_mode="sst_mask_geo_season",
            data_manifest=data_manifest,
        )
        return {
            "ready": True,
            "condition_mode": dataset.condition_mode,
            "condition_schema_version": (
                dataset.condition_schema_version
            ),
            "condition_chans": dataset.condition_chans,
            "condition_channel_names": list(
                dataset.condition_channel_names
            ),
            "calendar_encoding": dataset.calendar_encoding,
            "time_units_reference": dataset.time_units_reference,
            "geospatial_summary": dataset.geospatial_summary,
            "image_shape": list(dataset.image_shape),
            "val_split_sequences_per_window": (
                dataset.sequences_per_window
            ),
        }
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        return {
            "ready": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    finally:
        if dataset is not None:
            dataset.close()


def write_audit_report(output_path, payload):
    """Write the manifest JSON, refusing to overwrite anything."""
    if os.path.exists(output_path):
        raise RuntimeError(
            f"refusing to overwrite existing audit report: "
            f"{output_path}"
        )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-days", type=int, default=7)
    parser.add_argument("--output-days", type=int, default=15)
    parser.add_argument("--checksum", action="store_true",
                        help="stream a full-file sha256 (slow for "
                             "multi-GB files)")
    parser.add_argument(
        "--source-netcdf",
        default=None,
        help=(
            "upstream NetCDF whose time axis proves the true daily "
            "offsets; produces the training-ready data manifest"
        ),
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help=(
            "write the training-ready data manifest here (refuses to "
            "overwrite); requires --source-netcdf"
        ),
    )
    args = parser.parse_args()
    if args.manifest_out is not None and args.source_netcdf is None:
        raise ValueError("--manifest-out requires --source-netcdf")
    for output_path in (args.out, args.manifest_out):
        if output_path is not None and os.path.exists(output_path):
            raise RuntimeError(
                "refusing to overwrite existing audit output: "
                f"{output_path}"
            )
    payload = audit_h5_to_json(
        args.h5_path,
        input_days=args.input_days,
        output_days=args.output_days,
        checksum=args.checksum,
    )
    if args.source_netcdf is not None:
        data_manifest = build_data_manifest_payload(
            args.source_netcdf,
            args.h5_path,
            payload,
        )
        summary = {
            "ready": True,
            "schema_version": data_manifest["schema_version"],
            "n_days": data_manifest["n_days"],
            "units": data_manifest["units"],
            "calendar": data_manifest["calendar"],
            "gaps": data_manifest["gaps"],
            "day_offset_sha256": data_manifest[
                "day_offset_sha256"
            ],
            "manifest_sha256": __import__(
                "diafno.data.manifest",
                fromlist=["canonical_manifest_sha256"],
            ).canonical_manifest_sha256(data_manifest),
            "source_netcdf": data_manifest["source_netcdf"],
        }
        if args.manifest_out is not None:
            write_audit_report(args.manifest_out, data_manifest)
            summary["file"] = os.path.abspath(args.manifest_out)
            # Re-run the exact geo-season construction using the
            # upstream-proven real-day manifest.  This supersedes the
            # expected fail-closed result produced from metadata-poor
            # HDF5 files before the manifest exists.
            payload["geo_dataset_precheck"] = (
                _geo_dataset_precheck(
                    args.h5_path,
                    input_days=args.input_days,
                    output_days=args.output_days,
                    data_manifest=args.manifest_out,
                )
            )
        payload["data_manifest"] = summary
    write_audit_report(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
