"""Download WFM training samples from OCI and materialize Cosmos post-train layout."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wfm_post_training.camera_names import (
    CAPTION_FOLDER,
    SDS_CAMERA_SHORT_NAMES,
    short_camera_name,
)
from wfm_post_training.oci_helpers import download_file, object_exists

if TYPE_CHECKING:
    import boto3

logger = logging.getLogger(__name__)

CONTROL_BUNDLE_PREFIX = "control_bundles"
RGB_PREFIX = "rgb"
CAPTIONS_PREFIX = "captions"
FINETUNING_JOBS_PREFIX = "finetuning_jobs"
# Legacy layout prefix; used only when use_legacy_sds_paths is True.
SDS_PREFIX = "sds"


@dataclass(frozen=True)
class ManifestEntry:
    control_bundle_id: str
    segment_id: str
    caption_version: str


@dataclass(frozen=True)
class FinetuningMappingEntry:
    segment_id: str
    annotation_hash: str
    control_bundle_id: str


@dataclass(frozen=True)
class MaterializeResult:
    dataset_dir: Path
    valid_count: int
    skipped_count: int


def parse_finetuning_mapping_line(line: str, line_num: int) -> FinetuningMappingEntry | None:
    """Parse one segment_annotation_control_bundle.txt line; return None for blank lines."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) != 3:
        raise ValueError(
            f"mapping line {line_num}: expected 3 fields "
            f"(segment_id annotation_hash control_bundle_id), got {len(parts)}"
        )
    return FinetuningMappingEntry(
        segment_id=parts[0],
        annotation_hash=parts[1],
        control_bundle_id=parts[2],
    )


def load_finetuning_mapping(mapping_text: str) -> list[FinetuningMappingEntry]:
    """Parse a full segment_annotation_control_bundle.txt string into entries."""
    entries: list[FinetuningMappingEntry] = []
    for line_num, line in enumerate(mapping_text.splitlines(), start=1):
        entry = parse_finetuning_mapping_line(line, line_num)
        if entry is not None:
            entries.append(entry)
    return entries


def finetuning_mapping_key(flyte_job_id: str) -> str:
    """Return the OCI key for a finetuning run's segment-to-bundle mapping file."""
    return (
        f"{FINETUNING_JOBS_PREFIX}/{flyte_job_id}/segment_annotation_control_bundle.txt"
    )


def download_finetuning_mapping(
    client: "boto3.client",
    bucket: str,
    flyte_job_id: str,
) -> list[FinetuningMappingEntry]:
    """Download and parse segment_annotation_control_bundle.txt from OCI."""
    key = finetuning_mapping_key(flyte_job_id)
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt") as tmp:
        client.download_file(bucket, key, tmp.name)
        return load_finetuning_mapping(Path(tmp.name).read_text(encoding="utf-8"))


def finetuning_mapping_to_manifest_entries(
    mapping_entries: list[FinetuningMappingEntry],
    caption_version: str,
) -> list[ManifestEntry]:
    """Convert finetuning mapping lines to ManifestEntry for materialization."""
    return [
        ManifestEntry(
            control_bundle_id=entry.control_bundle_id,
            segment_id=entry.segment_id,
            caption_version=caption_version,
        )
        for entry in mapping_entries
    ]


def parse_manifest_line(line: str, line_num: int) -> ManifestEntry | None:
    """Parse one JSONL manifest line; return None for blank lines."""
    stripped = line.strip()
    if not stripped:
        return None
    data = json.loads(stripped)
    for field in ("control_bundle_id", "segment_id"):
        if not data.get(field):
            raise ValueError(f"manifest line {line_num}: missing required field {field!r}")
    caption_version = data.get("caption_version") or data.get("caption_id")
    if not caption_version:
        raise ValueError(
            f"manifest line {line_num}: missing required field 'caption_version'"
        )
    return ManifestEntry(
        control_bundle_id=str(data["control_bundle_id"]),
        segment_id=str(data["segment_id"]),
        caption_version=str(caption_version),
    )


def load_manifest(manifest_text: str) -> list[ManifestEntry]:
    """Parse a full JSONL manifest string into entries."""
    entries: list[ManifestEntry] = []
    for line_num, line in enumerate(manifest_text.splitlines(), start=1):
        entry = parse_manifest_line(line, line_num)
        if entry is not None:
            entries.append(entry)
    return entries


def download_manifest(
    client: "boto3.client",
    bucket: str,
    key: str,
) -> list[ManifestEntry]:
    """Download and parse a JSONL manifest from OCI."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".jsonl") as tmp:
        client.download_file(bucket, key, tmp.name)
        return load_manifest(Path(tmp.name).read_text(encoding="utf-8"))


def _control_paths_from_spec(
    client: "boto3.client",
    bucket: str,
    bundle_id: str,
) -> dict[str, str] | None:
    """Return short_name -> OCI key for bbox control videos from spec.json."""
    spec_key = f"{CONTROL_BUNDLE_PREFIX}/{bundle_id}/spec.json"
    if not object_exists(client, bucket, spec_key):
        return None

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json") as tmp:
        client.download_file(bucket, spec_key, tmp.name)
        spec = json.loads(Path(tmp.name).read_text(encoding="utf-8"))

    bundle_prefix = f"{CONTROL_BUNDLE_PREFIX}/{bundle_id}/"
    result: dict[str, str] = {}
    for raw_key, value in spec.items():
        if raw_key == "name" or not isinstance(value, dict):
            continue
        control_path = value.get("control_path")
        if not control_path:
            continue
        short = short_camera_name(raw_key)
        result[short] = bundle_prefix + control_path.lstrip("/")
    return result or None


def _control_paths_from_listing(
    client: "boto3.client",
    bucket: str,
    bundle_id: str,
) -> dict[str, str]:
    """Fallback: list cameras/*/bbox.mp4 and map raw folder names to short names."""
    prefix = f"{CONTROL_BUNDLE_PREFIX}/{bundle_id}/cameras/"
    result: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/bbox.mp4"):
                continue
            # control_bundles/<id>/cameras/<raw>/bbox.mp4
            parts = key.split("/")
            if len(parts) < 2:
                continue
            raw_name = parts[-2]
            short = short_camera_name(raw_name)
            result[short] = key
    return result


def _resolve_control_paths(
    client: "boto3.client",
    bucket: str,
    bundle_id: str,
) -> dict[str, str]:
    """Resolve short_name -> OCI key for all control bbox videos in a bundle."""
    from_spec = _control_paths_from_spec(client, bucket, bundle_id)
    if from_spec is not None:
        return from_spec
    return _control_paths_from_listing(client, bucket, bundle_id)


def _rgb_key(segment_id: str, short_name: str, *, use_legacy_sds_paths: bool) -> str:
    if use_legacy_sds_paths:
        return f"{SDS_PREFIX}/{segment_id}/rgb/{short_name}.mp4"
    return f"{RGB_PREFIX}/{segment_id}/{short_name}.mp4"


def _caption_key(
    segment_id: str,
    caption_version: str,
    *,
    use_legacy_sds_paths: bool,
) -> str:
    if use_legacy_sds_paths:
        return f"{SDS_PREFIX}/{segment_id}/captions/{caption_version}"
    return f"{CAPTIONS_PREFIX}/{segment_id}/{CAPTION_FOLDER}/{caption_version}.json"


def _read_caption_text(caption_path: Path, *, use_legacy_sds_paths: bool) -> str:
    raw = caption_path.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    if use_legacy_sds_paths:
        return raw
    data = json.loads(raw)
    if isinstance(data, dict):
        return str(data.get("caption", "")).strip()
    return raw


def _materialize_sample(
    client: "boto3.client",
    bucket: str,
    entry: ManifestEntry,
    dataset_dir: Path,
    *,
    use_legacy_sds_paths: bool = False,
) -> bool:
    """Download and layout one sample. Returns True on success."""
    sample_id = entry.control_bundle_id
    control_paths = _resolve_control_paths(client, bucket, entry.control_bundle_id)

    missing: list[str] = []
    for short_name in SDS_CAMERA_SHORT_NAMES:
        control_key = control_paths.get(short_name)
        rgb_key = _rgb_key(entry.segment_id, short_name, use_legacy_sds_paths=use_legacy_sds_paths)
        if control_key is None or not object_exists(client, bucket, control_key):
            missing.append(f"control/{short_name}")
        elif not object_exists(client, bucket, rgb_key):
            missing.append(f"rgb/{short_name}")

    caption_key = _caption_key(
        entry.segment_id,
        entry.caption_version,
        use_legacy_sds_paths=use_legacy_sds_paths,
    )
    if not object_exists(client, bucket, caption_key):
        missing.append("caption")

    if missing:
        logger.warning(
            "Skipping sample control_bundle_id=%s segment_id=%s: missing %s",
            entry.control_bundle_id,
            entry.segment_id,
            ", ".join(missing),
        )
        return False

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt") as caption_tmp:
        client.download_file(bucket, caption_key, caption_tmp.name)
        caption_text = _read_caption_text(
            Path(caption_tmp.name),
            use_legacy_sds_paths=use_legacy_sds_paths,
        )
    if not caption_text:
        logger.warning(
            "Skipping sample control_bundle_id=%s: empty caption at s3://%s/%s",
            entry.control_bundle_id,
            bucket,
            caption_key,
        )
        return False

    for short_name in SDS_CAMERA_SHORT_NAMES:
        control_dest = dataset_dir / "control_input_hdmap_bbox" / short_name / f"{sample_id}.mp4"
        rgb_dest = dataset_dir / "videos" / short_name / f"{sample_id}.mp4"
        download_file(client, bucket, control_paths[short_name], control_dest)
        download_file(
            client,
            bucket,
            _rgb_key(entry.segment_id, short_name, use_legacy_sds_paths=use_legacy_sds_paths),
            rgb_dest,
        )

    caption_dest = dataset_dir / "captions" / CAPTION_FOLDER / f"{sample_id}.json"
    caption_dest.parent.mkdir(parents=True, exist_ok=True)
    caption_dest.write_text(
        json.dumps({"caption": caption_text}, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def materialize_dataset(
    client: "boto3.client",
    bucket: str,
    entries: list[ManifestEntry],
    dataset_dir: Path,
    *,
    use_legacy_sds_paths: bool = False,
) -> MaterializeResult:
    """Materialize all manifest entries into dataset_dir."""
    for subdir in ("videos", "control_input_hdmap_bbox", "captions"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)
    for short_name in SDS_CAMERA_SHORT_NAMES:
        (dataset_dir / "videos" / short_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "control_input_hdmap_bbox" / short_name).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "captions" / CAPTION_FOLDER).mkdir(parents=True, exist_ok=True)

    valid = 0
    skipped = 0
    for i, entry in enumerate(entries):
        logger.info(
            "Materializing sample %d/%d: control_bundle_id=%s",
            i + 1,
            len(entries),
            entry.control_bundle_id,
        )
        if _materialize_sample(
            client,
            bucket,
            entry,
            dataset_dir,
            use_legacy_sds_paths=use_legacy_sds_paths,
        ):
            valid += 1
        else:
            skipped += 1

    if valid == 0:
        raise RuntimeError(
            f"No valid samples materialized from {len(entries)} manifest entries "
            f"(skipped {skipped})"
        )

    logger.info(
        "Materialized %d samples to %s (skipped %d)",
        valid,
        dataset_dir,
        skipped,
    )
    return MaterializeResult(dataset_dir=dataset_dir, valid_count=valid, skipped_count=skipped)
