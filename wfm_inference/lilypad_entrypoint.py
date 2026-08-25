"""Lilypad entrypoint for Cosmos Transfer 2.5 WFM inference."""

import fractions
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import boto3
import botocore.config
import ray
from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

logger = logging.getLogger(__name__)

# OCI S3-compat requires payload signing and disables the default AWS SDK v4
# checksum headers that OCI doesn't support.
_OCI_BOTO_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)

# Persistent directory on the worker node for shared resources that survive
# across jobs in a batch (checkpoint, HF cache). Lives outside tempdir so it
# is not cleaned up between jobs.
_WORKER_CACHE_DIR = Path("/tmp/wfm_worker_cache")

# Subdirectory (alongside the spec.json file) where ground-truth RGB videos are
# staged. Kept distinct from the control bundle tree so RGB never collides with
# a bundle file. Staged next to the spec file (not the assets root) because
# multiview resolves input_path relative to the spec file's directory, and the
# spec need not sit at the assets root.
_RGB_SUBDIR = "_rgb"

# Suffixes accepted in place of a per-camera RGB video. A single still is
# expanded into a video matching that camera's control track, because inference
# reads the same frame window from the input and the control file.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

# Canonical short camera name (the spec.json key) -> the raw sensor stems the
# RGB videos may be named with under rgb/sds/<segment_id>/. RGB is uploaded
# under the raw sensor name (e.g. FRONT_CENTER.mp4), but the short name and the
# NVIDIA rig stems are accepted as fallbacks. Mirrors
# adp/services/wfm/async_job_runners/conditioning/core/bundle/camera_short_names.go
# and adp/services/wfm/scripts/generate_overlay_videos.py.
_SHORT_TO_RAW_CAMERA_STEMS: dict[str, list[str]] = {
    "front_wide": ["FRONT_CENTER", "camera_front_wide_120fov", "front_wide"],
    "front_tele": ["FRONT_CENTER_NARROW", "camera_front_tele_30fov", "front_tele"],
    "cross_left": ["FRONT_LEFT", "camera_cross_left_120fov", "cross_left"],
    "cross_right": ["FRONT_RIGHT", "camera_cross_right_120fov", "cross_right"],
    "rear": ["REAR_CENTER", "camera_rear_tele_30fov", "rear"],
    "rear_left": ["REAR_LEFT", "camera_rear_left_70fov", "rear_left"],
    "rear_right": ["REAR_RIGHT", "camera_rear_right_70fov", "rear_right"],
}


def _apply_recipe_overrides(spec: dict, recipe_overrides: dict) -> None:
    """Apply recipe overrides from the WFM InferenceRecipe onto a spec.json dict in-place.

    camera_conditional_frames maps camera names to frame counts and is merged into
    each camera sub-dict (creating it if absent). An inline prompt replaces prompt_path.
    All other keys are applied at the top level.
    """
    for key, value in recipe_overrides.items():
        if key == "camera_conditional_frames":
            for camera, frame_count in value.items():
                if camera not in spec:
                    spec[camera] = {}
                spec[camera]["num_conditional_frames_per_view"] = frame_count
        elif key == "prompt":
            spec["prompt"] = value
            spec.pop("prompt_path", None)
        else:
            spec[key] = value


def _download_rgb_inputs(
    plain_client: "boto3.client",
    rgb_bucket: str,
    rgb_prefix: str,
    spec_dir: Path,
    logger: "logging.Logger",
) -> dict[str, str]:
    """Download the ground-truth RGB videos under rgb_prefix next to the spec.

    spec_dir is the directory that holds the spec.json file. Videos are staged
    into spec_dir/_RGB_SUBDIR and the returned relative paths are computed
    against spec_dir, because multiview resolves input_path relative to the
    spec file's directory (which need not be the assets root).

    Returns a map of file stem (e.g. "FRONT_CENTER") to the downloaded file's
    path relative to spec_dir, which is the form spec.json input_path values
    take. A stem carrying both a video and a still image resolves to the video:
    an image is only a stand-in for a video it cannot outrank.
    """
    rgb_root = spec_dir / _RGB_SUBDIR
    stem_to_relpath: dict[str, str] = {}
    paginator = plain_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=rgb_bucket, Prefix=rgb_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(rgb_prefix):].lstrip("/")
            # Skip the prefix "directory" placeholder key some listings return.
            if not relative:
                continue
            dest = rgb_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            plain_client.download_file(rgb_bucket, key, str(dest))
            stem = Path(relative).stem
            claimed = stem_to_relpath.get(stem)
            if claimed is not None and not _is_image(Path(claimed)):
                continue
            stem_to_relpath[stem] = str(dest.relative_to(spec_dir))
    logger.info(
        "Downloaded %d RGB input file(s) from s3://%s/%s",
        len(stem_to_relpath), rgb_bucket, rgb_prefix,
    )
    return stem_to_relpath


def _is_image(path: Path) -> bool:
    """Whether path names a still image rather than a video."""
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _resolve_spec_path(spec_dir: Path, raw: str) -> Path:
    """Resolve a spec.json path value, which may be relative to the spec file."""
    path = Path(raw)
    return path if path.is_absolute() else spec_dir / path


class _ControlGeometry(NamedTuple):
    """Frame geometry of a bundle's control videos, as inference will read it."""

    num_frames: int
    rate: fractions.Fraction
    fps: float
    sizes_hw: dict[str, tuple[int, int]]


def _control_video_geometry(spec: dict, spec_dir: Path) -> _ControlGeometry:
    """Read the frame geometry of the active cameras' control videos.

    num_frames is the minimum across control videos, which is the same bound
    inference derives its read window from (min_control_frames in
    cosmos_transfer2/multiview.py). Matching it means a generated input video
    covers whatever window the chunk math settles on, without this code having
    to know state_t, num_chunks, chunk_overlap, or the fps downsample factor.

    rate is PyAV's exact rational average_rate, used for encoding: inference
    collects get_avg_fps() from every input and control file into a set and
    rejects the sample unless all of them agree, so the generated file has to
    carry the control's rate exactly rather than a float round-trip of it.
    """
    import av
    import decord

    frame_counts: list[int] = []
    sizes_hw: dict[str, tuple[int, int]] = {}
    # Keyed by camera so a disagreement can name the offending views.
    signatures: dict[str, tuple[fractions.Fraction, float]] = {}
    for camera, entry in _active_cameras(spec):
        path = _resolve_spec_path(spec_dir, entry["control_path"])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            sizes_hw[camera] = (stream.height, stream.width)
        reader = decord.VideoReader(str(path))
        frame_counts.append(len(reader))
        signatures[camera] = (rate, reader.get_avg_fps())

    if not frame_counts:
        raise ValueError("spec.json has no active camera with a control_path")
    distinct = set(signatures.values())
    if len(distinct) != 1:
        raise ValueError(
            "control videos disagree on frame rate, which inference rejects: "
            + ", ".join(f"{camera}={fps}" for camera, (_, fps) in sorted(signatures.items()))
        )
    rate, fps = distinct.pop()
    return _ControlGeometry(min(frame_counts), rate, fps, sizes_hw)


def _write_still_video(
    image_path: Path,
    dest: Path,
    num_frames: int,
    rate: fractions.Fraction,
    size_hw: tuple[int, int],
) -> None:
    """Encode num_frames copies of image_path into dest at exactly `rate`.

    The image is resized to the control video's dimensions so the generated
    output keeps the control-only baseline's geometry: inference records the
    input video's resolution as original_hw and restores each view to it.
    """
    import av
    import numpy as np
    from PIL import Image

    height, width = size_hw
    # H.264 with yuv420p cannot encode odd dimensions.
    height -= height % 2
    width -= width % 2
    with Image.open(image_path) as opened:
        frame_rgb = np.asarray(opened.convert("RGB").resize((width, height), Image.BILINEAR))

    with av.open(str(dest), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # A fresh VideoFrame per iteration: encode() stamps a pts onto the frame
        # it is given, so reusing one object would re-submit a used timestamp.
        for _ in range(num_frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _verify_still_video(dest: Path, geometry: _ControlGeometry) -> None:
    """Fail loudly unless dest is long enough and carries the control's fps.

    Checked with decord because that is what the dataloader reads with: it asks
    for a fixed index range and puts every file's get_avg_fps() into a set. A
    file that is short or off-rate would otherwise surface far downstream as an
    out-of-bound decord index reported as a NoneType error from the dataset's
    collate function.
    """
    import decord

    reader = decord.VideoReader(str(dest))
    frames, fps = len(reader), reader.get_avg_fps()
    if frames < geometry.num_frames or fps != geometry.fps:
        raise RuntimeError(
            f"generated {dest.name} has {frames} frames at {fps} fps; "
            f"need at least {geometry.num_frames} frames at {geometry.fps} fps "
            "to match the control videos"
        )


def _expand_images_to_videos(
    spec: dict,
    stem_to_relpath: dict[str, str],
    spec_dir: Path,
    logger: "logging.Logger",
) -> None:
    """Replace staged still images with videos matched to the control videos.

    Mutates stem_to_relpath so an expanded stem points at the generated mp4,
    leaving stems that were staged as real videos untouched. Lets a caller
    supply one image per camera instead of full-length RGB; only the leading
    frames are ever used as conditioning, so the rest of the video is padding
    that exists purely to satisfy the dataloader's fixed read window.
    """
    if not any(_is_image(Path(rel)) for rel in stem_to_relpath.values()):
        return

    geometry = _control_video_geometry(spec, spec_dir)
    for camera, _ in _active_cameras(spec):
        matched = next(
            (s for s in _SHORT_TO_RAW_CAMERA_STEMS[camera] if s in stem_to_relpath), None
        )
        if matched is None:
            continue
        source = spec_dir / stem_to_relpath[matched]
        if not _is_image(source):
            continue
        dest = source.with_suffix(".mp4")
        _write_still_video(source, dest, geometry.num_frames, geometry.rate, geometry.sizes_hw[camera])
        _verify_still_video(dest, geometry)
        stem_to_relpath[matched] = str(dest.relative_to(spec_dir))
        logger.info(
            "Expanded %s into %d frame(s) at %s fps for camera %s",
            source.name, geometry.num_frames, geometry.fps, camera,
        )


def _active_cameras(spec: dict):
    """Yield (camera, entry) for each spec camera carrying a control_path.

    A control_path is what makes a view active for the model, so this is the
    set every conditioning input has to cover.
    """
    for camera in _SHORT_TO_RAW_CAMERA_STEMS:
        entry = spec.get(camera)
        if isinstance(entry, dict) and "control_path" in entry:
            yield camera, entry


def _require_conditioning_enabled(spec: dict) -> None:
    """Reject a job that staged conditioning inputs but disabled conditioning.

    Inference swaps the control videos in for the input videos when the global
    num_conditional_frames is 0, so the staged inputs would be ignored and the
    model would condition on the hdmap/bbox render instead. Per-camera counts
    do not lift this: the swap happens before they are read.
    """
    if spec.get("num_conditional_frames", 1) == 0:
        raise ValueError(
            "conditioning inputs were staged but num_conditional_frames is 0, so "
            "inference would silently condition on the control videos instead; "
            "set num_conditional_frames to 1 or 5"
        )


def _require_rgb_for_active_cameras(spec: dict) -> None:
    """Fail loudly when an active camera has no conditioning input staged.

    Inference needs an input_path on every view it treats as active. Naming the
    gap here beats what happens otherwise: a missing per-camera video key raises
    a bare KeyError from the dataset augmentor, several layers from the cause.
    """
    missing = [camera for camera, entry in _active_cameras(spec) if "input_path" not in entry]
    if missing:
        raise ValueError(
            "no conditioning input staged for active camera(s): " + ", ".join(missing)
        )


def _inject_rgb_input_paths(
    spec: dict,
    stem_to_relpath: dict[str, str],
    logger: "logging.Logger",
) -> None:
    """Set input_path on each active camera in spec to its staged RGB video.

    Only cameras that already carry a control_path are touched (those are the
    views the model treats as active). A camera with no matching RGB file has
    any stale input_path cleared, so a missing input is never papered over by a
    leftover bundled path; _require_rgb_for_active_cameras then reports it.
    """
    for camera, entry in _active_cameras(spec):
        matched = next(
            (s for s in _SHORT_TO_RAW_CAMERA_STEMS[camera] if s in stem_to_relpath), None
        )
        if matched is None:
            entry.pop("input_path", None)
            logger.warning(
                "No RGB input video found for camera %s; leaving it control-only", camera,
            )
            continue
        entry["input_path"] = stem_to_relpath[matched]


def _remap_hf_snapshot(
    hf_cache_dir: Path,
    repo: str,
    expected_revision: str,
    logger: "logging.Logger",
) -> None:
    """Copy snapshots/<actual_rev>/ to snapshots/<expected_rev>/ when they differ.

    The OCI cache may have been staged at a different commit than what
    checkpoint_db.py requests. Since file content is identical, we can just
    alias the snapshot directory. HF hub with HF_HUB_OFFLINE=1 looks up files
    by snapshot path, not by blob hash, so this is sufficient.
    """
    import shutil

    model_dir = hf_cache_dir / ("models--" + repo.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        logger.warning("_remap_hf_snapshot: refs/main not found for %s, skipping", repo)
        return

    actual_revision = refs_main.read_text().strip()
    if actual_revision == expected_revision:
        return

    actual_snapshot = model_dir / "snapshots" / actual_revision
    expected_snapshot = model_dir / "snapshots" / expected_revision

    if not actual_snapshot.exists():
        logger.warning("_remap_hf_snapshot: snapshot %s not found for %s, skipping", actual_revision[:8], repo)
        return

    if not expected_snapshot.exists():
        shutil.copytree(str(actual_snapshot), str(expected_snapshot), symlinks=True)
        logger.info("Remapped %s: %s -> %s", repo, actual_revision[:8], expected_revision[:8])


def _download_checkpoint(
    cached_client: "boto3.client",
    bucket: str,
    key: str,
    logger: "logging.Logger",
) -> Path:
    """Download model checkpoint to a persistent cache; skip if already present."""
    dest = _WORKER_CACHE_DIR / "checkpoints" / bucket / key.lstrip("/")
    if dest.exists():
        logger.info("Checkpoint already cached at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading checkpoint s3://%s/%s -> %s", bucket, key, dest)
    cached_client.download_file(bucket, key, str(dest))
    return dest


def _setup_hf_cache(
    plain_client: "boto3.client",
    cached_client: "boto3.client",
    hf_cache_bucket: str,
    hf_cache_prefix: str,
    hf_cache_dir: Path,
    logger: "logging.Logger",
) -> None:
    """Download HF model cache from OCI; skip if expected snapshots already exist."""
    predict2b_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Predict2.5-2B"
        / "snapshots"
        / "6787e176dce74a101d922174a95dba29fa5f0c55"
    )
    reason1_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Reason1-7B"
        / "snapshots"
        / "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    )
    if predict2b_snapshot.exists() and reason1_snapshot.exists():
        logger.info("HF cache already populated, skipping download")
        return

    hf_cache_prefix = hf_cache_prefix.rstrip("/")
    logger.info("Downloading HF model cache from s3://%s/%s", hf_cache_bucket, hf_cache_prefix)
    paginator = plain_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=hf_cache_bucket, Prefix=hf_cache_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(hf_cache_prefix):].lstrip("/")
            dest = hf_cache_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached_client.download_file(hf_cache_bucket, key, str(dest))

    _remap_hf_snapshot(
        hf_cache_dir,
        repo="nvidia/Cosmos-Predict2.5-2B",
        expected_revision="6787e176dce74a101d922174a95dba29fa5f0c55",
        logger=logger,
    )
    _remap_hf_snapshot(
        hf_cache_dir,
        repo="nvidia/Cosmos-Reason1-7B",
        expected_revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        logger=logger,
    )


@ray.remote
def _run_batch_on_gpu(base_config: dict, jobs: list[dict]) -> None:
    """Runs on the GPU worker. Downloads shared resources once, then runs all jobs."""
    import json
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import boto3
    import botocore.config
    from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    _oci_config = botocore.config.Config(
        s3={"payload_signing_enabled": True},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    plain_client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=_oci_config,
    )
    cached_client = get_readonly_boto_client()

    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

    # Download shared resources once for the whole batch.
    checkpoint_path = _download_checkpoint(
        cached_client,
        base_config["checkpoint_bucket"],
        base_config["checkpoint_key"],
        logger,
    )
    _setup_hf_cache(
        plain_client,
        cached_client,
        base_config["hf_cache_bucket"],
        base_config["hf_cache_prefix"],
        hf_cache_dir,
        logger,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Shared resources ready; running %d job(s)", len(jobs))

    num_gpus = base_config.get("num_gpus", 8)
    experiment = base_config["experiment"]

    for i, job in enumerate(jobs):
        control_bucket = job["control_bucket"]
        control_prefix = job["control_prefix"]
        output_bucket = job["output_bucket"]
        output_prefix = job["output_prefix"]
        spec_json = job.get("spec_json", "spec.json")
        recipe_overrides = job.get("recipe_overrides", {})
        # Present only when the recipe requested RGB conditioning frames; the
        # RGB videos live outside the control bundle (rgb/sds/<segment_id>/).
        rgb_bucket = job.get("rgb_bucket")
        rgb_prefix = job.get("rgb_prefix")

        logger.info("Job %d/%d: s3://%s/%s -> s3://%s/%s",
                    i + 1, len(jobs), control_bucket, control_prefix, output_bucket, output_prefix)

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            assets_dir = work / "assets"
            output_dir = work / "outputs"
            output_dir.mkdir(parents=True)

            paginator = plain_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=control_bucket, Prefix=control_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(control_prefix):].lstrip("/")
                    dest = assets_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    plain_client.download_file(control_bucket, key, str(dest))

            has_rgb = bool(rgb_bucket and rgb_prefix)
            if recipe_overrides or has_rgb:
                spec_path = assets_dir / spec_json
                with open(spec_path) as f:
                    spec_data = json.load(f)
                if recipe_overrides:
                    _apply_recipe_overrides(spec_data, recipe_overrides)
                # Stage RGB and wire input_path after overrides so per-camera
                # conditioning counts (which may create camera entries) are
                # already in place when we decide which views are active.
                if has_rgb:
                    _require_conditioning_enabled(spec_data)
                    stem_to_relpath = _download_rgb_inputs(
                        plain_client, rgb_bucket, rgb_prefix, spec_path.parent, logger,
                    )
                    _expand_images_to_videos(
                        spec_data, stem_to_relpath, spec_path.parent, logger,
                    )
                    _inject_rgb_input_paths(spec_data, stem_to_relpath, logger)
                    _require_rgb_for_active_cameras(spec_data)
                with open(spec_path, "w") as f:
                    json.dump(spec_data, f, indent=2)
                logger.info(
                    "Updated %s (recipe_overrides=%s, rgb_conditioning=%s)",
                    spec_json, bool(recipe_overrides), has_rgb,
                )

            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "--master_port=12341",
                "-m", "examples.multiview",
                "-i", str(assets_dir / spec_json),
                "-o", str(output_dir),
                "--checkpoint_path", str(checkpoint_path),
                "--experiment", experiment,
                # Guardrail model (nvidia/Cosmos-Guardrail1) is not staged in OCI;
                # not needed for internal inference on controlled driving data.
                "--disable-guardrails",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=False)

            if result.returncode != 0:
                console_log = output_dir / "console.log"
                if console_log.exists():
                    debug_key = f"{output_prefix}/_debug/console.log"
                    try:
                        plain_client.upload_file(str(console_log), output_bucket, debug_key)
                        logger.info("Uploaded console.log to s3://%s/%s", output_bucket, debug_key)
                    except Exception as upload_err:
                        logger.warning("Could not upload console.log: %s", upload_err)
                raise RuntimeError(f"Job {i + 1}/{len(jobs)} torchrun exited with code {result.returncode}")

            logger.info("Job %d/%d torchrun finished successfully", i + 1, len(jobs))

            output_files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
            logger.info("Uploading %d file(s) to s3://%s/%s", len(output_files), output_bucket, output_prefix)
            for path in output_files:
                key = f"{output_prefix}/{path.relative_to(output_dir)}".lstrip("/")
                plain_client.upload_file(str(path), output_bucket, key)

            plain_client.put_object(
                Body=b"",
                Bucket=output_bucket,
                Key=f"{output_prefix}/succeed.txt",
            )
            logger.info("Job %d/%d upload complete", i + 1, len(jobs))


def run(config: dict) -> None:
    """Lilypad entrypoint for Cosmos Transfer 2.5 multiview inference.

    Accepts either a single job (flat config) or a batch (jobs list). Shared
    resources (model checkpoint, HF model cache) are downloaded once per batch
    and reused across all jobs.

    Base config keys (shared across all jobs):
        checkpoint_bucket:  OCI bucket containing the model checkpoint
        checkpoint_key:     full object key for model_ema_bf16.pt
        hf_cache_bucket:    OCI bucket containing the pre-staged HF model cache
        hf_cache_prefix:    prefix under which the HF cache tree is stored
        experiment:         --experiment arg passed to examples.multiview
        num_gpus:           number of GPUs to use (default: 8)

    Per-job keys (under the jobs list, or at top level for a single job):
        control_bucket:     OCI bucket containing the control bundle assets
        control_prefix:     prefix under which the assets/ tree is stored
        output_bucket:      OCI bucket to upload inference outputs to
        output_prefix:      prefix under which outputs will be written
        spec_json:          spec file path relative to assets root
                            (default: multiview_spec.json)
        rgb_bucket:         OCI bucket holding the conditioning inputs to
                            condition on (optional; RGB conditioning only)
        rgb_prefix:         prefix under which the per-camera conditioning
                            inputs live, e.g. rgb/sds/<segment_id> (optional).
                            When both rgb_bucket and rgb_prefix are set, the
                            files are staged and wired into spec.json as
                            per-camera input_path values. Each file may be a
                            full RGB video or a single still image, which is
                            expanded here into a video matching that camera's
                            control track.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = config.get("num_gpus", 8)

    if "jobs" in config:
        jobs = config["jobs"]
        base_config = {k: v for k, v in config.items() if k != "jobs"}
    else:
        # Single-job flat format for backward compatibility.
        job = {
            "control_bucket": config["control_bucket"],
            "control_prefix": config["control_prefix"],
            "output_bucket": config["output_bucket"],
            "output_prefix": config["output_prefix"],
            "spec_json": config.get("spec_json", "spec.json"),
        }
        # RGB conditioning is optional; carry the top-level keys through so the
        # flat format supports it too (the docstring advertises these as usable
        # at top level for a single job).
        for optional_key in ("rgb_bucket", "rgb_prefix", "recipe_overrides"):
            if optional_key in config:
                job[optional_key] = config[optional_key]
        jobs = [job]
        base_config = config

    ray.init(address="auto")
    logger.info("Dispatching batch of %d job(s) to GPU worker (num_gpus=%d)", len(jobs), num_gpus)
    ref = _run_batch_on_gpu.options(num_gpus=num_gpus).remote(base_config, jobs)
    ray.get(ref)
    logger.info("Batch complete.")
