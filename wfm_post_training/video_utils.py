"""Video helpers for WFM post-training dataset materialization."""

from __future__ import annotations

import logging
import subprocess
from fractions import Fraction
from pathlib import Path

# Cosmos multiview training compares decord avg FPS with exact equality.
CANONICAL_TRAINING_FPS = 10


def parse_frame_rate(rate: str) -> float:
    """Parse an ffprobe avg_frame_rate value like '10/1' or '2997/300'."""
    rate = rate.strip()
    if not rate or rate == "0/0":
        raise ValueError(f"invalid frame rate: {rate!r}")
    return float(Fraction(rate))


def get_avg_fps(path: Path) -> float:
    """Return average FPS for the first video stream in an MP4."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_frame_rate(result.stdout)


def get_decord_fps(path: Path) -> float:
    """Return decord's avg FPS — same probe used by Cosmos training."""
    import io

    from decord import VideoReader

    video_reader = VideoReader(io.BytesIO(path.read_bytes()))
    return float(video_reader.get_avg_fps())


def _reencode_constant_fps(
    path: Path,
    target_fps: int,
    logger: logging.Logger,
) -> None:
    """Re-encode path to a constant frame rate."""
    current_fps = get_avg_fps(path)
    tmp_path = path.with_suffix(".aligned.mp4")
    logger.info(
        "Re-encoding %s FPS %.3f -> %d",
        path.name,
        current_fps,
        target_fps,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            str(target_fps),
            str(tmp_path),
        ],
        check=True,
    )
    tmp_path.replace(path)


def normalize_training_video_pair(
    rgb_path: Path,
    control_path: Path,
    logger: logging.Logger,
) -> None:
    """Force RGB and control videos to the same CFR that Cosmos accepts."""
    try:
        if get_decord_fps(rgb_path) == get_decord_fps(control_path):
            return
    except Exception:
        # Fall through to re-encode when decord cannot open the download yet.
        logger.exception("decord FPS probe failed before normalize")

    for path in (rgb_path, control_path):
        _reencode_constant_fps(path, CANONICAL_TRAINING_FPS, logger)

    rgb_fps = get_decord_fps(rgb_path)
    control_fps = get_decord_fps(control_path)
    if rgb_fps != control_fps:
        raise RuntimeError(
            "decord FPS mismatch after normalize: "
            f"rgb={rgb_fps} control={control_fps} "
            f"(rgb={rgb_path.name}, control={control_path.name})"
        )
    logger.debug(
        "Normalized %s / %s to decord FPS %.3f",
        rgb_path.name,
        control_path.name,
        rgb_fps,
    )
