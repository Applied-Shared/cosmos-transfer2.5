"""Canonical short camera names for WFM post-training dataset materialization."""

from typing import Final

# Stable short names used as folder names in the materialized training dataset
# and as SDS rgb object keys (rgb/sds/<segment_id>/<short_name>.mp4).
SDS_CAMERA_SHORT_NAMES: Final[tuple[str, ...]] = (
    "cross_left",
    "cross_right",
    "front_tele",
    "front_wide",
    "rear_left",
    "rear_right",
    "rear",
)

# Local training layout always uses short-name caption folders.
CAPTION_FOLDER: Final[str] = "front_wide"

# OCI caption folders to try (WFM captioning writes FRONT_CENTER today).
CAPTION_SOURCE_FOLDERS: Final[tuple[str, ...]] = (
    "front_wide",
    "FRONT_CENTER",
)

# Mirror of adp/services/wfm/async_job_runners/conditioning/core/bundle/camera_short_names.go.
RAW_NAME_TO_SHORT: Final[dict[str, str]] = {
    # Applied ROG-102 rig.
    "FRONT_CENTER": "front_wide",
    "FRONT_CENTER_NARROW": "front_tele",
    "FRONT_LEFT": "cross_left",
    "FRONT_RIGHT": "cross_right",
    "REAR_CENTER": "rear",
    "REAR_LEFT": "rear_left",
    "REAR_RIGHT": "rear_right",
    # NVIDIA Cosmos-Transfer 2.5 default AV rig.
    "camera_front_wide_120fov": "front_wide",
    "camera_front_tele_30fov": "front_tele",
    "camera_cross_left_120fov": "cross_left",
    "camera_cross_right_120fov": "cross_right",
    "camera_rear_tele_30fov": "rear",
    "camera_rear_left_70fov": "rear_left",
    "camera_rear_right_70fov": "rear_right",
    # spec.json may already use short names as keys.
    "front_wide": "front_wide",
    "front_tele": "front_tele",
    "cross_left": "cross_left",
    "cross_right": "cross_right",
    "rear": "rear",
    "rear_left": "rear_left",
    "rear_right": "rear_right",
}


def short_camera_name(raw_name: str) -> str:
    """Return the canonical short name for a raw sensor or spec.json camera key."""
    return RAW_NAME_TO_SHORT.get(raw_name, raw_name)


def oci_stem_aliases(short_name: str) -> tuple[str, ...]:
    """Return object-key filename stems to try for a canonical short camera name."""
    stems: list[str] = [short_name]
    for raw_name, mapped_short in RAW_NAME_TO_SHORT.items():
        if mapped_short == short_name and raw_name not in stems:
            stems.append(raw_name)
    return tuple(stems)
