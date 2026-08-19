# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
ClipGT data loader implementation.

This module provides a loader for ClipGT format data, converting it to the
unified SceneData representation.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import SceneDataLoader, auto_register
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import (
    Crosswalk,
    DynamicObject,
    EgoPose,
    IntersectionArea,
    LaneBoundary,
    LaneLine,
    LaneLineColor,
    LaneLineStyle,
    ObjectType,
    Pole,
    RoadBoundary,
    RoadIsland,
    RoadMarking,
    SceneData,
    TrafficLight,
    TrafficSign,
    TrafficSignType,
    WaitLine,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.data_utils import (
    convert_points_flu_to_rdf,
    convert_quaternions_flu_to_rdf,
    normalize_quaternions,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.laneline_utils import build_lane_line_type

# An unwitnessed color is only assumed constant across a short window, in either
# direction: real signal phases rarely last less than ~3 s (a yellow phase), so
# extrapolating a color further than that from its nearest witness -- backward
# into a lead-in or forward past the last observation -- is a coin flip. Forward,
# the window also comfortably covers benign short occlusion gaps (observed real
# gaps within a healthy track run well under 3 s), so held colors don't flicker.
_TL_STATE_HOLD_MAX_MICROS = 3_000_000
# Signals closer than this are treated as observing the same physical head (or
# the same mast): duplicate track fragments of one head sit well within a couple
# of meters of each other, while heads for different approaches sit farther apart.
# The radius cannot be widened to catch every re-acquired fragment: genuinely
# distinct heads (concurrently witnessed for many seconds) occur within ~4 m of
# each other, while depth noise can put fragments of one head ~8 m apart.
_TL_COLOCATED_DISTANCE_M = 2.5
# A track starting within this window after another ends may be the same
# physical head re-acquired under a new id. Two hold windows is the bound: it
# spans every observed re-acquisition gap, and it guarantees the two nearest
# witnesses' hold windows cover a merged seam end to end, so a merge never
# introduces interior UNKNOWN frames. Concurrently witnessed tracks never
# merge regardless of proximity: the labeler saw both heads lit at once.
_TL_MERGE_MAX_GAP_MICROS = 2 * _TL_STATE_HOLD_MAX_MICROS
# Depth on a small, distant, thin object is the auto-labeler's unreliable
# axis: a re-acquired head can drift ~8 m in 3D, nearly all of it along the
# viewing ray. Bearing survives that noise, so fragments of one head stay
# within a fraction of a degree of the same ray (observed: 0.4 deg at ~55 m)
# while genuinely distinct heads -- even on the same mast -- keep multi-degree
# separation (observed: >= 2.2 deg). The distance cap rejects a farther head
# that happens to line up with the same ray, e.g. the next signal down the
# corridor near the vanishing point.
_TL_MERGE_MAX_VIEW_ANGLE_DEG = 1.0
_TL_MERGE_MAX_DISTANCE_M = 20.0
# Heads serving other approaches can hang within a meter of each other (back
# to back on one span wire, or on one pole corner) yet face elsewhere, so a
# merge also requires roughly agreeing facings when both are known.
_TL_MERGE_MAX_FACING_DEG = 45.0


class _TrafficLightTrackFragment:
    """One signal track's timed-observation summary: the unit of merge detection."""

    def __init__(
        self,
        group_id: str,
        t_first: int,
        t_last: int,
        first_state: str,
        num_observations: int,
        center: np.ndarray,
        facing: Optional[np.ndarray],
    ) -> None:
        self.group_id = group_id
        self.t_first = t_first
        self.t_last = t_last
        self.first_state = first_state
        self.num_observations = num_observations
        self.center = center
        self.facing = facing


def _traffic_light_facings_agree(facing_a: Optional[np.ndarray], facing_b: Optional[np.ndarray]) -> bool:
    """Whether two facing unit vectors agree within ``_TL_MERGE_MAX_FACING_DEG``.

    An unknown facing (identity-quaternion placeholder) doesn't block a merge:
    only a witnessed disagreement is evidence of two distinct heads.
    """
    if facing_a is None or facing_b is None:
        return True
    return float(np.dot(facing_a, facing_b)) >= float(np.cos(np.radians(_TL_MERGE_MAX_FACING_DEG)))


def _traffic_light_view_angle_deg(
    center_a: np.ndarray,
    center_b: np.ndarray,
    reference_ts: float,
    frame_timestamps: Optional[np.ndarray],
    ego_positions: Optional[np.ndarray],
) -> Optional[float]:
    """Angle between the ego->center viewing rays at ``reference_ts``, in degrees.

    This is screen-space proximity in camera-independent form: for any camera
    at the ego position, two centers separated by a small bearing project to
    (nearly) the same pixels, whatever their depths. Returns ``None`` when no
    ego trajectory is available to anchor the rays.
    """
    if (
        frame_timestamps is None
        or len(frame_timestamps) == 0
        or ego_positions is None
        or len(ego_positions) == 0
    ):
        return None
    frame = int(np.clip(np.searchsorted(frame_timestamps, reference_ts), 0, len(ego_positions) - 1))
    ego = np.asarray(ego_positions[frame], dtype=np.float64)
    ray_a = np.asarray(center_a, dtype=np.float64) - ego
    ray_b = np.asarray(center_b, dtype=np.float64) - ego
    norm_a = float(np.linalg.norm(ray_a))
    norm_b = float(np.linalg.norm(ray_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    cos_angle = np.clip(np.dot(ray_a, ray_b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _find_traffic_light_merge_clusters(
    fragments: List[_TrafficLightTrackFragment],
    frame_timestamps: Optional[np.ndarray],
    ego_positions: Optional[np.ndarray],
) -> List[List[_TrafficLightTrackFragment]]:
    """Cluster track fragments that are very likely the same physical head.

    The auto-labeler sometimes loses a signal and re-acquires it under a new
    track id whose estimated 3D position has drifted along the viewing ray.
    Left separate, the fragments each render a box -- two boxes for one
    physical head. A fragment joins an existing cluster only when all hold:

    - its observations start strictly after the cluster's last observation,
      within ``_TL_MERGE_MAX_GAP_MICROS`` (a track that overlaps the cluster
      in time is a genuinely distinct head, never a re-acquisition);
    - some cluster member is spatially the same head: within
      ``_TL_COLOCATED_DISTANCE_M`` in 3D, or within
      ``_TL_MERGE_MAX_VIEW_ANGLE_DEG`` of the same viewing ray from the ego
      at the gap midpoint while no farther than ``_TL_MERGE_MAX_DISTANCE_M``
      (the angular test, not 3D distance, is what tolerates depth noise);
    - that member's facing agrees with the fragment's when both are known.

    Ambiguity between clusters resolves to the smallest viewing angle (or
    smallest 3D distance when no ego trajectory is available -- angle
    availability is uniform across one invocation, so the score units never
    mix). Returns every fragment exactly once, grouped into clusters ordered
    by first observation; unmerged fragments come back as singletons.
    """
    clusters: List[List[_TrafficLightTrackFragment]] = []
    for fragment in sorted(fragments, key=lambda f: (f.t_first, f.group_id)):
        best_cluster: Optional[List[_TrafficLightTrackFragment]] = None
        best_score: Optional[float] = None
        for cluster in clusters:
            cluster_last_ts = max(member.t_last for member in cluster)
            gap = fragment.t_first - cluster_last_ts
            if not 0 < gap <= _TL_MERGE_MAX_GAP_MICROS:
                continue
            reference_ts = (cluster_last_ts + fragment.t_first) / 2.0
            for member in cluster:
                if not _traffic_light_facings_agree(member.facing, fragment.facing):
                    continue
                distance = float(np.linalg.norm(member.center - fragment.center))
                if distance > _TL_MERGE_MAX_DISTANCE_M:
                    continue
                angle = _traffic_light_view_angle_deg(
                    member.center, fragment.center, reference_ts, frame_timestamps, ego_positions
                )
                if distance > _TL_COLOCATED_DISTANCE_M and (
                    angle is None or angle > _TL_MERGE_MAX_VIEW_ANGLE_DEG
                ):
                    continue
                score = angle if angle is not None else distance
                if best_score is None or score < best_score:
                    best_score = score
                    best_cluster = cluster
        if best_cluster is not None:
            best_cluster.append(fragment)
        else:
            clusters.append([fragment])
    return clusters


def _apply_traffic_light_merge_seam(
    states: List[Optional[str]],
    frame_timestamps: Optional[np.ndarray],
    seam_start_ts: int,
    seam_end_ts: int,
    seam_end_state: str,
) -> None:
    """Rewrite the unwitnessed frames between two merged fragments' spans.

    The sequence builder's hold-last fill carries the earlier fragment's final
    color across the whole seam, which goes stale the moment the head actually
    changed (unwitnessed). Instead, each seam frame takes its temporally
    nearest witness's color -- the earlier fragment's last color up to the gap
    midpoint, the later fragment's first color after it -- and only within
    ``_TL_STATE_HOLD_MAX_MICROS`` of that witness (the same signal-phase bound
    the lead-in and trailing windows use); frames supported by neither witness
    become UNKNOWN. ``_TL_MERGE_MAX_GAP_MICROS`` currently guarantees the two
    windows meet, so merged seams render fully colored. Mutates ``states``.
    """
    if frame_timestamps is None or len(frame_timestamps) == 0:
        return
    seam_mid_ts = (seam_start_ts + seam_end_ts) / 2.0
    lo = int(np.searchsorted(frame_timestamps, seam_start_ts, side="right"))
    hi = int(np.searchsorted(frame_timestamps, seam_end_ts, side="left"))
    for frame in range(lo, min(hi, len(states))):
        ts = int(frame_timestamps[frame])
        if ts <= seam_mid_ts:
            if ts - seam_start_ts > _TL_STATE_HOLD_MAX_MICROS:
                states[frame] = None
        else:
            states[frame] = seam_end_state if seam_end_ts - ts <= _TL_STATE_HOLD_MAX_MICROS else None


class _TrafficLightSequence:
    """Per-signal state sequence plus what's needed to vet its unwitnessed guesses.

    ``states[first_observed_frame:last_observed_end]`` spans first to last
    observation. For a single track the whole span is witnessed (real
    observations plus hold-last interpolation between them); for a merged light
    only ``witnessed_intervals`` -- each member fragment's own span -- is, and
    the seam frames between fragments carry midpoint-rule guesses (see
    ``_apply_traffic_light_merge_seam``). Frames outside the span carry lead-in
    / trailing guesses. The resolvers below decide whether each guess is kept;
    guesses never count as evidence for other lights, so evidence scans must go
    through ``witnessed_frames`` rather than the raw span.
    """

    def __init__(
        self,
        center: np.ndarray,
        states: List[Optional[str]],
        first_observed_frame: int,
        last_observed_end: int,
        last_observed_ts: Optional[int],
        witnessed_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.center = center
        self.states = states
        self.first_observed_frame = first_observed_frame
        self.last_observed_end = last_observed_end
        self.last_observed_ts = last_observed_ts
        self.witnessed_intervals = witnessed_intervals or [(first_observed_frame, last_observed_end)]

    def witnessed_frames(self, lo: int, hi: int) -> Iterator[int]:
        """Genuinely witnessed frame indices within ``[lo, hi)``, ascending."""
        for start, end in self.witnessed_intervals:
            yield from range(max(start, lo), min(end, hi))


def _resolve_traffic_light_lead_ins(
    lights: List[_TrafficLightSequence], frame_timestamps: Optional[np.ndarray]
) -> None:
    """Decide how much of each signal's leading-frame color guess to keep.

    Frames before a signal's first observation carry its first observed color as
    a guess, not data. A guess is only trustworthy as far back as evidence
    supports it, so for each signal with a lead-in gap this weighs the states
    that co-located signals (within ``_TL_COLOCATED_DISTANCE_M`` -- duplicate
    track fragments or heads on the same mast) actually observed during the gap:

    - Any co-located signal showing a DIFFERENT color during the gap means the
      guess is wrong for part of the gap, and rendering it would draw two
      contradictory boxes on one physical head. The whole lead-in becomes
      UNKNOWN (``None`` entries, rendered gray).
    - A co-located signal showing the SAME color pushes the witnessed boundary
      back to that evidence, keeping the guess through the corroborated span.
    - With no evidence either way, the guess is kept only for
      ``_TL_STATE_HOLD_MAX_MICROS`` before the earliest witness; earlier
      frames become UNKNOWN.

    Only genuinely witnessed frames of another signal count as evidence: a
    lead-in, trailing, or merge-seam guess can't corroborate or contradict
    another guess. Mutates each ``states`` list in place.
    """
    if frame_timestamps is None or len(frame_timestamps) == 0:
        return
    for light in lights:
        gap_frames = min(light.first_observed_frame, len(light.states))
        if gap_frames <= 0:
            continue
        guess = light.states[0]
        contradicted = False
        witnessed_frame = gap_frames  # frame index of the earliest supporting evidence
        for other in lights:
            if other is light:
                continue
            if float(np.linalg.norm(light.center - other.center)) > _TL_COLOCATED_DISTANCE_M:
                continue
            for frame in other.witnessed_frames(0, gap_frames):
                if other.states[frame] != guess:
                    contradicted = True
                    break
                witnessed_frame = min(witnessed_frame, frame)
            if contradicted:
                break
        if contradicted:
            light.states[:gap_frames] = [None] * gap_frames
            continue
        witnessed_ts = int(frame_timestamps[min(witnessed_frame, len(frame_timestamps) - 1)])
        for frame in range(gap_frames):
            if int(frame_timestamps[frame]) < witnessed_ts - _TL_STATE_HOLD_MAX_MICROS:
                light.states[frame] = None


def _resolve_traffic_light_trailing_holds(
    lights: List[_TrafficLightSequence], frame_timestamps: Optional[np.ndarray]
) -> None:
    """Decide how much of each signal's trailing hold-forward guess to keep.

    Frames after a signal's last observation carry its last observed color as a
    guess, not data. Held forever, a stale guess overlaps reality when the
    labeler re-acquires the same physical head under a new track id (with enough
    position noise to defeat co-location): the old track keeps painting its old
    color on top of the new track's witnessed one. So the forward hold is vetted
    exactly like the lead-in guess, against co-located signals' genuinely
    witnessed frames:

    - A co-located signal witnessing a DIFFERENT color during the hold proves the
      head had changed by that frame; the hold is cut there (later frames become
      UNKNOWN). Frames the window keeps before the change stay: the color was
      still witnessed-true at the hold's start.
    - A co-located signal witnessing the SAME color pushes the witnessed
      boundary forward, keeping the hold through the corroborated span.
    - Beyond the latest supporting witness the guess survives only
      ``_TL_STATE_HOLD_MAX_MICROS``; later frames become UNKNOWN.

    UNKNOWN clears only the color: the light keeps rendering as a gray box, so
    presence is unaffected. Mutates each ``states`` list in place.
    """
    if frame_timestamps is None or len(frame_timestamps) == 0:
        return
    for light in lights:
        num_frames = len(light.states)
        trail_start = light.last_observed_end
        if trail_start >= num_frames or light.last_observed_ts is None:
            continue
        guess = light.states[trail_start]
        contradiction_frame = num_frames
        corroborated_frames: List[int] = []
        for other in lights:
            if other is light:
                continue
            if float(np.linalg.norm(light.center - other.center)) > _TL_COLOCATED_DISTANCE_M:
                continue
            for frame in other.witnessed_frames(trail_start, num_frames):
                if other.states[frame] != guess:
                    contradiction_frame = min(contradiction_frame, frame)
                    break
                corroborated_frames.append(frame)
        # Corroboration past a proven change can't resurrect the hold.
        witnessed_ts = int(light.last_observed_ts)
        for frame in corroborated_frames:
            if frame < contradiction_frame:
                witnessed_ts = max(witnessed_ts, int(frame_timestamps[frame]))
        for frame in range(trail_start, num_frames):
            if frame >= contradiction_frame or int(frame_timestamps[frame]) > witnessed_ts + _TL_STATE_HOLD_MAX_MICROS:
                light.states[frame] = None


@auto_register(priority=10)  # High priority for ClipGT format
class ClipGTLoader(SceneDataLoader):
    """Loader for ClipGT format data."""

    @property
    def name(self) -> str:
        """Get the loader name."""
        return "clipgt"

    @property
    def description(self) -> str:
        """Get loader description."""
        return "Loader for ClipGT parquet-based scene data format"

    def _detect_clip_id(self, path: Path) -> Optional[str]:
        """
        Detect the clip_id from a directory.
        First tries to use the directory name, then scans for parquet files.
        """
        # Try directory name as clip_id first
        clip_id = path.name
        required_files = [
            f"{clip_id}.calibration_estimate.parquet",
            f"{clip_id}.egomotion_estimate.parquet",
        ]
        if all((path / f).exists() for f in required_files):
            return clip_id

        # If not found, scan for any calibration_estimate.parquet files
        calib_files = list(path.glob("*.calibration_estimate.parquet"))
        for calib_file in calib_files:
            # Extract clip_id from filename: {clip_id}.calibration_estimate.parquet
            clip_id = calib_file.name.replace(".calibration_estimate.parquet", "")
            required_files = [
                f"{clip_id}.calibration_estimate.parquet",
                f"{clip_id}.egomotion_estimate.parquet",
            ]
            if all((path / f).exists() for f in required_files):
                return clip_id

        return None

    def can_load(self, source: Union[Path, str, Dict[str, Any]]) -> bool:
        """Check if source is a ClipGT directory."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return self._detect_clip_id(path) is not None
        return False

    def load(
        self,
        source: Union[Path, str, Dict[str, Any]],
        camera_names: Optional[List[str]] = None,
        max_frames: int = -1,
        input_pose_fps: int = 30,
        resize_resolution_hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> SceneData:
        """
        Load ClipGT scene data.

        Args:
            source: ClipGT directory path
            camera_names: Optional list of camera names to load
            max_frames: Maximum frames to load
            input_pose_fps: Target frame rate for interpolation
            resize_resolution_hw: Optional camera resize resolution
            **kwargs: Additional arguments

        Returns:
            Loaded scene data
        """
        if not isinstance(source, (str, Path)):
            raise TypeError(f"ClipGTLoader only supports string or Path sources, got {type(source)!r}")

        clipgt_path = Path(source)
        clip_id = self._detect_clip_id(clipgt_path)

        if clip_id is None:
            raise ValueError(
                f"Could not detect clip_id from directory: {clipgt_path}. "
                "Expected files: {{clip_id}}.calibration_estimate.parquet and {{clip_id}}.egomotion_estimate.parquet"
            )

        logger.debug(f"Loading ClipGT data from: {clipgt_path} (clip_id: {clip_id})")

        # Initialize scene data
        scene_data = SceneData(scene_id=clip_id, frame_rate=input_pose_fps, duration_seconds=0.0)

        # Define file paths
        files = {
            "calibration": clipgt_path / f"{clip_id}.calibration_estimate.parquet",
            "egomotion": clipgt_path / f"{clip_id}.egomotion_estimate.parquet",
            "obstacle": clipgt_path / f"{clip_id}.obstacle.parquet",
            "lane": clipgt_path / f"{clip_id}.lane.parquet",
            "lane_line": clipgt_path / f"{clip_id}.lane_line.parquet",
            "road_boundary": clipgt_path / f"{clip_id}.road_boundary.parquet",
            "crosswalk": clipgt_path / f"{clip_id}.crosswalk.parquet",
            "pole": clipgt_path / f"{clip_id}.pole.parquet",
            "road_marking": clipgt_path / f"{clip_id}.road_marking.parquet",
            "wait_line": clipgt_path / f"{clip_id}.wait_line.parquet",
            "traffic_light": clipgt_path / f"{clip_id}.traffic_light.parquet",
            "traffic_sign": clipgt_path / f"{clip_id}.traffic_sign.parquet",
            "intersection_area": clipgt_path / f"{clip_id}.intersection_area.parquet",
            "road_island": clipgt_path / f"{clip_id}.road_island.parquet",
            "buffer_zone": clipgt_path / f"{clip_id}.buffer_zone.parquet",
            "camera_timestamps": clipgt_path / f"{clip_id}.camera_front_wide_120fov.json",
        }

        # Load ego poses (use camera timestamps if available for frame-accurate sync)
        if files["egomotion"].exists():
            self._load_ego_poses(
                scene_data,
                files["egomotion"],
                input_pose_fps,
                max_frames,
                # Use camera timestamps as reference (cameras are assumed to be synchronized)
                camera_timestamps_file=files["camera_timestamps"] if files["camera_timestamps"].exists() else None,
            )

        # Load camera calibrations
        if files["calibration"].exists():
            self._load_camera_calibrations(scene_data, files["calibration"], camera_names, resize_resolution_hw)

        # Load dynamic objects
        if files["obstacle"].exists():
            self._load_dynamic_objects(scene_data, files["obstacle"])

        # Load map elements
        self._load_map_elements(scene_data, files)

        return scene_data

    def _load_ego_poses(
        self,
        scene_data: SceneData,
        ego_file: Path,
        target_fps: int,
        max_frames: int,
        camera_timestamps_file: Optional[Path] = None,
    ) -> None:
        """Load and interpolate ego poses.

        If a camera timestamp JSON file is provided, the poses will be interpolated
        to match camera frame timestamps for frame-accurate synchronization.
        Otherwise, poses are interpolated to a uniform target_fps.

        Args:
            scene_data: Scene data to populate with ego poses
            ego_file: Path to egomotion parquet file
            target_fps: Target frame rate for interpolation (used if no camera timestamps)
            max_frames: Maximum number of frames to load (-1 for all)
            camera_timestamps_file: Optional path to camera timestamp JSON file
        """
        ego_df = pd.read_parquet(ego_file)

        positions = []
        quaternions = []
        timestamps = []

        for _, row in ego_df.iterrows():
            ego_data = row["egomotion_estimate"]
            key = row["key"]

            if "location" in ego_data and "orientation" in ego_data:
                loc = ego_data["location"]
                ori = ego_data["orientation"]

                positions.append([loc["x"], loc["y"], loc["z"]])
                quaternions.append([ori["x"], ori["y"], ori["z"], ori["w"]])

                if isinstance(key, dict) and "timestamp_micros" in key:
                    timestamps.append(key["timestamp_micros"])

        if not timestamps:
            logger.warning("No ego poses found")
            return

        positions = np.array(positions)
        quaternions = np.array(quaternions)
        timestamps = np.array(timestamps)

        # Try to load camera timestamps for frame-accurate synchronization
        camera_timestamps = self._load_camera_timestamps(camera_timestamps_file)

        if camera_timestamps is not None:
            # Use camera timestamps for interpolation (frame-accurate sync)
            target_timestamps_micros = camera_timestamps.astype(np.float64)

            # Apply max_frames limit
            if max_frames > 0 and len(target_timestamps_micros) > max_frames:
                target_timestamps_micros = target_timestamps_micros[:max_frames]

            # Log if extrapolation will be needed
            ego_start, ego_end = timestamps[0], timestamps[-1]
            n_before = int(np.sum(target_timestamps_micros < ego_start))
            n_after = int(np.sum(target_timestamps_micros > ego_end))
            if n_before > 0 or n_after > 0:
                logger.warning(
                    f"Extrapolating {n_before} frames before and {n_after} frames after ego pose range "
                    f"(using constant velocity assumption)"
                )

            num_frames = len(target_timestamps_micros)
            duration = (target_timestamps_micros[-1] - target_timestamps_micros[0]) / 1e6
            sync_mode = "camera timestamps"
        else:
            # Fall back to uniform FPS interpolation
            duration = (timestamps[-1] - timestamps[0]) / 1e6  # seconds
            num_frames = int(duration * target_fps) + 1

            if max_frames > 0:
                num_frames = min(num_frames, max_frames)

            # Create target timestamps at the desired frame rate
            target_timestamps_seconds = np.linspace(0, (num_frames - 1) / target_fps, num_frames)
            target_timestamps_micros = timestamps[0] + (target_timestamps_seconds * 1e6)
            sync_mode = f"{target_fps} Hz"

        # Interpolate positions
        interp_positions = []
        for i in range(3):  # x, y, z
            f = interp1d(
                timestamps,
                positions[:, i],
                kind="linear",
                fill_value=cast(Any, "extrapolate"),
            )
            interp_positions.append(f(target_timestamps_micros))
        interp_positions = np.array(interp_positions).T

        # Interpolate quaternions using SLERP with extrapolation support
        interp_quaternions = self._interpolate_quaternions_with_extrapolation(
            timestamps, quaternions, target_timestamps_micros
        )

        # Create EgoPose objects with proper microsecond timestamps (OpenCV RDF)
        for i in range(num_frames):
            # Convert position
            pos_flu = interp_positions[i].astype(np.float32)
            pos_rdf = convert_points_flu_to_rdf(pos_flu.reshape(1, 3))[0]

            # Convert orientation via basis change: R_rdf = S * R_flu * S^T
            quat_rdf = convert_quaternions_flu_to_rdf(
                interp_quaternions[i].reshape(1, 4),
                double_sided=True,
            )[0]

            # Maintain original microsecond resolution
            timestamp_us = int(target_timestamps_micros[i])

            scene_data.ego_poses.append(
                EgoPose(
                    timestamp=timestamp_us,
                    position=pos_rdf,
                    orientation=quat_rdf,
                )
            )

        scene_data.duration_seconds = duration
        scene_data.metadata["coordinate_frame"] = "opencv_rdf"
        logger.debug(f"Loaded {num_frames} ego poses at {sync_mode} (OpenCV RDF)")

    def _load_camera_timestamps(self, camera_timestamps_file: Optional[Path]) -> Optional[np.ndarray]:
        """Load camera frame timestamps from JSON file.

        Args:
            camera_timestamps_file: Path to camera timestamp JSON file

        Returns:
            Array of timestamps in microseconds, or None if file not provided or invalid
        """
        if camera_timestamps_file is None:
            return None

        try:
            with open(camera_timestamps_file) as f:
                cam_data = json.load(f)

            timestamps = np.array([frame["timestamp"] for frame in cam_data], dtype=np.int64)
            logger.info(
                f"Using camera timestamps from {camera_timestamps_file.name} "
                f"({len(timestamps)} frames) for pose interpolation"
            )
            return timestamps
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load camera timestamps from {camera_timestamps_file}: {e}")
            return None

    def _interpolate_quaternions_with_extrapolation(
        self,
        keyframe_timestamps: np.ndarray,
        keyframe_quaternions: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Interpolate quaternions with constant angular velocity extrapolation.

        Uses SLERP for interpolation within the keyframe range, and constant
        angular velocity extrapolation for timestamps outside the range.

        Args:
            keyframe_timestamps: Timestamps of keyframe poses (microseconds)
            keyframe_quaternions: Quaternions at keyframe poses, shape (N, 4)
            target_timestamps: Target timestamps to interpolate to (microseconds)

        Returns:
            Interpolated quaternions, shape (M, 4)
        """
        t_start, t_end = keyframe_timestamps[0], keyframe_timestamps[-1]

        # Identify which timestamps need extrapolation vs interpolation
        before_mask = target_timestamps < t_start
        after_mask = target_timestamps > t_end
        interp_mask = ~before_mask & ~after_mask

        result = np.zeros((len(target_timestamps), 4), dtype=np.float64)

        # Handle interpolation (within range) using SLERP
        if np.any(interp_mask):
            rotations = Rotation.from_quat(keyframe_quaternions)
            slerp = Slerp(keyframe_timestamps.astype(np.float64), rotations)
            interp_rotations = slerp(target_timestamps[interp_mask])
            result[interp_mask] = interp_rotations.as_quat()

        # Handle extrapolation before the first keyframe
        if np.any(before_mask):
            # Compute angular velocity from first two keyframes
            r0 = Rotation.from_quat(keyframe_quaternions[0])
            r1 = Rotation.from_quat(keyframe_quaternions[1])
            dt = keyframe_timestamps[1] - keyframe_timestamps[0]  # microseconds
            # Relative rotation from r0 to r1
            delta_r = r0.inv() * r1
            # Angular velocity as rotation vector (radians per microsecond)
            omega = delta_r.as_rotvec() / dt

            # Extrapolate backwards
            for i in np.where(before_mask)[0]:
                dt_extrap = target_timestamps[i] - t_start  # negative value
                delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
                result[i] = (r0 * delta_extrap).as_quat()

        # Handle extrapolation after the last keyframe
        if np.any(after_mask):
            # Compute angular velocity from last two keyframes
            r_m1 = Rotation.from_quat(keyframe_quaternions[-2])
            r_m0 = Rotation.from_quat(keyframe_quaternions[-1])
            dt = keyframe_timestamps[-1] - keyframe_timestamps[-2]  # microseconds
            # Relative rotation from r_m1 to r_m0
            delta_r = r_m1.inv() * r_m0
            # Angular velocity as rotation vector (radians per microsecond)
            omega = delta_r.as_rotvec() / dt

            # Extrapolate forwards
            for i in np.where(after_mask)[0]:
                dt_extrap = target_timestamps[i] - t_end  # positive value
                delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
                result[i] = (r_m0 * delta_extrap).as_quat()

        return result

    def _load_camera_calibrations(
        self,
        scene_data: SceneData,
        cal_file: Path,
        camera_names: Optional[List[str]],
        resize_hw: Optional[Tuple[int, int]],
    ) -> None:
        """Load camera calibration data."""
        cal_df = pd.read_parquet(cal_file)
        cal_data = cal_df.iloc[0]["calibration_estimate"]
        rig_data = json.loads(str(cal_data["rig_json"]))

        # Build a map of sensor name variants: colon and underscore
        sensors = rig_data["rig"]["sensors"]
        name_to_sensor: dict[str, dict] = {}
        for sensor in sensors:
            name = sensor.get("name", "")
            if name:
                name_to_sensor[name] = sensor
                name_to_sensor[name.replace(":", "_")] = sensor

        # If no specific cameras requested, load all available cameras (underscore form)
        if camera_names is None:
            camera_names = [s["name"].replace(":", "_") for s in sensors if s.get("name", "").startswith("camera:")]

        for camera_name in camera_names:
            # Accept either underscore or colon input names
            camera_dict = name_to_sensor.get(camera_name)
            if camera_dict is None:
                camera_dict = name_to_sensor.get(camera_name.replace("_", ":"))

            if camera_dict is None:
                logger.warning(f"Camera {camera_name} not found in calibration")
                continue

            props = camera_dict["properties"]

            # Get polynomial coefficients
            poly_key = "polynomial" if "polynomial" in props else "bw-poly"
            if poly_key not in props:
                logger.warning(f"No polynomial coefficients for {camera_name}")
                continue

            poly_str = props[poly_key]
            poly_coeffs = np.array([float(x) for x in poly_str.split()], dtype=np.float32)
            if len(poly_coeffs) == 5:
                poly_coeffs = np.append(poly_coeffs, 0.0)

            # Get intrinsics
            cx = float(props["cx"])
            cy = float(props["cy"])
            width = int(props["width"])
            height = int(props["height"])

            # Determine polynomial direction from polynomial-type field
            # pixeldistance-to-angle = backward (r → θ) = needs inversion
            # angle-to-pixeldistance = forward (θ → r) = use directly
            poly_type = props.get("polynomial-type", "")
            if poly_type == "angle-to-pixeldistance":
                is_bw_poly = False  # Forward polynomial
            elif poly_type == "pixeldistance-to-angle":
                is_bw_poly = True  # Backward polynomial
            elif poly_key == "bw-poly":
                is_bw_poly = True  # Explicitly named backward poly
            else:
                # Heuristic fallback: backward poly has small c1 (radians/pixel)
                # Forward poly has large c1 (pixels/radian, ~focal length)
                is_bw_poly = len(poly_coeffs) > 1 and abs(poly_coeffs[1]) < 1.0

            # Get linear affine term [[C,D],[D,E]] - defaults to identity
            linear_c = float(props.get("linear-c", 1.0))
            linear_d = float(props.get("linear-d", 0.0))
            linear_e = float(props.get("linear-e", 0.0))
            linear_cde = np.array([linear_c, linear_d, linear_e], dtype=np.float32)

            camera_model = FThetaCamera(
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                poly=poly_coeffs.copy(),
                is_bw_poly=is_bw_poly,
                linear_cde=linear_cde.copy(),
            )

            # Apply resize if specified
            if resize_hw:
                resize_h, resize_w = resize_hw
                scale_h = resize_h / height
                scale_w = resize_w / width

                camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)

            cx = float(camera_model._center[0])
            cy = float(camera_model._center[1])
            width = int(camera_model.width)
            height = int(camera_model.height)
            poly_coeffs = camera_model._intrinsics[4:10].astype(np.float32)
            linear_cde = camera_model.linear_cde.astype(np.float32)

            # Get extrinsics
            extrinsics = camera_dict.get("nominalSensor2Rig_FLU", {})
            camera_to_vehicle = np.eye(4, dtype=np.float32)

            if extrinsics:
                if "t" in extrinsics:
                    camera_to_vehicle[:3, 3] = extrinsics["t"]
                if "roll-pitch-yaw" in extrinsics:
                    rpy = extrinsics["roll-pitch-yaw"]
                    rot = Rotation.from_euler("xyz", np.radians(rpy))
                    camera_to_vehicle[:3, :3] = rot.as_matrix()

            # Convert camera_to_vehicle from FLU to OpenCV RDF
            S = np.array(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            R_flu = camera_to_vehicle[:3, :3]
            t_flu = camera_to_vehicle[:3, 3]
            R_rdf = S @ R_flu @ S.T
            t_rdf = S @ t_flu
            camera_to_vehicle[:3, :3] = R_rdf
            camera_to_vehicle[:3, 3] = t_rdf

            # Use underscore canonical name for calibration key and record
            canon_name = camera_name.replace(":", "_")
            scene_data.camera_models[canon_name] = camera_model
            scene_data.camera_extrinsics[canon_name] = camera_to_vehicle

        logger.debug(f"Loaded calibrations for {len(scene_data.camera_models)} cameras")

    def _load_dynamic_objects(self, scene_data: SceneData, obstacle_file: Path) -> None:
        """Load dynamic object tracks."""
        df = pd.read_parquet(obstacle_file)

        # Group observations by track ID
        tracks = defaultdict(list)

        for _, row in df.iterrows():
            obstacle = row["obstacle"]
            key = row["key"]

            if "trackline_id" in obstacle and "timestamp_micros" in key:
                track_id = str(obstacle["trackline_id"])
                if track_id.startswith("labelstore:"):
                    # Skip duplicated labelstore entries (numeric counterpart exists)
                    continue
                timestamp = key["timestamp_micros"]

                tracks[track_id].append(
                    {
                        "timestamp": timestamp,
                        "obstacle": obstacle,
                    }
                )

        # Process each track
        for track_id, observations in tracks.items():
            if len(observations) < 2:
                continue  # Skip single-observation tracks

            # Sort by timestamp
            observations.sort(key=lambda x: x["timestamp"])

            # Extract trajectory data
            timestamps = []
            centers = []
            dimensions = []
            orientations = []

            for obs in observations:
                obstacle = obs["obstacle"]

                timestamps.append(obs["timestamp"])

                center = obstacle["center"]
                centers.append([center["x"], center["y"], center["z"]])

                size = obstacle["size"]
                dimensions.append([size["x"], size["y"], size["z"]])

                ori = obstacle["orientation"]
                orientations.append([ori["x"], ori["y"], ori["z"], ori["w"]])

            # Convert to numpy arrays
            timestamps = np.array(timestamps, dtype=np.int64)
            centers = np.array(centers, dtype=np.float32)
            dimensions = np.array(dimensions, dtype=np.float32)
            orientations = np.array(orientations, dtype=np.float32)

            normalized_quats, valid_mask = normalize_quaternions(orientations)
            if not np.all(valid_mask):
                logger.warning(
                    "Skipping %d zero-norm quaternions for track %s",
                    np.count_nonzero(~valid_mask),
                    track_id,
                )
            timestamps = timestamps[valid_mask]
            centers = centers[valid_mask]
            dimensions = dimensions[valid_mask]

            if (
                len(timestamps) < 2
                or np.isnan(centers).any()
                or np.isnan(dimensions).any()
                or np.isnan(orientations).any()
            ):
                continue

            centers = convert_points_flu_to_rdf(centers)
            orientations = convert_quaternions_flu_to_rdf(normalized_quats)

            # Map object type
            first_obs = observations[0]["obstacle"]
            category = first_obs.get("category", "unknown").lower()

            type_map = {
                "automobile": ObjectType.CAR,
                "other_vehicle": ObjectType.CAR,
                "vehicle": ObjectType.CAR,
                "car": ObjectType.CAR,
                "pedestrian": ObjectType.PEDESTRIAN,
                "person": ObjectType.PEDESTRIAN,
                "bicycle": ObjectType.CYCLIST,
                "cyclist": ObjectType.CYCLIST,
                "motorcycle": ObjectType.CYCLIST,
                "rider": ObjectType.CYCLIST,
                "bus": ObjectType.TRUCK,
                "truck": ObjectType.TRUCK,
                "heavy_truck": ObjectType.TRUCK,
                "train_or_tram_car": ObjectType.TRUCK,
                "trolley_bus": ObjectType.TRUCK,
                "trailer": ObjectType.TRUCK,
            }
            object_type = type_map.get(category, ObjectType.OTHER)

            # Create dynamic object
            scene_data.dynamic_objects[track_id] = DynamicObject(
                track_id=track_id,
                object_type=object_type,
                timestamps=timestamps,
                centers=centers,
                dimensions=dimensions,
                orientations=orientations,
                is_moving=True,
                max_extrapolation_us=500_000.0,
            )

        logger.debug(f"Loaded {len(scene_data.dynamic_objects)} dynamic object tracks")

    def _load_map_elements(self, scene_data: SceneData, files: Dict[str, Path]) -> None:
        """Load all map elements."""

        # Load lane boundaries
        if files["lane"].exists():
            self._load_lane_boundaries(scene_data, files["lane"])

        # Load lane lines
        if files["lane_line"].exists():
            self._load_lane_lines(scene_data, files["lane_line"])

        # Load road boundaries
        if files["road_boundary"].exists():
            self._load_road_boundaries(scene_data, files["road_boundary"])

        # Load crosswalks
        if files["crosswalk"].exists():
            self._load_crosswalks(scene_data, files["crosswalk"])

        # Load poles
        if files["pole"].exists():
            self._load_poles(scene_data, files["pole"])

        # Load road markings
        if files["road_marking"].exists():
            self._load_road_markings(scene_data, files["road_marking"])

        # Load wait lines
        if files["wait_line"].exists():
            self._load_wait_lines(scene_data, files["wait_line"])

        # Load traffic lights
        if files["traffic_light"].exists():
            self._load_traffic_lights(scene_data, files["traffic_light"])

        # Load traffic signs
        if files["traffic_sign"].exists():
            self._load_traffic_signs(scene_data, files["traffic_sign"])

        # Load intersection areas
        if files["intersection_area"].exists():
            self._load_intersection_areas(scene_data, files["intersection_area"])

        # Load road islands
        if files["road_island"].exists():
            self._load_road_islands(scene_data, files["road_island"])

    def _load_lane_boundaries(self, scene_data: SceneData, lane_file: Path) -> None:
        """Load lane boundaries."""
        df = pd.read_parquet(lane_file)

        for idx, row in df.iterrows():
            lane = row["lane"]
            lane_id = str(idx)

            # Process left rail
            if "left_rail" in lane and lane["left_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane["left_rail"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.lane_boundaries.append(
                        LaneBoundary(
                            element_id=f"{lane_id}_left",
                            points=points,
                            is_left_boundary=True,
                            lane_id=lane_id,
                        )
                    )

            # Process right rail
            if "right_rail" in lane and lane["right_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane["right_rail"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.lane_boundaries.append(
                        LaneBoundary(
                            element_id=f"{lane_id}_right",
                            points=points,
                            is_left_boundary=False,
                            lane_id=lane_id,
                        )
                    )

    def _load_lane_lines(self, scene_data: SceneData, lane_line_file: Path) -> None:
        """Load lane lines."""
        df = pd.read_parquet(lane_line_file)

        for idx, row in df.iterrows():
            lane_line = row["lane_line"]

            # Get points (check both new and old format)
            points = None
            if "line_rail" in lane_line and lane_line["line_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane_line["line_rail"]],
                    dtype=np.float32,
                )
            elif "path" in lane_line and lane_line["path"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane_line["path"]],
                    dtype=np.float32,
                )

            if points is None or len(points) < 2 or np.isnan(points).any():
                continue

            color = LaneLineColor.WHITE
            style = LaneLineStyle.SOLID_SINGLE

            if "colors" in lane_line and "styles" in lane_line:
                colors = lane_line["colors"]
                styles = lane_line["styles"]

                if len(colors) > 0 and len(styles) > 0:
                    # Find most common combination
                    combinations = list(zip(colors, styles, strict=False))
                    most_common = Counter(combinations).most_common(1)[0][0]
                    color_str, style_str = most_common

                    if color_str:
                        try:
                            color = LaneLineColor[color_str.upper()]
                        except KeyError:
                            color = LaneLineColor.UNKNOWN
                            logger.debug(f"Unknown lane line color: {color_str}, using UNKNOWN")
                    if style_str:
                        try:
                            style = LaneLineStyle[style_str.upper()]
                        except KeyError:
                            style = LaneLineStyle.UNKNOWN
                            logger.debug(f"Unknown lane line style: {style_str}, using UNKNOWN")

            points = convert_points_flu_to_rdf(points)
            scene_data.lane_lines.append(
                LaneLine(
                    element_id=f"lane_line_{idx}",
                    points=points,
                    lane_type=build_lane_line_type(color=color, style=style),
                )
            )

    def _load_road_boundaries(self, scene_data: SceneData, road_boundary_file: Path) -> None:
        """Load road boundaries."""
        df = pd.read_parquet(road_boundary_file)

        for idx, row in df.iterrows():
            boundary = row["road_boundary"]

            if "location" in boundary:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in boundary["location"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.road_boundaries.append(
                        RoadBoundary(
                            element_id=f"road_boundary_{idx}",
                            points=points,
                        )
                    )

    def _load_crosswalks(self, scene_data: SceneData, crosswalk_file: Path) -> None:
        """Load crosswalks."""
        df = pd.read_parquet(crosswalk_file)

        for idx, row in df.iterrows():
            crosswalk = row["crosswalk"]

            if "location" in crosswalk:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in crosswalk["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.crosswalks.append(
                        Crosswalk(
                            element_id=f"crosswalk_{idx}",
                            vertices=vertices,
                        )
                    )

    def _load_poles(self, scene_data: SceneData, pole_file: Path) -> None:
        """Load poles."""
        df = pd.read_parquet(pole_file)

        for idx, row in df.iterrows():
            pole = row["pole"]

            if "location" in pole:
                loc = pole["location"]
                if len(loc) >= 2:
                    # Use provided points
                    points = np.array(
                        [[pt["x"], pt["y"], pt["z"]] for pt in loc],
                        dtype=np.float32,
                    )
                    if np.isnan(points).any():
                        continue
                elif len(loc) == 1:
                    # Create vertical pole from single point
                    base = np.array([loc[0]["x"], loc[0]["y"], loc[0]["z"]], dtype=np.float32)
                    base_rdf = convert_points_flu_to_rdf(base.reshape(1, 3))[0]
                    scene_data.poles.append(Pole.from_base_point(f"pole_{idx}", base_rdf, height=3.0))
                    continue
                else:
                    continue

                points = convert_points_flu_to_rdf(points)
                scene_data.poles.append(Pole(element_id=f"pole_{idx}", points=points))

    def _load_road_markings(self, scene_data: SceneData, road_marking_file: Path) -> None:
        """Load road markings."""
        df = pd.read_parquet(road_marking_file)

        for idx, row in df.iterrows():
            marking = cast(Dict[str, Any], row["road_marking"])

            if "location" in marking:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in marking["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.road_markings.append(
                        RoadMarking(
                            element_id=f"road_marking_{idx}",
                            vertices=vertices,
                            marking_type=marking.get("type", "unknown"),
                        )
                    )

    def _load_wait_lines(self, scene_data: SceneData, wait_line_file: Path) -> None:
        """Load wait lines."""
        df = pd.read_parquet(wait_line_file)

        for idx, row in df.iterrows():
            wait_line = cast(Dict[str, Any], row["wait_line"])

            if "location" in wait_line:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in wait_line["location"]],
                    dtype=np.float32,
                )
                if len(points) >= 2 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.wait_lines.append(
                        WaitLine(
                            element_id=f"wait_line_{idx}",
                            points=points,
                        )
                    )

    def _load_traffic_lights(self, scene_data: SceneData, traffic_light_file: Path) -> None:
        """Load traffic lights, carrying each signal's per-frame color onto the light.

        The parquet may store either a single representative ``state`` per signal
        (one row, ``timestamp_micros`` null) or a per-frame series (one row per
        ``(label_class_id, timestamp_micros)``, mirroring the obstacle parquet).
        Rows are grouped by ``label_class_id`` -- the stable per-signal id -- so a
        signal that changes color mid-clip becomes a single light whose
        ``metadata["state_sequence"]`` steps through its colors, aligned onto the
        ego/render frame grid by holding the last observed state between updates.
        Frames outside a signal's witnessed span carry its nearest observed color
        only as far as corroborating evidence supports (see
        ``_resolve_traffic_light_lead_ins`` and
        ``_resolve_traffic_light_trailing_holds``); beyond that they stay UNKNOWN.
        A signal with no usable state stays UNKNOWN (gray). Orientation is
        optional: a missing or null quaternion falls back to identity instead of
        raising.

        Before lights are built, temporally disjoint tracks that are very
        likely re-acquisitions of one physical head (see
        ``_find_traffic_light_merge_clusters``) collapse into a single light
        with one continuous state timeline, so exactly one box renders per
        physical head; the unwitnessed seam between two merged fragments takes
        each side's nearest witnessed color (see
        ``_apply_traffic_light_merge_seam``).
        """
        df = pd.read_parquet(traffic_light_file)

        num_frames = max(1, scene_data.num_frames)
        frame_timestamps = scene_data.timestamps  # microseconds, one per frame
        ego_positions = (
            np.array([pose.position for pose in scene_data.ego_poses], dtype=np.float64)
            if scene_data.ego_poses
            else None
        )
        pending: List[_TrafficLightSequence] = []

        # Group rows by the stable per-signal id so multiple per-timestamp rows
        # collapse into one light. Fall back to the row index when the key is
        # absent (legacy single-row-per-signal parquets).
        grouped: Dict[str, List[Tuple[Optional[int], Dict[str, Any]]]] = defaultdict(list)
        for idx, row in df.iterrows():
            light = cast(Dict[str, Any], row["traffic_light"])
            key = row["key"] if "key" in row else None
            label_id = key.get("label_class_id") if isinstance(key, dict) else None
            timestamp = key.get("timestamp_micros") if isinstance(key, dict) else None
            group_id = str(label_id) if label_id is not None else f"row_{idx}"
            grouped[group_id].append((timestamp, light))

        # Geometry is static per signal: parse it once per group, from the
        # group's first observation.
        geometry: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}
        fragments: List[_TrafficLightTrackFragment] = []
        for group_id, observations in grouped.items():
            light = observations[0][1]

            center = np.array(
                [light["center"]["x"], light["center"]["y"], light["center"]["z"]],
                dtype=np.float32,
            )

            dimensions = np.array([0.6, 0.6, 1.0], dtype=np.float32)  # Default
            if "dimensions" in light:
                dims = light["dimensions"]
                if dims is not None and all(dims[k] is not None for k in ["x", "y", "z"]):
                    dimensions = np.array([dims["x"], dims["y"], dims["z"]], dtype=np.float32)

            # Orientation is optional: a missing or null quaternion (e.g. signals
            # from sources that record no facing) defaults to identity. Record
            # whether it was known so downstream consumers (e.g. the renderer's
            # facing cull) don't treat the identity placeholder as a real facing.
            orient = light["orientation"] if "orientation" in light else None
            orientation_known = orient is not None and all(orient[k] is not None for k in ("x", "y", "z", "w"))
            if orientation_known:
                orientation = np.array(
                    [orient["x"], orient["y"], orient["z"], orient["w"]],
                    dtype=np.float32,
                )
            else:
                orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if np.isnan(center).any() or np.isnan(dimensions).any() or np.isnan(orientation).any():
                continue

            center = convert_points_flu_to_rdf(center.reshape(1, 3))[0]
            orientation = convert_quaternions_flu_to_rdf(orientation.reshape(1, 4))[0]
            geometry[group_id] = (center, dimensions, orientation, orientation_known)

            # Same facing convention as the renderer's cull: local +x axis.
            facing = Rotation.from_quat(orientation).apply([1.0, 0.0, 0.0]) if orientation_known else None
            timed = sorted(
                (int(ts), obs.get("state"))
                for ts, obs in observations
                if ts is not None and isinstance(obs.get("state"), str) and obs.get("state").strip()
            )
            if timed:
                fragments.append(
                    _TrafficLightTrackFragment(
                        group_id=group_id,
                        t_first=timed[0][0],
                        t_last=timed[-1][0],
                        first_state=timed[0][1],
                        num_observations=len(timed),
                        center=center,
                        facing=facing,
                    )
                )

        clusters = _find_traffic_light_merge_clusters(fragments, frame_timestamps, ego_positions)
        # The best-witnessed fragment names the merged light and provides its
        # rendered geometry; the other fragments' rows only feed its timeline.
        primary_of: Dict[str, str] = {}
        cluster_of_primary: Dict[str, List[_TrafficLightTrackFragment]] = {}
        for cluster in clusters:
            primary = max(cluster, key=lambda f: (f.num_observations, f.t_last - f.t_first, f.group_id))
            cluster_of_primary[primary.group_id] = cluster
            for member in cluster:
                primary_of[member.group_id] = primary.group_id

        for group_id in grouped:
            if group_id not in geometry:
                continue
            if primary_of.get(group_id, group_id) != group_id:
                continue  # absorbed into another fragment's light
            cluster = cluster_of_primary.get(group_id, [])
            member_ids = [member.group_id for member in cluster] or [group_id]
            observations = [obs for member_id in member_ids for obs in grouped[member_id]]

            center, dimensions, orientation, orientation_known = geometry[group_id]
            traffic_light = TrafficLight(
                element_id=f"traffic_light_{group_id}",
                center=center,
                dimensions=dimensions,
                orientation=orientation,
            )
            traffic_light.metadata["orientation_known"] = orientation_known
            if len(member_ids) > 1:
                traffic_light.metadata["merged_track_ids"] = member_ids

            # Build the per-frame color sequence. Timestamped observations step
            # through colors aligned to the frame grid (hold-last); a single
            # untimed row broadcasts its state. An absent state stays gray.
            built = self._build_traffic_light_state_sequence(observations, frame_timestamps, num_frames)
            if built is not None:
                states, first_observed_frame, last_observed_end, last_observed_ts = built
                seam_start_ts = cluster[0].t_last if cluster else 0
                for member in cluster[1:]:
                    _apply_traffic_light_merge_seam(
                        states, frame_timestamps, seam_start_ts, member.t_first, member.first_state
                    )
                    seam_start_ts = max(seam_start_ts, member.t_last)
                # A merged light is only genuinely witnessed inside each member
                # fragment's own span; the seam frames between fragments hold
                # guesses that must not vouch for other lights' guesses.
                witnessed_intervals: Optional[List[Tuple[int, int]]] = None
                if len(cluster) > 1 and frame_timestamps is not None and len(frame_timestamps) > 0:
                    witnessed_intervals = []
                    for member in cluster:
                        lo = int(np.searchsorted(frame_timestamps, member.t_first, side="left"))
                        hi = int(np.searchsorted(frame_timestamps, member.t_last, side="right"))
                        if witnessed_intervals and lo <= witnessed_intervals[-1][1]:
                            prev_lo, prev_hi = witnessed_intervals[-1]
                            witnessed_intervals[-1] = (prev_lo, max(prev_hi, hi))
                        else:
                            witnessed_intervals.append((lo, hi))
                pending.append(
                    _TrafficLightSequence(
                        center=center,
                        states=states,
                        first_observed_frame=first_observed_frame,
                        last_observed_end=last_observed_end,
                        last_observed_ts=last_observed_ts,
                        witnessed_intervals=witnessed_intervals,
                    )
                )
                traffic_light.metadata["state_sequence"] = states

            scene_data.traffic_lights.append(traffic_light)

        _resolve_traffic_light_lead_ins(pending, frame_timestamps)
        _resolve_traffic_light_trailing_holds(pending, frame_timestamps)

    @staticmethod
    def _build_traffic_light_state_sequence(
        observations: List[Tuple[Optional[int], Dict[str, Any]]],
        frame_timestamps: np.ndarray,
        num_frames: int,
    ) -> Optional[Tuple[List[Optional[str]], int, int, Optional[int]]]:
        """Build a length-``num_frames`` list of traffic-light state strings.

        ``observations`` are ``(timestamp_micros, light)`` tuples for one signal.
        Observations carrying a usable state and a timestamp are sorted and mapped
        onto ``frame_timestamps`` by holding the most recent state at or before
        each frame; frames outside the observed span provisionally carry the
        nearest observed state (``_resolve_traffic_light_lead_ins`` and
        ``_resolve_traffic_light_trailing_holds`` decide how much of those
        guesses to keep). When no observation carries a timestamp, the single
        representative state is broadcast across all frames. Returns ``(states,
        first_observed_frame, last_observed_end, last_observed_ts)`` where frames
        before ``first_observed_frame`` or at/after ``last_observed_end`` are
        guesses rather than observations and ``last_observed_ts`` is the last
        observation's timestamp (``None`` for broadcast states), or ``None`` when
        the signal has no usable (non-empty string) state, leaving the light
        UNKNOWN (gray).
        """
        timed: List[Tuple[int, str]] = []
        untimed_state: Optional[str] = None
        for timestamp, light in observations:
            state = light.get("state") if isinstance(light, dict) else None
            if not (isinstance(state, str) and state.strip()):
                continue
            if timestamp is None:
                untimed_state = state
            else:
                timed.append((int(timestamp), state))

        if timed:
            timed.sort(key=lambda ts_state: ts_state[0])
            observed_ts = np.array([ts for ts, _ in timed], dtype=np.int64)
            observed_states = [state for _, state in timed]
            if frame_timestamps is None or len(frame_timestamps) == 0:
                # No frame grid to align to: hold the earliest observed state.
                return [observed_states[0]] * num_frames, 0, num_frames, None
            # Most recent observation at or before each frame; frames before the
            # first observation provisionally take the first state.
            indices = np.searchsorted(observed_ts, frame_timestamps, side="right") - 1
            indices = np.clip(indices, 0, len(observed_states) - 1)
            first_observed_frame = int(np.searchsorted(frame_timestamps, observed_ts[0], side="left"))
            last_observed_end = int(np.searchsorted(frame_timestamps, observed_ts[-1], side="right"))
            return [observed_states[int(i)] for i in indices], first_observed_frame, last_observed_end, int(observed_ts[-1])

        if untimed_state is not None:
            return [untimed_state] * num_frames, 0, num_frames, None

        return None

    def _load_traffic_signs(self, scene_data: SceneData, traffic_sign_file: Path) -> None:
        """Load traffic signs."""
        df = pd.read_parquet(traffic_sign_file)

        for idx, row in df.iterrows():
            sign = cast(Dict[str, Any], row["traffic_sign"])

            center = np.array(
                [sign["center"]["x"], sign["center"]["y"], sign["center"]["z"]],
                dtype=np.float32,
            )

            dimensions = np.array([0.8, 0.3, 0.8], dtype=np.float32)  # Default
            if "dimensions" in sign:
                dims = sign["dimensions"]
                if dims is not None and all(dims[k] is not None for k in ["x", "y", "z"]):
                    dimensions = np.array([dims["x"], dims["y"], dims["z"]], dtype=np.float32)

            orientation = np.array(
                [
                    sign["orientation"]["x"],
                    sign["orientation"]["y"],
                    sign["orientation"]["z"],
                    sign["orientation"]["w"],
                ],
                dtype=np.float32,
            )

            # Map sign type
            category = sign.get("category", "")
            if category is not None:
                category = category.upper()
            sign_type_map = {
                "STOP": TrafficSignType.STOP,
                "YIELD": TrafficSignType.YIELD,
                "SPEED_LIMIT": TrafficSignType.SPEED_LIMIT,
            }
            sign_type = sign_type_map.get(category, TrafficSignType.UNKNOWN)

            if np.isnan(center).any() or np.isnan(dimensions).any() or np.isnan(orientation).any():
                continue

            center = convert_points_flu_to_rdf(center.reshape(1, 3))[0]
            orientation = convert_quaternions_flu_to_rdf(orientation.reshape(1, 4))[0]

            scene_data.traffic_signs.append(
                TrafficSign(
                    element_id=f"traffic_sign_{idx}",
                    center=center,
                    dimensions=dimensions,
                    orientation=orientation,
                    sign_type=sign_type,
                )
            )

    def _load_intersection_areas(self, scene_data: SceneData, intersection_file: Path) -> None:
        """Load intersection areas."""
        df = pd.read_parquet(intersection_file)

        for idx, row in df.iterrows():
            area = cast(Dict[str, Any], row["intersection_area"])

            if "location" in area:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in area["location"]],
                    dtype=np.float32,
                )

                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.intersection_areas.append(
                        IntersectionArea(
                            element_id=f"intersection_{idx}",
                            vertices=vertices,
                        )
                    )

    def _load_road_islands(self, scene_data: SceneData, road_island_file: Path) -> None:
        """Load road islands."""
        df = pd.read_parquet(road_island_file)

        for idx, row in df.iterrows():
            island = row["road_island"]

            if "location" in island:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in island["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.road_islands.append(
                        RoadIsland(
                            element_id=f"road_island_{idx}",
                            vertices=vertices,
                        )
                    )
