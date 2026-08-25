# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Tests for ``ClipGTLoader._load_traffic_lights`` state + orientation handling."""

from pathlib import Path

import numpy as np
import pandas as pd

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario import data_types
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders import clipgt_loader, data_utils

_CENTER = {"x": 1.0, "y": 2.0, "z": 3.0}
_IDENTITY = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _scene_with_frames(num_frames: int) -> "data_types.SceneData":
    """A SceneData whose only relevant property is num_frames (== len(ego_poses))."""
    scene = data_types.SceneData(scene_id="test_clip", duration_seconds=0.0)
    scene.ego_poses = [
        data_types.EgoPose(
            timestamp=i,
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for i in range(num_frames)
    ]
    return scene


def _load(monkeypatch, scene, rows) -> None:
    """Run _load_traffic_lights against an in-memory parquet of ``rows``."""
    df = pd.DataFrame({"traffic_light": rows})
    monkeypatch.setattr(clipgt_loader.pd, "read_parquet", lambda _path: df)
    clipgt_loader.ClipGTLoader()._load_traffic_lights(scene, Path("unused.traffic_light.parquet"))


def test_should_broadcast_state_across_all_frames_when_state_present(monkeypatch):
    # Precondition.
    scene = _scene_with_frames(4)

    # Under test.
    _load(monkeypatch, scene, [{"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}])

    # Postcondition.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 4


def test_should_omit_state_sequence_when_state_absent(monkeypatch):
    # Precondition.
    scene = _scene_with_frames(3)

    # Under test.
    _load(monkeypatch, scene, [{"center": _CENTER, "orientation": _IDENTITY}])

    # Postcondition.
    assert "state_sequence" not in scene.traffic_lights[0].metadata


def test_should_omit_state_sequence_when_state_empty(monkeypatch):
    # Precondition.
    scene = _scene_with_frames(3)

    # Under test.
    _load(monkeypatch, scene, [{"center": _CENTER, "orientation": _IDENTITY, "state": ""}])

    # Postcondition.
    assert "state_sequence" not in scene.traffic_lights[0].metadata


def test_should_default_to_identity_when_orientation_null(monkeypatch):
    # Precondition.
    scene = _scene_with_frames(2)

    # Under test.
    _load(monkeypatch, scene, [{"center": _CENTER, "orientation": None, "state": "GREEN"}])

    # Postcondition.
    assert len(scene.traffic_lights) == 1
    assert np.isfinite(scene.traffic_lights[0].orientation).all()


def test_should_default_to_identity_when_orientation_component_null(monkeypatch):
    # Precondition.
    scene = _scene_with_frames(2)
    rows = [{"center": _CENTER, "orientation": {"x": None, "y": 0.0, "z": 0.0, "w": 1.0}, "state": "RED"}]

    # Under test.
    _load(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 1
    assert np.isfinite(scene.traffic_lights[0].orientation).all()


def test_should_default_dimensions_when_dimensions_null(monkeypatch):
    # Precondition. The parquet always carries a "dimensions" column, but
    # sim-bag-sourced lights leave its value null; the key is present, value None.
    scene = _scene_with_frames(2)
    rows = [{"center": _CENTER, "orientation": _IDENTITY, "state": "RED", "dimensions": None}]

    # Under test.
    _load(monkeypatch, scene, rows)

    # Postcondition. Falls back to the default box (no crash) and still colors the light.
    assert len(scene.traffic_lights) == 1
    np.testing.assert_array_equal(scene.traffic_lights[0].dimensions, [0.6, 0.6, 1.0])
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 2


def _scene_with_frame_timestamps(frame_timestamps) -> "data_types.SceneData":
    """A SceneData whose ego-pose timestamps define the render frame grid."""
    scene = data_types.SceneData(scene_id="test_clip", duration_seconds=0.0)
    scene.ego_poses = [
        data_types.EgoPose(
            timestamp=int(ts),
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for ts in frame_timestamps
    ]
    return scene


def _load_keyed(monkeypatch, scene, rows) -> None:
    """Run _load_traffic_lights against a parquet of ``rows`` (traffic_light + key)."""
    df = pd.DataFrame(
        {
            "traffic_light": [r["traffic_light"] for r in rows],
            "key": [r["key"] for r in rows],
        }
    )
    monkeypatch.setattr(clipgt_loader.pd, "read_parquet", lambda _path: df)
    clipgt_loader.ClipGTLoader()._load_traffic_lights(scene, Path("unused.traffic_light.parquet"))


def _key(label_id, timestamp):
    return {"label_class_id": label_id, "timestamp_micros": timestamp}


def test_should_step_state_per_frame_when_observations_timestamped(monkeypatch):
    # Precondition. One signal, RED at t=0 then GREEN at t=20; frames at 0,10,20,30.
    scene = _scene_with_frame_timestamps([0, 10, 20, 30])
    rows = [
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}, "key": _key("7", 0)},
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"}, "key": _key("7", 20)},
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Hold-last: RED until the GREEN observation at t=20.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED", "RED", "GREEN", "GREEN"]


def test_should_collapse_rows_with_same_label_class_id_into_one_light(monkeypatch):
    # Precondition. Two timestamped rows for the same signal.
    scene = _scene_with_frame_timestamps([0, 10])
    rows = [
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}, "key": _key("7", 0)},
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"}, "key": _key("7", 10)},
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. One physical light, not two.
    assert len(scene.traffic_lights) == 1


def test_should_emit_separate_lights_per_label_class_id(monkeypatch):
    # Precondition. Two distinct signals.
    scene = _scene_with_frame_timestamps([0, 10])
    rows = [
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}, "key": _key("7", 0)},
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"}, "key": _key("8", 0)},
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 2


def test_should_hold_first_state_for_frames_before_first_observation(monkeypatch):
    # Precondition. First observation is at t=15, after frames at 0 and 10.
    scene = _scene_with_frame_timestamps([0, 10, 20])
    rows = [
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"}, "key": _key("7", 15)},
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}, "key": _key("7", 18)},
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Leading frames clamp to the earliest observation.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["GREEN", "GREEN", "RED"]


def test_should_broadcast_when_label_present_but_timestamp_null(monkeypatch):
    # Precondition. Legacy/static shape: one row per signal, null timestamp.
    scene = _scene_with_frame_timestamps([0, 10, 20])
    rows = [
        {"traffic_light": {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"}, "key": _key("7", None)},
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Single representative state broadcast across all frames.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 3


def _light_row(label_id, timestamp, state, center=None, orientation=None):
    light = {"center": center or _CENTER, "orientation": orientation or _IDENTITY, "state": state}
    return {"traffic_light": light, "key": _key(label_id, timestamp)}


def test_should_keep_lead_in_guess_when_gap_is_short(monkeypatch):
    # Precondition. First observation 1.5s in -- well under the backfill window.
    scene = _scene_with_frame_timestamps([0, 1_000_000, 2_000_000])

    # Under test.
    _load_keyed(monkeypatch, scene, [_light_row("7", 1_500_000, "GREEN")])

    # Postcondition. The short lead-in carries the first observed color.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["GREEN"] * 3


def test_should_mark_lead_in_unknown_when_gap_exceeds_backfill_window(monkeypatch):
    # Precondition. First observation 9s in; no other signal nearby.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])

    # Under test.
    _load_keyed(monkeypatch, scene, [_light_row("7", 9_000_000, "RED")])

    # Postcondition. Only the 3s window before the observation keeps the guess.
    assert scene.traffic_lights[0].metadata["state_sequence"] == [None] * 6 + ["RED"] * 4


def test_should_keep_full_lead_in_when_colocated_signal_corroborates(monkeypatch):
    # Precondition. A co-located signal (0.5m away) observed the same color from t=0.
    scene = _scene_with_frame_timestamps([i * 2_000_000 for i in range(11)])
    near = {"x": 1.0, "y": 2.5, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 20_000_000, "RED", center=near)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Corroborated: the 20s lead-in keeps the guess end to end.
    late = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_8")
    assert late.metadata["state_sequence"] == ["RED"] * 11


def test_should_mark_lead_in_unknown_when_colocated_signal_contradicts(monkeypatch):
    # Precondition. A co-located signal observed a DIFFERENT color during the gap.
    scene = _scene_with_frame_timestamps([i * 2_000_000 for i in range(11)])
    near = {"x": 1.0, "y": 2.5, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 20_000_000, "GREEN", center=near)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. The contradicted guess is dropped; only the observation stays.
    late = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_8")
    assert late.metadata["state_sequence"] == [None] * 10 + ["GREEN"]


def test_should_not_use_far_signal_as_lead_in_evidence(monkeypatch):
    # Precondition. A conflicting signal exists but 100m away -- a different head.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    far = {"x": 101.0, "y": 2.0, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 9_000_000, "GREEN", center=far)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. No contradiction applied; the plain backfill window governs.
    late = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_8")
    assert late.metadata["state_sequence"] == [None] * 6 + ["GREEN"] * 4


def test_should_decay_trailing_hold_when_silence_exceeds_hold_window(monkeypatch):
    # Precondition. A signal observed only at t=0, then silent for 9s.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])

    # Under test.
    _load_keyed(monkeypatch, scene, [_light_row("7", 0, "RED")])

    # Postcondition. The color survives the 3s hold window, then goes UNKNOWN.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 4 + [None] * 6


def test_should_keep_trailing_hold_when_silence_within_window(monkeypatch):
    # Precondition. Last observation 2s before clip end -- inside the hold window.
    scene = _scene_with_frame_timestamps([0, 1_000_000, 2_000_000])

    # Under test.
    _load_keyed(monkeypatch, scene, [_light_row("7", 0, "RED")])

    # Postcondition. No decay near clip end.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 3


def test_should_hold_state_through_internal_gap_when_track_reobserved(monkeypatch):
    # Precondition. An 8s observation gap WITHIN the track (occlusion), not after it.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED"), _light_row("7", 8_000_000, "RED")]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Interior frames stay held; no flicker to UNKNOWN.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 10


def test_should_extend_trailing_hold_when_colocated_signal_corroborates(monkeypatch):
    # Precondition. A co-located signal witnesses the same color 20s later.
    scene = _scene_with_frame_timestamps([i * 2_000_000 for i in range(11)])
    near = {"x": 1.0, "y": 2.5, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 20_000_000, "RED", center=near)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Corroborated: the 20s trailing hold keeps the color end to end.
    early = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_7")
    assert early.metadata["state_sequence"] == ["RED"] * 11


def test_should_cut_trailing_hold_when_colocated_signal_contradicts(monkeypatch):
    # Precondition. A co-located signal witnesses a DIFFERENT color 20s later.
    scene = _scene_with_frame_timestamps([i * 2_000_000 for i in range(11)])
    near = {"x": 1.0, "y": 2.5, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 20_000_000, "GREEN", center=near)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. The hold window keeps 2s of RED; the contradicted rest is UNKNOWN.
    early = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_7")
    assert early.metadata["state_sequence"] == ["RED"] * 2 + [None] * 9


def test_should_not_use_far_signal_as_trailing_evidence(monkeypatch):
    # Precondition. A same-color signal exists 100m away -- a different head.
    scene = _scene_with_frame_timestamps([i * 2_000_000 for i in range(11)])
    far = {"x": 101.0, "y": 2.0, "z": 3.0}
    rows = [_light_row("7", 0, "RED"), _light_row("8", 20_000_000, "RED", center=far)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. No corroboration applied; the plain hold window governs.
    early = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_7")
    assert early.metadata["state_sequence"] == ["RED"] * 2 + [None] * 9


# Merge fixtures: ego sits at the origin, so a fragment at 50m forward and a
# re-acquisition ~8m deeper along almost the same bearing model the labeler's
# depth drift on a small distant head.
_NEAR = {"x": 50.0, "y": 0.0, "z": 0.0}
_DEEP_SAME_RAY = {"x": 58.0, "y": 0.5, "z": 0.6}  # 8m away in 3D, <1 deg off the ray
_NEAR_OFF_RAY = {"x": 50.0, "y": 8.0, "z": 0.0}  # 8m away in 3D, ~9 deg off the ray
_NEAR_COLOCATED = {"x": 50.0, "y": 0.5, "z": 0.6}  # within the co-location radius
_YAW_90 = {"x": 0.0, "y": 0.0, "z": 0.70710678, "w": 0.70710678}


def test_should_merge_disjoint_tracks_when_on_shared_viewing_ray(monkeypatch):
    # Precondition. Track 7 ends at t=1s; track 8 re-acquires at t=5s, 8m deeper
    # in 3D but under 1 degree off the same viewing ray from the ego.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 1_000_000, "RED", center=_NEAR)]
    rows += [_light_row("8", (5 + i) * 1_000_000, "GREEN", center=_DEEP_SAME_RAY) for i in range(5)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. One light (named for the better-witnessed fragment), not two.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].element_id == "traffic_light_8"
    assert scene.traffic_lights[0].metadata["merged_track_ids"] == ["7", "8"]


def test_should_split_merged_seam_colors_at_gap_midpoint(monkeypatch):
    # Precondition. Merged fragments RED-until-1s and GREEN-from-7s: the 6s seam
    # is unwitnessed, and the midpoint (t=4s) is where the nearest witness flips.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 1_000_000, "RED", center=_NEAR)]
    rows += [_light_row("8", (7 + i) * 1_000_000, "GREEN", center=_DEEP_SAME_RAY) for i in range(3)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Each seam frame carries its temporally nearest witness's color.
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 5 + ["GREEN"] * 5


def test_should_not_merge_tracks_when_temporally_overlapping(monkeypatch):
    # Precondition. Both tracks are witnessed simultaneously (overlap 5-6s): two
    # genuinely distinct heads, even though they sit on almost the same ray.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 6_000_000, "RED", center=_NEAR)]
    rows += [_light_row("8", (5 + i) * 1_000_000, "GREEN", center=_DEEP_SAME_RAY) for i in range(5)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 2


def test_should_not_merge_disjoint_tracks_when_off_shared_viewing_ray(monkeypatch):
    # Precondition. Same 8m separation and timing as the merge case, but lateral:
    # ~9 degrees off the ray, the signature of a different head, not depth noise.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 1_000_000, "RED", center=_NEAR)]
    rows += [_light_row("8", (5 + i) * 1_000_000, "GREEN", center=_NEAR_OFF_RAY) for i in range(5)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 2


def test_should_not_merge_disjoint_tracks_when_facings_disagree(monkeypatch):
    # Precondition. Co-located and disjoint, but facing 90 degrees apart -- e.g.
    # heads for two approaches sharing one pole corner.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 1_000_000, "RED", center=_NEAR)]
    rows += [
        _light_row("8", (5 + i) * 1_000_000, "GREEN", center=_NEAR_COLOCATED, orientation=_YAW_90)
        for i in range(5)
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 2


def test_should_not_merge_disjoint_tracks_when_gap_exceeds_merge_window(monkeypatch):
    # Precondition. Co-located, same facing, but silent for 7s between tracks.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR)]
    rows += [_light_row("8", (7 + i) * 1_000_000, "RED", center=_NEAR_COLOCATED) for i in range(3)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition.
    assert len(scene.traffic_lights) == 2


def test_should_ignore_merge_seam_guess_as_evidence_for_colocated_light(monkeypatch):
    # Precondition. Tracks 7+8 merge into one RED light whose seam (frames 2-6)
    # holds unwitnessed guesses; light 9 shares the mast (co-located, facing 90
    # degrees away so it can't merge itself) and is GREEN through t=1s.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(12)])
    rows = [_light_row("7", t * 1_000_000, "RED", center=_NEAR) for t in (0, 1)]
    rows += [_light_row("8", t * 1_000_000, "RED", center=_NEAR_COLOCATED) for t in (7, 8)]
    third_center = {"x": 50.0, "y": 1.0, "z": 0.6}
    rows += [_light_row("9", t * 1_000_000, "GREEN", center=third_center, orientation=_YAW_90) for t in (0, 1)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Light 9's trailing hold decays by its own 3s window; the
    # seam's RED guess is not witnessed evidence, so it must not cut the hold.
    assert len(scene.traffic_lights) == 2
    third = next(li for li in scene.traffic_lights if li.element_id == "traffic_light_9")
    assert third.metadata["state_sequence"] == ["GREEN"] * 5 + [None] * 7


def test_should_merge_colocated_disjoint_tracks_when_gap_within_window(monkeypatch):
    # Precondition. Fragments within the co-location radius, 3s apart: the 3D
    # tier merges them without needing the viewing-ray test.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", 0, "RED", center=_NEAR), _light_row("7", 1_000_000, "RED", center=_NEAR)]
    rows += [_light_row("8", (4 + i) * 1_000_000, "RED", center=_NEAR_COLOCATED) for i in range(6)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. One light, one uninterrupted color.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 10


def test_should_carry_per_frame_pose_when_rows_are_timestamped(monkeypatch):
    # Precondition. One signal whose box moves in the parquet between frames --
    # in real data this is the drifting ego pose that placed it, not the head.
    scene = _scene_with_frame_timestamps([0, 10, 20])
    rows = [
        _light_row("7", 0, "RED", center={"x": 50.0, "y": 0.0, "z": 3.0}),
        _light_row("7", 20, "RED", center={"x": 51.0, "y": 0.0, "z": 3.0}),
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Hold-last, one pose per frame: the t=0 box until t=20.
    light = scene.traffic_lights[0]
    expected = data_utils.convert_points_flu_to_rdf(
        np.array([[50.0, 0.0, 3.0], [50.0, 0.0, 3.0], [51.0, 0.0, 3.0]], dtype=np.float32)
    )
    assert light.num_pose_frames == 3
    np.testing.assert_allclose(light.centers, expected)


def test_should_hold_first_pose_for_frames_before_first_observation(monkeypatch):
    # Precondition. First observation at t=15, after frames at 0 and 10.
    scene = _scene_with_frame_timestamps([0, 10, 20])
    rows = [
        _light_row("7", 15, "RED", center={"x": 50.0, "y": 0.0, "z": 3.0}),
        _light_row("7", 18, "RED", center={"x": 52.0, "y": 0.0, "z": 3.0}),
    ]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Leading frames clamp to the earliest box, as state does.
    expected = data_utils.convert_points_flu_to_rdf(
        np.array([[50.0, 0.0, 3.0], [50.0, 0.0, 3.0], [52.0, 0.0, 3.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(scene.traffic_lights[0].centers, expected)


def test_should_leave_pose_static_when_rows_carry_no_timestamp(monkeypatch):
    # Precondition. Legacy shape: one untimed row per signal, nothing to place
    # per frame.
    scene = _scene_with_frame_timestamps([0, 10, 20])

    # Under test.
    _load_keyed(monkeypatch, scene, [_light_row("7", None, "RED")])

    # Postcondition. The renderer falls back to the single static box.
    light = scene.traffic_lights[0]
    assert light.num_pose_frames == 0
    assert light.centers is None


def test_should_take_each_frames_pose_from_the_member_that_saw_it(monkeypatch):
    # Precondition. Two co-located fragments of one head, 3s apart, each
    # reporting the box from its own frames. They merge into one light.
    scene = _scene_with_frame_timestamps([i * 1_000_000 for i in range(10)])
    rows = [_light_row("7", i * 1_000_000, "RED", center=_NEAR) for i in range(2)]
    rows += [_light_row("8", (4 + i) * 1_000_000, "RED", center=_NEAR_COLOCATED) for i in range(6)]

    # Under test.
    _load_keyed(monkeypatch, scene, rows)

    # Postcondition. Frames 0-3 hold fragment 7's box (frames 2-3 across the
    # unwitnessed seam); frames 4-9 take fragment 8's.
    assert len(scene.traffic_lights) == 1
    expected = data_utils.convert_points_flu_to_rdf(
        np.array(
            [[_NEAR["x"], _NEAR["y"], _NEAR["z"]]] * 4
            + [[_NEAR_COLOCATED["x"], _NEAR_COLOCATED["y"], _NEAR_COLOCATED["z"]]] * 6,
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(scene.traffic_lights[0].centers, expected)
