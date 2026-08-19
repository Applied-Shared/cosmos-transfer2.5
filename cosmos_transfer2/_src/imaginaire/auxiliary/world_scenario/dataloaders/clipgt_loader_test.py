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
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders import clipgt_loader

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


def _light_row(label_id, timestamp, state, center=None):
    light = {"center": center or _CENTER, "orientation": _IDENTITY, "state": state}
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
