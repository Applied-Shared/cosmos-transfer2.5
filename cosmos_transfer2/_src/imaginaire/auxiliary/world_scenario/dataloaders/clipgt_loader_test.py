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


def _load(monkeypatch, scene, rows, keys=None) -> None:
    """Run _load_traffic_lights against an in-memory parquet of ``rows``."""
    data: dict = {"traffic_light": rows}
    if keys is not None:
        data["key"] = keys
    df = pd.DataFrame(data)
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


def test_should_group_timestamped_rows_by_label_class_id(monkeypatch):
    # Precondition. SDS parquets emit one row per timestamp per signal.
    scene = _scene_with_frames(4)
    for frame_idx, timestamp_us in enumerate([0, 1_000_000, 2_000_000, 3_000_000]):
        scene.ego_poses[frame_idx].timestamp = timestamp_us
    rows = [
        {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"},
        {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"},
        {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"},
    ]
    keys = [
        {"label_class_id": "tl-1", "timestamp_micros": 0},
        {"label_class_id": "tl-1", "timestamp_micros": 2_000_000},
        {"label_class_id": "tl-1", "timestamp_micros": 3_000_000},
    ]

    # Under test.
    _load(monkeypatch, scene, rows, keys=keys)

    # Postcondition.
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["feature_id"] == "tl-1"
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED", "RED", "GREEN", "GREEN"]


def test_should_hold_state_at_ego_timestamps_for_sds_rows(monkeypatch):
    # Precondition. Ego timestamps are microseconds; state changes mid-clip.
    scene = _scene_with_frames(5)
    for frame_idx, timestamp_us in enumerate([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000]):
        scene.ego_poses[frame_idx].timestamp = timestamp_us
    rows = [
        {"center": _CENTER, "orientation": _IDENTITY, "state": "RED"},
        {"center": _CENTER, "orientation": _IDENTITY, "state": "GREEN"},
    ]
    keys = [
        {"timestamp_micros": 0},
        {"timestamp_micros": 3_000_000},
    ]

    # Under test.
    _load(monkeypatch, scene, rows, keys=keys)

    # Postcondition. RED through 2s, GREEN from 4s onward (hold-until-next).
    assert len(scene.traffic_lights) == 1
    assert scene.traffic_lights[0].metadata["state_sequence"] == [
        "RED",
        "RED",
        "RED",
        "GREEN",
        "GREEN",
    ]


def test_should_emit_one_light_per_sim_bag_row_without_timestamps(monkeypatch):
    # Precondition. Sim-bag writes one row per signal with no timestamp_micros.
    scene = _scene_with_frames(3)
    rows = [
        {"center": {"x": 1.0, "y": 2.0, "z": 3.0}, "orientation": _IDENTITY, "state": "RED"},
        {"center": {"x": 5.0, "y": 6.0, "z": 7.0}, "orientation": _IDENTITY, "state": "GREEN"},
    ]
    keys = [{"label_class_id": "sig-a"}, {"label_class_id": "sig-b"}]

    # Under test.
    _load(monkeypatch, scene, rows, keys=keys)

    # Postcondition.
    assert len(scene.traffic_lights) == 2
    assert scene.traffic_lights[0].metadata["state_sequence"] == ["RED"] * 3
    assert scene.traffic_lights[1].metadata["state_sequence"] == ["GREEN"] * 3
