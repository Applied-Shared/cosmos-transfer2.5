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
