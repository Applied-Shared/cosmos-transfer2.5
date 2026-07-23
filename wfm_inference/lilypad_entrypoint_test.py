"""Tests for the WFM inference Lilypad entrypoint's RGB conditioning wiring."""

import logging

import pytest

# The entrypoint module imports ray, boto3, and the lilypad SDK at import time;
# skip the whole module when those runtime-only deps are unavailable (e.g. a
# plain checkout outside the inference Docker image).
pytest.importorskip("ray")
pytest.importorskip("boto3")
pytest.importorskip("lilypad.public.sdk_py.cached_file_access.boto")

from wfm_inference import lilypad_entrypoint  # noqa: E402

_LOGGER = logging.getLogger(__name__)


def test_short_to_raw_camera_stems_covers_all_multiview_cameras() -> None:
    # The RGB mapping must cover exactly the seven multiview camera keys the
    # model accepts, so no active view is silently left without an RGB lookup.
    from cosmos_transfer2.multiview_config import MULTIVIEW_CAMERA_KEYS

    assert set(lilypad_entrypoint._SHORT_TO_RAW_CAMERA_STEMS) == set(MULTIVIEW_CAMERA_KEYS)


def test_inject_rgb_input_paths_sets_input_path_for_active_cameras() -> None:
    # Precondition: front_wide has a control_path and a matching raw RGB stem.
    spec = {
        "name": "multiview",
        "front_wide": {"control_path": "controls/front_wide.mp4"},
    }
    stem_to_relpath = {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}

    # Under test.
    lilypad_entrypoint._inject_rgb_input_paths(spec, stem_to_relpath, _LOGGER)

    # Postcondition: input_path points at the staged RGB file.
    assert spec["front_wide"]["input_path"] == "_rgb/FRONT_CENTER.mp4"


def test_inject_rgb_input_paths_prefers_raw_name_over_short_alias() -> None:
    # When both the raw sensor name and the short-name alias are present, the
    # raw sensor name (the canonical RGB upload name) must win.
    spec = {"cross_left": {"control_path": "controls/cross_left.mp4"}}
    stem_to_relpath = {
        "cross_left": "_rgb/cross_left.mp4",
        "FRONT_LEFT": "_rgb/FRONT_LEFT.mp4",
    }

    lilypad_entrypoint._inject_rgb_input_paths(spec, stem_to_relpath, _LOGGER)

    assert spec["cross_left"]["input_path"] == "_rgb/FRONT_LEFT.mp4"


def test_inject_rgb_input_paths_leaves_camera_control_only_when_no_rgb() -> None:
    # A camera with a control_path but no matching RGB file must be left as-is,
    # not given a bogus input_path.
    spec = {"rear": {"control_path": "controls/rear.mp4"}}

    lilypad_entrypoint._inject_rgb_input_paths(spec, {}, _LOGGER)

    assert "input_path" not in spec["rear"]


def test_inject_rgb_input_paths_ignores_cameras_without_control_path() -> None:
    # Non-camera keys and cameras without a control_path (inactive views) must
    # never receive an input_path.
    spec = {
        "name": "multiview",
        "front_tele": {"num_conditional_frames_per_view": 1},
    }
    stem_to_relpath = {"FRONT_CENTER_NARROW": "_rgb/FRONT_CENTER_NARROW.mp4"}

    lilypad_entrypoint._inject_rgb_input_paths(spec, stem_to_relpath, _LOGGER)

    assert "input_path" not in spec["front_tele"]
    assert isinstance(spec["name"], str)
