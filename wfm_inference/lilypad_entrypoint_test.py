"""Tests for the WFM inference Lilypad entrypoint's RGB conditioning wiring."""

import fractions
import logging
from pathlib import Path

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


class _FakePlainClient:
    """Minimal boto3-style stub: lists a fixed key set and writes empty files."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def get_paginator(self, _name: str) -> "_FakePlainClient._Paginator":
        return _FakePlainClient._Paginator(self._keys)

    def download_file(self, _bucket: str, _key: str, dest: str) -> None:
        with open(dest, "wb") as f:
            f.write(b"")

    class _Paginator:
        def __init__(self, keys: list[str]) -> None:
            self._keys = keys

        def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
            yield {"Contents": [{"Key": k} for k in self._keys]}


def test_download_rgb_inputs_stages_relative_to_spec_dir(tmp_path) -> None:
    # Precondition: the spec file lives in a nested subdirectory, not at the
    # assets root — multiview resolves input_path against the spec's directory.
    spec_dir = tmp_path / "assets" / "nested"
    spec_dir.mkdir(parents=True)
    client = _FakePlainClient(["rgb/sds/seg/FRONT_CENTER.mp4"])

    # Under test.
    stem_to_relpath = lilypad_entrypoint._download_rgb_inputs(
        client, "bucket", "rgb/sds/seg/", spec_dir, _LOGGER,
    )

    # Postcondition: paths are relative to the spec dir and the file is staged
    # there, so inference finds it after chdir to the spec's directory.
    assert stem_to_relpath == {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}
    assert (spec_dir / "_rgb" / "FRONT_CENTER.mp4").is_file()


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


def test_inject_rgb_input_paths_clears_stale_input_path_when_no_rgb() -> None:
    # A camera whose bundled spec already carries an input_path but has no
    # matching staged RGB must have that stale path cleared, so the model fails
    # validation loudly rather than conditioning on a leftover bundled path.
    spec = {
        "rear": {
            "control_path": "controls/rear.mp4",
            "input_path": "stale/rear.mp4",
        }
    }

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


@pytest.mark.parametrize(
    "keys",
    [
        ["rgb/seg/FRONT_CENTER.png", "rgb/seg/FRONT_CENTER.mp4"],
        ["rgb/seg/FRONT_CENTER.mp4", "rgb/seg/FRONT_CENTER.png"],
    ],
)
def test_download_rgb_inputs_prefers_video_over_still_image(tmp_path, keys) -> None:
    # A stem carrying both a video and a still resolves to the video regardless
    # of listing order: the still is only a stand-in for a video.
    spec_dir = tmp_path / "assets"
    spec_dir.mkdir()

    stem_to_relpath = lilypad_entrypoint._download_rgb_inputs(
        _FakePlainClient(keys), "bucket", "rgb/seg/", spec_dir, _LOGGER,
    )

    assert stem_to_relpath == {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}


def test_require_conditioning_enabled_rejects_zero_conditional_frames() -> None:
    # Inference swaps the control videos in for the input videos when the global
    # count is 0, so a staged input would be silently ignored.
    with pytest.raises(ValueError, match="num_conditional_frames is 0"):
        lilypad_entrypoint._require_conditioning_enabled({"num_conditional_frames": 0})


@pytest.mark.parametrize("spec", [{}, {"num_conditional_frames": 1}])
def test_require_conditioning_enabled_accepts_nonzero_and_default(spec) -> None:
    # An absent key means the model's default of 1, which is conditioning-on.
    lilypad_entrypoint._require_conditioning_enabled(spec)


def test_require_rgb_for_active_cameras_names_the_camera_without_input() -> None:
    spec = {
        "front_wide": {"control_path": "c.mp4", "input_path": "_rgb/FRONT_CENTER.mp4"},
        "rear": {"control_path": "c.mp4"},
    }

    with pytest.raises(ValueError, match="rear"):
        lilypad_entrypoint._require_rgb_for_active_cameras(spec)


def test_require_rgb_for_active_cameras_ignores_inactive_views() -> None:
    # A view with no control_path is inactive and needs no conditioning input.
    lilypad_entrypoint._require_rgb_for_active_cameras(
        {"front_wide": {"num_conditional_frames_per_view": 1}}
    )


def _write_video(
    path: Path,
    num_frames: int,
    rate: "fractions.Fraction",
    size_hw: tuple[int, int] = (64, 96),
) -> None:
    """Encode a blank video, standing in for a control track.

    Deliberately independent of the entrypoint's own encoder so a test that
    compares generated output against a control is not comparing a function
    with itself.
    """
    import av
    import numpy as np

    height, width = size_hw
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(num_frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(blank, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_image(path: Path, size_hw: tuple[int, int] = (200, 300)) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray(np.full((*size_hw, 3), 128, dtype=np.uint8)).save(path)


@pytest.fixture
def av_and_decord():
    """Skip when the video libraries the entrypoint needs are absent."""
    pytest.importorskip("av")
    pytest.importorskip("decord")
    pytest.importorskip("PIL")


def test_expand_images_to_videos_matches_control_length_and_rate(
    tmp_path, av_and_decord,
) -> None:
    # Precondition: two active cameras whose controls differ in length. The
    # shorter one bounds the window inference reads, so it is what a generated
    # input has to cover.
    import decord

    spec_dir = tmp_path / "assets"
    (spec_dir / "_rgb").mkdir(parents=True)
    rate = fractions.Fraction(30, 1)
    _write_video(spec_dir / "front_wide_control.mp4", 71, rate)
    _write_video(spec_dir / "rear_control.mp4", 61, rate)
    spec = {
        "num_conditional_frames": 1,
        "front_wide": {"control_path": "front_wide_control.mp4"},
        "rear": {"control_path": "rear_control.mp4"},
    }
    _write_image(spec_dir / "_rgb" / "FRONT_CENTER.png")
    _write_image(spec_dir / "_rgb" / "REAR_CENTER.jpg")
    stem_to_relpath = {
        "FRONT_CENTER": "_rgb/FRONT_CENTER.png",
        "REAR_CENTER": "_rgb/REAR_CENTER.jpg",
    }

    # Under test.
    lilypad_entrypoint._expand_images_to_videos(spec, stem_to_relpath, spec_dir, _LOGGER)

    # Postcondition: both stems now point at videos as long as the shortest
    # control and reporting the control's exact frame rate, which inference
    # requires to be identical across every input and control file.
    assert stem_to_relpath == {
        "FRONT_CENTER": "_rgb/FRONT_CENTER.mp4",
        "REAR_CENTER": "_rgb/REAR_CENTER.mp4",
    }
    control_fps = decord.VideoReader(str(spec_dir / "rear_control.mp4")).get_avg_fps()
    for relpath in stem_to_relpath.values():
        reader = decord.VideoReader(str(spec_dir / relpath))
        assert len(reader) == 61
        assert reader.get_avg_fps() == control_fps


def test_expand_images_to_videos_resizes_to_the_control_frame_size(
    tmp_path, av_and_decord,
) -> None:
    # Inference records the input video's resolution as original_hw and restores
    # each view to it, so a generated input must carry the control's frame size
    # rather than the uploaded image's.
    import decord

    spec_dir = tmp_path / "assets"
    (spec_dir / "_rgb").mkdir(parents=True)
    _write_video(spec_dir / "control.mp4", 20, fractions.Fraction(10, 1), size_hw=(64, 96))
    spec = {"front_wide": {"control_path": "control.mp4"}}
    _write_image(spec_dir / "_rgb" / "FRONT_CENTER.png", size_hw=(200, 300))
    stem_to_relpath = {"FRONT_CENTER": "_rgb/FRONT_CENTER.png"}

    lilypad_entrypoint._expand_images_to_videos(spec, stem_to_relpath, spec_dir, _LOGGER)

    frame = decord.VideoReader(str(spec_dir / "_rgb" / "FRONT_CENTER.mp4"))[0].asnumpy()
    assert frame.shape[:2] == (64, 96)


def test_expand_images_to_videos_leaves_staged_videos_untouched(
    tmp_path, av_and_decord,
) -> None:
    # A camera staged with a real RGB video must not be re-encoded, so its full
    # frame content survives instead of collapsing to a still.
    spec_dir = tmp_path / "assets"
    (spec_dir / "_rgb").mkdir(parents=True)
    _write_video(spec_dir / "control.mp4", 20, fractions.Fraction(10, 1))
    staged = spec_dir / "_rgb" / "FRONT_CENTER.mp4"
    _write_video(staged, 20, fractions.Fraction(10, 1))
    before = staged.read_bytes()
    spec = {"front_wide": {"control_path": "control.mp4"}}
    stem_to_relpath = {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}

    lilypad_entrypoint._expand_images_to_videos(spec, stem_to_relpath, spec_dir, _LOGGER)

    assert stem_to_relpath == {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}
    assert staged.read_bytes() == before


def test_expand_images_to_videos_rejects_controls_that_disagree_on_rate(
    tmp_path, av_and_decord,
) -> None:
    # Inference rejects a sample whose files report more than one frame rate, so
    # there is no single rate to generate at; the error names the cameras.
    spec_dir = tmp_path / "assets"
    (spec_dir / "_rgb").mkdir(parents=True)
    _write_video(spec_dir / "front.mp4", 20, fractions.Fraction(30, 1))
    _write_video(spec_dir / "rear.mp4", 20, fractions.Fraction(10, 1))
    spec = {
        "front_wide": {"control_path": "front.mp4"},
        "rear": {"control_path": "rear.mp4"},
    }
    _write_image(spec_dir / "_rgb" / "FRONT_CENTER.png")

    with pytest.raises(ValueError, match="front_wide=30.*rear=10"):
        lilypad_entrypoint._expand_images_to_videos(
            spec, {"FRONT_CENTER": "_rgb/FRONT_CENTER.png"}, spec_dir, _LOGGER,
        )


def test_expand_images_to_videos_is_a_no_op_without_images(tmp_path) -> None:
    # No staged image means no control probing at all, so a spec with no
    # readable control videos still passes through untouched.
    stem_to_relpath = {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}

    lilypad_entrypoint._expand_images_to_videos(
        {"front_wide": {"control_path": "missing.mp4"}},
        stem_to_relpath,
        tmp_path,
        _LOGGER,
    )

    assert stem_to_relpath == {"FRONT_CENTER": "_rgb/FRONT_CENTER.mp4"}
