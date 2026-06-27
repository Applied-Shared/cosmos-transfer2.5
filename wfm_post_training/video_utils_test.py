"""Tests for WFM post-training video helpers."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest import mock

from wfm_post_training.video_utils import (
    normalize_training_video_pair,
    parse_frame_rate,
)


class ParseFrameRateTest(unittest.TestCase):
    def test_should_parse_integer_rate(self) -> None:
        self.assertEqual(10.0, parse_frame_rate("10/1"))

    def test_should_parse_fractional_rate(self) -> None:
        self.assertAlmostEqual(9.99, parse_frame_rate("2997/300"), places=2)


class NormalizeTrainingVideoPairTest(unittest.TestCase):
    @mock.patch("wfm_post_training.video_utils.get_decord_fps", side_effect=[10.0, 10.0])
    @mock.patch("wfm_post_training.video_utils._reencode_constant_fps")
    def test_should_skip_when_decord_fps_already_matches(
        self,
        mock_reencode: mock.Mock,
        _mock_decord_fps: mock.Mock,
    ) -> None:
        rgb_path = Path("/tmp/rgb.mp4")
        control_path = Path("/tmp/control.mp4")

        normalize_training_video_pair(rgb_path, control_path, logging.getLogger("test"))

        mock_reencode.assert_not_called()

    @mock.patch("wfm_post_training.video_utils.get_decord_fps", side_effect=[9.979, 10.0, 10.0, 10.0])
    @mock.patch("wfm_post_training.video_utils._reencode_constant_fps")
    def test_should_reencode_both_when_decord_fps_differs(
        self,
        mock_reencode: mock.Mock,
        _mock_decord_fps: mock.Mock,
    ) -> None:
        rgb_path = Path("/tmp/rgb.mp4")
        control_path = Path("/tmp/control.mp4")

        normalize_training_video_pair(rgb_path, control_path, logging.getLogger("test"))

        self.assertEqual(2, mock_reencode.call_count)

    @mock.patch(
        "wfm_post_training.video_utils.get_decord_fps",
        side_effect=[10.009, 10.0, 10.009, 10.0],
    )
    @mock.patch("wfm_post_training.video_utils._reencode_constant_fps")
    def test_should_raise_when_decord_fps_still_mismatched_after_reencode(
        self,
        _mock_reencode: mock.Mock,
        _mock_decord_fps: mock.Mock,
    ) -> None:
        rgb_path = Path("/tmp/rgb.mp4")
        control_path = Path("/tmp/control.mp4")

        with self.assertRaisesRegex(RuntimeError, "decord FPS mismatch after normalize"):
            normalize_training_video_pair(rgb_path, control_path, logging.getLogger("test"))


if __name__ == "__main__":
    unittest.main()
