# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Tests for the traffic-light per-camera facing-cull mask."""

import numpy as np

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.traffic_light_utils import (
    compute_traffic_light_facing_mask,
)

_LENS_NORMAL_X = np.array([1.0, 0.0, 0.0])  # lit lens points +x
_ORIGIN = np.array([0.0, 0.0, 0.0])
_IN_FRONT = np.array([10.0, 0.0, 0.0])  # camera in front of a +x-facing lens
_BEHIND = np.array([-10.0, 0.0, 0.0])  # camera behind a +x-facing lens


def test_should_return_none_when_cutoff_not_positive():
    # Precondition: a head is present but the cull is disabled by a non-positive cutoff.
    # Under test.
    mask = compute_traffic_light_facing_mask([_ORIGIN], [_LENS_NORMAL_X], [True], _IN_FRONT, 0.0)
    # Postcondition: no mask means no cull.
    assert mask is None


def test_should_return_none_when_no_heads():
    # Precondition: the cull is enabled but there are no heads.
    # Under test.
    mask = compute_traffic_light_facing_mask([], [], [], _IN_FRONT, 90.0)
    # Postcondition.
    assert mask is None


def test_should_keep_head_facing_the_camera():
    # Precondition: the head's lens points +x and the camera sits in front of it.
    # Under test.
    mask = compute_traffic_light_facing_mask([_ORIGIN], [_LENS_NORMAL_X], [True], _IN_FRONT, 90.0)
    # Postcondition.
    assert mask == [True]


def test_should_cull_head_facing_away_from_the_camera():
    # Precondition: the head's lens points +x and the camera sits behind it.
    # Under test.
    mask = compute_traffic_light_facing_mask([_ORIGIN], [_LENS_NORMAL_X], [True], _BEHIND, 90.0)
    # Postcondition.
    assert mask == [False]


def test_should_keep_unknown_orientation_head_even_when_facing_away():
    # Precondition: lens points +x with the camera behind, but the orientation is
    # unknown, so the normal is not a real facing.
    # Under test.
    mask = compute_traffic_light_facing_mask([_ORIGIN], [_LENS_NORMAL_X], [False], _BEHIND, 90.0)
    # Postcondition: unknown-orientation heads are never culled.
    assert mask == [True]
