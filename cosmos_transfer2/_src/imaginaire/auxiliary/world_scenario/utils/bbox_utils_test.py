# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Tests for the bbox object-type simplification."""

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.bbox_utils import simplify_object_type

_CANONICAL_CATEGORIES = ["Car", "Truck", "Pedestrian", "Cyclist", "Others"]


def test_should_map_every_canonical_category_to_itself():
    # Precondition: the input is already one of the canonical categories.
    for category in _CANONICAL_CATEGORIES:
        # Under test.
        simplified = simplify_object_type(category)
        # Postcondition: canonical categories survive re-simplification unchanged.
        assert simplified == category


def test_should_be_idempotent_for_any_input():
    # Precondition: a mix of synonyms, canonical names, and unmapped strings.
    inputs = ["Truck", "Bus", "Vehicle", "Person", "Rider", "Other", "Garbage_Type", ""]
    for object_type in inputs:
        # Under test.
        once = simplify_object_type(object_type)
        twice = simplify_object_type(once)
        # Postcondition: simplifying an already-simplified value is a no-op.
        assert twice == once


def test_should_map_synonyms_to_their_category():
    # Precondition: known synonym spellings per category.
    synonyms = {
        "Bus": "Truck",
        "heavy_truck": "Truck",
        "Trailer": "Truck",
        "Vehicle": "Car",
        "other_vehicle": "Car",
        "Person": "Pedestrian",
        "Rider": "Cyclist",
        "Motorcycle": "Cyclist",
    }
    for object_type, expected in synonyms.items():
        # Under test.
        simplified = simplify_object_type(object_type)
        # Postcondition.
        assert simplified == expected


def test_should_normalize_separators_and_case():
    # Precondition: the same synonym in dash, space, and upper-case spellings.
    for object_type in ["heavy-truck", "heavy truck", "HEAVY_TRUCK"]:
        # Under test.
        simplified = simplify_object_type(object_type)
        # Postcondition.
        assert simplified == "Truck"


def test_should_map_unknown_type_to_others():
    # Precondition: a string outside every category bucket.
    # Under test.
    simplified = simplify_object_type("Unicycle")
    # Postcondition.
    assert simplified == "Others"
