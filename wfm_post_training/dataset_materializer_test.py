"""Tests for WFM post-training dataset materializer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from wfm_post_training.dataset_materializer import (
    _caption_key,
    _read_caption_text,
    _rgb_key,
    finetuning_mapping_key,
    finetuning_mapping_to_manifest_entries,
    load_finetuning_mapping,
    parse_finetuning_mapping_line,
)


class ParseFinetuningMappingLineTest(unittest.TestCase):
    def test_should_parse_valid_line(self) -> None:
        entry = parse_finetuning_mapping_line("seg-a hash-a cb-123", 1)
        assert entry is not None
        self.assertEqual("seg-a", entry.segment_id)
        self.assertEqual("hash-a", entry.annotation_hash)
        self.assertEqual("cb-123", entry.control_bundle_id)

    def test_should_skip_blank_lines(self) -> None:
        self.assertIsNone(parse_finetuning_mapping_line("", 1))
        self.assertIsNone(parse_finetuning_mapping_line("  ", 1))
        self.assertIsNone(parse_finetuning_mapping_line("# comment", 1))

    def test_should_reject_invalid_field_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_finetuning_mapping_line("only-two fields", 2)


class LoadFinetuningMappingTest(unittest.TestCase):
    def test_should_load_multiple_lines(self) -> None:
        text = "seg-a hash-a cb-1\n\nseg-b hash-b cb-2\n"
        entries = load_finetuning_mapping(text)
        self.assertEqual(2, len(entries))
        self.assertEqual("cb-2", entries[1].control_bundle_id)


class FinetuningMappingHelpersTest(unittest.TestCase):
    def test_finetuning_mapping_key(self) -> None:
        self.assertEqual(
            "finetuning_jobs/flyte-abc/segment_annotation_control_bundle.txt",
            finetuning_mapping_key("flyte-abc"),
        )

    def test_finetuning_mapping_to_manifest_entries(self) -> None:
        mapping = load_finetuning_mapping("seg-a hash-a cb-1\n")
        entries = finetuning_mapping_to_manifest_entries(
            mapping,
            "cosmos-reason2-2b_prompts-v1",
        )
        self.assertEqual(1, len(entries))
        self.assertEqual("cb-1", entries[0].control_bundle_id)
        self.assertEqual("seg-a", entries[0].segment_id)
        self.assertEqual("cosmos-reason2-2b_prompts-v1", entries[0].caption_id)


class OciPathHelpersTest(unittest.TestCase):
    def test_rgb_key_canonical_layout(self) -> None:
        self.assertEqual(
            "rgb/seg-1/front_wide.mp4",
            _rgb_key("seg-1", "front_wide", use_legacy_sds_paths=False),
        )

    def test_rgb_key_legacy_layout(self) -> None:
        self.assertEqual(
            "sds/seg-1/rgb/front_wide.mp4",
            _rgb_key("seg-1", "front_wide", use_legacy_sds_paths=True),
        )

    def test_caption_key_canonical_layout(self) -> None:
        self.assertEqual(
            "captions/seg-1/front_wide/cosmos-reason2-2b_prompts-v1.json",
            _caption_key(
                "seg-1",
                "cosmos-reason2-2b_prompts-v1",
                use_legacy_sds_paths=False,
            ),
        )

    def test_caption_key_legacy_layout(self) -> None:
        self.assertEqual(
            "sds/seg-1/captions/v1.txt",
            _caption_key("seg-1", "v1.txt", use_legacy_sds_paths=True),
        )


class ReadCaptionTextTest(unittest.TestCase):
    def test_should_read_plain_text_legacy(self) -> None:
        path = Path("/tmp/caption.txt")
        with mock.patch.object(Path, "read_text", return_value=" hello "):
            self.assertEqual("hello", _read_caption_text(path, use_legacy_sds_paths=True))

    def test_should_read_json_canonical(self) -> None:
        path = Path("/tmp/caption.json")
        payload = json.dumps({"caption": "scene text"})
        with mock.patch.object(Path, "read_text", return_value=payload):
            self.assertEqual(
                "scene text",
                _read_caption_text(path, use_legacy_sds_paths=False),
            )


if __name__ == "__main__":
    unittest.main()
