"""Tests for post-training entry loading."""

from __future__ import annotations

import logging
import unittest
from unittest import mock
from unittest.mock import ANY

from wfm_post_training.dataset_materializer import ManifestEntry
from wfm_post_training.lilypad_post_training_entrypoint import _load_training_entries


class LoadTrainingEntriesTest(unittest.TestCase):
    def test_should_load_finetuning_mapping_when_flyte_job_id_set(self) -> None:
        config = {
            "manifest_bucket": "sensor-sim-wfm",
            "flyte_job_id": "flyte-abc",
            "caption_version": "cosmos-reason2-2b_prompts-v1",
        }
        mapping_entry = mock.Mock()
        expected = [
            ManifestEntry(
                control_bundle_id="cb-1",
                segment_id="seg-1",
                caption_version="cosmos-reason2-2b_prompts-v1",
            )
        ]
        with mock.patch(
            "wfm_post_training.lilypad_post_training_entrypoint.download_finetuning_mapping",
            return_value=[mapping_entry],
        ) as mock_download, mock.patch(
            "wfm_post_training.lilypad_post_training_entrypoint.finetuning_mapping_to_manifest_entries",
            return_value=expected,
        ) as mock_convert:
            entries, use_legacy = _load_training_entries(
                config,
                mock.Mock(),
                logging.getLogger("test"),
            )

        self.assertFalse(use_legacy)
        self.assertEqual(expected, entries)
        mock_download.assert_called_once_with(ANY, "sensor-sim-wfm", "flyte-abc")
        mock_convert.assert_called_once_with(
            [mapping_entry],
            "cosmos-reason2-2b_prompts-v1",
        )

    def test_should_accept_deprecated_caption_id_alias(self) -> None:
        config = {
            "manifest_bucket": "sensor-sim-wfm",
            "flyte_job_id": "flyte-abc",
            "caption_id": "legacy-caption-v1",
        }
        with mock.patch(
            "wfm_post_training.lilypad_post_training_entrypoint.download_finetuning_mapping",
            return_value=[],
        ), mock.patch(
            "wfm_post_training.lilypad_post_training_entrypoint.finetuning_mapping_to_manifest_entries",
            return_value=[],
        ) as mock_convert:
            _load_training_entries(config, mock.Mock(), logging.getLogger("test"))

        mock_convert.assert_called_once_with([], "legacy-caption-v1")

    def test_should_load_legacy_manifest_when_manifest_key_set(self) -> None:
        config = {
            "manifest_bucket": "sensor-sim-wfm",
            "manifest_key": "post_training/run/manifest.jsonl",
        }
        expected = [
            ManifestEntry(
                control_bundle_id="cb-1",
                segment_id="seg-1",
                caption_version="v1.txt",
            )
        ]
        with mock.patch(
            "wfm_post_training.lilypad_post_training_entrypoint.download_manifest",
            return_value=expected,
        ) as mock_download:
            entries, use_legacy = _load_training_entries(
                config,
                mock.Mock(),
                logging.getLogger("test"),
            )

        self.assertTrue(use_legacy)
        self.assertEqual(expected, entries)
        mock_download.assert_called_once_with(
            ANY,
            "sensor-sim-wfm",
            "post_training/run/manifest.jsonl",
        )

    def test_should_require_caption_version_with_flyte_job_id(self) -> None:
        config = {
            "manifest_bucket": "sensor-sim-wfm",
            "flyte_job_id": "flyte-abc",
        }
        with self.assertRaisesRegex(ValueError, "caption_version is required"):
            _load_training_entries(config, mock.Mock(), logging.getLogger("test"))

    def test_should_require_flyte_job_id_or_manifest_key(self) -> None:
        config = {"manifest_bucket": "sensor-sim-wfm"}
        with self.assertRaisesRegex(ValueError, "either flyte_job_id or manifest_key"):
            _load_training_entries(config, mock.Mock(), logging.getLogger("test"))


if __name__ == "__main__":
    unittest.main()
