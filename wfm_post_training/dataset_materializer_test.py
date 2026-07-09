"""Tests for dataset materializer path helpers."""

from __future__ import annotations

import unittest

from wfm_post_training.dataset_materializer import (
    _caption_key,
    finetuning_mapping_key,
)


class FinetuningMappingKeyTest(unittest.TestCase):
    def test_should_use_flyte_job_and_conditioning_batch_id(self) -> None:
        self.assertEqual(
            finetuning_mapping_key(
                "flyte-job-abc",
                "018f1234-5678-7abc-def0-123456789abc",
            ),
            "finetuning_datasets/flyte-job-abc/018f1234-5678-7abc-def0-123456789abc.txt",
            "mapping file must live under finetuning_datasets/<flyte_job_id>/",
        )


class CaptionKeyTest(unittest.TestCase):
    def test_should_use_captions_sds_prefix(self) -> None:
        self.assertEqual(
            _caption_key(
                "seg-001",
                "cosmos-reason2-2b_prompts-v1",
                use_legacy_sds_paths=False,
            ),
            "captions/sds/seg-001/front_wide/cosmos-reason2-2b_prompts-v1.json",
            "canonical captions live under captions/sds/<segment_id>/",
        )


if __name__ == "__main__":
    unittest.main()
