"""Tests for dataset materializer path helpers."""

from __future__ import annotations

import unittest

from wfm_post_training.dataset_materializer import finetuning_mapping_key


class FinetuningMappingKeyTest(unittest.TestCase):
    def test_should_use_finetuning_datasets_prefix(self) -> None:
        self.assertEqual(
            finetuning_mapping_key("flyte-job-abc"),
            "finetuning_datasets/flyte-job-abc/segment_annotation_control_bundle.txt",
            "mapping file must live under finetuning_datasets/<flyte_job_id>/",
        )


if __name__ == "__main__":
    unittest.main()
