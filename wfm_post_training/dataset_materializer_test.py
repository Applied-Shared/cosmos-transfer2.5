"""Tests for dataset materializer path helpers."""

from __future__ import annotations

import unittest

from wfm_post_training.dataset_materializer import finetuning_mapping_key


class FinetuningMappingKeyTest(unittest.TestCase):
    def test_should_use_conditioning_batch_id_filename(self) -> None:
        batch_id = "018f1234-5678-7abc-def0-123456789abc"
        self.assertEqual(
            finetuning_mapping_key(batch_id),
            f"finetuning_datasets/{batch_id}.txt",
            "mapping file must be keyed by conditioning_batch_id",
        )


if __name__ == "__main__":
    unittest.main()
