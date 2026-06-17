#!/usr/bin/env python3
"""Upload a local ROG Cosmos post-train dataset to OCI for Lilypad post-training tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3
import botocore.config

ROG_TO_SHORT = {
    "FRONT_CENTER": "front_wide",
    "FRONT_CENTER_NARROW": "front_tele",
    "FRONT_LEFT": "cross_left",
    "FRONT_RIGHT": "cross_right",
    "REAR_CENTER": "rear",
    "REAR_LEFT": "rear_left",
    "REAR_RIGHT": "rear_right",
}
ROG_CAMERAS = list(ROG_TO_SHORT.keys())

OCI_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)


def s3_client() -> "boto3.client":
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "AWS_ENDPOINT_URL_S3",
            "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com",
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-phoenix-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=OCI_CONFIG,
    )


def upload_file(client: "boto3.client", bucket: str, key: str, local_path: Path) -> None:
    print(f"  PUT s3://{bucket}/{key}")
    client.upload_file(str(local_path), bucket, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="sensor-sim-wfm")
    parser.add_argument("--run-id", default="rog102-v2-test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument(
        "--caption-id",
        default="v1.txt",
        help=(
            "Caption object name under sds/<segment_id>/captions/. "
            "Use a version label (e.g. v1.txt, v2.txt) so multiple caption "
            "generations can coexist for the same segment."
        ),
    )
    args = parser.parse_args()

    dataset = args.dataset_dir
    client = s3_client()
    manifest_lines: list[str] = []

    front_center = dataset / "videos" / "FRONT_CENTER"
    if not front_center.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {front_center}")

    samples = sorted(p.stem for p in front_center.glob("*.mp4"))
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No samples found under {front_center}")

    print(f"Uploading {len(samples)} sample(s) to s3://{args.bucket}/")
    print(f"Caption id: {args.caption_id}")

    for i, segment_id in enumerate(samples, 1):
        # For manual testing, control_bundle_id matches segment_id.
        control_bundle_id = segment_id
        caption_id = args.caption_id
        print(f"[{i}/{len(samples)}] segment_id={segment_id} caption_id={caption_id}")

        for rog, short in ROG_TO_SHORT.items():
            rgb_local = dataset / "videos" / rog / f"{segment_id}.mp4"
            if not rgb_local.exists():
                raise FileNotFoundError(rgb_local)
            upload_file(client, args.bucket, f"sds/{segment_id}/rgb/{short}.mp4", rgb_local)

        caption_json = dataset / "captions" / "FRONT_CENTER" / f"{segment_id}.json"
        if not caption_json.exists():
            raise FileNotFoundError(caption_json)
        caption_text = json.loads(caption_json.read_text(encoding="utf-8"))["caption"].strip()
        if not caption_text:
            raise ValueError(f"Empty caption in {caption_json}")
        caption_tmp = Path(f"/tmp/caption_{segment_id}_{caption_id.replace('/', '_')}")
        caption_tmp.write_text(caption_text + "\n", encoding="utf-8")
        upload_file(client, args.bucket, f"sds/{segment_id}/captions/{caption_id}", caption_tmp)

        spec: dict[str, object] = {"name": "multiview"}
        for rog in ROG_CAMERAS:
            control_local = dataset / "control_input_hdmap_bbox" / rog / f"{segment_id}.mp4"
            if not control_local.exists():
                raise FileNotFoundError(control_local)
            rel = f"cameras/{rog}/bbox.mp4"
            upload_file(
                client,
                args.bucket,
                f"control_bundles/{control_bundle_id}/{rel}",
                control_local,
            )
            spec[ROG_TO_SHORT[rog]] = {"control_path": rel}

        spec_tmp = Path(f"/tmp/spec_{control_bundle_id}.json")
        spec_tmp.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        upload_file(
            client,
            args.bucket,
            f"control_bundles/{control_bundle_id}/spec.json",
            spec_tmp,
        )

        manifest_lines.append(
            json.dumps(
                {
                    "control_bundle_id": control_bundle_id,
                    "segment_id": segment_id,
                    "caption_id": caption_id,
                }
            )
        )

    manifest_tmp = Path("/tmp/manifest.jsonl")
    manifest_tmp.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    manifest_key = f"post_training/{args.run_id}/manifest.jsonl"
    upload_file(client, args.bucket, manifest_key, manifest_tmp)

    print("\nDone.")
    print(f"Manifest: s3://{args.bucket}/{manifest_key}")
    print("Update cosmos_transfer_post_training.yaml:")
    print(f"  training_run_id: {args.run_id}")
    print(f"  manifest_key: {manifest_key}")
    print(f"  output_prefix: post_training/{args.run_id}")


if __name__ == "__main__":
    main()
