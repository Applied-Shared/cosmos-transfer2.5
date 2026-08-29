"""Lilypad entrypoint for Cosmos DiffusionRenderer (Inverse) smoke inference.

Mirrors wfm_inference/lilypad_entrypoint.py (Ray head -> GPU worker, OCI boto
download/upload) but runs the cosmos_predict1 DiffusionRenderer instead of
cosmos_transfer2 multiview. DR inverse is single-GPU, single-process: the worker
claims num_gpus=1 and runs dr_smoke_inference directly (no torchrun, no HF cache
remap — DR needs only its two OCI-staged weight dirs).

Config keys (entrypoint_fn_config):
    weights_bucket:  OCI bucket holding the DR weights         (e.g. sensor-sim-wfm)
    weights_prefix:  prefix containing the two model subdirs   (e.g. checkpoints/diffusion_renderer)
    output_bucket:   OCI bucket for outputs
    output_prefix:   prefix for outputs
    num_steps:       diffusion steps (default 15)
    num_frames:      video frames (default 57)
"""
import logging
import os

import ray

logger = logging.getLogger(__name__)

# Persistent dir on the worker so the ~29 GB weights survive across jobs.
_WORKER_CACHE_DIR = "/tmp/dr_worker_cache"


@ray.remote(num_gpus=1)
def _run_dr_on_gpu(config: dict) -> None:
    """Runs on the GPU worker: download weights once, then run DR smoke inference."""
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import boto3
    import botocore.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # OCI S3-compat requires payload signing and disables AWS v4 checksum headers.
    oci_config = botocore.config.Config(
        s3={"payload_signing_enabled": True},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=oci_config,
    )

    weights_bucket = config["weights_bucket"]
    weights_prefix = config["weights_prefix"].rstrip("/")
    checkpoint_dir = Path(_WORKER_CACHE_DIR) / "weights"

    # Download both model subdirs once; skip if already present.
    sentinel = checkpoint_dir / "Diffusion_Renderer_Inverse_Cosmos_7B"
    if sentinel.exists():
        log.info("DR weights already cached at %s", checkpoint_dir)
    else:
        log.info("Downloading DR weights s3://%s/%s -> %s",
                 weights_bucket, weights_prefix, checkpoint_dir)
        paginator = client.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=weights_bucket, Prefix=weights_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(weights_prefix):].lstrip("/")
                dest = checkpoint_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(weights_bucket, key, str(dest))
                n += 1
        log.info("Downloaded %d weight file(s)", n)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "outputs"
        output_dir.mkdir(parents=True)

        cmd = [
            "python", "-m", "wfm_inference.dr_smoke_inference",
            "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(output_dir),
            "--num-steps", str(config.get("num_steps", 15)),
            "--num-frames", str(config.get("num_frames", 57)),
        ]
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"dr_smoke_inference exited with code {result.returncode}")

        output_bucket = config["output_bucket"]
        output_prefix = config["output_prefix"].rstrip("/")
        files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
        log.info("Uploading %d file(s) to s3://%s/%s", len(files), output_bucket, output_prefix)
        for path in files:
            key = f"{output_prefix}/{path.relative_to(output_dir)}".lstrip("/")
            client.upload_file(str(path), output_bucket, key)
        client.put_object(Body=b"", Bucket=output_bucket, Key=f"{output_prefix}/succeed.txt")
        log.info("Upload complete.")


def run(config: dict) -> None:
    """Lilypad entrypoint: dispatch DR smoke inference to a single GPU worker."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ray.init(address="auto")
    logger.info("Dispatching DR inverse smoke inference to GPU worker (num_gpus=1)")
    ray.get(_run_dr_on_gpu.remote(config))
    logger.info("DR inference complete.")
