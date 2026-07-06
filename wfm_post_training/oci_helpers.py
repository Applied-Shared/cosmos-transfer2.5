"""OCI S3 helpers for WFM post-training Lilypad entrypoint."""

import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import botocore.config

# OCI S3-compat requires payload signing and disables the default AWS SDK v4
# checksum headers that OCI doesn't support.
OCI_BOTO_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)

WORKER_CACHE_DIR = Path("/tmp/wfm_worker_cache")

_THREAD_LOCAL = threading.local()


def _worker_client() -> "boto3.client":
    """Return a thread-local OCI client; boto3 clients are not thread-safe."""
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = make_plain_client()
        _THREAD_LOCAL.client = client
    return client


def make_plain_client() -> "boto3.client":
    """Build a direct OCI S3 client from standard AWS env vars."""
    import os

    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=OCI_BOTO_CONFIG,
    )


def remap_hf_snapshot(
    hf_cache_dir: Path,
    repo: str,
    expected_revision: str,
    logger: logging.Logger,
) -> None:
    """Copy snapshots/<actual_rev>/ to snapshots/<expected_rev>/ when they differ."""
    model_dir = hf_cache_dir / ("models--" + repo.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        logger.warning("remap_hf_snapshot: refs/main not found for %s, skipping", repo)
        return

    actual_revision = refs_main.read_text().strip()
    if actual_revision == expected_revision:
        return

    actual_snapshot = model_dir / "snapshots" / actual_revision
    expected_snapshot = model_dir / "snapshots" / expected_revision

    if not actual_snapshot.exists():
        logger.warning(
            "remap_hf_snapshot: snapshot %s not found for %s, skipping",
            actual_revision[:8],
            repo,
        )
        return

    if not expected_snapshot.exists():
        shutil.copytree(str(actual_snapshot), str(expected_snapshot), symlinks=True)
        logger.info("Remapped %s: %s -> %s", repo, actual_revision[:8], expected_revision[:8])


def download_checkpoint(
    cached_client: "boto3.client",
    bucket: str,
    key: str,
    logger: logging.Logger,
) -> Path:
    """Download model checkpoint to a persistent cache; skip if already present."""
    dest = WORKER_CACHE_DIR / "checkpoints" / bucket / key.lstrip("/")
    if dest.exists():
        logger.info("Checkpoint already cached at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading checkpoint s3://%s/%s -> %s", bucket, key, dest)
    cached_client.download_file(bucket, key, str(dest))
    return dest


def setup_hf_cache(
    plain_client: "boto3.client",
    cached_client: "boto3.client",
    hf_cache_bucket: str,
    hf_cache_prefix: str,
    hf_cache_dir: Path,
    logger: logging.Logger,
) -> None:
    """Download HF model cache from OCI; skip if expected snapshots already exist."""
    predict2b_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Predict2.5-2B"
        / "snapshots"
        / "6787e176dce74a101d922174a95dba29fa5f0c55"
    )
    reason1_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Reason1-7B"
        / "snapshots"
        / "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    )
    if predict2b_snapshot.exists() and reason1_snapshot.exists():
        logger.info("HF cache already populated, skipping download")
        return

    hf_cache_prefix = hf_cache_prefix.rstrip("/")
    logger.info("Downloading HF model cache from s3://%s/%s", hf_cache_bucket, hf_cache_prefix)
    paginator = plain_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=hf_cache_bucket, Prefix=hf_cache_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(hf_cache_prefix):].lstrip("/")
            dest = hf_cache_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached_client.download_file(hf_cache_bucket, key, str(dest))

    remap_hf_snapshot(
        hf_cache_dir,
        repo="nvidia/Cosmos-Predict2.5-2B",
        expected_revision="6787e176dce74a101d922174a95dba29fa5f0c55",
        logger=logger,
    )
    remap_hf_snapshot(
        hf_cache_dir,
        repo="nvidia/Cosmos-Reason1-7B",
        expected_revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        logger=logger,
    )


def object_exists(client: "boto3.client", bucket: str, key: str) -> bool:
    """Return True when an object exists at bucket/key."""
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def download_file(client: "boto3.client", bucket: str, key: str, dest: Path) -> None:
    """Download a single object to dest, creating parent dirs as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))


def upload_file(client: "boto3.client", local_path: Path, bucket: str, key: str) -> None:
    """Upload a local file to bucket/key."""
    client.upload_file(str(local_path), bucket, key)


def upload_directory(
    client: "boto3.client",
    local_dir: Path,
    bucket: str,
    prefix: str,
    logger: logging.Logger,
    *,
    max_workers: int = 10,
) -> None:
    """Recursively upload local_dir to s3://bucket/prefix/."""
    prefix = prefix.rstrip("/")
    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    logger.info("Uploading %d file(s) from %s to s3://%s/%s/", len(files), local_dir, bucket, prefix)

    if max_workers <= 1:
        for path in files:
            key = f"{prefix}/{path.relative_to(local_dir)}".replace("\\", "/")
            client.upload_file(str(path), bucket, key)
        return

    def _upload_one(path: Path) -> None:
        key = f"{prefix}/{path.relative_to(local_dir)}".replace("\\", "/")
        upload_file(_worker_client(), path, bucket, key)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_upload_one, files))


def download_directory(
    client: "boto3.client",
    bucket: str,
    prefix: str,
    dest_dir: Path,
    logger: logging.Logger,
) -> None:
    """Recursively download s3://bucket/prefix/ to dest_dir."""
    prefix = prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix):].lstrip("/")
            dest = dest_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    logger.info("Downloaded %d file(s) from s3://%s/%s to %s", count, bucket, prefix, dest_dir)
