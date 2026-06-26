"""Lilypad entrypoint for Cosmos Transfer 2.5 WFM post-training."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

import ray
from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

from wfm_post_training.dataset_materializer import (
    download_finetuning_mapping,
    download_manifest,
    finetuning_mapping_to_manifest_entries,
    materialize_dataset,
)
from wfm_post_training.oci_helpers import (
    WORKER_CACHE_DIR,
    download_checkpoint,
    download_directory,
    make_plain_client,
    setup_hf_cache,
    upload_directory,
    upload_file,
)

logger = logging.getLogger(__name__)

JOB_PROJECT = "cosmos_transfer_v2p5"
JOB_GROUP = "auto_multiview"
TRAIN_CONFIG = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"


def _worker_cache_dir(training_run_id: str) -> Path:
    return WORKER_CACHE_DIR / "post_training" / training_run_id


def _imaginaire_output_root(training_run_id: str) -> Path:
    return _worker_cache_dir(training_run_id) / "output"


def _local_checkpoints_dir(training_run_id: str) -> Path:
    return (
        _imaginaire_output_root(training_run_id)
        / JOB_PROJECT
        / JOB_GROUP
        / training_run_id
        / "checkpoints"
    )


def _read_latest_checkpoint_iter(checkpoints_dir: Path) -> str | None:
    latest_file = checkpoints_dir / "latest_checkpoint.txt"
    if not latest_file.exists():
        return None
    return latest_file.read_text(encoding="utf-8").strip()


def _resolve_local_resume_path(training_run_id: str) -> Path | None:
    checkpoints_dir = _local_checkpoints_dir(training_run_id)
    latest_iter = _read_latest_checkpoint_iter(checkpoints_dir)
    if not latest_iter:
        return None
    resume_path = checkpoints_dir / latest_iter
    if not resume_path.exists():
        return None
    return resume_path


def _download_oci_resume(
    plain_client,
    config: dict,
    training_run_id: str,
    worker_logger: logging.Logger,
) -> Path | None:
    """Download the latest OCI checkpoint iter for cross-submission resume."""
    output_bucket = config["output_bucket"]
    output_prefix = config["output_prefix"].rstrip("/")
    prefix = f"{output_prefix}/checkpoints/"

    paginator = plain_client.get_paginator("list_objects_v2")
    latest_iter: str | None = None
    for page in paginator.paginate(Bucket=output_bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            name = common_prefix["Prefix"][len(prefix):].rstrip("/")
            if name.startswith("iter_"):
                if latest_iter is None or name > latest_iter:
                    latest_iter = name

    if latest_iter is None:
        worker_logger.info("No OCI checkpoints found under s3://%s/%s", output_bucket, prefix)
        return None

    local_dir = _local_checkpoints_dir(training_run_id)
    local_iter_dir = local_dir / latest_iter
    worker_logger.info(
        "Downloading OCI resume checkpoint s3://%s/%s%s -> %s",
        output_bucket,
        prefix,
        latest_iter,
        local_iter_dir,
    )
    download_directory(
        plain_client,
        output_bucket,
        f"{prefix}{latest_iter}",
        local_iter_dir,
        worker_logger,
    )
    (local_dir / "latest_checkpoint.txt").write_text(latest_iter + "\n", encoding="utf-8")
    return local_iter_dir


def _wandb_login(worker_logger: logging.Logger, wandb_dir: Path) -> None:
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is required for post-training")
    host = os.environ.get("WANDB_BASE_URL", "https://appliedintuition.wandb.io")

    # Ray may set WANDB_DIR under /tmp/ray/wandb, which is not writable on GPU workers.
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)

    # Lilypad GPU workers mount /tmp/netrc read-only; keep netrc under worker cache.
    netrc_path = wandb_dir / ".netrc"
    os.environ["NETRC"] = str(netrc_path)

    worker_logger.info("Logging in to W&B at %s (dir=%s)", host, wandb_dir)
    import wandb

    wandb.login(key=api_key, host=host, relogin=True)


class _CheckpointWatcher:
    """Poll latest_checkpoint.txt and upload new iters to OCI."""

    def __init__(
        self,
        plain_client,
        checkpoints_dir: Path,
        output_bucket: str,
        output_prefix: str,
        worker_logger: logging.Logger,
        poll_interval_s: float = 30.0,
    ) -> None:
        self._client = plain_client
        self._checkpoints_dir = checkpoints_dir
        self._output_bucket = output_bucket
        self._output_prefix = output_prefix.rstrip("/")
        self._logger = worker_logger
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._uploaded_iters: set[str] = set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)

    def _run(self) -> None:
        while not self._stop.is_set():
            latest_iter = _read_latest_checkpoint_iter(self._checkpoints_dir)
            if latest_iter and latest_iter not in self._uploaded_iters:
                local_iter_dir = self._checkpoints_dir / latest_iter
                if local_iter_dir.is_dir():
                    oci_prefix = f"{self._output_prefix}/checkpoints/{latest_iter}"
                    self._logger.info(
                        "Uploading checkpoint iter %s to s3://%s/%s/",
                        latest_iter,
                        self._output_bucket,
                        oci_prefix,
                    )
                    upload_directory(
                        self._client,
                        local_iter_dir,
                        self._output_bucket,
                        oci_prefix,
                        self._logger,
                    )
                    self._uploaded_iters.add(latest_iter)
            self._stop.wait(self._poll_interval_s)


def _build_train_cmd(
    config: dict,
    dataset_dir: Path,
    training_run_id: str,
    resume_path: Path | None,
    initial_checkpoint_path: Path | None = None,
) -> list[str]:
    num_gpus = config.get("num_gpus", 8)
    experiment = config.get("experiment", "transfer2_auto_multiview_post_train_example")
    data_train = config.get("data_train", "example_multiview_train_data_control_input_hdmap_sds")
    max_iter = config.get("max_iter", 5000)
    save_iter = config.get("save_iter", 200)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=12341",
        "-m",
        "scripts.train",
        f"--config={TRAIN_CONFIG}",
        "--",
        f"experiment={experiment}",
        f"data_train={data_train}",
        f"dataloader_train.dataset.dataset_dir={dataset_dir}",
        f"dataloader_train.sampler.dataset.dataset_dir={dataset_dir}",
        f"job.name={training_run_id}",
        "job.wandb_mode=online",
        f"trainer.max_iter={max_iter}",
        f"checkpoint.save_iter={save_iter}",
    ]

    if resume_path is not None:
        cmd.append(f"checkpoint.load_path={resume_path}")
        cmd.append("checkpoint.load_training_state=True")
        cmd.append("checkpoint.strict_resume=False")
    elif initial_checkpoint_path is not None:
        cmd.append(f"checkpoint.load_path={initial_checkpoint_path}")
    elif config.get("checkpoint_load_path"):
        cmd.append(f"checkpoint.load_path={config['checkpoint_load_path']}")

    return cmd


def _convert_and_upload_final_checkpoint(
    plain_client,
    config: dict,
    training_run_id: str,
    worker_logger: logging.Logger,
) -> None:
    checkpoints_dir = _local_checkpoints_dir(training_run_id)
    latest_iter = _read_latest_checkpoint_iter(checkpoints_dir)
    if not latest_iter:
        worker_logger.warning("No latest checkpoint found; skipping final .pt conversion")
        return

    checkpoint_dir = checkpoints_dir / latest_iter
    model_distcp = checkpoint_dir / "model"
    if not model_distcp.is_dir():
        worker_logger.warning("Checkpoint model dir missing at %s; skipping conversion", model_distcp)
        return

    convert_dir = checkpoint_dir / "converted"
    convert_dir.mkdir(parents=True, exist_ok=True)
    convert_cmd = [
        "python",
        "scripts/convert_distcp_to_pt.py",
        str(model_distcp),
        str(convert_dir),
    ]
    worker_logger.info("Converting DCP checkpoint: %s", " ".join(convert_cmd))
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        worker_logger.error("convert_distcp_to_pt failed: %s", result.stderr)
        return

    output_bucket = config["output_bucket"]
    output_prefix = config["output_prefix"].rstrip("/")
    for name in ("model_ema_bf16.pt", "model_ema_fp32.pt", "model.pt"):
        local_path = convert_dir / name
        if local_path.exists():
            key = f"{output_prefix}/{name}"
            worker_logger.info("Uploading %s to s3://%s/%s", name, output_bucket, key)
            upload_file(plain_client, local_path, output_bucket, key)


def _load_training_entries(config: dict, plain_client, worker_logger: logging.Logger):
    """Load manifest entries from finetuning mapping or legacy JSONL manifest."""
    manifest_bucket = config["manifest_bucket"]
    flyte_job_id = (config.get("flyte_job_id") or "").strip()
    manifest_key = (config.get("manifest_key") or "").strip()

    if flyte_job_id:
        caption_id = (config.get("caption_id") or "").strip()
        if not caption_id:
            raise ValueError("caption_id is required when flyte_job_id is set")
        worker_logger.info(
            "Downloading finetuning mapping s3://%s/finetuning_jobs/%s/"
            "segment_annotation_control_bundle.txt",
            manifest_bucket,
            flyte_job_id,
        )
        mapping_entries = download_finetuning_mapping(
            plain_client,
            manifest_bucket,
            flyte_job_id,
        )
        entries = finetuning_mapping_to_manifest_entries(mapping_entries, caption_id)
        worker_logger.info(
            "Finetuning mapping contains %d entries (caption_id=%s)",
            len(entries),
            caption_id,
        )
        return entries, False

    if not manifest_key:
        raise ValueError("either flyte_job_id or manifest_key is required")

    worker_logger.info(
        "Downloading legacy manifest s3://%s/%s",
        manifest_bucket,
        manifest_key,
    )
    entries = download_manifest(plain_client, manifest_bucket, manifest_key)
    worker_logger.info("Manifest contains %d entries", len(entries))
    return entries, True


@ray.remote
def _run_post_training_on_gpu(config: dict) -> None:
    """Runs on the GPU worker: materialize dataset, train, upload checkpoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker_logger = logging.getLogger(__name__)

    plain_client = make_plain_client()
    cached_client = get_readonly_boto_client()

    training_run_id = config["training_run_id"]
    output_bucket = config["output_bucket"]
    output_prefix = config["output_prefix"]

    worker_dir = _worker_cache_dir(training_run_id)
    dataset_dir = worker_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    imaginaire_root = _imaginaire_output_root(training_run_id)
    imaginaire_root.mkdir(parents=True, exist_ok=True)
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(imaginaire_root)

    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    setup_hf_cache(
        plain_client,
        cached_client,
        config["hf_cache_bucket"],
        config["hf_cache_prefix"],
        hf_cache_dir,
        worker_logger,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"

    initial_checkpoint_path: Path | None = None
    if config.get("checkpoint_bucket") and config.get("checkpoint_key"):
        initial_checkpoint_path = download_checkpoint(
            cached_client,
            config["checkpoint_bucket"],
            config["checkpoint_key"],
            worker_logger,
        )

    entries, use_legacy_sds_paths = _load_training_entries(config, plain_client, worker_logger)
    materialize_dataset(
        plain_client,
        config["manifest_bucket"],
        entries,
        dataset_dir,
        use_legacy_sds_paths=use_legacy_sds_paths,
    )

    resume_path: Path | None = None
    if config.get("resume_from_oci"):
        resume_path = _download_oci_resume(plain_client, config, training_run_id, worker_logger)
    if resume_path is None:
        resume_path = _resolve_local_resume_path(training_run_id)
    if resume_path is not None:
        worker_logger.info("Resuming training from checkpoint %s", resume_path)

    _wandb_login(worker_logger, worker_dir / "wandb")

    checkpoints_dir = _local_checkpoints_dir(training_run_id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    watcher = _CheckpointWatcher(
        plain_client,
        checkpoints_dir,
        output_bucket,
        output_prefix,
        worker_logger,
    )
    watcher.start()

    train_cmd = _build_train_cmd(
        config,
        dataset_dir,
        training_run_id,
        resume_path,
        initial_checkpoint_path,
    )
    worker_logger.info("Running: %s", " ".join(train_cmd))
    log_path = worker_dir / "console.log"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                train_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    finally:
        watcher.stop()

    if result.returncode != 0:
        debug_key = f"{output_prefix.rstrip('/')}/_debug/console.log"
        try:
            upload_file(plain_client, log_path, output_bucket, debug_key)
            worker_logger.info("Uploaded console.log to s3://%s/%s", output_bucket, debug_key)
        except Exception as upload_err:
            worker_logger.warning("Could not upload console.log: %s", upload_err)
        plain_client.put_object(
            Body=b"",
            Bucket=output_bucket,
            Key=f"{output_prefix.rstrip('/')}/failed.txt",
        )
        raise RuntimeError(f"torchrun exited with code {result.returncode}")

    worker_logger.info("Training finished successfully")

    latest_iter = _read_latest_checkpoint_iter(checkpoints_dir)
    if latest_iter and latest_iter not in watcher._uploaded_iters:
        local_iter_dir = checkpoints_dir / latest_iter
        oci_prefix = f"{output_prefix.rstrip('/')}/checkpoints/{latest_iter}"
        upload_directory(plain_client, local_iter_dir, output_bucket, oci_prefix, worker_logger)

    _convert_and_upload_final_checkpoint(plain_client, config, training_run_id, worker_logger)
    plain_client.put_object(
        Body=b"",
        Bucket=output_bucket,
        Key=f"{output_prefix.rstrip('/')}/succeed.txt",
    )
    worker_logger.info("Post-training complete for run %s", training_run_id)


def run(config: dict) -> None:
    """Lilypad entrypoint for Cosmos Transfer 2.5 multiview post-training.

    Required config keys:
        training_run_id:    unique run id; used as job.name and W&B run name
        manifest_bucket:    OCI bucket containing training inputs
        output_bucket:        OCI bucket for checkpoints and final .pt uploads
        output_prefix:        prefix under output_bucket (e.g. post_training/run-001)
        hf_cache_bucket:      OCI bucket with pre-staged HF model cache
        hf_cache_prefix:      prefix under hf_cache_bucket

    Input manifest — provide either:
        flyte_job_id + caption_id: reads finetuning_jobs/<flyte_job_id>/
            segment_annotation_control_bundle.txt and WFM canonical OCI paths
        manifest_key: legacy JSONL manifest at manifest_bucket/manifest_key

    Optional config keys:
        checkpoint_bucket / checkpoint_key: base model .pt to cache locally (default base)
        experiment:           Hydra experiment name (default transfer2_auto_multiview_post_train_example)
        data_train:           Hydra data_train name (default example_multiview_train_data_control_input_hdmap_sds)
        num_gpus:             GPUs to use (default 8)
        max_iter:             training iterations (default 5000)
        save_iter:            checkpoint save frequency (default 200)
        resume_from_oci:      download latest OCI checkpoint before training (default false)
        checkpoint_load_path: override initial checkpoint.load_path (local path or URI)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = config.get("num_gpus", 8)
    ray.init(address="auto")
    logger.info(
        "Dispatching post-training run %s to GPU worker (num_gpus=%d)",
        config["training_run_id"],
        num_gpus,
    )
    ref = _run_post_training_on_gpu.options(num_gpus=num_gpus).remote(config)
    ray.get(ref)
    logger.info("Post-training run complete.")
