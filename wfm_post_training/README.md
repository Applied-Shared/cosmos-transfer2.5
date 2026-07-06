# WFM Post-Training — Lilypad Entrypoint

Runs Cosmos Transfer 2.5 multiview HDMap post-training on Lilypad generic workloads.
The workload config lives at `adp/services/wfm/lilypad_workload_configs/cosmos_transfer_post_training.yaml`
in the applied3 repo.

## Architecture

Same Ray head/GPU-worker split as inference (`wfm_inference/lilypad_entrypoint.py`):

- **Head node** — runs `lilypad_post_training_entrypoint.run()`.
- **GPU worker** — downloads manifest, materializes dataset, runs `torchrun scripts.train`,
  uploads checkpoints to OCI, converts final DCP to `model_ema_bf16.pt`.

## Manual launch

```bash
export WANDB_API_KEY=<from 1Password item "WFM W&B API Key (sensor-platform)">
export WANDB_ENTITY=sensor-platform
export AWS_ACCESS_KEY_ID=<oci-access-key>
export AWS_SECRET_ACCESS_KEY=<oci-secret-key>

lilypad workload launch adp/services/wfm/lilypad_workload_configs/cosmos_transfer_post_training.yaml
```

Create a personal W&B API key at https://appliedintuition.wandb.io/settings if needed.
For team runs, use the shared key from 1Password item **WFM W&B API Key (sensor-platform)**.

## Manifest format

### Finetuning mapping (recommended)

When `conditioning_batch_id` and `caption_version` are set in the workload config, the entrypoint
reads:

```
finetuning_datasets/<flyte_job_id>/<conditioning_batch_id>.txt
```

Each non-empty line: `<segment_id> <annotation_hash> <control_bundle_id>`

OCI sources resolved per sample (WFM canonical layout):

| Field | OCI path |
|-------|----------|
| `control_bundle_id` | `control_bundles/<id>/` via `spec.json` or `cameras/*/bbox.mp4` |
| `segment_id` | `rgb/sds/<segment_id>/{short_name\|ROG\|rig}.mp4` — tries lowercase short name first, then ROG names like `FRONT_CENTER`, then Cosmos rig names |
| `caption_version` | `captions/<segment_id>/{front_wide\|FRONT_CENTER}/<caption_version>.json` |

Example `caption_version`: `cosmos-reason2-2b_prompts-v1`

### Legacy JSONL manifest

Upload to OCI before launch (one sample per line):

```json
{"control_bundle_id":"<uuid>","segment_id":"<segment>","caption_version":"<caption>"}
```

Example key: `post_training/example-run-001/manifest.jsonl`

Each line references three OCI sources (legacy `sds/` layout):

| Field | OCI path |
|-------|----------|
| `control_bundle_id` | `control_bundles/<id>/cameras/*/bbox.mp4` |
| `segment_id` | `sds/<segment_id>/rgb/<short_name>.mp4` (7 cameras) |
| `caption_version` | `sds/<segment_id>/captions/<caption_version>` (e.g. `v1.txt`) |

On-disk sample stem is `control_bundle_id`. Invalid samples (missing files or empty caption)
are skipped with a warning; the job fails only if zero valid samples remain.

## Materialized dataset layout

Written to `/tmp/wfm_worker_cache/post_training/{training_run_id}/dataset/`:

```
dataset/
├── videos/{short_name}/{control_bundle_id}.mp4
├── control_input_hdmap_bbox/{short_name}/{control_bundle_id}.mp4
└── captions/front_wide/{control_bundle_id}.json
```

Short camera folder names: `cross_left`, `cross_right`, `front_tele`, `front_wide`,
`rear_left`, `rear_right`, `rear`.

## Config keys

| Key | Description |
|-----|-------------|
| `training_run_id` | `finetuning_runs.uuid` for this Lilypad submission; used as `job.name`, W&B run name, and local cache dir |
| `input_bucket` | OCI bucket for training inputs |
| `flyte_job_id` | Flyte campaign id (informational when submitted via WFM) |
| `conditioning_batch_id` | Conditioning batch UUID; reads `finetuning_datasets/<flyte_job_id>/<conditioning_batch_id>.txt` |
| `caption_version` | Caption version filename (required with `conditioning_batch_id`), e.g. `cosmos-reason2-2b_prompts-v1` |
| `manifest_key` | Legacy JSONL manifest key (use when `conditioning_batch_id` is unset) |
| `output_bucket` / `output_prefix` | OCI destination for checkpoints and final `.pt` (`finetuning_jobs/<finetuning_run_id>/` when submitted via WFM) |
| `checkpoint_bucket` / `checkpoint_key` | Base model `.pt` cached locally (default: base) |
| `hf_cache_bucket` / `hf_cache_prefix` | Pre-staged HuggingFace cache |
| `experiment` | Hydra experiment (default: `transfer2_auto_multiview_post_train_example`) |
| `data_train` | Hydra dataloader name (default: `example_multiview_train_data_control_input_hdmap_sds`) |
| `max_iter` / `save_iter` | Training length and checkpoint frequency |
| `resume_from_oci` | Download latest OCI checkpoint before training (default: false) |
| `checkpoint_load_path` | Optional override for initial `checkpoint.load_path` |

## Checkpoints and resume

Training writes DCP checkpoints under:

```
/tmp/wfm_worker_cache/post_training/{training_run_id}/output/
  cosmos_transfer_v2p5/auto_multiview/{training_run_id}/checkpoints/
```

(`IMAGINAIRE_OUTPUT_ROOT` is set to the `output/` directory above.)

| Scenario | Behavior |
|----------|----------|
| Same pod retry | Resumes from local `latest_checkpoint.txt` if present |
| New submission | Set `resume_from_oci: true` to pull latest iter from `output_prefix/checkpoints/` |
| During training | Background watcher uploads each new iter to OCI |
| On success | Converts latest DCP to `model_ema_bf16.pt` and uploads to `output_prefix/` |

## W&B from WFM service submit

The workload YAML lists `WANDB_API_KEY` under `required_environment_variables`. When WFM
submits this workload, `buildEnvVars` forwards keys from the WFM pod environment. Mount
`WANDB_API_KEY` as a K8s secret on the WFM deployment (same pattern as OCI creds). Source
the value from 1Password item **WFM W&B API Key (sensor-platform)**. Set `WANDB_ENTITY`
to `sensor-platform` in the workload YAML (or rely on the entrypoint default).

## Upload local dataset to OCI

Use `upload_local_dataset_to_oci.py` to transform a local ROG-layout dataset and upload
manifest + SDS + control bundle objects:

```bash
python3 wfm_post_training/upload_local_dataset_to_oci.py \
  --dataset-dir /path/to/rog102_v2 \
  --run-id rog102-v2-test \
  --max-samples 5 \
  --caption-version v1.txt
```

`--caption-version` is the object name under `sds/<segment_id>/captions/` (default `v1.txt`).
Use a new version (e.g. `v2.txt`) when uploading a revised caption for the same segment; point
the manifest at the desired `caption_version` per sample.

## Fake data for testing

Before the SDS pipeline is live, seed minimal objects in `sensor-sim-wfm`:

```
sds/test-segment-001/rgb/front_wide.mp4       (+ 6 other short names)
sds/test-segment-001/captions/cap-v1.txt
control_bundles/<uuid>/cameras/FRONT_CENTER/bbox.mp4   (+ 6 other ROG camera dirs)
post_training/test-run/manifest.jsonl
```

Manifest line:

```json
{"control_bundle_id":"<uuid>","segment_id":"test-segment-001","caption_version":"cap-v1.txt"}
```

Update `training_run_id`, `manifest_key`, and `output_prefix` in the YAML to match.

## Building and pushing the Docker image

1. Find the current image tag in applied3:

   ```bash
   grep docker_image applied3/adp/services/wfm/lilypad_workload_configs/cosmos_transfer_post_training.yaml
   ```

   The tag is the suffix after `sds:` (for example `cosmos_transfer2.5_v0.0.32`).

2. Bump the patch version for your new build (for example `v0.0.32` → `v0.0.33`).

3. Build and push with the bumped tag:

```bash
cd /home/yun/cosmos-transfer2.5
VERSION=cosmos_transfer2.5_v0.0.NN  # bump from step 1
IMAGE=us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:${VERSION}

docker build -f Dockerfile \
  --no-cache \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true \
  -t "${IMAGE}" .

docker push "${IMAGE}"
```

   Use `--no-cache` when rebuilding after changing files under `wfm_post_training/`
   so Ray head and GPU workers get the same code.

4. Update `docker_image` in
   `applied3/adp/services/wfm/lilypad_workload_configs/cosmos_transfer_post_training.yaml`
   to the new tag.

## OCI S3-compat gotcha

Same as inference — any boto3 client used against OCI must use payload signing and
`when_required` checksum settings. See `wfm_inference/README.md`.
