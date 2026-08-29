"""In-container smoke inference for Cosmos DiffusionRenderer (Inverse).

Runs single-GPU, single-process (no torchrun). Builds the pipeline, runs the
inverse renderer on a synthetic zeros video, and saves the predicted albedo
(basecolor) of frame 0 as a .npy plus a peak-VRAM report.

Local (RTX 5090 / sm_120) is expected to fail at the denoise step with either
OOM or "no kernel image available for sm_120" — both are caught and reported as
"image OK, needs A100". On an A100/H100 this completes and writes the artefact.

Usage:
    python -m wfm_inference.dr_smoke_inference \
        --checkpoint-dir /weights --output-dir /tmp/out

--checkpoint-dir must contain both model subdirs:
    Diffusion_Renderer_Inverse_Cosmos_7B/
    Cosmos-Tokenize1-CV8x8x8-720p/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# DiffusionRenderer source clone location (configs load relative to repo root).
_DR_REPO_ROOT = "/cosmos-dr"
_DR_MODEL_NAME = "Diffusion_Renderer_Inverse_Cosmos_7B"

# basecolor/albedo is index 0 in GBUFFER_INDEX_MAPPING (rendering_utils.py).
_BASECOLOR_INDEX = 0


def _build_dummy_batch(num_frames: int, height: int, width: int) -> dict:
    """Synthetic data_batch matching DiffusionRendererPipeline.generate_video.

    The dataset normally returns 4-D tensors and relies on dict_collation_fn to
    add the batch dim; we skip the dataloader, so add it explicitly (unsqueeze(0)).
    Zero T5 embeddings => no text encoder needed at inference.
    """
    dummy_video = torch.zeros(3, num_frames, height, width, dtype=torch.float32)
    return {
        "rgb": dummy_video.unsqueeze(0),
        "context_index": torch.LongTensor([_BASECOLOR_INDEX]).unsqueeze(0),
        "t5_text_embeddings": torch.zeros(512, 1024).unsqueeze(0),
        "t5_text_mask": torch.zeros(512).unsqueeze(0),
        "image_size": torch.tensor([height, width]).unsqueeze(0),
        "num_frames": torch.tensor(float(num_frames)).unsqueeze(0),
        "fps": torch.tensor(24.0).unsqueeze(0),
        "padding_mask": torch.zeros(1, height, width).unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos DiffusionRenderer inverse smoke inference.")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Dir containing the DR DiT + tokenizer subdirs.")
    parser.add_argument("--output-dir", required=True, help="Where to write artefacts.")
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    weights_path = os.path.join(args.checkpoint_dir, _DR_MODEL_NAME)
    if not os.path.isdir(weights_path):
        print(f"ERROR: DR weights not found at {weights_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[1/4] DR weights found at {weights_path}: OK")

    # Pipeline loads its config from paths relative to the repo root.
    os.chdir(_DR_REPO_ROOT)
    from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import (
        DiffusionRendererPipeline,
    )
    print("[2/4] cosmos_predict1 DiffusionRendererPipeline import: OK")

    torch.cuda.reset_peak_memory_stats()
    print(f"[3/4] Loading pipeline ({_DR_MODEL_NAME}) ...")
    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=_DR_MODEL_NAME,
        # Full offload keeps peak VRAM low; on a >=27 GB card these can be False
        # for realistic throughput. We measure VRAM either way and report it.
        offload_network=True,
        offload_tokenizer=True,
        offload_text_encoder_model=True,
        offload_guardrail_models=True,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        num_video_frames=args.num_frames,
    )
    print("    Pipeline loaded.")

    data_batch = _build_dummy_batch(args.num_frames, args.height, args.width)
    print(f"    Running inference on dummy {args.num_frames}x{args.height}x{args.width} video ...")
    try:
        output = pipeline.generate_video(data_batch=data_batch)  # [T,H,W,3] uint8
    except torch.OutOfMemoryError:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[4/4] Inference OOM -- GPU has {vram:.1f} GB; DR needs ~27 GB. "
              "Re-run on a larger GPU. Image is OK.")
        return
    except RuntimeError as exc:
        if "no kernel image" in str(exc).lower() or "sm_120" in str(exc).lower():
            cap = torch.cuda.get_device_capability(0)
            print(f"[4/4] Missing kernel for sm_{cap[0]}{cap[1]} (TE/megatron ship "
                  "sm_80/90 only). Image is OK; run on A100/H100.")
            return
        raise

    print(f"    Output shape: {output.shape}  dtype: {output.dtype}")
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[4/4] Peak VRAM: {vram_gb:.2f} GB")
    out_path = os.path.join(args.output_dir, "albedo_frame0.npy")
    np.save(out_path, output[0].cpu().numpy() if torch.is_tensor(output) else output[0])
    print(f"Saved albedo frame 0 -> {out_path}")
    print("Smoke inference PASSED.")


if __name__ == "__main__":
    main()
