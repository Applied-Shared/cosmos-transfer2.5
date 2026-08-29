# SPDX-License-Identifier: Apache-2.0
#
# Standalone Cosmos DiffusionRenderer (Inverse) image for Lilypad generic workloads.
#
# This is intentionally SEPARATE from the cosmos_transfer2 image (../Dockerfile):
# DiffusionRenderer is cosmos_predict1 lineage (transformer_engine 1.12.0,
# megatron-core 0.10.0, diffusers 0.32.2, torch 2.6.0/cu124) whose dependency
# tree does not coexist with the uv-locked cosmos_transfer2 env. We build a clean
# image from the CUDA base and reuse only the Lilypad plumbing (Ray + lilypad-py +
# boto), mirroring wfm_inference/lilypad_entrypoint.py.
#
# Dependency recipe proven in applied3 PR #62714 (which built fine; only its
# bazel/wheel integration failed — this image avoids that path entirely).
#
# Weights are NOT baked in. They are downloaded at runtime from OCI by
# wfm_inference.dr_lilypad_entrypoint, keeping the image lean.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f docker/diffusion_renderer.Dockerfile \
#     -t us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_dr_inverse_v0.0.1 .

# cudnn-devel base => cuDNN headers + nvcc present, so TE's C++ extension links at
# build time without the dpkg header dance #62714 needed on the appliedgs base.
# ubuntu22.04 => python3.10 native (DiffusionRenderer requires 3.10).
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
# torch ext builds: target the cluster GPUs (A100=80, H100=90). sm_120 (RTX 5090)
# is intentionally omitted — TE/megatron don't ship sm_120 kernels, so local runs
# stop at the kernel wall. Final verification is on A100/H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV PIP_NO_CACHE_DIR=1

# DiffusionRenderer pinned commit (package: cosmos_predict1).
ARG DR_COMMIT=0f3e2dc435032ecbad654c2fc2153df85384b138

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git git-lfs ffmpeg wget curl ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python3.10 -m pip install --upgrade pip setuptools wheel packaging pybind11

WORKDIR /workspace

# --- 1. Torch first (TE compiles against it) -----------------------------------
RUN python3.10 -m pip install \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# numpy must stay <2 for cosmos_predict1.
RUN python3.10 -m pip install "numpy==1.26.4"

# --- 2. transformer_engine 1.12.0 (cosmos_predict1 imports its private _RMSNorm) -
# --no-build-isolation so it sees the torch installed above; cuDNN headers come
# from the -devel base.
RUN python3.10 -m pip install --no-build-isolation "transformer_engine[pytorch]==1.12.0"

# Make TE's bundled cuDNN libs discoverable at runtime.
RUN CUDNN_LIB=$(python3.10 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')") \
    && echo "$CUDNN_LIB" > /etc/ld.so.conf.d/nvidia-cudnn-te.conf && ldconfig

# --- 3. megatron-core (its metadata pins a conflicting TE; --no-deps avoids it) -
RUN python3.10 -m pip install --no-deps "megatron-core==0.10.0"

# --- 4. DiffusionRenderer source (configs loaded relative to repo root) ---------
RUN git clone https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer.git /cosmos-dr \
    && cd /cosmos-dr && git checkout ${DR_COMMIT} \
    && python3.10 -m pip install --no-deps /cosmos-dr

# --- 5. cosmos_predict1 Python deps (pinned, consistent set from the DR repo) ---
# Using the vendored requirements (not a hand-picked subset) keeps the env exactly
# at cosmos's tested pins — notably numpy==1.26.4 (a latest opencv/imageio would
# otherwise drag numpy to 2.x and break ABI for compiled extensions).
COPY docker/dr_requirements.txt /tmp/dr_requirements.txt
RUN python3.10 -m pip install -r /tmp/dr_requirements.txt

# --- 6. Lilypad plumbing (mirrors ../Dockerfile) -------------------------------
# Ray is Applied-internal; click 8.2.1 avoids the Ray deepcopy crash on 8.3.x.
RUN python3.10 -m pip install "ray[default]==2.50.1.7" \
        --extra-index-url https://ursa.pypi.applied.dev/simple \
    && python3.10 -m pip install "click==8.2.1" \
    && python3.10 -m pip install "lilypad-py==2.27.0" \
        --extra-index-url https://ursa.pypi.applied.dev/simple \
    && python3.10 -m pip install "boto3" "botocore"

# Re-assert numpy after Ray/lilypad resolution so nothing dragged it to 2.x.
# cosmos_predict1 + opencv 4.10 require numpy 1.x; this guarantees a consistent ABI.
RUN python3.10 -m pip install "numpy==1.26.4" "einops==0.8.1"

# --- 7. Project code -----------------------------------------------------------
# Only the inference glue is needed; cosmos_transfer2 source is deliberately absent.
COPY wfm_inference /workspace/wfm_inference
ENV PYTHONPATH=/workspace

# NOTE: no build-time `import transformer_engine.pytorch` check — it initializes a
# CUDA context at import, which requires a live GPU (absent during docker build).
# Verification is the smoke test (wfm_inference/dr_smoke_inference.py) on a GPU.

CMD ["/bin/bash"]
