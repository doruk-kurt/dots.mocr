FROM vllm/vllm-openai:v0.17.1

ARG DOTS_MOCR_REF=23f3e5612fb8066d4034d5ecfc8f33a9243533eb
ARG MODEL_REVISION=f5a115b

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_NAME=rednote-hilab/dots.mocr \
    MODEL_REVISION=${MODEL_REVISION} \
    SERVE_MODEL_NAME=model \
    VLLM_PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.9 \
    INLINE_RESPONSE_MAX_BYTES=1500000 \
    INLINE_RESPONSE_MAX_PAGES=2 \
    RESULTS_BASE_DIR=/tmp/dots_mocr_results \
    UPLOAD_ARTIFACTS_BY_DEFAULT=1 \
    KEEP_LOCAL_RESULTS_BY_DEFAULT=0

# git is needed for the pinned upstream parser install.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install the official dots.mocr parser layer plus the RunPod worker runtime.
RUN python3 -m pip install -U \
    "git+https://github.com/rednote-hilab/dots.mocr.git@${DOTS_MOCR_REF}" \
    "runpod==1.9.0" \
    "requests>=2.32,<3"

# Pre-download a pinned model revision so cold starts do not hit Hugging Face.
RUN python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_NAME'], revision=os.environ['MODEL_REVISION'])"
ENV HF_HUB_OFFLINE=1

COPY handler.py /handler.py

EXPOSE 8000

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
