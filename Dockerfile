FROM vllm/vllm-openai:v0.17.1

ARG MODEL_NAME=rednote-hilab/dots.mocr
ARG MODEL_REVISION=f5a115b
ARG TOKENIZER_NAME=
ARG TOKENIZER_REVISION=
ARG BASE_PATH=/models

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    BASE_PATH=${BASE_PATH} \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_REVISION=${MODEL_REVISION} \
    TOKENIZER_NAME=${TOKENIZER_NAME} \
    TOKENIZER_REVISION=${TOKENIZER_REVISION} \
    HF_HOME=${BASE_PATH}/huggingface-cache \
    HF_DATASETS_CACHE=${BASE_PATH}/huggingface-cache/datasets \
    HUGGINGFACE_HUB_CACHE=${BASE_PATH}/huggingface-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    SERVE_MODEL_NAME=model \
    OPENAI_SERVED_MODEL_NAME_OVERRIDE=model \
    VLLM_PORT=8000 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.9

RUN python3 -m pip install -U \
    "huggingface_hub>=0.30,<1" \
    "runpod==1.9.0" \
    "requests>=2.32,<3"

COPY download_model.py /download_model.py
RUN mkdir -p "${BASE_PATH}" \
 && python3 /download_model.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY handler.py /handler.py

EXPOSE 8000

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
