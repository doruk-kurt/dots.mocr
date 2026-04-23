# dots.mocr RunPod Serverless

RunPod Serverless worker for serving [`rednote-hilab/dots.mocr`](https://huggingface.co/rednote-hilab/dots.mocr) through plain `vLLM`.

This repo is intentionally limited to the OpenAI-compatible `vLLM` path:

- serve `rednote-hilab/dots.mocr` through `vLLM`
- bake the pinned model snapshot into the image at build time
- keep notebook / SDK usage unchanged through RunPod's `openai/v1` URL

There is no custom `dots.mocr` parser pipeline in this version of the worker.

## What stays the same

If you already call the endpoint like this, you do not need to change your client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<RUNPOD_API_KEY>",
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
```

Use `model="model"` by default, or call `client.models.list()` and use the served model name returned by the endpoint.

## Build behavior

The Docker build downloads the pinned model snapshot into an internal Hugging Face cache under `BASE_PATH`. At runtime, the container starts `vLLM` in offline mode so worker startup uses the baked snapshot instead of downloading weights again.

Pinned defaults:

- base image: `vllm/vllm-openai:v0.17.1`
- model: `rednote-hilab/dots.mocr`
- revision: `f5a115b`

Useful build args:

- `MODEL_NAME`: default `rednote-hilab/dots.mocr`
- `MODEL_REVISION`: default `f5a115b`
- `BASE_PATH`: default `/models`
- `TOKENIZER_NAME`
- `TOKENIZER_REVISION`

Useful runtime env vars:

- `OPENAI_SERVED_MODEL_NAME_OVERRIDE`: default `model`
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`
- `CHAT_TEMPLATE_CONTENT_FORMAT`
- `TRUST_REMOTE_CODE`
- `VLLM_STARTUP_TIMEOUT`
- `VLLM_REQUEST_TIMEOUT`

## RunPod deployment

1. Put this folder's contents at the root of a GitHub repo.
2. In RunPod Serverless, choose **Build from GitHub repo**.
3. Point it to that repo.
4. No custom start command is needed. The `Dockerfile` starts the worker.
5. Use the endpoint's `openai/v1` route from notebooks, apps, or the local test script.

## Native API

The worker also keeps a minimal RunPod native handler for queue-based requests:

- `{"input": {"messages": [...]}}` proxies to `/v1/chat/completions`
- `{"input": {"prompt": "..."}}` proxies to `/v1/completions`
- `{"input": {"list_models": true}}` proxies to `/v1/models`

This is only a thin wrapper around local `vLLM`. It does not implement parser-specific behavior.
