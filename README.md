# dots.mocr RunPod Serverless

RunPod Serverless worker for [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr).

This worker follows the current official `dots.mocr` deployment shape:

- serve `rednote-hilab/dots.mocr` through `vLLM`
- run the official `dots.mocr` parser layer against the local `vLLM` server for PDF/image parsing

The image is pinned for reproducibility:

- `vllm/vllm-openai:v0.17.1`
- `rednote-hilab/dots.mocr` repo at commit `23f3e5612fb8066d4034d5ecfc8f33a9243533eb`
- model revision `f5a115b`

## Why a separate repo

RunPod GitHub builds expect the deployment files at the repository root. This folder is meant to become the root of a dedicated deployment repo or branch.

The files that matter are:

- `Dockerfile`
- `handler.py`

## What it exposes

The worker supports two request modes.

### 1. Parser mode

Best for the full `dots.mocr` document pipeline.

Input shape:

```json
{
  "input": {
    "url": "https://example.com/document.pdf",
    "prompt_mode": "prompt_layout_all_en"
  }
}
```

Supported source keys:

- `source`
- `url`
- `file`
- `image`
- `pdf`
- `path`

Supported prompt modes:

- `prompt_layout_all_en`
- `prompt_layout_only_en`
- `prompt_ocr`
- `prompt_grounding_ocr`
- `prompt_web_parsing`
- `prompt_scene_spotting`
- `prompt_image_to_svg`
- `prompt_general`

Optional parser fields:

- `bbox`: `[x1, y1, x2, y2]` for `prompt_grounding_ocr`
- `custom_prompt`: useful with `prompt_general`
- `dpi`: default `200`
- `num_thread`: default `64`
- `temperature`: default `0.1`
- `top_p`: default `0.9`
- `max_completion_tokens`: default `16384`
- `fitz_preprocess`: default `true`
- `include_layout_image`: default `false`
- `response_mode`: `auto`, `inline`, or `manifest` with default `auto`

Response behavior:

- `response_mode=auto`: small jobs return inline markdown / JSON; larger jobs return an artifact manifest
- `response_mode=inline`: always inline parser content into the response
- `response_mode=manifest`: always return artifact metadata instead of embedding content

This worker is stateless. Your backend should persist the returned parser output if you need durable storage.

Example:

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "url": "https://arxiv.org/pdf/1706.03762.pdf",
      "prompt_mode": "prompt_layout_all_en",
      "response_mode": "auto"
    }
  }'
```

Typical manifest response:

```json
{
  "mode": "parser",
  "source": "...",
  "prompt_mode": "prompt_layout_all_en",
  "page_count": 3,
  "response_mode": "manifest",
  "pages": [
    {
      "page_no": 0,
      "filtered": false,
      "artifacts": {
        "markdown": {
          "relative_path": "document/document_page_0.md",
          "size_bytes": 18234
        },
        "layout_json": {
          "relative_path": "document/document_page_0.json",
          "size_bytes": 94421
        }
      }
    }
  ]
}
```

### 2. Raw chat mode

Best if you want plain `vLLM` OpenAI-compatible chat completions.

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "model",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/page.png"}},
            {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>Extract the text content from this image."}
          ]
        }
      ]
    }
  }'
```

## RunPod deployment

1. Put this folder's contents at the root of a GitHub repo.
2. In RunPod Serverless, choose **Build from GitHub repo**.
3. Point it to that repo.
4. No custom start command is needed. The `Dockerfile` starts the worker.
5. Have your backend call this endpoint and persist the returned results if you need durable storage.

Useful optional runtime env vars:

- `INLINE_RESPONSE_MAX_BYTES`
- `INLINE_RESPONSE_MAX_PAGES`
- `RESULTS_BASE_DIR`

## Build-time notes

- `dots.mocr` is public, so no Hugging Face token is needed for the default build.
- The Dockerfile downloads model weights during the image build. RunPod endpoint env vars are runtime settings; they do not change that build step.
- If you need gated or private model access, build and push your own image from Docker Hub or another registry where you can control build secrets or build args.

## Notes

- This setup uses only `dots.mocr`. No additional layout model is required.
- The Docker image pre-downloads the pinned model revision and enables `HF_HUB_OFFLINE=1` for more predictable cold starts.
- The worker prefers the official `dots_mocr` package and falls back to `dots_ocr` only for backward compatibility.
- For larger documents, prefer `/run` over `/runsync` and let your backend poll the job result before saving it.
