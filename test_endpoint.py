#!/usr/bin/env python3
"""
Minimal RunPod test client for a plain vLLM dots.mocr endpoint.

Examples:

1. List models:
   python3 test_endpoint.py --list-models

2. Test with a public image URL:
   python3 test_endpoint.py \
     --source-url https://example.com/page.png

3. Test with a local image file encoded as a data URL:
   python3 test_endpoint.py \
     --file /absolute/path/to/page.png
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


API_BASE = "https://api.runpod.ai/v2"

# Optional local defaults. Set these if you want to run the script without
# passing --endpoint-id / --api-key every time.
DEFAULT_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
DEFAULT_RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
DEFAULT_MODEL = (
    os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
    or os.getenv("SERVE_MODEL_NAME")
    or "model"
)
DEFAULT_PROMPT = "Extract the text content from this image."


def parse_args():
    parser = argparse.ArgumentParser(description="Test a dots.mocr RunPod vLLM endpoint.")
    parser.add_argument(
        "--endpoint-id",
        default=DEFAULT_ENDPOINT_ID,
        help="RunPod endpoint ID.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_RUNPOD_API_KEY or os.getenv("RUNPOD_API_KEY"),
        help="RunPod API key. Defaults to DEFAULT_RUNPOD_API_KEY or RUNPOD_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to send in the OpenAI-compatible request.",
    )
    parser.add_argument(
        "--source-url",
        help="Public HTTP(S) URL of an input image.",
    )
    parser.add_argument(
        "--file",
        help="Local image file to base64-encode as a data URL before sending.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send with the image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Default: 0.1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value. Default: 0.9.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens. Default: 2048.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models exposed by the endpoint and exit.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the JSON response.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "--api-key is required unless DEFAULT_RUNPOD_API_KEY or RUNPOD_API_KEY is set."
        )
    if not args.endpoint_id:
        parser.error("--endpoint-id is required unless DEFAULT_ENDPOINT_ID is set.")
    if not args.list_models and bool(args.source_url) == bool(args.file):
        parser.error("Provide exactly one of --source-url or --file unless --list-models is set.")
    return args


def make_data_url(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def get_base_url(endpoint_id):
    return f"{API_BASE}/{endpoint_id}/openai/v1"


def load_json(request):
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def get_json(url, api_key):
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return load_json(request)


def post_json(url, api_key, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return load_json(request)


def build_chat_payload(args):
    image_url = args.source_url if args.source_url else make_data_url(args.file)
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": f"<|img|><|imgpad|><|endofimg|>{args.prompt}",
                    },
                ],
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
    }


def list_models(args):
    return get_json(f"{get_base_url(args.endpoint_id)}/models", args.api_key)


def run_chat_completion(args):
    payload = build_chat_payload(args)
    return post_json(
        f"{get_base_url(args.endpoint_id)}/chat/completions",
        args.api_key,
        payload,
    )


def main():
    args = parse_args()
    result = list_models(args) if args.list_models else run_chat_completion(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
