#!/usr/bin/env python3
"""
Minimal RunPod test client for the dots.mocr serverless endpoint.

Examples:

1. Test with a public PDF URL:
   python3 test_endpoint.py \
     --endpoint-id <ENDPOINT_ID> \
     --api-key <RUNPOD_API_KEY> \
     --source-url https://arxiv.org/pdf/1706.03762.pdf

2. Test with a local file encoded as a data URL:
   python3 test_endpoint.py \
     --endpoint-id <ENDPOINT_ID> \
     --api-key <RUNPOD_API_KEY> \
     --file /absolute/path/to/document.pdf

3. Run async and poll for completion:
   python3 test_endpoint.py \
     --endpoint-id <ENDPOINT_ID> \
     --api-key <RUNPOD_API_KEY> \
     --file /absolute/path/to/page.png \
     --async
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


API_BASE = "https://api.runpod.ai/v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Test a dots.mocr RunPod endpoint.")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("RUNPOD_API_KEY"),
        help="RunPod API key. Defaults to RUNPOD_API_KEY.",
    )
    parser.add_argument(
        "--source-url",
        help="Public HTTP(S) URL of an input file/image to parse.",
    )
    parser.add_argument(
        "--file",
        help="Local file to base64-encode as a data URL before sending.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="prompt_layout_all_en",
        help="dots.mocr prompt mode. Default: prompt_layout_all_en.",
    )
    parser.add_argument(
        "--response-mode",
        default="auto",
        choices=["auto", "inline", "manifest"],
        help="Parser response mode. Default: auto.",
    )
    parser.add_argument(
        "--include-layout-image",
        action="store_true",
        help="Inline layout preview image as base64 when response mode permits it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Parser DPI. Default: 200.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=32,
        help="Parser thread count. Default: 32.",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use /run and poll status instead of /runsync.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval for async mode in seconds. Default: 5.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Overall timeout for async polling in seconds. Default: 900.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the JSON response.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required unless RUNPOD_API_KEY is set.")
    if bool(args.source_url) == bool(args.file):
        parser.error("Provide exactly one of --source-url or --file.")
    return args


def make_data_url(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


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


def get_json(url, api_key):
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return load_json(request)


def load_json(request):
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def build_payload(args):
    source = args.source_url if args.source_url else make_data_url(args.file)
    return {
        "input": {
            "source": source,
            "prompt_mode": args.prompt_mode,
            "response_mode": args.response_mode,
            "include_layout_image": args.include_layout_image,
            "dpi": args.dpi,
            "num_thread": args.num_thread,
        }
    }


def run_sync(args, payload):
    url = f"{API_BASE}/{args.endpoint_id}/runsync"
    return post_json(url, args.api_key, payload)


def run_async(args, payload):
    submit_url = f"{API_BASE}/{args.endpoint_id}/run"
    submitted = post_json(submit_url, args.api_key, payload)
    job_id = submitted.get("id")
    if not job_id:
        raise RuntimeError(f"Async submit did not return a job id: {submitted}")

    status_url = f"{API_BASE}/{args.endpoint_id}/status/{job_id}"
    start = time.time()

    while True:
        if time.time() - start > args.timeout:
            raise TimeoutError(f"Timed out waiting for async job {job_id}.")

        status = get_json(status_url, args.api_key)
        state = (status.get("status") or "").upper()
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return status
        time.sleep(args.poll_interval)


def main():
    args = parse_args()
    payload = build_payload(args)
    result = run_async(args, payload) if args.async_mode else run_sync(args, payload)
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
