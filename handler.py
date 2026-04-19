"""
RunPod Serverless worker for dots.mocr via vLLM.

This worker starts a local vLLM server with the official dots.mocr settings,
then exposes two request modes through RunPod's queue API:

1. Parser mode: {"input": {"url": "...", "prompt_mode": "prompt_layout_all_en"}}
2. Raw chat mode: {"input": {"messages": [...], "model": "model"}}

Parser mode is size-aware:

- small jobs return inline markdown / JSON for convenience
- larger jobs return a manifest of generated artifacts instead of forcing the
  entire parser output into the response body
- if RunPod bucket credentials are configured, artifacts can be uploaded and
  returned as durable URLs
"""

import base64
import inspect
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlparse

import requests
import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")


def _as_bool(value, default=False):
    """Coerce common JSON/string boolean values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


MODEL_NAME = os.getenv("MODEL_NAME", "rednote-hilab/dots.mocr")
MODEL_REVISION = os.getenv("MODEL_REVISION", "f5a115b")
SERVE_MODEL_NAME = os.getenv("SERVE_MODEL_NAME", "model")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_URL = f"http://localhost:{VLLM_PORT}"
TENSOR_PARALLEL_SIZE = os.getenv("TENSOR_PARALLEL_SIZE", "1")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.9")
INLINE_RESPONSE_MAX_BYTES = int(os.getenv("INLINE_RESPONSE_MAX_BYTES", "1500000"))
INLINE_RESPONSE_MAX_PAGES = int(os.getenv("INLINE_RESPONSE_MAX_PAGES", "2"))
RESULTS_BASE_DIR = os.getenv("RESULTS_BASE_DIR", "/tmp/dots_mocr_results")
UPLOAD_ARTIFACTS_BY_DEFAULT = _as_bool(
    os.getenv("UPLOAD_ARTIFACTS_BY_DEFAULT"),
    default=True,
)
KEEP_LOCAL_RESULTS_BY_DEFAULT = _as_bool(
    os.getenv("KEEP_LOCAL_RESULTS_BY_DEFAULT"),
    default=False,
)

ARTIFACT_KEY_ALIASES = {
    "md_content_path": "markdown",
    "md_content_nohf_path": "markdown_nohf",
    "layout_info_path": "layout_json",
    "layout_image_path": "layout_image",
}

VLLM_PROCESS = None
VLLM_STARTUP_ERROR = None
PARSER_RUNTIME = None


def _truncate_text(value, limit=1000):
    """Return a compact single-line string for logs and error payloads."""
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _set_vllm_startup_error(message):
    global VLLM_STARTUP_ERROR
    VLLM_STARTUP_ERROR = message


def _vllm_error_payload(job_id, message, **extra):
    payload = {
        "error": message,
        "job_id": job_id,
        "vllm_url": f"{VLLM_URL}/v1/chat/completions",
    }
    if VLLM_STARTUP_ERROR:
        payload["startup_error"] = VLLM_STARTUP_ERROR
    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is not None:
        payload["vllm_exit_code"] = VLLM_PROCESS.returncode
    payload.update({k: v for k, v in extra.items() if v not in {None, ""}})
    return payload


def stream_output(pipe):
    """Stream vLLM logs into worker logs."""
    try:
        for line in pipe:
            line = line.strip()
            if line:
                log.info("[vllm] %s", line)
    except Exception as exc:
        log.exception("Error while streaming vLLM logs: %s", exc)
    finally:
        pipe.close()


def start_vllm():
    """Start the official dots.mocr vLLM server in the background."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        TENSOR_PARALLEL_SIZE,
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--chat-template-content-format",
        "string",
        "--served-model-name",
        SERVE_MODEL_NAME,
        "--trust-remote-code",
    ]
    if MODEL_REVISION:
        cmd.extend(
            [
                "--revision",
                MODEL_REVISION,
                "--code-revision",
                MODEL_REVISION,
                "--tokenizer-revision",
                MODEL_REVISION,
            ]
        )

    log.info("Starting vLLM: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        thread.start()

    return process


def wait_for_vllm(process, timeout=600):
    """Wait for the local vLLM API to become ready."""
    start = time.time()
    last_error = ""
    while time.time() - start < timeout:
        if process.poll() is not None:
            message = f"vLLM exited during startup with code {process.returncode}."
            _set_vllm_startup_error(message)
            raise RuntimeError(message)

        try:
            health = requests.get(f"{VLLM_URL}/health", timeout=2)
            if health.status_code == 200:
                models = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
                if models.status_code == 200:
                    log.info("vLLM is ready")
                    return True
                last_error = (
                    f"/v1/models returned {models.status_code}: "
                    f"{_truncate_text(models.text)}"
                )
            else:
                last_error = (
                    f"/health returned {health.status_code}: "
                    f"{_truncate_text(health.text)}"
                )
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)

    message = f"vLLM did not become ready within {timeout}s"
    if last_error:
        message = f"{message}. Last readiness check: {last_error}"
    _set_vllm_startup_error(message)
    raise TimeoutError(message)


def _guess_suffix(source, content_type=None):
    """Guess a file suffix from a URL/path and optional content type."""
    if content_type:
        guess = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guess:
            return guess

    parsed = urlparse(source)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    return ".bin"


def _write_temp_bytes(payload, suffix):
    """Persist bytes to a temp file and return its path."""
    temp_dir = tempfile.mkdtemp(prefix="dots_mocr_")
    path = os.path.join(temp_dir, f"input{suffix}")
    with open(path, "wb") as handle:
        handle.write(payload)
    return path, temp_dir


def _materialize_source(source):
    """
    Convert a supported source reference into a local file path.
    Returns (path, cleanup_dir_or_none, original_source).
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Input source must be a non-empty string.")

    source = source.strip()
    parsed = urlparse(source)

    if parsed.scheme in {"http", "https"}:
        response = requests.get(source, timeout=120)
        response.raise_for_status()
        suffix = _guess_suffix(source, response.headers.get("Content-Type"))
        path, cleanup_dir = _write_temp_bytes(response.content, suffix)
        return path, cleanup_dir, source

    if parsed.scheme == "data":
        header, sep, payload = source.partition(",")
        if not sep:
            raise ValueError("Malformed data URL: missing comma separator.")
        if ";base64" in header:
            data = base64.b64decode(payload, validate=True)
        else:
            data = unquote_to_bytes(payload)
        mime = header[5:].split(";")[0] if header.startswith("data:") else ""
        suffix = mimetypes.guess_extension(mime) or ".bin"
        path, cleanup_dir = _write_temp_bytes(data, suffix)
        return path, cleanup_dir, source

    if parsed.scheme == "file":
        return parsed.path, None, source

    if parsed.scheme == "" and source.startswith("/"):
        return source, None, source

    raise ValueError(
        "Unsupported input source. Use an http(s) URL, data URL, file:// URL, or absolute path."
    )


def _normalize_response_mode(value):
    mode = (value or "auto").strip().lower()
    if mode not in {"auto", "inline", "manifest"}:
        raise ValueError("response_mode must be one of: auto, inline, manifest.")
    return mode


def _extract_parser_job(job_input):
    """Extract parser-mode arguments from the RunPod input payload."""
    if isinstance(job_input, str):
        job_input = {"source": job_input}

    if not isinstance(job_input, dict):
        return None

    if "messages" in job_input:
        return None

    source = (
        job_input.get("source")
        or job_input.get("url")
        or job_input.get("file")
        or job_input.get("image")
        or job_input.get("pdf")
        or job_input.get("path")
    )
    if not source:
        return None

    prompt_mode = job_input.get("prompt_mode") or job_input.get("prompt")
    if not prompt_mode:
        prompt_mode = "prompt_layout_all_en"

    return {
        "source": source,
        "prompt_mode": prompt_mode,
        "bbox": job_input.get("bbox"),
        "dpi": int(job_input.get("dpi", 200)),
        "num_thread": int(job_input.get("num_thread", 64)),
        "temperature": float(job_input.get("temperature", 0.1)),
        "top_p": float(job_input.get("top_p", 0.9)),
        "max_completion_tokens": int(job_input.get("max_completion_tokens", 16384)),
        "fitz_preprocess": _as_bool(job_input.get("fitz_preprocess"), default=True),
        "include_layout_image": _as_bool(
            job_input.get("include_layout_image"),
            default=False,
        ),
        "response_mode": _normalize_response_mode(job_input.get("response_mode")),
        "upload_artifacts": job_input.get("upload_artifacts"),
        "keep_results": job_input.get("keep_results"),
        "custom_prompt": job_input.get("custom_prompt"),
    }


def _load_parser_runtime():
    """Load the official parser package, preferring dots_mocr."""
    global PARSER_RUNTIME
    if PARSER_RUNTIME is not None:
        return PARSER_RUNTIME

    import_errors = []
    candidates = [
        ("dots_mocr.parser", "DotsMOCRParser", "dots_mocr.utils.prompts"),
        ("dots_ocr.parser", "DotsOCRParser", "dots_ocr.utils.prompts"),
    ]

    for parser_module_name, parser_class_name, prompt_module_name in candidates:
        try:
            parser_module = __import__(parser_module_name, fromlist=[parser_class_name])
            prompt_module = __import__(
                prompt_module_name,
                fromlist=["dict_promptmode_to_prompt"],
            )
            PARSER_RUNTIME = {
                "parser_class": getattr(parser_module, parser_class_name),
                "prompt_modes": getattr(prompt_module, "dict_promptmode_to_prompt"),
                "package": parser_module_name.split(".")[0],
            }
            return PARSER_RUNTIME
        except ImportError as exc:
            import_errors.append(f"{parser_module_name}: {exc}")

    raise ImportError(
        "Unable to import dots.mocr parser package. "
        + " | ".join(import_errors)
    )


def _make_output_dir(job_id):
    safe_job_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(job_id)
    )
    output_dir = os.path.join(RESULTS_BASE_DIR, safe_job_id)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _artifact_alias(path_key):
    return ARTIFACT_KEY_ALIASES.get(path_key, path_key[: -len("_path")])


def _artifact_kind(path):
    mime_type = mimetypes.guess_type(path)[0] or ""
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".md", ".txt", ".svg", ".xml", ".html", ".csv"}:
        return "text"
    if mime_type.startswith("image/"):
        return "image"
    return "binary"


def _read_text_if_exists(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return None


def _read_json_if_exists(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _read_base64_if_exists(path):
    if path and os.path.isfile(path):
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    return None


def _collect_parser_pages(results, output_dir):
    """Build a response-friendly manifest for each parser page."""
    pages = []
    artifact_records = []

    for raw_page in results:
        page_payload = {}
        artifacts = {}

        for key, value in raw_page.items():
            if key == "file_path":
                continue

            if key.endswith("_path") and value:
                abs_path = os.path.abspath(value)
                if not os.path.isfile(abs_path):
                    continue

                artifact_payload = {
                    "relative_path": os.path.relpath(abs_path, start=output_dir),
                    "size_bytes": os.path.getsize(abs_path),
                    "kind": _artifact_kind(abs_path),
                    "mime_type": mimetypes.guess_type(abs_path)[0]
                    or "application/octet-stream",
                }
                alias = _artifact_alias(key)
                artifacts[alias] = artifact_payload
                artifact_records.append(
                    {
                        "alias": alias,
                        "abs_path": abs_path,
                        "relative_path": artifact_payload["relative_path"],
                        "kind": artifact_payload["kind"],
                        "size_bytes": artifact_payload["size_bytes"],
                        "artifact_payload": artifact_payload,
                        "page_payload": page_payload,
                    }
                )
                continue

            page_payload[key] = value

        page_payload["filtered"] = bool(page_payload.get("filtered", False))
        if artifacts:
            page_payload["artifacts"] = artifacts
        pages.append(page_payload)

    return pages, artifact_records


def _bucket_env_configured():
    required = (
        "BUCKET_ENDPOINT_URL",
        "BUCKET_ACCESS_KEY_ID",
        "BUCKET_SECRET_ACCESS_KEY",
    )
    return all(os.getenv(name) for name in required)


def _resolve_upload_artifacts(parser_job):
    requested = parser_job.get("upload_artifacts")
    if requested is None:
        return UPLOAD_ARTIFACTS_BY_DEFAULT and _bucket_env_configured()

    enabled = _as_bool(requested)
    if enabled and not _bucket_env_configured():
        raise ValueError(
            "upload_artifacts=true requires BUCKET_ENDPOINT_URL, "
            "BUCKET_ACCESS_KEY_ID, and BUCKET_SECRET_ACCESS_KEY."
        )
    return enabled


def _resolve_keep_results(parser_job, response_mode, upload_artifacts):
    requested = parser_job.get("keep_results")
    if requested is not None:
        return _as_bool(requested)
    if response_mode == "manifest" and not upload_artifacts:
        return True
    return KEEP_LOCAL_RESULTS_BY_DEFAULT


def _estimate_inline_bytes(artifact_records, include_layout_image):
    total = 0
    for record in artifact_records:
        if record["kind"] == "binary":
            continue
        if record["kind"] == "image" and not include_layout_image:
            continue
        total += record["size_bytes"]
    return total


def _resolve_response_mode(parser_job, artifact_records, page_count):
    warnings = []
    requested_mode = parser_job["response_mode"]
    if requested_mode != "auto":
        return requested_mode, warnings

    estimated_bytes = _estimate_inline_bytes(
        artifact_records,
        include_layout_image=parser_job["include_layout_image"],
    )

    if page_count > INLINE_RESPONSE_MAX_PAGES:
        warnings.append(
            "Returning a manifest instead of inline page content because "
            f"page_count={page_count} exceeds INLINE_RESPONSE_MAX_PAGES="
            f"{INLINE_RESPONSE_MAX_PAGES}."
        )
        return "manifest", warnings

    if estimated_bytes > INLINE_RESPONSE_MAX_BYTES:
        warnings.append(
            "Returning a manifest instead of inline page content because the "
            f"estimated response size ({estimated_bytes} bytes) exceeds "
            f"INLINE_RESPONSE_MAX_BYTES={INLINE_RESPONSE_MAX_BYTES}."
        )
        return "manifest", warnings

    return "inline", warnings


def _add_local_paths(artifact_records):
    for record in artifact_records:
        record["artifact_payload"]["local_path"] = record["abs_path"]


def _attach_inline_content(artifact_records, include_layout_image):
    for record in artifact_records:
        content = None

        if record["kind"] == "json":
            content = _read_json_if_exists(record["abs_path"])
        elif record["kind"] == "text":
            content = _read_text_if_exists(record["abs_path"])
        elif record["kind"] == "image":
            if not include_layout_image:
                continue
            content = _read_base64_if_exists(record["abs_path"])
            if content is not None:
                record["artifact_payload"]["encoding"] = "base64"
                record["page_payload"]["layout_image_base64"] = content

        if content is None:
            continue

        record["artifact_payload"]["content"] = content
        if record["alias"] in {"markdown", "markdown_nohf", "layout_json"}:
            record["page_payload"][record["alias"]] = content


def _upload_artifacts(job_id, artifact_records):
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket

    uploaded_urls = {}
    seen_paths = set()
    safe_job_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(job_id)
    )

    for record in artifact_records:
        relative_path = record["relative_path"]
        if relative_path in seen_paths:
            continue
        seen_paths.add(relative_path)
        object_name = f"dots-mocr/{safe_job_id}/{relative_path.replace(os.sep, '/')}"
        uploaded_urls[relative_path] = upload_file_to_bucket(object_name, record["abs_path"])

    for record in artifact_records:
        url = uploaded_urls.get(record["relative_path"])
        if url:
            record["artifact_payload"]["url"] = url

    return uploaded_urls


def _parse_with_dots_parser(parser_job, job_id):
    """Run the official dots.mocr parser layer against the local vLLM server."""
    runtime = _load_parser_runtime()
    parser_class = runtime["parser_class"]
    prompt_modes = runtime["prompt_modes"]
    prompt_mode = parser_job["prompt_mode"]
    keep_results = False
    output_dir = None

    if prompt_mode not in prompt_modes:
        raise ValueError(
            f"Unsupported prompt_mode '{prompt_mode}'. Supported values: {sorted(prompt_modes)}"
        )

    bbox = parser_job.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be a list of four integers: [x1, y1, x2, y2].")
        bbox = [int(value) for value in bbox]

    source_path, cleanup_dir, original_source = _materialize_source(parser_job["source"])

    try:
        output_dir = _make_output_dir(job_id)
        parser = parser_class(
            protocol="http",
            ip="localhost",
            port=VLLM_PORT,
            model_name=SERVE_MODEL_NAME,
            temperature=parser_job["temperature"],
            top_p=parser_job["top_p"],
            max_completion_tokens=parser_job["max_completion_tokens"],
            num_thread=parser_job["num_thread"],
            dpi=parser_job["dpi"],
            output_dir=output_dir,
            use_hf=False,
        )
        log.info(
            "Job %s: parsing source with dots.mocr (prompt_mode=%s)",
            job_id,
            prompt_mode,
        )

        parse_kwargs = {
            "input_path": source_path,
            "output_dir": output_dir,
            "prompt_mode": prompt_mode,
            "bbox": bbox,
            "fitz_preprocess": parser_job["fitz_preprocess"],
        }
        if parser_job.get("custom_prompt") is not None:
            signature = inspect.signature(parser.parse_file)
            if "custom_prompt" in signature.parameters:
                parse_kwargs["custom_prompt"] = parser_job["custom_prompt"]

        results = parser.parse_file(**parse_kwargs)
        pages, artifact_records = _collect_parser_pages(results, output_dir)
        response_mode, warnings = _resolve_response_mode(
            parser_job,
            artifact_records,
            page_count=len(pages),
        )

        upload_artifacts = _resolve_upload_artifacts(parser_job)
        keep_results = _resolve_keep_results(
            parser_job,
            response_mode=response_mode,
            upload_artifacts=upload_artifacts,
        )
        if keep_results:
            _add_local_paths(artifact_records)

        upload_error = None
        uploaded_urls = {}
        if upload_artifacts:
            try:
                uploaded_urls = _upload_artifacts(job_id, artifact_records)
            except Exception as exc:
                upload_error = str(exc)
                warnings.append(
                    "Artifact upload failed; returning local artifact references instead. "
                    f"upload_error={_truncate_text(exc, 500)}"
                )
                if not keep_results:
                    keep_results = True
                    _add_local_paths(artifact_records)

        if response_mode == "inline":
            _attach_inline_content(
                artifact_records,
                include_layout_image=parser_job["include_layout_image"],
            )
        elif not uploaded_urls and not keep_results:
            warnings.append(
                "The response contains only artifact metadata. Configure RunPod bucket "
                "credentials or set keep_results=true if you need durable access to the "
                "full parser outputs for larger jobs."
            )

        response = {
            "mode": "parser",
            "source": original_source,
            "prompt_mode": prompt_mode,
            "page_count": len(pages),
            "response_mode": response_mode,
            "parser_package": runtime["package"],
            "pages": pages,
            "artifacts": {
                "bucket_configured": _bucket_env_configured(),
                "uploaded_file_count": len(uploaded_urls),
                "result_dir": output_dir if keep_results else None,
                "inline_limits": {
                    "max_bytes": INLINE_RESPONSE_MAX_BYTES,
                    "max_pages": INLINE_RESPONSE_MAX_PAGES,
                },
            },
        }
        if warnings:
            response["warnings"] = warnings
        if upload_error:
            response["upload_error"] = upload_error
        if len(pages) == 1:
            first_page = pages[0]
            if "markdown" in first_page:
                response["markdown"] = first_page["markdown"]
            if "layout_json" in first_page:
                response["layout_json"] = first_page["layout_json"]
            if "layout_image_base64" in first_page:
                response["layout_image_base64"] = first_page["layout_image_base64"]
        return response
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        if output_dir and not keep_results:
            shutil.rmtree(output_dir, ignore_errors=True)


def _proxy_chat_completion(job_input, job_id):
    """Forward an OpenAI-style chat completion payload to local vLLM."""
    payload = dict(job_input)
    if "model" not in payload:
        payload["model"] = SERVE_MODEL_NAME
    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    if response.status_code != 200:
        body = _truncate_text(response.text, 4000)
        log.error(
            "Job %s: vLLM returned %s with body: %s",
            job_id,
            response.status_code,
            body,
        )
        return _vllm_error_payload(
            job_id,
            f"Local vLLM server returned HTTP {response.status_code}.",
            http_status=response.status_code,
            response_body=body,
        )
    return response.json()


def handler(job):
    """RunPod queue handler."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input")
    log.info("Job %s: received request", job_id)

    if VLLM_STARTUP_ERROR:
        log.error("Job %s: refusing request because vLLM is unavailable", job_id)
        return _vllm_error_payload(job_id, "Local vLLM server failed to initialize.")

    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is not None:
        message = f"Local vLLM process exited with code {VLLM_PROCESS.returncode}."
        _set_vllm_startup_error(message)
        log.error("Job %s: %s", job_id, message)
        return _vllm_error_payload(job_id, "Local vLLM process is not running.")

    parser_job = _extract_parser_job(job_input)
    if parser_job is not None:
        try:
            return _parse_with_dots_parser(parser_job, job_id)
        except Exception as exc:
            log.exception("Job %s: dots.mocr parser failed: %s", job_id, exc)
            return {"error": str(exc), "job_id": job_id, "mode": "parser"}

    if not isinstance(job_input, dict):
        return {
            "error": (
                "Invalid request format. Use a parser payload such as "
                "{'input': {'url': 'https://...', 'prompt_mode': 'prompt_layout_all_en'}} "
                "or an OpenAI-style chat completions payload wrapped in 'input'."
            )
        }

    if "messages" not in job_input:
        return {
            "error": (
                "Input must contain either a parser source field "
                "('source', 'url', 'file', 'image', 'pdf', 'path') or a "
                "'messages' field for raw chat completions."
            )
        }

    try:
        return _proxy_chat_completion(job_input, job_id)
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" | response_body={_truncate_text(exc.response.text, 4000)}"
        log.error("Job %s: failed - %s%s", job_id, exc, detail)
        return _vllm_error_payload(
            job_id,
            "Request to local vLLM server failed.",
            request_error=str(exc),
            response_body=_truncate_text(
                getattr(getattr(exc, "response", None), "text", ""),
                4000,
            ),
        )


if __name__ == "__main__":
    VLLM_PROCESS = start_vllm()
    try:
        wait_for_vllm(VLLM_PROCESS)
    except Exception as exc:
        _set_vllm_startup_error(str(exc))
        log.error("Worker startup aborted: %s", exc)
        raise
    runpod.serverless.start({"handler": handler})
