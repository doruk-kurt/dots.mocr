"""Plain vLLM RunPod worker for a baked-in dots.mocr model."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

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
SERVE_MODEL_NAME = (
    os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
    or os.getenv("SERVE_MODEL_NAME", "model")
)
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_URL = f"http://localhost:{VLLM_PORT}"
TENSOR_PARALLEL_SIZE = os.getenv("TENSOR_PARALLEL_SIZE", "1")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.9")
CHAT_TEMPLATE_CONTENT_FORMAT = os.getenv("CHAT_TEMPLATE_CONTENT_FORMAT", "string")
TRUST_REMOTE_CODE = _as_bool(os.getenv("TRUST_REMOTE_CODE"), default=True)
VLLM_STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "600"))
VLLM_REQUEST_TIMEOUT = int(os.getenv("VLLM_REQUEST_TIMEOUT", "600"))

VLLM_PROCESS = None
VLLM_STARTUP_ERROR = None


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


def _stream_output(pipe):
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


def _model_uses_local_path(model_name):
    """Return whether MODEL_NAME is a filesystem path instead of an HF repo id."""
    if not model_name:
        return False
    if model_name.startswith(("file://", "/", "./", "../")):
        return True
    return Path(model_name).exists()


def start_vllm():
    """Start the local vLLM server in the background."""
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
        "--served-model-name",
        SERVE_MODEL_NAME,
    ]
    if CHAT_TEMPLATE_CONTENT_FORMAT:
        cmd.extend(["--chat-template-content-format", CHAT_TEMPLATE_CONTENT_FORMAT])
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    if MODEL_REVISION and not _model_uses_local_path(MODEL_NAME):
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
        thread = threading.Thread(target=_stream_output, args=(process.stdout,), daemon=True)
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


def _normalize_job_input(job_input):
    """Normalize RunPod native input into an OpenAI-compatible vLLM request."""
    if not isinstance(job_input, dict):
        raise ValueError(
            "Input must be an object containing either 'messages', 'prompt', or 'list_models'."
        )

    payload = dict(job_input)
    if payload.pop("list_models", False):
        return "models", payload
    if "messages" in payload:
        payload.setdefault("model", SERVE_MODEL_NAME)
        return "chat/completions", payload
    if "prompt" in payload:
        payload.setdefault("model", SERVE_MODEL_NAME)
        return "completions", payload

    raise ValueError(
        "Input must contain either 'messages', 'prompt', or 'list_models'. "
        "For OpenAI SDK usage, call the endpoint's /openai/v1 route directly."
    )


def _proxy_request(route, payload, job_id):
    """Proxy a normalized request to the local vLLM server."""
    url = f"{VLLM_URL}/v1/{route}"
    if route == "models":
        response = requests.get(url, timeout=VLLM_REQUEST_TIMEOUT)
    else:
        response = requests.post(url, json=payload, timeout=VLLM_REQUEST_TIMEOUT)

    if response.status_code != 200:
        body = _truncate_text(response.text, 4000)
        log.error(
            "Job %s: vLLM returned %s on %s with body: %s",
            job_id,
            response.status_code,
            route,
            body,
        )
        return _vllm_error_payload(
            job_id,
            f"Local vLLM server returned HTTP {response.status_code}.",
            http_status=response.status_code,
            response_body=body,
        )

    try:
        return response.json()
    except json.JSONDecodeError:
        return {
            "job_id": job_id,
            "route": route,
            "text": response.text,
        }


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

    try:
        route, payload = _normalize_job_input(job_input)
        return _proxy_request(route, payload, job_id)
    except ValueError as exc:
        return {"error": str(exc), "job_id": job_id}
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
        wait_for_vllm(VLLM_PROCESS, timeout=VLLM_STARTUP_TIMEOUT)
    except Exception as exc:
        _set_vllm_startup_error(str(exc))
        log.error("Worker startup aborted: %s", exc)
        raise
    runpod.serverless.start({"handler": handler})
