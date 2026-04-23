"""Bake the configured Hugging Face model into the container image."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download


def _download(repo_id: str, revision: str, token: str | None, cache_dir: str) -> str:
    return snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        token=token or None,
        cache_dir=cache_dir,
    )


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "").strip()
    model_revision = os.getenv("MODEL_REVISION", "").strip()
    tokenizer_name = os.getenv("TOKENIZER_NAME", "").strip()
    tokenizer_revision = os.getenv("TOKENIZER_REVISION", "").strip()
    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME")
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not model_name:
        raise SystemExit("MODEL_NAME must be set for build-time model baking.")
    if not cache_dir:
        raise SystemExit("HUGGINGFACE_HUB_CACHE or HF_HOME must be set.")

    model_snapshot = _download(model_name, model_revision, hf_token, cache_dir)
    print(f"Downloaded model snapshot to {model_snapshot}")

    if tokenizer_name and tokenizer_name != model_name:
        tokenizer_snapshot = _download(
            tokenizer_name,
            tokenizer_revision,
            hf_token,
            cache_dir,
        )
        print(f"Downloaded tokenizer snapshot to {tokenizer_snapshot}")


if __name__ == "__main__":
    main()
