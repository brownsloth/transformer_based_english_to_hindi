#!/usr/bin/env python3
"""Download model weights + tokenizers from Hugging Face Hub into serving/artifacts/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "1starun8-research/en-hi-translation"


def download_artifacts(
    repo_id: str | None = None,
    dest: Path | str | None = None,
    token: str | None = None,
) -> Path:
    repo_id = repo_id or os.environ.get("HF_ARTIFACTS_REPO", DEFAULT_REPO)
    dest = Path(dest or os.environ.get("SERVE_ARTIFACTS_DIR", "serving/artifacts"))
    token = token or os.environ.get("HF_TOKEN")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(dest),
        allow_patterns=["student/*", "teacher/*", "tokenizers/*"],
        token=token,
    )

    required = [
        dest / "student" / "dict_14_fp16.pt",
        dest / "teacher" / "tmodel_10_dynamic_int8_linear.pt",
        dest / "tokenizers" / "tokenizer_en.json",
        dest / "tokenizers" / "tokenizer_hi.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Download finished but required files are missing:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    print("Artifacts ready:")
    for p in required:
        print(f"  {p} ({p.stat().st_size // (1024 * 1024)} MB)")
    return dest


def main() -> None:
    try:
        download_artifacts()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
