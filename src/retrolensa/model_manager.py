"""Automatic offline model setup for MediaPipe HandLandmarker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_MODEL_URL, FALLBACK_MODEL_URL

ProgressCallback = Callable[[int, int | None], None]


def _download_file(
    url: str,
    destination: Path,
    progress_callback: ProgressCallback | None = None,
    timeout: float = 30.0,
) -> None:
    """Download a URL to destination atomically via a .part temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": "Retrolensa/0.1"})

    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                raise RuntimeError(
                    f"Unexpected HTML response when downloading model from {url}"
                )

            length_header = response.headers.get("Content-Length")
            total_bytes = int(length_header) if length_header and length_header.isdigit() else None
            downloaded = 0

            with part_path.open("wb") as out_file:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded, total_bytes)

        if downloaded < 100_000:
            raise RuntimeError(
                f"Downloaded model from {url} is unexpectedly small ({downloaded} bytes)."
            )

        os.replace(part_path, destination)
    except Exception:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        raise


def ensure_hand_landmarker_model(
    model_path: Path,
    model_url: str = DEFAULT_MODEL_URL,
    fallback_url: str = FALLBACK_MODEL_URL,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Return a verified local hand_landmarker.task path, downloading on first launch."""
    model_path = Path(model_path)
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    urls = [model_url]
    if fallback_url and fallback_url != model_url:
        urls.append(fallback_url)

    last_error: Exception | None = None
    for url in urls:
        try:
            _download_file(url, model_path, progress_callback=progress_callback)
            return model_path
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to download MediaPipe hand_landmarker.task to '{model_path}'. "
        f"Download it manually from {model_url} and place it at '{model_path}'. "
        f"Last error: {last_error}"
    )
