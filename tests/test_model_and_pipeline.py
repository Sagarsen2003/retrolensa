"""Tests for auto-model download and end-to-end RetrolensaPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from retrolensa.app import RetrolensaPipeline, _resolve_initial_filter
from retrolensa.config import AppConfig
from retrolensa.model_manager import ensure_hand_landmarker_model
from retrolensa.portal import DetectedHand


def _make_open_palm_landmarks(cx: float = 0.5, cy: float = 0.5, scale: float = 0.15) -> np.ndarray:
    """Create a self-contained upright open-palm 21-point normalized landmark set."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = [0.0, 0.45]
    pts[1] = [-0.22, 0.28]
    pts[2] = [-0.38, 0.14]
    pts[3] = [-0.54, 0.02]
    pts[4] = [-0.70, -0.10]
    pts[5], pts[6], pts[7], pts[8] = [-0.26, -0.08], [-0.28, -0.38], [-0.28, -0.60], [-0.28, -0.80]
    pts[9], pts[10], pts[11], pts[12] = [-0.06, -0.10], [-0.07, -0.44], [-0.07, -0.69], [-0.07, -0.92]
    pts[13], pts[14], pts[15], pts[16] = [0.12, -0.07], [0.14, -0.39], [0.14, -0.61], [0.14, -0.82]
    pts[17], pts[18], pts[19], pts[20] = [0.28, -0.02], [0.34, -0.27], [0.34, -0.45], [0.34, -0.62]
    pts[:, 0] = cx + pts[:, 0] * scale
    pts[:, 1] = cy + pts[:, 1] * scale
    return pts


class _FakeHTTPResponse:
    def __init__(self, payload: bytes, content_type: str = "application/octet-stream") -> None:
        self._payload = payload
        self._offset = 0
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload)),
        }

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_ensure_model_uses_existing_file_offline(tmp_path: Path) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    model_file.write_bytes(b"mock-task-model")

    with patch("retrolensa.model_manager.urlopen") as mock_open:
        resolved = ensure_hand_landmarker_model(model_file)

    assert resolved == model_file
    mock_open.assert_not_called()


def test_ensure_model_downloads_atomically_on_first_launch(tmp_path: Path) -> None:
    model_file = tmp_path / "models" / "hand_landmarker.task"
    payload = b"PK\x03\x04" + b"0" * 120_000
    progress_calls: list[tuple[int, int | None]] = []

    with patch(
        "retrolensa.model_manager.urlopen",
        return_value=_FakeHTTPResponse(payload),
    ):
        resolved = ensure_hand_landmarker_model(
            model_file,
            progress_callback=lambda done, total: progress_calls.append((done, total)),
        )

    assert resolved == model_file
    assert model_file.read_bytes() == payload
    assert not model_file.with_name("hand_landmarker.task.part").exists()
    assert progress_calls and progress_calls[-1] == (len(payload), len(payload))


def test_ensure_model_rejects_html_and_cleans_part_file(tmp_path: Path) -> None:
    model_file = tmp_path / "hand_landmarker.task"
    with patch(
        "retrolensa.model_manager.urlopen",
        return_value=_FakeHTTPResponse(b"<html>error</html>", content_type="text/html"),
    ):
        with pytest.raises(RuntimeError, match="Failed to download"):
            ensure_hand_landmarker_model(model_file, fallback_url="")

    assert not model_file.exists()
    assert not model_file.with_name("hand_landmarker.task.part").exists()


class _FakeTracker:
    active_delegate = "CPU"

    def __init__(self) -> None:
        self.next_hands: list[DetectedHand] = []

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[DetectedHand]:
        return list(self.next_hands)

    def close(self) -> None:
        return None


def test_pipeline_end_to_end_portal_and_palm_switch() -> None:
    tracker = _FakeTracker()
    cfg = AppConfig(
        mirror=False,
        gesture_hold_frames=2,
        gesture_cooldown_seconds=1.0,
        two_hand_release_grace_seconds=0.0,
        show_landmarks=True,
    )
    pipeline = RetrolensaPipeline(tracker=tracker, config=cfg, initial_filter_index=0)
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)

    # 1) No hands: portal inactive
    out0, portal0, sw0 = pipeline.process_frame(frame, now=1.0)
    assert out0.shape == frame.shape
    assert portal0 is None
    assert sw0.active_filter.key == "thermal"

    # 2) Two hands: portal opens and filters inside quad
    h_left = DetectedHand.from_landmarks(_make_open_palm_landmarks(cx=0.35, cy=0.50), 640, 360)
    h_right = DetectedHand.from_landmarks(_make_open_palm_landmarks(cx=0.65, cy=0.50), 640, 360)
    tracker.next_hands = [h_left, h_right]
    out1, portal1, _ = pipeline.process_frame(frame, now=1.1)
    assert portal1 is not None
    # Inside thermal portal, neutral gray (100,100,100) maps to COLORMAP_JET != (100,100,100)
    assert not np.array_equal(out1[180, 320], frame[180, 320])

    # 3) Single open palm for 2 frames cycles filter to sketch
    tracker.next_hands = [h_left]
    pipeline.portal_tracker.reset()
    pipeline.process_frame(frame, now=2.0)
    _, _, sw2 = pipeline.process_frame(frame, now=2.1)
    assert sw2.just_switched is True
    assert sw2.active_filter.key == "sketch"


def test_resolve_initial_filter() -> None:
    assert _resolve_initial_filter("thermal") == 0
    assert _resolve_initial_filter("Dual-Tone") == 4
    assert _resolve_initial_filter("8") == 7
    with pytest.raises(ValueError):
        _resolve_initial_filter("nonexistent")
