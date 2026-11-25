"""MediaPipe HandLandmarker wrapper with automatic model download and CPU/GPU setup."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import AppConfig
from .model_manager import ProgressCallback, ensure_hand_landmarker_model
from .portal import DetectedHand


class MediaPipeHandTracker:
    """Offline MediaPipe HandLandmarker wrapper for real-time video frames."""

    def __init__(
        self,
        model_path: Path,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.55,
        min_hand_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        delegate: str = "auto",
        running_mode: Literal["VIDEO", "IMAGE"] = "VIDEO",
    ) -> None:
        self.model_path = Path(model_path)
        self.running_mode = running_mode
        self._last_timestamp_ms = -1
        self.active_delegate, self._landmarker = self._create_landmarker(
            model_path=self.model_path,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=delegate,
            running_mode=running_mode,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        progress_callback: ProgressCallback | None = None,
        running_mode: Literal["VIDEO", "IMAGE"] = "VIDEO",
    ) -> MediaPipeHandTracker:
        """Ensure the model exists locally and build a configured tracker."""
        model_path = ensure_hand_landmarker_model(
            model_path=config.resolved_model_path(),
            model_url=config.model_url,
            progress_callback=progress_callback,
        )
        return cls(
            model_path=model_path,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            delegate=config.delegate,
            running_mode=running_mode,
        )

    @staticmethod
    def _create_landmarker(
        model_path: Path,
        num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
        delegate: str,
        running_mode: Literal["VIDEO", "IMAGE"],
    ) -> tuple[str, mp_vision.HandLandmarker]:
        mode_enum = (
            mp_vision.RunningMode.VIDEO
            if running_mode == "VIDEO"
            else mp_vision.RunningMode.IMAGE
        )
        delegate_norm = delegate.lower().strip()
        if delegate_norm == "gpu":
            candidates = [("GPU", mp_python.BaseOptions.Delegate.GPU)]
        elif delegate_norm == "cpu":
            candidates = [("CPU", mp_python.BaseOptions.Delegate.CPU)]
        elif delegate_norm == "auto":
            # MediaPipe pip wheels on Windows do not ship an OpenGL GPU delegate;
            # skipping the failing GPU probe saves ~2.7s on startup.
            if sys.platform.startswith("win"):
                candidates = [("CPU", mp_python.BaseOptions.Delegate.CPU)]
            else:
                candidates = [
                    ("GPU", mp_python.BaseOptions.Delegate.GPU),
                    ("CPU", mp_python.BaseOptions.Delegate.CPU),
                ]
        else:
            raise ValueError(f"Unsupported delegate '{delegate}'. Use 'auto', 'cpu', or 'gpu'.")

        last_exc: Exception | None = None
        for label, delegate_enum in candidates:
            try:
                base_options = mp_python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=delegate_enum,
                )
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mode_enum,
                    num_hands=num_hands,
                    min_hand_detection_confidence=min_hand_detection_confidence,
                    min_hand_presence_confidence=min_hand_presence_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                return label, mp_vision.HandLandmarker.create_from_options(options)
            except Exception as exc:
                last_exc = exc
                if delegate_norm != "auto":
                    raise

        raise RuntimeError(f"Failed to initialize MediaPipe HandLandmarker: {last_exc}")

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[DetectedHand]:
        """Run MediaPipe hand landmark detection on a BGR frame."""
        if frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "VIDEO":
            ts = max(int(timestamp_ms), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)

        detected: list[DetectedHand] = []
        for idx, hand_lms in enumerate(result.hand_landmarks):
            coords = np.array([[lm.x, lm.y] for lm in hand_lms], dtype=np.float32)
            label = "Unknown"
            score = 1.0
            if idx < len(result.handedness) and result.handedness[idx]:
                cat = result.handedness[idx][0]
                label = cat.category_name or "Unknown"
                score = float(cat.score if cat.score is not None else 1.0)
            detected.append(
                DetectedHand.from_landmarks(
                    landmarks_norm=coords,
                    frame_width=w,
                    frame_height=h,
                    handedness=label,
                    score=score,
                )
            )
        return detected

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> MediaPipeHandTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
