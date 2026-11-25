"""Configuration defaults for Retrolensa."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FALLBACK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def default_model_path() -> Path:
    """Return the default local path for the hand_landmarker.task asset."""
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return repo_root / "models" / "hand_landmarker.task"
    return Path.home() / ".cache" / "retrolensa" / "hand_landmarker.task"


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for the Retrolensa camera, tracker, portal, and HUD."""

    camera_index: int = 0
    source: str | None = None
    frame_width: int = 1280
    frame_height: int = 720
    mirror: bool = True
    delegate: str = "auto"  # "auto", "cpu", or "gpu"
    model_path: Path | None = None
    model_url: str = DEFAULT_MODEL_URL

    # MediaPipe detection thresholds
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55

    # Portal geometry and EMA jitter smoothing
    ema_alpha: float = 0.35
    min_portal_width_px: float = 64.0
    min_portal_height_px: float = 64.0
    portal_missing_grace_frames: int = 4

    # Single open-palm gesture switching
    gesture_hold_frames: int = 5
    gesture_cooldown_seconds: float = 1.5
    two_hand_release_grace_seconds: float = 0.35

    # Visual overlays
    show_hud: bool = True
    show_landmarks: bool = False
    window_title: str = "Retrolensa - Spatial Hand Portal"
    snapshot_dir: Path = Path("snapshots")

    def resolved_model_path(self) -> Path:
        """Return explicit model_path if set, otherwise the default local path."""
        return Path(self.model_path) if self.model_path is not None else default_model_path()
