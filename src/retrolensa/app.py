"""Main Retrolensa frame processing pipeline, webcam runner, and CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Protocol

import cv2
import numpy as np

from .config import AppConfig
from .filters import FILTER_KEYS, FILTERS, render_portal_filter
from .hud import draw_hand_landmarks, draw_hud, draw_portal_frame
from .portal import DetectedHand, FilterSwitcher, PortalState, PortalTracker, SwitchState
from .tracker import MediaPipeHandTracker


class HandTrackerProtocol(Protocol):
    """Minimal interface expected by RetrolensaPipeline for tracking or testing."""

    active_delegate: str

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[DetectedHand]:
        ...

    def close(self) -> None:
        ...


class RetrolensaPipeline:
    """Orchestrates tracking, gesture switching, EMA portal geometry, filters, and HUD."""

    def __init__(
        self,
        tracker: HandTrackerProtocol,
        config: AppConfig,
        initial_filter_index: int = 0,
    ) -> None:
        self.tracker = tracker
        self.config = config
        self.portal_tracker = PortalTracker(
            ema_alpha=config.ema_alpha,
            min_width_px=config.min_portal_width_px,
            min_height_px=config.min_portal_height_px,
            missing_grace_frames=config.portal_missing_grace_frames,
        )
        self.switcher = FilterSwitcher(
            initial_index=initial_filter_index,
            hold_frames=config.gesture_hold_frames,
            cooldown_seconds=config.gesture_cooldown_seconds,
            two_hand_release_grace_seconds=config.two_hand_release_grace_seconds,
        )
        self.show_hud = config.show_hud
        self.show_landmarks = config.show_landmarks
        self.mirror = config.mirror
        self.frame_index = 0
        self.fps = 0.0
        self._last_frame_time: float | None = None
        self._toast_text: str | None = None
        self._toast_until = 0.0

    def trigger_toast(self, text: str, now: float, duration: float = 1.2) -> None:
        """Display a transient HUD banner."""
        self._toast_text = text
        self._toast_until = now + duration

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        now: float | None = None,
    ) -> tuple[np.ndarray, PortalState | None, SwitchState]:
        """Run one end-to-end frame through tracking, portal filter rendering, and HUD."""
        ts = time.perf_counter() if now is None else float(now)
        if self._last_frame_time is not None:
            dt = ts - self._last_frame_time
            if dt > 1e-4:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.fps == 0.0 else (0.15 * inst_fps + 0.85 * self.fps)
        self._last_frame_time = ts

        frame = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        h, w = frame.shape[:2]

        hands = self.tracker.detect(frame, timestamp_ms=int(ts * 1000.0))
        portal = self.portal_tracker.update(hands, frame_width=w, frame_height=h)
        switch_state = self.switcher.update(hands, now=ts)

        if switch_state.just_switched:
            self.trigger_toast(
                f"FILTER -> {switch_state.active_filter.name.upper()}", now=ts
            )

        if portal is not None:
            output = render_portal_filter(
                frame,
                portal.corners,
                switch_state.active_filter.key,
                frame_index=self.frame_index,
            )
            draw_portal_frame(output, portal, switch_state)
        else:
            output = frame.copy()

        if self.show_landmarks and hands:
            draw_hand_landmarks(output, hands)

        active_toast = self._toast_text if ts < self._toast_until else None
        if self.show_hud:
            delegate_label = getattr(self.tracker, "active_delegate", "CPU")
            draw_hud(
                output,
                switch_state=switch_state,
                portal=portal,
                hand_count=len(hands),
                fps=self.fps,
                delegate_label=delegate_label,
                toast_text=active_toast,
            )

        self.frame_index += 1
        return output, portal, switch_state


def _resolve_initial_filter(filter_arg: str | None) -> int:
    if not filter_arg:
        return 0
    cleaned = filter_arg.strip().lower().replace("-", "_")
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(FILTERS):
            return idx
        raise ValueError(f"Filter number must be between 1 and {len(FILTERS)}.")
    if cleaned in FILTER_KEYS:
        return FILTER_KEYS.index(cleaned)
    raise ValueError(f"Unknown filter '{filter_arg}'. Choose from: {', '.join(FILTER_KEYS)}")


def _open_capture(config: AppConfig) -> cv2.VideoCapture:
    if config.source is not None:
        cap = cv2.VideoCapture(config.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {config.source}")
        return cap

    # Prefer DirectShow on Windows for fast camera startup, with generic fallback.
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            _configure_camera(cap, config)
            return cap
        cap.release()

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {config.camera_index}")
    _configure_camera(cap, config)
    return cap


def _configure_camera(cap: cv2.VideoCapture, config: AppConfig) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _save_snapshot(frame: np.ndarray, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = snapshot_dir / f"retrolensa_{stamp}.png"
    cv2.imwrite(str(out_path), frame)
    return out_path


def _print_download_progress(downloaded: int, total: int | None) -> None:
    if total:
        pct = min(100, int(downloaded * 100 / total))
        mb_done = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        print(
            f"\rDownloading hand_landmarker.task: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)",
            end="",
            flush=True,
        )
        if downloaded >= total:
            print()
    else:
        mb_done = downloaded / (1024 * 1024)
        print(f"\rDownloading hand_landmarker.task: {mb_done:.1f} MB", end="", flush=True)


def _make_status_frame(width: int, height: int, title: str, subtitle: str) -> np.ndarray:
    """Create a dark startup/status canvas for immediate window visibility."""
    w = max(480, min(1280, int(width)))
    h = max(320, min(720, int(height)))
    canvas = np.full((h, w, 3), (18, 18, 24), dtype=np.uint8)
    cv2.rectangle(canvas, (24, 24), (w - 24, h - 24), (65, 72, 90), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        title,
        (48, h // 2 - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (245, 245, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (48, h // 2 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _init_display_window(window_title: str, width: int, height: int) -> None:
    """Create and surface the OpenCV window immediately so it is never hidden behind a terminal."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, max(640, min(1280, width)), max(480, min(720, height)))
    splash = _make_status_frame(
        width,
        height,
        "RETROLENSA",
        "Starting MediaPipe HandLandmarker & camera...",
    )
    cv2.imshow(window_title, splash)
    # Brief topmost toggle brings the new window to front over PowerShell/terminal.
    try:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        cv2.waitKey(1)


def run_app(
    config: AppConfig,
    initial_filter_index: int = 0,
    headless: bool = False,
    max_frames: int | None = None,
) -> int:
    """Launch the Retrolensa live tracking and filter loop."""
    if not headless:
        _init_display_window(config.window_title, config.frame_width, config.frame_height)

    model_path = config.resolved_model_path()
    needs_download = not (model_path.is_file() and model_path.stat().st_size > 0)
    if needs_download:
        print(f"First launch: preparing MediaPipe model at {model_path} ...")

    with MediaPipeHandTracker.from_config(
        config,
        progress_callback=_print_download_progress if needs_download else None,
        running_mode="VIDEO",
    ) as tracker:
        if needs_download:
            print()
        print(
            f"Retrolensa ready | Delegate: {tracker.active_delegate} | "
            f"Model: {tracker.model_path}"
        )
        pipeline = RetrolensaPipeline(
            tracker=tracker,
            config=config,
            initial_filter_index=initial_filter_index,
        )

        # Single static image source convenience path
        if config.source and Path(config.source).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".webp"
        }:
            image = cv2.imread(config.source)
            if image is None:
                raise RuntimeError(f"Could not read image source: {config.source}")
            rendered, _, _ = pipeline.process_frame(image, now=time.perf_counter())
            if not headless:
                cv2.imshow(config.window_title, rendered)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 0

        cap = _open_capture(config)
        processed = 0
        empty_reads = 0
        try:
            while max_frames is None or processed < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if config.source is not None:
                        break
                    empty_reads += 1
                    if not headless:
                        if empty_reads % 15 == 1:
                            wait_img = _make_status_frame(
                                config.frame_width,
                                config.frame_height,
                                "WAITING FOR WEBCAM...",
                                "Close other apps using the camera or press Q to quit.",
                            )
                            cv2.imshow(config.window_title, wait_img)
                        key = cv2.waitKey(30) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            break
                    continue

                empty_reads = 0
                now = time.perf_counter()
                rendered, _, _ = pipeline.process_frame(frame, now=now)
                processed += 1

                if headless:
                    continue

                cv2.imshow(config.window_title, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(config.window_title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                if ord("1") <= key <= ord("8"):
                    spec = pipeline.switcher.set_filter_index(key - ord("1"), now=now)
                    pipeline.trigger_toast(f"FILTER -> {spec.name.upper()}", now=now)
                elif key in (ord(" "), ord("n"), ord("N")):
                    spec = pipeline.switcher.next_filter(now=now)
                    pipeline.trigger_toast(f"FILTER -> {spec.name.upper()}", now=now)
                elif key in (ord("p"), ord("P")):
                    spec = pipeline.switcher.prev_filter(now=now)
                    pipeline.trigger_toast(f"FILTER -> {spec.name.upper()}", now=now)
                elif key in (ord("h"), ord("H")):
                    pipeline.show_hud = not pipeline.show_hud
                elif key in (ord("l"), ord("L")):
                    pipeline.show_landmarks = not pipeline.show_landmarks
                elif key in (ord("m"), ord("M")):
                    pipeline.mirror = not pipeline.mirror
                    pipeline.portal_tracker.reset()
                elif key in (ord("s"), ord("S")):
                    saved = _save_snapshot(rendered, config.snapshot_dir)
                    pipeline.trigger_toast(f"SAVED {saved.name}", now=now)
        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for Retrolensa."""
    parser = argparse.ArgumentParser(
        prog="retrolensa",
        description=(
            "Retrolensa: real-time two-hand spatial tracking portal with 8 OpenCV filters."
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--source", type=str, default=None, help="Optional video or image path")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal selfie flip")
    parser.add_argument(
        "--delegate",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="MediaPipe inference delegate (default: auto)",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Custom .task model path")
    parser.add_argument(
        "--filter",
        type=str,
        default="thermal",
        help="Initial filter name or 1-8 index (default: thermal)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing factor in (0, 1] (default: 0.35)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="Cooldown in seconds between palm gesture switches (default: 1.5)",
    )
    parser.add_argument("--landmarks", action="store_true", help="Show hand landmarks overlay")
    parser.add_argument("--no-hud", action="store_true", help="Hide HUD overlay on startup")
    parser.add_argument("--headless", action="store_true", help="Run without opening a GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Retrolensa."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    initial_idx = _resolve_initial_filter(args.filter)
    config = AppConfig(
        camera_index=args.camera,
        source=args.source,
        frame_width=args.width,
        frame_height=args.height,
        mirror=not args.no_mirror,
        delegate=args.delegate,
        model_path=args.model_path,
        ema_alpha=args.ema_alpha,
        gesture_cooldown_seconds=args.cooldown,
        show_hud=not args.no_hud,
        show_landmarks=args.landmarks,
    )
    try:
        return run_app(
            config=config,
            initial_filter_index=initial_idx,
            headless=args.headless,
            max_frames=args.max_frames,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Retrolensa error: {exc}", file=sys.stderr)
        return 1
