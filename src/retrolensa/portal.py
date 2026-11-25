"""Hand gesture classification, filter switching cooldown, and two-hand portal geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .filters import FILTERS, FilterSpec
from .smoothing import HandPairSmoother

# MediaPipe 21-landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

PALM_CORE_INDICES = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_TRIPLES = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)


@dataclass(frozen=True, slots=True)
class DetectedHand:
    """Processed MediaPipe hand detection in pixel and normalized spaces."""

    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    palm_center_px: np.ndarray
    palm_span_px: float
    handedness: str = "Unknown"
    score: float = 1.0

    @classmethod
    def from_landmarks(
        cls,
        landmarks_norm: np.ndarray,
        frame_width: int,
        frame_height: int,
        handedness: str = "Unknown",
        score: float = 1.0,
    ) -> DetectedHand:
        """Build a DetectedHand from normalized (21, 2+) landmarks."""
        norm = np.asarray(landmarks_norm, dtype=np.float32)[:, :2]
        if norm.shape != (21, 2):
            raise ValueError(f"Expected (21, 2) landmarks, got {norm.shape}")

        scale = np.array([float(frame_width), float(frame_height)], dtype=np.float32)
        px = norm * scale
        palm_center = np.mean(px[list(PALM_CORE_INDICES)], axis=0).astype(np.float32)
        palm_span = float(np.linalg.norm(px[MIDDLE_MCP] - px[WRIST]))
        return cls(
            landmarks_px=px,
            landmarks_norm=norm,
            palm_center_px=palm_center,
            palm_span_px=palm_span,
            handedness=handedness,
            score=float(score),
        )


@dataclass(frozen=True, slots=True)
class PortalState:
    """Active 4-corner dynamic portal state."""

    corners: np.ndarray  # (4, 2) float32 in [TL, TR, BR, BL] order
    left_anchor: np.ndarray  # (2,) float32
    right_anchor: np.ndarray  # (2,) float32
    width_px: float
    height_px: float
    angle_deg: float
    is_grace: bool = False


@dataclass(frozen=True, slots=True)
class SwitchState:
    """Snapshot of the single-palm gesture filter switcher."""

    active_filter: FilterSpec
    filter_index: int
    total_filters: int
    just_switched: bool
    open_palm_detected: bool
    hold_progress: float
    cooldown_remaining: float
    cooldown_total: float


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def is_open_palm(hand: DetectedHand) -> bool:
    """Return True when a hand is held upright with thumb and all four fingers extended."""
    pts = hand.landmarks_px
    palm_span = hand.palm_span_px
    if palm_span < 12.0:
        return False

    wrist = pts[WRIST]
    # Require the hand to be raised (middle fingertip above wrist in image coords).
    if pts[MIDDLE_TIP, 1] >= wrist[1] - 0.30 * palm_span:
        return False

    for mcp_idx, pip_idx, tip_idx in FINGER_TRIPLES:
        d_tip = _dist(wrist, pts[tip_idx])
        d_pip = _dist(wrist, pts[pip_idx])
        d_mcp = _dist(wrist, pts[mcp_idx])
        d_finger = _dist(pts[mcp_idx], pts[tip_idx])
        if not (
            d_tip > d_pip * 1.10
            and d_tip > d_mcp * 1.32
            and d_finger > palm_span * 0.60
        ):
            return False

    # Thumb extension & abduction relative to pinky base (mirror/handedness invariant).
    thumb_abducted = _dist(pts[PINKY_MCP], pts[THUMB_TIP]) > _dist(
        pts[PINKY_MCP], pts[THUMB_MCP]
    ) * 1.18
    thumb_extended = _dist(wrist, pts[THUMB_TIP]) > _dist(wrist, pts[THUMB_IP]) * 1.04
    return bool(thumb_abducted and thumb_extended)


class FilterSwitcher:
    """Cycles through the 8 filters when a single raised open palm is held."""

    def __init__(
        self,
        initial_index: int = 0,
        hold_frames: int = 5,
        cooldown_seconds: float = 1.5,
        two_hand_release_grace_seconds: float = 0.35,
    ) -> None:
        self.filter_index = initial_index % len(FILTERS)
        self.hold_frames = max(1, int(hold_frames))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.two_hand_release_grace_seconds = max(0.0, float(two_hand_release_grace_seconds))
        self._open_palm_streak = 0
        self._last_switch_time = -1e9
        self._last_two_hand_time = -1e9

    @property
    def active_filter(self) -> FilterSpec:
        return FILTERS[self.filter_index]

    def set_filter_index(self, index: int, now: float | None = None) -> FilterSpec:
        self.filter_index = index % len(FILTERS)
        self._open_palm_streak = 0
        if now is not None:
            self._last_switch_time = now
        return self.active_filter

    def next_filter(self, now: float | None = None) -> FilterSpec:
        return self.set_filter_index(self.filter_index + 1, now=now)

    def prev_filter(self, now: float | None = None) -> FilterSpec:
        return self.set_filter_index(self.filter_index - 1, now=now)

    def update(self, hands: list[DetectedHand], now: float) -> SwitchState:
        """Update gesture streak and cooldown; returns current SwitchState."""
        if len(hands) >= 2:
            self._last_two_hand_time = now
            self._open_palm_streak = 0
            return self._snapshot(now=now, just_switched=False, open_palm=False)

        in_release_grace = (now - self._last_two_hand_time) < self.two_hand_release_grace_seconds
        cooldown_elapsed = now - self._last_switch_time
        in_cooldown = cooldown_elapsed < self.cooldown_seconds

        open_palm = len(hands) == 1 and is_open_palm(hands[0])
        if not open_palm or in_release_grace or in_cooldown:
            self._open_palm_streak = 0
            return self._snapshot(now=now, just_switched=False, open_palm=open_palm)

        self._open_palm_streak += 1
        if self._open_palm_streak >= self.hold_frames:
            self.filter_index = (self.filter_index + 1) % len(FILTERS)
            self._last_switch_time = now
            self._open_palm_streak = 0
            return self._snapshot(now=now, just_switched=True, open_palm=True)

        return self._snapshot(now=now, just_switched=False, open_palm=True)

    def _snapshot(self, now: float, just_switched: bool, open_palm: bool) -> SwitchState:
        cooldown_remaining = max(0.0, self.cooldown_seconds - (now - self._last_switch_time))
        hold_progress = min(1.0, self._open_palm_streak / float(self.hold_frames))
        return SwitchState(
            active_filter=self.active_filter,
            filter_index=self.filter_index,
            total_filters=len(FILTERS),
            just_switched=just_switched,
            open_palm_detected=open_palm,
            hold_progress=hold_progress,
            cooldown_remaining=cooldown_remaining,
            cooldown_total=self.cooldown_seconds,
        )


class PortalTracker:
    """Computes an EMA-smoothed 4-corner portal frame anchored between two hands."""

    def __init__(
        self,
        ema_alpha: float = 0.35,
        min_width_px: float = 64.0,
        min_height_px: float = 64.0,
        missing_grace_frames: int = 4,
    ) -> None:
        self._smoother = HandPairSmoother(alpha=ema_alpha)
        self.min_width_px = float(min_width_px)
        self.min_height_px = float(min_height_px)
        self.missing_grace_frames = max(0, int(missing_grace_frames))
        self._last_portal: PortalState | None = None
        self._missing_count = 0

    def reset(self) -> None:
        self._smoother.reset()
        self._last_portal = None
        self._missing_count = 0

    def update(
        self,
        hands: list[DetectedHand],
        frame_width: int,
        frame_height: int,
    ) -> PortalState | None:
        """Update portal corners from detected hands; returns None when inactive."""
        if len(hands) < 2:
            if self._last_portal is not None and self._missing_count < self.missing_grace_frames:
                self._missing_count += 1
                return PortalState(
                    corners=self._last_portal.corners.copy(),
                    left_anchor=self._last_portal.left_anchor.copy(),
                    right_anchor=self._last_portal.right_anchor.copy(),
                    width_px=self._last_portal.width_px,
                    height_px=self._last_portal.height_px,
                    angle_deg=self._last_portal.angle_deg,
                    is_grace=True,
                )
            self.reset()
            return None

        self._missing_count = 0
        # If >2 hands were ever returned, pick the two highest-confidence hands.
        ordered = sorted(hands, key=lambda h: h.score, reverse=True)[:2]
        h0, h1 = ordered[0], ordered[1]

        s_left, s_left_span, s_right, s_right_span = self._smoother.update(
            h0.palm_center_px,
            h0.palm_span_px,
            h1.palm_center_px,
            h1.palm_span_px,
        )

        delta = s_right - s_left
        raw_dist = float(np.linalg.norm(delta))
        if raw_dist < 1e-3:
            u = np.array([1.0, 0.0], dtype=np.float32)
        else:
            u = delta / raw_dist

        # Perpendicular unit vector oriented downward when left hand is left of right hand.
        n = np.array([-u[1], u[0]], dtype=np.float32)

        width = max(self.min_width_px, raw_dist)
        midpoint = 0.5 * (s_left + s_right)
        half_w = 0.5 * width
        left_edge = midpoint - half_w * u
        right_edge = midpoint + half_w * u

        avg_span = 0.5 * (s_left_span + s_right_span)
        max_height = max(self.min_height_px, float(frame_height) * 0.92)
        height = float(
            np.clip(avg_span * 1.45 + width * 0.38, self.min_height_px, max_height)
        )
        half_h = 0.5 * height

        tl = left_edge - half_h * n
        tr = right_edge - half_h * n
        br = right_edge + half_h * n
        bl = left_edge + half_h * n

        corners = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
        corners[:, 0] = np.clip(corners[:, 0], 0.0, float(max(1, frame_width - 1)))
        corners[:, 1] = np.clip(corners[:, 1], 0.0, float(max(1, frame_height - 1)))

        angle_deg = math.degrees(math.atan2(float(u[1]), float(u[0])))
        portal = PortalState(
            corners=corners,
            left_anchor=s_left,
            right_anchor=s_right,
            width_px=width,
            height_px=height,
            angle_deg=angle_deg,
            is_grace=False,
        )
        self._last_portal = portal
        return portal
