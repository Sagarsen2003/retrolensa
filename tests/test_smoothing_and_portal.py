"""Tests for EMA smoothing, hand gestures, cooldown switching, and portal geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from retrolensa.portal import DetectedHand, FilterSwitcher, PortalTracker, is_open_palm
from retrolensa.smoothing import ExponentialMovingAverage, HandPairSmoother


def _make_hand_landmarks(
    cx: float = 0.5,
    cy: float = 0.5,
    scale: float = 0.15,
    open_fingers: bool = True,
    upright: bool = True,
    mirror_x: float = 1.0,
) -> np.ndarray:
    """Synthesize a clean 21-point normalized MediaPipe hand landmark array."""
    pts = np.zeros((21, 2), dtype=np.float32)
    y_dir = -1.0 if upright else 1.0

    # Wrist & MCP knuckles
    pts[0] = [0.0, 0.45 * (-y_dir)]
    pts[5] = [-0.26 * mirror_x, -0.08 * (-y_dir)]
    pts[9] = [-0.06 * mirror_x, -0.10 * (-y_dir)]
    pts[13] = [0.12 * mirror_x, -0.07 * (-y_dir)]
    pts[17] = [0.28 * mirror_x, -0.02 * (-y_dir)]

    # Thumb
    pts[1] = [-0.22 * mirror_x, 0.28 * (-y_dir)]
    pts[2] = [-0.38 * mirror_x, 0.14 * (-y_dir)]
    if open_fingers:
        pts[3] = [-0.54 * mirror_x, 0.02 * (-y_dir)]
        pts[4] = [-0.70 * mirror_x, -0.10 * (-y_dir)]
    else:
        pts[3] = [-0.24 * mirror_x, 0.08 * (-y_dir)]
        pts[4] = [-0.12 * mirror_x, 0.10 * (-y_dir)]

    # Index, Middle, Ring, Pinky
    finger_specs = [
        (5, 6, 7, 8, -0.28, 0.72),
        (9, 10, 11, 12, -0.07, 0.82),
        (13, 14, 15, 16, 0.14, 0.75),
        (17, 18, 19, 20, 0.34, 0.60),
    ]
    for mcp, pip, dip, tip, x_off, reach in finger_specs:
        if open_fingers:
            pts[pip] = [x_off * mirror_x, pts[mcp, 1] + 0.42 * reach * y_dir]
            pts[dip] = [x_off * mirror_x, pts[mcp, 1] + 0.72 * reach * y_dir]
            pts[tip] = [x_off * mirror_x, pts[mcp, 1] + 1.00 * reach * y_dir]
        else:
            # Curled back toward palm/wrist
            pts[pip] = [x_off * mirror_x, pts[mcp, 1] + 0.18 * y_dir]
            pts[dip] = [x_off * mirror_x, pts[mcp, 1] + 0.04 * y_dir]
            pts[tip] = [x_off * mirror_x, pts[mcp, 1] - 0.12 * y_dir]

    # Translate and scale into [0, 1] normalized frame coords
    pts[:, 0] = cx + pts[:, 0] * scale
    pts[:, 1] = cy + pts[:, 1] * scale
    return pts


def test_ema_reduces_jitter() -> None:
    ema = ExponentialMovingAverage(alpha=0.3)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(alpha=0.0)

    rng = np.random.default_rng(0)
    raw = np.array([100.0, 200.0], dtype=np.float32) + rng.normal(0.0, 6.0, size=(40, 2))
    smoothed = np.stack([ema.update(pt) for pt in raw], axis=0)
    assert np.std(smoothed[10:], axis=0).mean() < np.std(raw[10:], axis=0).mean()


def test_hand_pair_smoother_keeps_consistent_assignment() -> None:
    smoother = HandPairSmoother(alpha=0.5)
    l1, _, r1, _ = smoother.update(np.array([200.0, 300.0]), 80.0, np.array([600.0, 300.0]), 80.0)
    assert l1[0] < r1[0]

    # Swap input ordering on next frame: smoother should preserve physical assignment
    l2, _, r2, _ = smoother.update(np.array([610.0, 305.0]), 80.0, np.array([205.0, 295.0]), 80.0)
    assert l2[0] < 300.0
    assert r2[0] > 500.0


def test_open_palm_detection_left_right_and_negatives() -> None:
    open_right = DetectedHand.from_landmarks(_make_hand_landmarks(open_fingers=True, upright=True, mirror_x=1.0), 1280, 720)
    open_left = DetectedHand.from_landmarks(_make_hand_landmarks(open_fingers=True, upright=True, mirror_x=-1.0), 1280, 720)
    fist = DetectedHand.from_landmarks(_make_hand_landmarks(open_fingers=False, upright=True), 1280, 720)
    downward = DetectedHand.from_landmarks(_make_hand_landmarks(open_fingers=True, upright=False), 1280, 720)

    assert is_open_palm(open_right) is True
    assert is_open_palm(open_left) is True
    assert is_open_palm(fist) is False
    assert is_open_palm(downward) is False


def test_filter_switcher_hold_and_cooldown() -> None:
    switcher = FilterSwitcher(
        initial_index=0,
        hold_frames=3,
        cooldown_seconds=1.5,
        two_hand_release_grace_seconds=0.35,
    )
    palm = [DetectedHand.from_landmarks(_make_hand_landmarks(open_fingers=True), 1280, 720)]

    # Needs 3 consecutive open-palm frames
    assert switcher.update(palm, now=1.00).just_switched is False
    assert switcher.update(palm, now=1.03).just_switched is False
    st = switcher.update(palm, now=1.06)
    assert st.just_switched is True
    assert st.active_filter.key == "sketch"

    # Cooldown prevents accidental rapid skips
    for dt in (1.10, 1.20, 1.40, 1.80, 2.20):
        assert switcher.update(palm, now=dt).just_switched is False
    assert switcher.active_filter.key == "sketch"

    # After cooldown expires + hold_frames, advances again
    assert switcher.update(palm, now=2.60).just_switched is False
    assert switcher.update(palm, now=2.63).just_switched is False
    st2 = switcher.update(palm, now=2.66)
    assert st2.just_switched is True
    assert st2.active_filter.key == "glitch"


def test_filter_switcher_ignores_two_hands_and_release_grace() -> None:
    switcher = FilterSwitcher(hold_frames=2, cooldown_seconds=0.5, two_hand_release_grace_seconds=0.3)
    h1 = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.3, open_fingers=True), 1280, 720)
    h2 = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.7, open_fingers=True), 1280, 720)

    # Two hands visible never triggers switch
    assert switcher.update([h1, h2], now=1.0).just_switched is False
    # Dropping to 1 hand inside grace period does not misfire
    assert switcher.update([h1], now=1.10).just_switched is False
    assert switcher.update([h1], now=1.20).just_switched is False
    assert switcher.active_filter.key == "thermal"


def test_portal_tracker_resize_rotate_move_and_grace() -> None:
    tracker = PortalTracker(ema_alpha=1.0, min_width_px=64.0, min_height_px=64.0, missing_grace_frames=2)

    # Inactive initially
    assert tracker.update([], 1280, 720) is None

    # Narrow pair -> wide pair stretches portal
    left_narrow = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.40, cy=0.50), 1280, 720)
    right_narrow = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.60, cy=0.50), 1280, 720)
    p_narrow = tracker.update([left_narrow, right_narrow], 1280, 720)
    assert p_narrow is not None
    assert abs(p_narrow.angle_deg) < 2.0

    left_wide = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.25, cy=0.50), 1280, 720)
    right_wide = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.75, cy=0.50), 1280, 720)
    p_wide = tracker.update([left_wide, right_wide], 1280, 720)
    assert p_wide is not None
    assert p_wide.width_px > p_narrow.width_px * 2.0

    # Tilting right hand higher/lower rotates portal
    left_tilt = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.35, cy=0.40), 1280, 720)
    right_tilt = DetectedHand.from_landmarks(_make_hand_landmarks(cx=0.65, cy=0.60), 1280, 720)
    p_tilt = tracker.update([left_tilt, right_tilt], 1280, 720)
    assert p_tilt is not None
    assert p_tilt.angle_deg > 10.0

    # Brief drop uses grace frames, then closes
    grace1 = tracker.update([], 1280, 720)
    grace2 = tracker.update([], 1280, 720)
    expired = tracker.update([], 1280, 720)
    assert grace1 is not None and grace1.is_grace is True
    assert grace2 is not None and grace2.is_grace is True
    assert expired is None
