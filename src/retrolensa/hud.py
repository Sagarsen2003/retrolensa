"""Minimal HUD and 4-corner portal frame renderer."""

from __future__ import annotations

import cv2
import numpy as np

from .filters import FILTERS
from .portal import DetectedHand, PortalState, SwitchState

# Standard MediaPipe 21-point hand skeleton bone edges
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _blend_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color_bgr: tuple[int, int, int] = (18, 18, 22),
    alpha: float = 0.68,
) -> None:
    """In-place translucent dark pill fill clipped cleanly to frame bounds."""
    h, w = frame.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = frame[cy0:cy1, cx0:cx1]
    overlay = np.empty_like(roi)
    overlay[:] = color_bgr
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def draw_hand_landmarks(frame: np.ndarray, hands: list[DetectedHand]) -> None:
    """Draw optional lightweight hand skeleton and palm centers."""
    for hand in hands:
        pts = np.round(hand.landmarks_px).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (190, 220, 235), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)
        palm = tuple(np.round(hand.palm_center_px).astype(np.int32))
        cv2.circle(frame, palm, 6, (60, 225, 255), 2, cv2.LINE_AA)


def draw_portal_frame(
    frame: np.ndarray,
    portal: PortalState,
    switch_state: SwitchState,
) -> None:
    """Draw the 4-corner portal frame, corner brackets, anchors, and filter tag."""
    accent = switch_state.active_filter.accent_bgr
    pts = np.round(portal.corners).astype(np.int32)

    # Thin outer border around the dynamic 4-corner quad.
    cv2.polylines(frame, [pts], isClosed=True, color=accent, thickness=2, lineType=cv2.LINE_AA)

    # Corner L-brackets along edge vectors so brackets stay aligned under rotation.
    bracket_len = max(14.0, min(32.0, min(portal.width_px, portal.height_px) * 0.16))
    corners_f = portal.corners.astype(np.float32)
    for idx in range(4):
        curr = corners_f[idx]
        prev_pt = corners_f[(idx - 1) % 4]
        next_pt = corners_f[(idx + 1) % 4]
        for neighbor in (prev_pt, next_pt):
            vec = neighbor - curr
            norm = float(np.linalg.norm(vec))
            if norm > 1e-3:
                end = curr + (vec / norm) * bracket_len
                cv2.line(
                    frame,
                    tuple(np.round(curr).astype(np.int32)),
                    tuple(np.round(end).astype(np.int32)),
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

    # Left/right hand anchor markers.
    for anchor in (portal.left_anchor, portal.right_anchor):
        center = tuple(np.round(anchor).astype(np.int32))
        cv2.circle(frame, center, 6, accent, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 9, (255, 255, 255), 1, cv2.LINE_AA)

    # Compact filter badge tucked above top-left corner of the portal.
    tl = pts[0]
    label = f"{switch_state.filter_index + 1:02d} // {switch_state.active_filter.name.upper()}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    bx0 = max(6, min(frame.shape[1] - tw - 20, int(tl[0])))
    by1 = max(th + 14, int(tl[1]) - 8)
    by0 = by1 - (th + 12)
    bx1 = bx0 + tw + 14
    _blend_rect(frame, bx0, by0, bx1, by1, color_bgr=(16, 16, 22), alpha=0.78)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), accent, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (bx0 + 7, by1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    switch_state: SwitchState,
    portal: PortalState | None,
    hand_count: int,
    fps: float,
    delegate_label: str,
    toast_text: str | None = None,
) -> None:
    """Draw minimal top status bar and bottom gesture controls bar."""
    h, w = frame.shape[:2]
    accent = switch_state.active_filter.accent_bgr
    spec = switch_state.active_filter

    # Top bar
    bar_w = min(w - 24, 580)
    _blend_rect(frame, 12, 12, 12 + bar_w, 86, color_bgr=(14, 15, 20), alpha=0.72)
    cv2.rectangle(frame, (12, 12), (12 + bar_w, 86), (62, 66, 78), 1, cv2.LINE_AA)

    header = (
        f"RETROLENSA   |   [{switch_state.filter_index + 1}/{switch_state.total_filters}] "
        f"{spec.name.upper()} - {spec.description}"
    )
    cv2.putText(
        frame,
        header,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 250),
        1,
        cv2.LINE_AA,
    )

    # 8 filter selector pips
    pip_x = 24
    pip_y = 52
    for idx, f_spec in enumerate(FILTERS):
        is_active = idx == switch_state.filter_index
        color = f_spec.accent_bgr if is_active else (82, 86, 98)
        cv2.rectangle(
            frame,
            (pip_x, pip_y),
            (pip_x + (18 if is_active else 12), pip_y + 6),
            color,
            -1,
        )
        pip_x += 24 if is_active else 18

    # Portal & gesture state text
    if portal is not None:
        status_text = (
            f"PORTAL {int(portal.width_px)}x{int(portal.height_px)}  "
            f"{portal.angle_deg:+.0f}deg"
        )
        status_color = accent
    elif switch_state.cooldown_remaining > 0.0:
        status_text = f"COOLDOWN {switch_state.cooldown_remaining:.1f}s"
        status_color = (120, 210, 255)
    elif switch_state.open_palm_detected and switch_state.hold_progress > 0.0:
        pct = int(switch_state.hold_progress * 100)
        status_text = f"OPEN PALM {pct}% -> NEXT"
        status_color = (130, 255, 175)
    elif hand_count == 1:
        status_text = "1 HAND: RAISE OPEN PALM TO SWITCH  |  SHOW 2 HANDS FOR PORTAL"
        status_color = (210, 215, 225)
    else:
        status_text = "SHOW BOTH HANDS TO OPEN PORTAL"
        status_color = (190, 195, 205)

    cv2.putText(
        frame,
        status_text,
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Top-right compact telemetry pill (FPS + Delegate)
    perf_text = f"{fps:4.1f} FPS  |  {delegate_label}"
    (pw, ph), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    px0 = max(12 + bar_w + 10, w - pw - 34)
    if px0 < w - 60:
        _blend_rect(frame, px0, 12, w - 12, 44, color_bgr=(14, 15, 20), alpha=0.72)
        cv2.putText(
            frame,
            perf_text,
            (px0 + 10, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 245, 200),
            1,
            cv2.LINE_AA,
        )

    # Cooldown / palm-hold micro bar under the top HUD
    if switch_state.cooldown_remaining > 0.0 and switch_state.cooldown_total > 0.0:
        ratio = switch_state.cooldown_remaining / switch_state.cooldown_total
        fill_w = int((bar_w - 4) * max(0.0, min(1.0, ratio)))
        cv2.rectangle(frame, (14, 83), (14 + fill_w, 86), accent, -1)
    elif switch_state.open_palm_detected and switch_state.hold_progress > 0.0:
        fill_w = int((bar_w - 4) * switch_state.hold_progress)
        cv2.rectangle(frame, (14, 83), (14 + fill_w, 86), (120, 255, 165), -1)

    # Switch toast banner
    if toast_text:
        (tw, th), _ = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        tx0 = max(16, (w - tw) // 2 - 18)
        tx1 = min(w - 16, tx0 + tw + 36)
        ty0 = 98
        ty1 = ty0 + th + 20
        _blend_rect(frame, tx0, ty0, tx1, ty1, color_bgr=(18, 18, 24), alpha=0.82)
        cv2.rectangle(frame, (tx0, ty0), (tx1, ty1), accent, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            toast_text,
            (tx0 + 18, ty1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Bottom minimal gesture controls bar
    controls = (
        "2 Hands: Frame / Stretch / Tilt   |   1 Open Palm: Next Filter   |   "
        "[1-8] Select   [Space] Next   [L] Landmarks   [H] HUD   [S] Snap   [Q] Quit"
    )
    if w >= 640:
        _blend_rect(frame, 12, h - 40, w - 12, h - 10, color_bgr=(14, 15, 20), alpha=0.68)
        cv2.putText(
            frame,
            controls,
            (24, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (215, 220, 230),
            1,
            cv2.LINE_AA,
        )
