"""Tests for the 8 OpenCV + NumPy visual filters and ROI portal compositing."""

from __future__ import annotations

import numpy as np
import pytest

from retrolensa.filters import FILTER_KEYS, FILTERS, apply_filter, render_portal_filter


def _sample_frame(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)
    # Draw crisp geometric structure for edge/blur/sketch checks
    img[40:120, 60:180] = (240, 240, 240)
    img[70:100, 90:150] = (20, 20, 20)
    return img


def test_exact_eight_filters_order() -> None:
    assert len(FILTERS) == 8
    assert FILTER_KEYS == (
        "thermal",
        "sketch",
        "glitch",
        "mono",
        "dual_tone",
        "invert",
        "sepia",
        "blur",
    )


@pytest.mark.parametrize("filter_key", FILTER_KEYS)
def test_all_filters_preserve_shape_and_dtype(filter_key: str) -> None:
    frame = _sample_frame()
    out = apply_filter(frame, filter_key, frame_index=3)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8

    tiny = _sample_frame(12, 12)
    out_tiny = apply_filter(tiny, filter_key, frame_index=1)
    assert out_tiny.shape == tiny.shape
    assert out_tiny.dtype == np.uint8

    empty = np.empty((0, 0, 3), dtype=np.uint8)
    assert apply_filter(empty, filter_key).size == 0


def test_filter_specific_behaviors() -> None:
    frame = _sample_frame()

    # Invert is exact bitwise complement
    inv = apply_filter(frame, "invert")
    np.testing.assert_array_equal(inv, 255 - frame)

    # Mono has identical B, G, R channels
    mono = apply_filter(frame, "mono")
    np.testing.assert_array_equal(mono[:, :, 0], mono[:, :, 1])
    np.testing.assert_array_equal(mono[:, :, 1], mono[:, :, 2])

    # Dual-tone maps dark shadows to pink/magenta and bright highlights to yellow
    ramp = np.zeros((10, 256, 3), dtype=np.uint8)
    ramp[:, :, :] = np.arange(256, dtype=np.uint8)[None, :, None]
    dual = apply_filter(ramp, "dual_tone")
    shadow_bgr = dual[0, 5]
    highlight_bgr = dual[0, 250]
    assert shadow_bgr[2] > shadow_bgr[1] and shadow_bgr[0] > shadow_bgr[1]
    assert highlight_bgr[2] > 200 and highlight_bgr[1] > 200 and highlight_bgr[0] < 80

    # Sepia outputs warm BGR (R >= G >= B) on neutral mid-gray
    gray_patch = np.full((32, 32, 3), 128, dtype=np.uint8)
    sepia = apply_filter(gray_patch, "sepia")
    b, g, r = sepia[16, 16]
    assert r >= g >= b

    # Blur smooths high-frequency checkerboard noise
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    blurred = apply_filter(checker, "blur")
    assert float(np.std(blurred)) < float(np.std(checker)) * 0.25

    # Glitch shifts Blue and Red channels laterally relative to Green
    stripe = np.zeros((64, 128, 3), dtype=np.uint8)
    stripe[:, 60:68, :] = 255
    glitched = apply_filter(stripe, "glitch", frame_index=0)
    assert not np.array_equal(glitched[:, :, 0], glitched[:, :, 2])


def test_render_portal_filter_only_modifies_inside_quad() -> None:
    frame = _sample_frame(240, 320)
    corners = np.array(
        [[80, 60], [240, 60], [240, 180], [80, 180]],
        dtype=np.float32,
    )
    rendered = render_portal_filter(frame, corners, "invert", frame_index=0)

    # Outside the portal remains untouched
    np.testing.assert_array_equal(rendered[:55, :], frame[:55, :])
    np.testing.assert_array_equal(rendered[185:, :], frame[185:, :])
    np.testing.assert_array_equal(rendered[:, :75], frame[:, :75])
    np.testing.assert_array_equal(rendered[:, 245:], frame[:, 245:])

    # Strictly inside the portal matches full invert
    np.testing.assert_array_equal(rendered[70:170, 90:230], 255 - frame[70:170, 90:230])
