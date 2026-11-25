"""The 8 real-time OpenCV + NumPy visual filters and ROI portal compositor."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FilterSpec:
    """Metadata for a single Retrolensa visual filter."""

    key: str
    name: str
    description: str
    accent_bgr: tuple[int, int, int]


FILTERS: tuple[FilterSpec, ...] = (
    FilterSpec(
        key="thermal",
        name="Thermal",
        description="Infrared heat-vision mapping",
        accent_bgr=(40, 140, 255),
    ),
    FilterSpec(
        key="sketch",
        name="Sketch",
        description="High-contrast pencil drawing",
        accent_bgr=(230, 230, 230),
    ),
    FilterSpec(
        key="glitch",
        name="Glitch",
        description="RGB chromatic aberration shift",
        accent_bgr=(255, 110, 220),
    ),
    FilterSpec(
        key="mono",
        name="Mono",
        description="High-contrast black & white",
        accent_bgr=(200, 200, 200),
    ),
    FilterSpec(
        key="dual_tone",
        name="Dual-Tone",
        description="Pop-art pink & yellow gradient",
        accent_bgr=(180, 70, 255),
    ),
    FilterSpec(
        key="invert",
        name="Invert",
        description="Full negative color inversion",
        accent_bgr=(255, 220, 70),
    ),
    FilterSpec(
        key="sepia",
        name="Sepia",
        description="Warm vintage retro film tone",
        accent_bgr=(90, 175, 225),
    ),
    FilterSpec(
        key="blur",
        name="Blur",
        description="Heavy Gaussian privacy blur",
        accent_bgr=(255, 195, 90),
    ),
)

FILTER_KEYS: tuple[str, ...] = tuple(spec.key for spec in FILTERS)
_FILTER_MAP: dict[str, FilterSpec] = {spec.key: spec for spec in FILTERS}


def _build_scurve_lut(gain: float = 6.0) -> np.ndarray:
    """Build a 256-entry uint8 sigmoid contrast lookup table."""
    x = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-gain * x))
    norm = (sig - sig[0]) / (sig[-1] - sig[0])
    return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)


def _build_sketch_lut() -> np.ndarray:
    """Darken midtone graphite lines slightly for crisp pencil contrast."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    curved = np.power(x, 1.65)
    return np.clip(np.round(curved * 255.0), 0, 255).astype(np.uint8)


def _build_dual_tone_lut() -> np.ndarray:
    """Build a (256, 3) BGR gradient LUT from hot pop-art pink to bright yellow."""
    anchors_t = np.array([0.0, 0.45, 1.0], dtype=np.float32)
    # BGR stops: deep magenta -> electric hot pink -> vivid yellow
    anchors_bgr = np.array(
        [
            [95.0, 8.0, 165.0],
            [175.0, 35.0, 255.0],
            [30.0, 248.0, 255.0],
        ],
        dtype=np.float32,
    )
    t_curve = _build_scurve_lut(gain=3.8).astype(np.float32) / 255.0
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(
            np.round(np.interp(t_curve, anchors_t, anchors_bgr[:, ch])), 0, 255
        ).astype(np.uint8)
    return lut


_MONO_LUT = _build_scurve_lut(gain=5.0)
_SKETCH_LUT = _build_sketch_lut()
_DUAL_TONE_LUT = _build_dual_tone_lut()
_SEPIA_KERNEL = np.array(
    [
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ],
    dtype=np.float32,
)


def get_filter_spec(filter_key: str) -> FilterSpec:
    """Look up filter metadata by key."""
    if filter_key not in _FILTER_MAP:
        raise ValueError(f"Unknown filter key '{filter_key}'. Valid: {FILTER_KEYS}")
    return _FILTER_MAP[filter_key]


def _filter_thermal(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def _filter_sketch(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    crisp = cv2.LUT(sketch, _SKETCH_LUT)
    return cv2.cvtColor(crisp, cv2.COLOR_GRAY2BGR)


def _filter_glitch(bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
    h, w = bgr.shape[:2]
    shift = max(4, min(18, w // 45))
    out = bgr.copy()

    if shift < w:
        # Chromatic aberration: shift Blue left and Red right without edge wraparound.
        out[:, :-shift, 0] = bgr[:, shift:, 0]
        out[:, -shift:, 0] = bgr[:, -1:, 0]
        out[:, shift:, 2] = bgr[:, :-shift, 2]
        out[:, :shift, 2] = bgr[:, :1, 2]

    # Subtle deterministic horizontal scanline bands for digital distortion.
    if h >= 24 and w >= 16:
        band_offset = max(4, shift // 2)
        for band_i in range(3):
            y_start = ((frame_index * 19 + band_i * (h // 3)) % max(1, h - 10))
            band_h = min(max(3, h // 28), h - y_start)
            direction = 1 if (band_i + frame_index) % 2 == 0 else -1
            dx = band_offset * direction
            if dx > 0 and dx < w:
                out[y_start : y_start + band_h, dx:] = out[
                    y_start : y_start + band_h, :-dx
                ]
            elif dx < 0 and -dx < w:
                out[y_start : y_start + band_h, :dx] = out[
                    y_start : y_start + band_h, -dx:
                ]

    return out


def _filter_mono(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrasted = cv2.LUT(gray, _MONO_LUT)
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)


def _filter_dual_tone(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _DUAL_TONE_LUT[gray]


def _filter_invert(bgr: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(bgr)


def _filter_sepia(bgr: np.ndarray) -> np.ndarray:
    return cv2.transform(bgr, _SEPIA_KERNEL)


def _filter_blur(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h >= 32 and w >= 32:
        small = cv2.resize(bgr, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (17, 17), 0)
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(bgr, (25, 25), 0)


def apply_filter(bgr: np.ndarray, filter_key: str, frame_index: int = 0) -> np.ndarray:
    """Apply one of the 8 real-time visual filters to a uint8 BGR image."""
    if bgr.size == 0:
        return bgr.copy()

    if filter_key == "thermal":
        return _filter_thermal(bgr)
    if filter_key == "sketch":
        return _filter_sketch(bgr)
    if filter_key == "glitch":
        return _filter_glitch(bgr, frame_index=frame_index)
    if filter_key == "mono":
        return _filter_mono(bgr)
    if filter_key == "dual_tone":
        return _filter_dual_tone(bgr)
    if filter_key == "invert":
        return _filter_invert(bgr)
    if filter_key == "sepia":
        return _filter_sepia(bgr)
    if filter_key == "blur":
        return _filter_blur(bgr)
    raise ValueError(f"Unknown filter key '{filter_key}'. Valid: {FILTER_KEYS}")


def render_portal_filter(
    frame: np.ndarray,
    corners: np.ndarray,
    filter_key: str,
    frame_index: int = 0,
    roi_padding: int = 24,
) -> np.ndarray:
    """Render the active filter strictly inside the 4-corner portal polygon."""
    output = frame.copy()
    h, w = frame.shape[:2]
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {pts.shape}")

    x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
    y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
    x1 = min(w, int(np.ceil(np.max(pts[:, 0]))))
    y1 = min(h, int(np.ceil(np.max(pts[:, 1]))))
    if x1 <= x0 or y1 <= y0:
        return output

    px0 = max(0, x0 - roi_padding)
    py0 = max(0, y0 - roi_padding)
    px1 = min(w, x1 + roi_padding)
    py1 = min(h, y1 + roi_padding)

    padded_roi = frame[py0:py1, px0:px1]
    filtered_padded = apply_filter(padded_roi, filter_key, frame_index=frame_index)

    filtered_tight = filtered_padded[(y0 - py0) : (y1 - py0), (x0 - px0) : (x1 - px0)]
    local_poly = np.round(pts - np.array([x0, y0], dtype=np.float32)).astype(np.int32)

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_poly, 255)
    mask_bool = mask > 0

    target = output[y0:y1, x0:x1]
    target[mask_bool] = filtered_tight[mask_bool]
    return output
