"""Exponential moving average (EMA) jitter smoothing for hand and portal coordinates."""

from __future__ import annotations

import numpy as np


class ExponentialMovingAverage:
    """Vector-valued exponential moving average filter."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        """Whether at least one sample has been recorded."""
        return self._state is not None

    @property
    def value(self) -> np.ndarray | None:
        """Current smoothed state copy, or None if uninitialized."""
        return None if self._state is None else self._state.copy()

    def update(self, sample: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Update the EMA state with a new sample and return the smoothed array."""
        arr = np.asarray(sample, dtype=np.float32)
        if self._state is None:
            self._state = arr.copy()
        else:
            self._state = self.alpha * arr + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        """Clear smoothed state so the next update re-initializes cleanly."""
        self._state = None


class HandPairSmoother:
    """Smooths left/right hand anchors and spans with consistent hand association."""

    def __init__(self, alpha: float = 0.35) -> None:
        self._ema = ExponentialMovingAverage(alpha=alpha)
        self._prev_left: np.ndarray | None = None
        self._prev_right: np.ndarray | None = None

    def _order_pair(
        self,
        anchor_a: np.ndarray,
        span_a: float,
        anchor_b: np.ndarray,
        span_b: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Match two anchors consistently to left/right slots to prevent flips."""
        if self._prev_left is None or self._prev_right is None:
            if anchor_a[0] <= anchor_b[0]:
                return anchor_a, span_a, anchor_b, span_b
            return anchor_b, span_b, anchor_a, span_a

        keep_cost = float(
            np.sum((anchor_a - self._prev_left) ** 2)
            + np.sum((anchor_b - self._prev_right) ** 2)
        )
        swap_cost = float(
            np.sum((anchor_b - self._prev_left) ** 2)
            + np.sum((anchor_a - self._prev_right) ** 2)
        )
        if keep_cost <= swap_cost:
            return anchor_a, span_a, anchor_b, span_b
        return anchor_b, span_b, anchor_a, span_a

    def update(
        self,
        anchor_a: np.ndarray,
        span_a: float,
        anchor_b: np.ndarray,
        span_b: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Smooth and return `(left_anchor, left_span, right_anchor, right_span)`."""
        la, ls, ra, rs = self._order_pair(
            np.asarray(anchor_a, dtype=np.float32),
            float(span_a),
            np.asarray(anchor_b, dtype=np.float32),
            float(span_b),
        )
        packed = np.array([la[0], la[1], ls, ra[0], ra[1], rs], dtype=np.float32)
        smoothed = self._ema.update(packed)
        s_left = smoothed[0:2]
        s_right = smoothed[3:5]
        self._prev_left = s_left.copy()
        self._prev_right = s_right.copy()
        return s_left, float(smoothed[2]), s_right, float(smoothed[5])

    def reset(self) -> None:
        """Reset tracking continuity and EMA history."""
        self._ema.reset()
        self._prev_left = None
        self._prev_right = None
