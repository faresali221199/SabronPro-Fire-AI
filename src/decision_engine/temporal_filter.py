"""Temporal filtering for multi-frame confirmation.

NEVER trigger an alarm based on a single frame.
Requires consistent detection over multiple frames within a time window.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalState:
    """Current temporal filtering state."""
    fire_smoothed: float = 0.0
    smoke_smoothed: float = 0.0
    fire_consecutive: int = 0
    smoke_consecutive: int = 0
    fire_confirmed: bool = False
    smoke_confirmed: bool = False
    last_alarm_time: float = 0.0
    in_cooldown: bool = False
    status: str = "NORMAL"  # NORMAL, MONITORING, PRE_ALARM, ALARM


class TemporalFilter:
    """Multi-frame temporal confirmation filter.

    Implements:
    - Exponential moving average smoothing
    - Consecutive frame counting
    - Time-windowed detection counting
    - Cooldown period after alarm
    - Hysteresis for alarm maintenance
    """

    def __init__(
        self,
        min_consecutive_frames: int = 8,
        temporal_window_seconds: float = 3.0,
        smoothing_alpha: float = 0.3,
        cooldown_seconds: float = 30.0,
        hysteresis_margin: float = 0.10,
        fire_threshold: float = 0.70,
        smoke_threshold: float = 0.65,
        min_detections_in_window: int = 3,
    ):
        self.min_consecutive_frames = min_consecutive_frames
        self.temporal_window = temporal_window_seconds
        self.smoothing_alpha = smoothing_alpha
        self.cooldown_seconds = cooldown_seconds
        self.hysteresis_margin = hysteresis_margin
        self.fire_threshold = fire_threshold
        self.smoke_threshold = smoke_threshold
        self.min_detections_in_window = min_detections_in_window

        # Internal state
        self._fire_history: deque = deque(maxlen=300)  # (timestamp, confidence)
        self._smoke_history: deque = deque(maxlen=300)
        self._state = TemporalState()

    def update(self, fire_confidence: float, smoke_confidence: float) -> TemporalState:
        """Update temporal filter with new frame confidences."""
        now = time.time()

        # Exponential moving average smoothing
        self._state.fire_smoothed = (
            self.smoothing_alpha * fire_confidence
            + (1 - self.smoothing_alpha) * self._state.fire_smoothed
        )
        self._state.smoke_smoothed = (
            self.smoothing_alpha * smoke_confidence
            + (1 - self.smoothing_alpha) * self._state.smoke_smoothed
        )

        # Record to history
        self._fire_history.append((now, fire_confidence))
        self._smoke_history.append((now, smoke_confidence))

        # Check cooldown — only applies AFTER an alarm has cleared.
        # An ongoing confirmed alarm is NOT suppressed by cooldown.
        if self._state.last_alarm_time > 0:
            elapsed = now - self._state.last_alarm_time
            self._state.in_cooldown = elapsed < self.cooldown_seconds
        else:
            self._state.in_cooldown = False

        # Determine effective thresholds (with hysteresis)
        fire_thresh = self.fire_threshold
        smoke_thresh = self.smoke_threshold
        if self._state.fire_confirmed:
            fire_thresh -= self.hysteresis_margin
        if self._state.smoke_confirmed:
            smoke_thresh -= self.hysteresis_margin

        # Count consecutive detections
        if self._state.fire_smoothed >= fire_thresh:
            self._state.fire_consecutive += 1
        else:
            self._state.fire_consecutive = 0
            self._state.fire_confirmed = False

        if self._state.smoke_smoothed >= smoke_thresh:
            self._state.smoke_consecutive += 1
        else:
            self._state.smoke_consecutive = 0
            self._state.smoke_confirmed = False

        # Count detections in temporal window
        fire_in_window = self._count_in_window(self._fire_history, now, fire_thresh)
        smoke_in_window = self._count_in_window(self._smoke_history, now, smoke_thresh)

        # Determine confirmation
        fire_confirmed = (
            self._state.fire_consecutive >= self.min_consecutive_frames
            and fire_in_window >= self.min_detections_in_window
        )
        smoke_confirmed = (
            self._state.smoke_consecutive >= self.min_consecutive_frames
            and smoke_in_window >= self.min_detections_in_window
        )

        self._state.fire_confirmed = fire_confirmed
        self._state.smoke_confirmed = smoke_confirmed

        # Determine status
        # An active, ongoing alarm stays in ALARM even during cooldown.
        # Cooldown only prevents a NEW alarm after the previous one cleared.
        if fire_confirmed or smoke_confirmed:
            if self._state.status == "ALARM" or not self._state.in_cooldown:
                self._state.status = "ALARM"
                self._state.last_alarm_time = now
            else:
                self._state.status = "MONITORING"
        elif self._state.fire_consecutive > 0 or self._state.smoke_consecutive > 0:
            if self._state.fire_consecutive >= 2 or self._state.smoke_consecutive >= 2:
                self._state.status = "PRE_ALARM"
            else:
                self._state.status = "MONITORING"
        else:
            self._state.status = "NORMAL"

        return self._state

    def _count_in_window(
        self, history: deque, now: float, threshold: float
    ) -> int:
        """Count detections above threshold in temporal window."""
        count = 0
        for ts, conf in reversed(history):
            if now - ts > self.temporal_window:
                break
            if conf >= threshold:
                count += 1
        return count

    def reset(self) -> None:
        """Reset all temporal state."""
        self._fire_history.clear()
        self._smoke_history.clear()
        self._state = TemporalState()
        logger.info("Temporal filter reset")

    @property
    def state(self) -> TemporalState:
        return self._state
