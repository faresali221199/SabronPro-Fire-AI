"""Decision engine combining temporal filtering, zone mapping, and sensor fusion."""

import time
from dataclasses import dataclass
from typing import List, Optional

from src.decision_engine.temporal_filter import TemporalFilter, TemporalState
from src.inference.engine import InferenceResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemStatus:
    """Overall system status."""
    status: str = "NORMAL"  # NORMAL, MONITORING, PRE_ALARM, ALARM, FAULT
    fire_detected: bool = False
    smoke_detected: bool = False
    fire_confidence: float = 0.0
    smoke_confidence: float = 0.0
    fire_smoothed: float = 0.0
    smoke_smoothed: float = 0.0
    active_zones: list = None
    camera_status: str = "UNKNOWN"
    ai_status: str = "UNKNOWN"
    fps: float = 0.0
    inference_latency_ms: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.active_zones is None:
            self.active_zones = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "fire_detected": self.fire_detected,
            "smoke_detected": self.smoke_detected,
            "fire_confidence": round(self.fire_confidence, 4),
            "smoke_confidence": round(self.smoke_confidence, 4),
            "fire_smoothed": round(self.fire_smoothed, 4),
            "smoke_smoothed": round(self.smoke_smoothed, 4),
            "active_zones": self.active_zones,
            "camera_status": self.camera_status,
            "ai_status": self.ai_status,
            "fps": round(self.fps, 1),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "timestamp": self.timestamp,
        }


class DecisionEngine:
    """Main decision engine coordinating all detection logic."""

    def __init__(
        self,
        temporal_filter: Optional[TemporalFilter] = None,
        fire_threshold: float = 0.70,
        smoke_threshold: float = 0.65,
    ):
        self.temporal_filter = temporal_filter or TemporalFilter(
            fire_threshold=fire_threshold,
            smoke_threshold=smoke_threshold,
        )
        self._camera_online = False
        self._ai_ready = False
        self._fault_reasons: List[str] = []

    def process(
        self,
        inference_result: Optional[InferenceResult],
        active_zones: Optional[List[int]] = None,
        fps: float = 0.0,
    ) -> SystemStatus:
        """Process inference result and produce system status."""
        status = SystemStatus()
        status.fps = fps
        status.camera_status = "ONLINE" if self._camera_online else "OFFLINE"
        status.ai_status = "RUNNING" if self._ai_ready else "NOT READY"

        # Check for faults
        if not self._camera_online:
            status.status = "FAULT"
            status.camera_status = "OFFLINE"
            return status

        if not self._ai_ready:
            status.status = "FAULT"
            status.ai_status = "NOT READY"
            return status

        if inference_result is None:
            status.status = "FAULT"
            return status

        # Update raw confidences
        status.fire_confidence = inference_result.fire_max_confidence
        status.smoke_confidence = inference_result.smoke_max_confidence
        status.inference_latency_ms = inference_result.inference_time_ms

        # Apply temporal filter
        temporal_state = self.temporal_filter.update(
            inference_result.fire_max_confidence,
            inference_result.smoke_max_confidence,
        )

        status.fire_smoothed = temporal_state.fire_smoothed
        status.smoke_smoothed = temporal_state.smoke_smoothed
        status.fire_detected = temporal_state.fire_confirmed
        status.smoke_detected = temporal_state.smoke_confirmed
        status.status = temporal_state.status
        status.active_zones = active_zones or []

        return status

    def set_camera_status(self, online: bool) -> None:
        self._camera_online = online

    def set_ai_status(self, ready: bool) -> None:
        self._ai_ready = ready

    def add_fault(self, reason: str) -> None:
        if reason not in self._fault_reasons:
            self._fault_reasons.append(reason)
            logger.warning(f"Fault added: {reason}")

    def clear_fault(self, reason: str) -> None:
        if reason in self._fault_reasons:
            self._fault_reasons.remove(reason)

    def reset(self) -> None:
        self.temporal_filter.reset()
        self._fault_reasons.clear()
        logger.info("Decision engine reset")
