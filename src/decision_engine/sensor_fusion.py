"""Sensor fusion engine combining vision and physical sensors.

Combines:
- Vision confidence (AI detection)
- Physical sensor state
- Zone state
- Temporal consistency

NOTE: Fusion weights are initial estimates and MUST be calibrated
through testing before production use.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.sensors.base import SensorReading, SensorState
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusionResult:
    """Result of sensor fusion."""
    combined_fire_score: float = 0.0
    combined_smoke_score: float = 0.0
    vision_contributed: bool = False
    sensors_contributed: bool = False
    contributing_sensors: list = None
    zone_id: Optional[int] = None
    recommendation: str = "NORMAL"  # NORMAL, MONITOR, ALARM

    def __post_init__(self):
        if self.contributing_sensors is None:
            self.contributing_sensors = []

    def to_dict(self) -> dict:
        return {
            "combined_fire_score": round(self.combined_fire_score, 4),
            "combined_smoke_score": round(self.combined_smoke_score, 4),
            "vision_contributed": self.vision_contributed,
            "sensors_contributed": self.sensors_contributed,
            "contributing_sensors": self.contributing_sensors,
            "zone_id": self.zone_id,
            "recommendation": self.recommendation,
        }


class SensorFusion:
    """Combines vision AI and physical sensor inputs.

    IMPORTANT: The weights below are initial estimates.
    They are NOT calibrated and should NOT be used in
    production without proper validation and testing.
    """

    def __init__(
        self,
        vision_weight: float = 0.6,
        sensor_weight: float = 0.3,
        temporal_weight: float = 0.1,
        alarm_threshold: float = 0.70,
    ):
        # Document that these are uncalibrated initial estimates
        self.vision_weight = vision_weight
        self.sensor_weight = sensor_weight
        self.temporal_weight = temporal_weight
        self.alarm_threshold = alarm_threshold

        logger.info(
            f"Sensor fusion initialized (UNCALIBRATED weights: "
            f"vision={vision_weight}, sensor={sensor_weight}, "
            f"temporal={temporal_weight})"
        )

    def fuse(
        self,
        fire_vision_confidence: float,
        smoke_vision_confidence: float,
        sensor_readings: List[SensorReading],
        temporal_fire_confirmed: bool = False,
        temporal_smoke_confirmed: bool = False,
        zone_id: Optional[int] = None,
    ) -> FusionResult:
        """Fuse vision and sensor inputs."""
        result = FusionResult(zone_id=zone_id)

        # Vision contribution
        vision_fire = fire_vision_confidence * self.vision_weight
        vision_smoke = smoke_vision_confidence * self.vision_weight
        result.vision_contributed = fire_vision_confidence > 0.1 or smoke_vision_confidence > 0.1

        # Sensor contribution
        sensor_fire = 0.0
        sensor_smoke = 0.0
        for reading in sensor_readings:
            if reading.state == SensorState.ACTIVE:
                result.sensors_contributed = True
                result.contributing_sensors.append(reading.sensor_id)
                if reading.sensor_type in ("flame", "fixed_heat", "rate_of_rise"):
                    sensor_fire = max(sensor_fire, self.sensor_weight)
                elif reading.sensor_type == "smoke":
                    sensor_smoke = max(sensor_smoke, self.sensor_weight)
                elif reading.sensor_type == "multi_sensor":
                    sensor_fire = max(sensor_fire, self.sensor_weight * 0.7)
                    sensor_smoke = max(sensor_smoke, self.sensor_weight * 0.7)
                elif reading.sensor_type == "gas":
                    sensor_fire = max(sensor_fire, self.sensor_weight * 0.5)

        # Temporal contribution
        temporal_fire = self.temporal_weight if temporal_fire_confirmed else 0.0
        temporal_smoke = self.temporal_weight if temporal_smoke_confirmed else 0.0

        # Combined scores
        result.combined_fire_score = min(1.0, vision_fire + sensor_fire + temporal_fire)
        result.combined_smoke_score = min(1.0, vision_smoke + sensor_smoke + temporal_smoke)

        # Recommendation
        if result.combined_fire_score >= self.alarm_threshold or result.combined_smoke_score >= self.alarm_threshold:
            result.recommendation = "ALARM"
        elif result.combined_fire_score >= self.alarm_threshold * 0.7 or result.combined_smoke_score >= self.alarm_threshold * 0.7:
            result.recommendation = "MONITOR"
        else:
            result.recommendation = "NORMAL"

        return result
