"""Simulation mode for testing without hardware."""

import math
import random
import time
from enum import Enum
from typing import Optional

import numpy as np

from src.inference.engine import Detection, InferenceResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimulationScenario(Enum):
    """Pre-defined simulation scenarios."""
    NORMAL = "normal"
    FIRE = "fire"
    SMOKE = "smoke"
    FIRE_AND_SMOKE = "fire_and_smoke"
    SENSOR_ALARM = "sensor_alarm"
    MANUAL_ALARM = "manual_alarm"
    FAULT = "fault"
    INTERMITTENT_FIRE = "intermittent_fire"  # For testing temporal filter
    FALSE_POSITIVE = "false_positive"  # Low confidence noise


class Simulator:
    """Simulates fire/smoke detections for testing."""

    def __init__(self, scenario: str = "normal", zone_id: int = 1):
        self.scenario = SimulationScenario(scenario)
        self.zone_id = zone_id
        self._frame_count = 0
        self._start_time = time.time()
        logger.info(f"Simulator initialized: scenario={scenario}, zone={zone_id}")

    def generate_frame(self) -> np.ndarray:
        """Generate a simulated camera frame (colored based on scenario)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._frame_count += 1

        if self.scenario == SimulationScenario.FIRE:
            # Red-orange flickering
            intensity = int(150 + 50 * math.sin(self._frame_count * 0.3))
            frame[:, :] = [0, 50, intensity]  # BGR
        elif self.scenario == SimulationScenario.SMOKE:
            # Grey haze
            intensity = int(100 + 30 * math.sin(self._frame_count * 0.1))
            frame[:, :] = [intensity, intensity, intensity]
        elif self.scenario == SimulationScenario.FIRE_AND_SMOKE:
            # Top half smoke, bottom half fire
            smoke_int = int(100 + 20 * math.sin(self._frame_count * 0.1))
            fire_int = int(150 + 50 * math.sin(self._frame_count * 0.3))
            frame[:240, :] = [smoke_int, smoke_int, smoke_int]
            frame[240:, :] = [0, 50, fire_int]
        elif self.scenario == SimulationScenario.FAULT:
            # Black (camera offline simulation)
            pass
        else:
            # Normal - green tinted
            frame[:, :] = [50, 80, 50]

        # Add text overlay
        import cv2
        text = f"SIM: {self.scenario.value} Frame:{self._frame_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def generate_inference_result(self) -> InferenceResult:
        """Generate simulated inference result based on scenario."""
        result = InferenceResult(model_name="simulation")
        noise = random.uniform(-0.05, 0.05)

        if self.scenario == SimulationScenario.FIRE:
            conf = min(1.0, max(0.0, 0.90 + noise))
            result.detections.append(
                Detection(
                    class_id=0, class_name="fire",
                    confidence=conf,
                    bbox=[200, 150, 440, 400],
                )
            )
            result.fire_max_confidence = conf

        elif self.scenario == SimulationScenario.SMOKE:
            conf = min(1.0, max(0.0, 0.85 + noise))
            result.detections.append(
                Detection(
                    class_id=1, class_name="smoke",
                    confidence=conf,
                    bbox=[100, 50, 540, 300],
                )
            )
            result.smoke_max_confidence = conf

        elif self.scenario == SimulationScenario.FIRE_AND_SMOKE:
            fire_conf = min(1.0, max(0.0, 0.92 + noise))
            smoke_conf = min(1.0, max(0.0, 0.87 + noise))
            result.detections.extend([
                Detection(class_id=0, class_name="fire", confidence=fire_conf, bbox=[200, 250, 440, 450]),
                Detection(class_id=1, class_name="smoke", confidence=smoke_conf, bbox=[100, 50, 540, 240]),
            ])
            result.fire_max_confidence = fire_conf
            result.smoke_max_confidence = smoke_conf

        elif self.scenario == SimulationScenario.INTERMITTENT_FIRE:
            # Fire appears every other frame (tests temporal filter)
            if self._frame_count % 3 != 0:
                conf = min(1.0, max(0.0, 0.88 + noise))
                result.detections.append(
                    Detection(class_id=0, class_name="fire", confidence=conf, bbox=[200, 150, 440, 400])
                )
                result.fire_max_confidence = conf

        elif self.scenario == SimulationScenario.FALSE_POSITIVE:
            # Low confidence noise
            conf = min(0.5, max(0.0, 0.30 + noise))
            result.detections.append(
                Detection(class_id=0, class_name="fire", confidence=conf, bbox=[300, 200, 350, 250])
            )
            result.fire_max_confidence = conf

        elif self.scenario == SimulationScenario.NORMAL:
            # No detections
            pass

        result.inference_time_ms = random.uniform(20, 50)
        return result

    def set_scenario(self, scenario: str) -> None:
        """Change simulation scenario."""
        self.scenario = SimulationScenario(scenario)
        self._frame_count = 0
        logger.info(f"Simulation scenario changed to: {scenario}")
