"""Sensor abstraction layer."""

from src.sensors.base import BaseSensor, SensorReading, SensorState
from src.sensors.simulated import SensorManager, SimulatedSensor

__all__ = [
    "BaseSensor",
    "SensorReading",
    "SensorState",
    "SimulatedSensor",
    "SensorManager",
]
