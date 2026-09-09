"""Base sensor abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SensorState(Enum):
    """Sensor operational state."""
    NORMAL = "NORMAL"
    ACTIVE = "ACTIVE"  # Detection/alarm triggered
    FAULT = "FAULT"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SensorReading:
    """A reading from a sensor."""
    sensor_id: str
    sensor_type: str
    state: SensorState
    zone_id: int
    value: Optional[float] = None  # Analog value if applicable
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "state": self.state.value,
            "zone_id": self.zone_id,
            "value": self.value,
            "description": self.description,
        }


class BaseSensor(ABC):
    """Abstract base class for all sensor types."""

    def __init__(self, sensor_id: str, sensor_type: str, zone_id: int):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.zone_id = zone_id
        self._state = SensorState.NORMAL

    @abstractmethod
    def read(self) -> SensorReading:
        """Read current sensor value."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset sensor to normal state."""
        ...

    @property
    def state(self) -> SensorState:
        return self._state

    @state.setter
    def state(self, value: SensorState) -> None:
        self._state = value
