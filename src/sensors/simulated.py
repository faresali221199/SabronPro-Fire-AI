"""Simulated sensors for development and testing."""

from src.sensors.base import BaseSensor, SensorReading, SensorState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimulatedSensor(BaseSensor):
    """A simulated sensor that can be manually triggered."""

    def __init__(self, sensor_id: str, sensor_type: str, zone_id: int):
        super().__init__(sensor_id, sensor_type, zone_id)
        self._value: float = 0.0

    def read(self) -> SensorReading:
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            state=self._state,
            zone_id=self.zone_id,
            value=self._value,
        )

    def trigger(self, value: float = 1.0) -> None:
        """Manually trigger the sensor (simulate alarm)."""
        self._state = SensorState.ACTIVE
        self._value = value
        logger.info(f"Sensor {self.sensor_id} triggered (zone {self.zone_id})")

    def set_fault(self) -> None:
        """Set sensor to fault state."""
        self._state = SensorState.FAULT
        logger.warning(f"Sensor {self.sensor_id} fault (zone {self.zone_id})")

    def reset(self) -> None:
        self._state = SensorState.NORMAL
        self._value = 0.0


class SensorManager:
    """Manages all sensors."""

    def __init__(self):
        self._sensors: dict[str, BaseSensor] = {}

    def add_sensor(self, sensor: BaseSensor) -> None:
        self._sensors[sensor.sensor_id] = sensor

    def get_sensor(self, sensor_id: str) -> BaseSensor | None:
        return self._sensors.get(sensor_id)

    def read_all(self) -> list[SensorReading]:
        return [s.read() for s in self._sensors.values()]

    def get_active_sensors(self) -> list[SensorReading]:
        return [s.read() for s in self._sensors.values() if s.state == SensorState.ACTIVE]

    def get_zone_sensors(self, zone_id: int) -> list[BaseSensor]:
        return [s for s in self._sensors.values() if s.zone_id == zone_id]

    def reset_all(self) -> None:
        for s in self._sensors.values():
            s.reset()

    def create_simulated_sensors(self, sensor_configs: dict) -> None:
        """Create simulated sensors from sensor configuration."""
        for sensor_type, config in sensor_configs.get("sensor_types", {}).items():
            for zone_id in config.get("zones", []):
                sensor_id = f"{sensor_type}_zone{zone_id}"
                sensor = SimulatedSensor(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    zone_id=zone_id,
                )
                self.add_sensor(sensor)
                logger.info(f"Created simulated sensor: {sensor_id}")

    @property
    def sensor_count(self) -> int:
        return len(self._sensors)

    @property
    def sensor_ids(self) -> list[str]:
        return list(self._sensors.keys())
