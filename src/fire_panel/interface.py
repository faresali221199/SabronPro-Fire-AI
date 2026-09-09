"""Fire alarm panel interface abstraction.

SAFETY NOTICE:
This is a MOCK interface for development purposes only.
Do NOT connect to a real fire alarm control panel until:
1. The exact panel model is identified
2. Communication protocol is documented
3. Electrical interface is verified
4. Proper isolation is in place
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PanelStatus:
    """Fire alarm panel status."""
    connected: bool = False
    status: str = "UNKNOWN"  # NORMAL, ALARM, FAULT, UNKNOWN
    zones: dict = None
    panel_type: str = "mock"

    def __post_init__(self):
        if self.zones is None:
            self.zones = {}


class FireAlarmInterface(ABC):
    """Abstract interface for fire alarm control panels."""

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_status(self) -> PanelStatus:
        ...

    @abstractmethod
    def get_zone_status(self, zone_id: int) -> str:
        ...

    @abstractmethod
    def send_alarm(self, zone_id: int, alarm_type: str, confidence: float) -> bool:
        ...

    @abstractmethod
    def send_fault(self, zone_id: int, fault_type: str) -> bool:
        ...

    @abstractmethod
    def reset(self) -> bool:
        ...


class MockFireAlarmPanel(FireAlarmInterface):
    """Mock fire alarm panel for development without hardware.

    Simulates panel behavior by logging all interactions.
    """

    def __init__(self, num_zones: int = 8):
        self._connected = False
        self._zones: Dict[int, str] = {i: "NORMAL" for i in range(1, num_zones + 1)}
        self._alarm_log: list = []
        logger.info(f"MockFireAlarmPanel created with {num_zones} zones")

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockFireAlarmPanel: Connected (simulated)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MockFireAlarmPanel: Disconnected")

    def get_status(self) -> PanelStatus:
        has_alarm = any(s == "ALARM" for s in self._zones.values())
        has_fault = any(s == "FAULT" for s in self._zones.values())
        if has_alarm:
            status = "ALARM"
        elif has_fault:
            status = "FAULT"
        else:
            status = "NORMAL"
        return PanelStatus(
            connected=self._connected,
            status=status,
            zones=dict(self._zones),
            panel_type="mock",
        )

    def get_zone_status(self, zone_id: int) -> str:
        return self._zones.get(zone_id, "UNKNOWN")

    def send_alarm(self, zone_id: int, alarm_type: str, confidence: float) -> bool:
        if not self._connected:
            logger.warning("MockFireAlarmPanel: Not connected, cannot send alarm")
            return False
        self._zones[zone_id] = "ALARM"
        entry = {
            "zone_id": zone_id,
            "alarm_type": alarm_type,
            "confidence": confidence,
            "action": "ALARM",
        }
        self._alarm_log.append(entry)
        logger.warning(
            f"MockFireAlarmPanel: ALARM Zone {zone_id} "
            f"Type={alarm_type} Confidence={confidence:.2f}"
        )
        return True

    def send_fault(self, zone_id: int, fault_type: str) -> bool:
        if not self._connected:
            return False
        self._zones[zone_id] = "FAULT"
        logger.warning(f"MockFireAlarmPanel: FAULT Zone {zone_id} Type={fault_type}")
        return True

    def reset(self) -> bool:
        for z in self._zones:
            self._zones[z] = "NORMAL"
        logger.info("MockFireAlarmPanel: All zones reset to NORMAL")
        return True

    @property
    def alarm_log(self) -> list:
        return list(self._alarm_log)
