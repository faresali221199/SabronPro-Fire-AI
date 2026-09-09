"""Fire alarm panel interface."""

from src.fire_panel.interface import (
    FireAlarmInterface,
    MockFireAlarmPanel,
    PanelStatus,
)

__all__ = ["FireAlarmInterface", "MockFireAlarmPanel", "PanelStatus"]
