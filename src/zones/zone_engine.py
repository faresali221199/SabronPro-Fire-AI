"""Zone engine for spatial zone mapping.

Maps detection coordinates to physical zones on the fire alarm training board.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from shapely.geometry import Point, Polygon

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Zone:
    """A physical zone on the training board."""
    id: int
    name: str
    detector_type: str
    polygon: Polygon
    color: Tuple[int, int, int]
    enabled: bool = True
    status: str = "NORMAL"  # NORMAL, ALARM, FAULT


class ZoneEngine:
    """Maps detection coordinates to training board zones."""

    def __init__(self, zone_configs: Optional[list] = None):
        self.zones: List[Zone] = []
        self._frame_width = 640
        self._frame_height = 480
        if zone_configs:
            self.load_zones(zone_configs)

    def load_zones(self, zone_configs: list) -> None:
        """Load zones from configuration.

        Args:
            zone_configs: List of zone dicts from zones.yaml
        """
        self.zones = []
        for zc in zone_configs:
            if not zc.get("enabled", True):
                continue
            try:
                poly_points = zc["polygon"]
                polygon = Polygon(poly_points)
                zone = Zone(
                    id=zc["id"],
                    name=zc["name"],
                    detector_type=zc.get("detector_type", "unknown"),
                    polygon=polygon,
                    color=tuple(zc.get("color", [255, 255, 255])),
                    enabled=zc.get("enabled", True),
                )
                self.zones.append(zone)
                logger.info(f"Zone {zone.id} loaded: {zone.name}")
            except Exception as e:
                logger.error(f"Failed to load zone config: {e}")

    def set_frame_size(self, width: int, height: int) -> None:
        """Set frame dimensions for coordinate scaling."""
        self._frame_width = width
        self._frame_height = height

    def get_zone_for_point(
        self, x: float, y: float, normalized: bool = False
    ) -> Optional[Zone]:
        """Find which zone contains the given point.

        Args:
            x: X coordinate (pixels or normalized 0-1)
            y: Y coordinate (pixels or normalized 0-1)
            normalized: If False, normalize pixel coords to 0-1
        """
        if not normalized:
            x = x / self._frame_width
            y = y / self._frame_height

        point = Point(x, y)
        for zone in self.zones:
            if zone.enabled and zone.polygon.contains(point):
                return zone
        return None

    def get_zone_by_id(self, zone_id: int) -> Optional[Zone]:
        """Get zone by ID."""
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        return None

    def get_zones_for_detections(
        self, detection_centers: List[Tuple[float, float]], normalized: Optional[bool] = None
    ) -> List[Optional[int]]:
        """Map a list of detection center points to zone IDs."""
        zone_ids = []
        for cx, cy in detection_centers:
            norm = normalized if normalized is not None else (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)
            zone = self.get_zone_for_point(cx, cy, normalized=norm)
            zone_ids.append(zone.id if zone else None)
        return zone_ids

    def set_zone_status(self, zone_id: int, status: str) -> None:
        """Update zone status."""
        zone = self.get_zone_by_id(zone_id)
        if zone:
            zone.status = status

    def get_all_statuses(self) -> dict:
        """Get status of all zones."""
        return {
            zone.id: {
                "name": zone.name,
                "status": zone.status,
                "detector_type": zone.detector_type,
            }
            for zone in self.zones
        }

    def reset_all(self) -> None:
        """Reset all zones to NORMAL."""
        for zone in self.zones:
            zone.status = "NORMAL"
