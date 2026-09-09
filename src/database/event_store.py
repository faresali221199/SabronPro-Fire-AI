"""SQLite event storage for SabronPro Fire AI."""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventStore:
    """SQLite-based event storage."""

    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    zone_id INTEGER,
                    fire_confidence REAL DEFAULT 0.0,
                    smoke_confidence REAL DEFAULT 0.0,
                    sensor_states TEXT,
                    decision TEXT,
                    system_status TEXT,
                    model_version TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id)
            """)
            self._conn.commit()
            logger.info(f"Event database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def add_event(
        self,
        event_type: str,
        zone_id: Optional[int] = None,
        fire_confidence: float = 0.0,
        smoke_confidence: float = 0.0,
        sensor_states: Optional[Dict] = None,
        decision: str = "",
        system_status: str = "",
        model_version: str = "",
        details: Optional[Dict] = None,
    ) -> str:
        """Add a new event. Returns event ID."""
        event_id = str(uuid.uuid4())[:8]
        timestamp = time.time()

        try:
            self._conn.execute(
                """
                INSERT INTO events
                (id, timestamp, event_type, zone_id, fire_confidence,
                 smoke_confidence, sensor_states, decision, system_status,
                 model_version, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    timestamp,
                    event_type,
                    zone_id,
                    fire_confidence,
                    smoke_confidence,
                    json.dumps(sensor_states or {}),
                    decision,
                    system_status,
                    model_version,
                    json.dumps(details or {}),
                ),
            )
            self._conn.commit()
            logger.info(f"Event stored: {event_id} type={event_type} zone={zone_id}")
            return event_id
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return ""

    def get_events(
        self,
        event_type: Optional[str] = None,
        zone_id: Optional[int] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query events with optional filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if zone_id is not None:
            query += " AND zone_id = ?"
            params.append(zone_id)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent events."""
        return self.get_events(limit=limit)

    def get_event_count(self) -> int:
        """Get total event count."""
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def cleanup(self, max_events: int = 100000) -> int:
        """Remove oldest events if exceeding max count."""
        count = self.get_event_count()
        if count <= max_events:
            return 0
        to_delete = count - max_events
        try:
            self._conn.execute(
                "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY timestamp ASC LIMIT ?)",
                (to_delete,),
            )
            self._conn.commit()
            logger.info(f"Cleaned up {to_delete} old events")
            return to_delete
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
