"""SabronPro Fire AI Dashboard - Flask web application.

Lightweight touchscreen-friendly dashboard for real-time monitoring.
"""

import json
import time
import threading
from typing import Optional

from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Dashboard:
    """Real-time monitoring dashboard."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, simulator=None):
        self.host = host
        self.port = port
        self.simulator = simulator
        self.app = Flask(
            __name__,
            template_folder=str(__file__).replace('app.py', 'templates'),
            static_folder=str(__file__).replace('app.py', 'static'),
        )
        self._setup_routes()

        # Shared state (thread-safe via GIL for simple reads)
        self._system_status: dict = {
            "status": "INITIALIZING",
            "camera_status": "UNKNOWN",
            "ai_status": "UNKNOWN",
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0,
            "fire_smoothed": 0.0,
            "smoke_smoothed": 0.0,
            "fire_detected": False,
            "smoke_detected": False,
            "fps": 0.0,
            "inference_latency_ms": 0.0,
            "active_zones": [],
            "timestamp": time.time(),
        }
        self._zone_statuses: dict = {}
        self._recent_events: list = []
        self._latest_frame: Optional[bytes] = None
        self._browser_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def _setup_routes(self):
        """Register Flask routes."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/api/status")
        def api_status():
            return jsonify(self._system_status)

        @self.app.route("/api/zones")
        def api_zones():
            return jsonify(self._zone_statuses)

        @self.app.route("/api/events")
        def api_events():
            return jsonify(self._recent_events[-50:])

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/api/simulation", methods=["POST"])
        def api_simulation():
            from flask import request
            data = request.get_json() or {}
            scenario = data.get("scenario", "normal")
            if self.simulator:
                try:
                    self.simulator.set_scenario(scenario)
                    logger.info(f"Simulator scenario switched via API to: {scenario}")
                    return jsonify({"status": "ok", "scenario": scenario})
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 400
            return jsonify({"status": "ok", "scenario": scenario, "note": "simulation mode inactive"})

        @self.app.route("/api/camera_frame", methods=["POST"])
        def api_camera_frame():
            from flask import request
            import base64
            data = request.get_json()
            if data and "image" in data:
                img_data = data["image"]
                # remove data:image/jpeg;base64,
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                try:
                    img_bytes = base64.b64decode(img_data)
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._frame_lock:
                            self._browser_frame = frame
                    return jsonify({"status": "ok"})
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 400
            return jsonify({"status": "error", "message": "no image provided"}), 400

    def get_browser_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._browser_frame


    def _generate_frames(self):
        """Generate MJPEG frames for video feed."""
        while True:
            with self._frame_lock:
                frame_bytes = self._latest_frame

            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.033)  # ~30fps max

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the latest frame for video feed."""
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._frame_lock:
                self._latest_frame = buffer.tobytes()
        except Exception as e:
            logger.error(f"Frame encode error: {e}")

    def update_status(self, status: dict) -> None:
        """Update system status."""
        self._system_status = status

    def update_zones(self, zones: dict) -> None:
        """Update zone statuses."""
        self._zone_statuses = zones

    def add_event(self, event: dict) -> None:
        """Add event to recent events list."""
        self._recent_events.append(event)
        if len(self._recent_events) > 200:
            self._recent_events = self._recent_events[-100:]

    def run(self, threaded: bool = True) -> Optional[threading.Thread]:
        """Start the dashboard server."""
        if threaded:
            thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="dashboard-server",
            )
            thread.start()
            logger.info(f"Dashboard started on http://{self.host}:{self.port}")
            return thread
        else:
            self._run_server()
            return None

    def _run_server(self):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
