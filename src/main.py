"""SabronPro Fire AI - Main Application Entry Point.

This is the main entry point that orchestrates all components:
- Camera capture
- AI inference
- Decision engine
- Zone engine
- Sensor management
- Dashboard
- Event logging
- Fire panel interface

Usage:
    python -m src.main
    python -m src.main --config configs/
    python -m src.main --simulation fire
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import yaml

from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger
from src.vision.camera import Camera
from src.inference.engine import InferenceEngine
from src.inference.postprocessing import draw_detections, get_detection_centers
from src.decision_engine.temporal_filter import TemporalFilter
from src.decision_engine.decision_engine import DecisionEngine
from src.zones.zone_engine import ZoneEngine
from src.sensors.simulated import SensorManager
from src.fire_panel.interface import MockFireAlarmPanel
from src.database.event_store import EventStore
from src.dashboard.app import Dashboard
from src.simulation.simulator import Simulator


class SabronProFireAI:
    """Main application class."""

    def __init__(self, config_dir: str = "configs", simulation: str = ""):
        self._running = False
        self._simulation_mode = simulation

        # Load configuration
        self.config = Config(config_dir)
        
        # Setup logging
        log_cfg = self.config.get("system.logging", {})
        self.logger = setup_logger(
            name="sabronpro",
            level=log_cfg.get("level", "INFO") if isinstance(log_cfg, dict) else "INFO",
            log_file=log_cfg.get("file") if isinstance(log_cfg, dict) else None,
            console=log_cfg.get("console", True) if isinstance(log_cfg, dict) else True,
        )
        self.logger.info("="*50)
        self.logger.info("SabronPro Fire AI Starting")
        self.logger.info("="*50)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all system components."""
        # Camera or Simulator
        if self._simulation_mode:
            self.logger.info(f"Simulation mode: {self._simulation_mode}")
            self.simulator = Simulator(
                scenario=self._simulation_mode,
                zone_id=1,
            )
            self.camera = None
        else:
            cam_cfg = self.config.get("system.camera", {})
            self.camera = Camera(
                source=cam_cfg.get("source", 0) if isinstance(cam_cfg, dict) else 0,
                width=cam_cfg.get("width", 640) if isinstance(cam_cfg, dict) else 640,
                height=cam_cfg.get("height", 480) if isinstance(cam_cfg, dict) else 480,
                fps=cam_cfg.get("fps", 30) if isinstance(cam_cfg, dict) else 30,
            )
            self.simulator = None

        # Inference engine
        inf_cfg = self.config.get("system.inference", {})
        self.inference_engine = InferenceEngine(
            backend=inf_cfg.get("backend", "onnx") if isinstance(inf_cfg, dict) else "onnx",
            model_path=inf_cfg.get("model_path", "") if isinstance(inf_cfg, dict) else "",
            input_size=inf_cfg.get("input_size", 640) if isinstance(inf_cfg, dict) else 640,
            confidence_threshold=inf_cfg.get("confidence_threshold", 0.25) if isinstance(inf_cfg, dict) else 0.25,
            nms_iou_threshold=inf_cfg.get("nms_iou_threshold", 0.45) if isinstance(inf_cfg, dict) else 0.45,
            device=inf_cfg.get("device", "cpu") if isinstance(inf_cfg, dict) else "cpu",
        )

        # Decision engine with temporal filter
        dec_cfg = self.config.get("decision", {})
        tf_cfg = dec_cfg.get("temporal_filter", {}) if isinstance(dec_cfg, dict) else {}
        d_cfg = dec_cfg.get("decision", {}) if isinstance(dec_cfg, dict) else {}
        
        temporal_filter = TemporalFilter(
            min_consecutive_frames=tf_cfg.get("min_consecutive_frames", 5) if isinstance(tf_cfg, dict) else 5,
            temporal_window_seconds=tf_cfg.get("temporal_window_seconds", 3.0) if isinstance(tf_cfg, dict) else 3.0,
            smoothing_alpha=tf_cfg.get("smoothing_alpha", 0.3) if isinstance(tf_cfg, dict) else 0.3,
            cooldown_seconds=tf_cfg.get("cooldown_seconds", 30.0) if isinstance(tf_cfg, dict) else 30.0,
            hysteresis_margin=tf_cfg.get("hysteresis_margin", 0.10) if isinstance(tf_cfg, dict) else 0.10,
            fire_threshold=d_cfg.get("fire_threshold", 0.70) if isinstance(d_cfg, dict) else 0.70,
            smoke_threshold=d_cfg.get("smoke_threshold", 0.65) if isinstance(d_cfg, dict) else 0.65,
            min_detections_in_window=d_cfg.get("min_detections_in_window", 3) if isinstance(d_cfg, dict) else 3,
        )
        self.decision_engine = DecisionEngine(temporal_filter=temporal_filter)

        # Zone engine
        zone_configs = self.config.get("zones.zones", [])
        self.zone_engine = ZoneEngine(zone_configs if isinstance(zone_configs, list) else [])

        # Sensor manager
        self.sensor_manager = SensorManager()
        sensor_cfg = self.config.get_section("sensors")
        self.sensor_manager.create_simulated_sensors(sensor_cfg)

        # Fire alarm panel (mock)
        self.fire_panel = MockFireAlarmPanel(num_zones=8)

        # Event store
        db_path = self.config.get("system.database.path", "data/events.db")
        self.event_store = EventStore(db_path=db_path if isinstance(db_path, str) else "data/events.db")

        # Dashboard
        dash_cfg = self.config.get("system.dashboard", {})
        self.dashboard = Dashboard(
            host=dash_cfg.get("host", "0.0.0.0") if isinstance(dash_cfg, dict) else "0.0.0.0",
            port=dash_cfg.get("port", 8080) if isinstance(dash_cfg, dict) else 8080,
            simulator=self.simulator,
        )

    def start(self):
        """Start the application."""
        self._running = True

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Connect fire panel
        self.fire_panel.connect()

        # Start camera
        camera_online = False
        if self.camera:
            camera_online = self.camera.open()
            if camera_online:
                self.zone_engine.set_frame_size(self.camera.width, self.camera.height)
        elif self.simulator:
            camera_online = True  # Simulation mode
            self.zone_engine.set_frame_size(640, 480)  # Match simulated frame size

        self.decision_engine.set_camera_status(camera_online)

        # Load AI model
        ai_ready = False
        if self.inference_engine._model_path:
            ai_ready = self.inference_engine.load()
        elif self._simulation_mode:
            ai_ready = True  # Simulation generates its own results

        self.decision_engine.set_ai_status(ai_ready or bool(self._simulation_mode))

        # Start dashboard
        dash_cfg = self.config.get("system.dashboard", {})
        if isinstance(dash_cfg, dict) and dash_cfg.get("enabled", True):
            self.dashboard.run(threaded=True)

        self.logger.info("SabronPro Fire AI is running")
        self.logger.info(f"Camera: {'ONLINE' if camera_online else 'OFFLINE'}")
        self.logger.info(f"AI: {'READY' if ai_ready else 'NOT READY (simulation)' if self._simulation_mode else 'NOT READY'}")
        if isinstance(dash_cfg, dict):
            self.logger.info(f"Dashboard: http://{dash_cfg.get('host', '0.0.0.0')}:{dash_cfg.get('port', 8080)}")

        # Main loop
        self._main_loop()

    def _main_loop(self):
        """Main processing loop."""
        frame_count = 0
        fps_start = time.time()
        fps = 0.0
        last_alarm_event_time = 0

        while self._running:
            try:
                # Get frame
                fps = self.camera.fps if self.camera else 30
                
                # Check for browser-injected live camera frame
                browser_frame = self.dashboard.get_browser_frame()
                if browser_frame is not None:
                    frame = browser_frame
                    if self.inference_engine.is_ready:
                        result = self.inference_engine.infer(frame)
                    else:
                        from src.inference.engine import InferenceResult
                        result = InferenceResult()
                    self.decision_engine.set_camera_status(True)
                elif self.simulator:
                    frame = self.simulator.generate_frame()
                    result = self.simulator.generate_inference_result()
                    self.decision_engine.set_camera_status(True)
                elif self.camera and self.camera.is_open:
                    ret, frame = self.camera.read()
                    if not ret or frame is None:
                        self.decision_engine.set_camera_status(False)
                        time.sleep(0.1)
                        continue
                    # Run inference
                    if self.inference_engine.is_ready:
                        result = self.inference_engine.infer(frame)
                    else:
                        from src.inference.engine import InferenceResult
                        result = InferenceResult()
                    self.decision_engine.set_camera_status(True)
                else:
                    time.sleep(0.5)
                    continue

                # Zone mapping
                active_zones = []
                if result.detections:
                    centers = get_detection_centers(result.detections)
                    zone_ids = self.zone_engine.get_zones_for_detections(centers)
                    active_zones = [z for z in zone_ids if z is not None]
                    for z_id in active_zones:
                        self.zone_engine.set_zone_status(z_id, "ALARM")

                # Decision engine
                system_status = self.decision_engine.process(
                    inference_result=result,
                    active_zones=active_zones,
                    fps=fps,
                )

                # Handle alarm events
                if system_status.status == "ALARM" and (time.time() - last_alarm_event_time) > 5:
                    # Use simulator zone as fallback
                    event_zone = active_zones[0] if active_zones else (
                        self.simulator.zone_id if self.simulator else None
                    )
                    if active_zones:
                        for z_id in active_zones:
                            alarm_type = "fire" if system_status.fire_detected else "smoke"
                            self.fire_panel.send_alarm(
                                zone_id=z_id,
                                alarm_type=alarm_type,
                                confidence=max(system_status.fire_confidence, system_status.smoke_confidence),
                            )
                    elif self.simulator:
                        alarm_type = "fire" if system_status.fire_detected else "smoke"
                        self.fire_panel.send_alarm(
                            zone_id=self.simulator.zone_id,
                            alarm_type=alarm_type,
                            confidence=max(system_status.fire_confidence, system_status.smoke_confidence),
                        )
                    event_id = self.event_store.add_event(
                        event_type="ALARM",
                        zone_id=event_zone,
                        fire_confidence=system_status.fire_confidence,
                        smoke_confidence=system_status.smoke_confidence,
                        decision=system_status.status,
                        system_status=system_status.status,
                    )
                    # Push to dashboard events
                    self.dashboard.add_event({
                        "event_type": "ALARM",
                        "zone_id": event_zone,
                        "fire_confidence": system_status.fire_confidence,
                        "smoke_confidence": system_status.smoke_confidence,
                        "timestamp": time.time(),
                    })
                    last_alarm_event_time = time.time()

                # Draw detections on frame
                if frame is not None and result.detections:
                    frame = draw_detections(frame, result)

                # Update dashboard
                self.dashboard.update_status(system_status.to_dict())
                self.dashboard.update_zones(self.zone_engine.get_all_statuses())
                if frame is not None:
                    self.dashboard.update_frame(frame)

                # Reset zone statuses for non-active zones
                for zone in self.zone_engine.zones:
                    if zone.id not in active_zones:
                        zone.status = "NORMAL"

                # FPS calculation
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                # Small delay for CPU
                time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(0.5)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self):
        """Stop the application gracefully."""
        self._running = False
        self.logger.info("Shutting down SabronPro Fire AI...")

        if self.camera:
            self.camera.release()
        self.fire_panel.disconnect()
        self.event_store.close()

        self.logger.info("SabronPro Fire AI stopped")


def main():
    parser = argparse.ArgumentParser(description="SabronPro Fire AI")
    parser.add_argument("--config", default="configs", help="Configuration directory")
    parser.add_argument(
        "--simulation",
        default="",
        choices=["", "normal", "fire", "smoke", "fire_and_smoke",
                 "intermittent_fire", "false_positive", "fault"],
        help="Run in simulation mode with specified scenario",
    )
    args = parser.parse_args()

    app = SabronProFireAI(
        config_dir=args.config,
        simulation=args.simulation,
    )
    app.start()


if __name__ == "__main__":
    main()
