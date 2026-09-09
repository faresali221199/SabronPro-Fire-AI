"""Camera capture module for SabronPro Fire AI."""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """Camera-related errors."""
    pass


class Camera:
    """Camera capture abstraction supporting USB, CSI, video file, and simulation."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        retry_interval: float = 5.0,
        max_retries: int = 10,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._actual_fps = 0.0

    def open(self) -> bool:
        """Open camera with retry logic and auto-search (Full Mode)."""
        sources_to_try = [self.source] if isinstance(self.source, str) else [self.source, 0, 1, 2, 3]
        
        for attempt in range(1, self.max_retries + 1):
            for src in sources_to_try:
                try:
                    logger.info(f"Scanning for camera at source={src} (attempt {attempt}/{self.max_retries})")
                    if isinstance(src, int):
                        # Use CAP_DSHOW on Windows for faster initialization if needed, but CAP_ANY is safer
                        self._cap = cv2.VideoCapture(src, cv2.CAP_ANY)
                    else:
                        self._cap = cv2.VideoCapture(str(src))

                    if self._cap is not None and self._cap.isOpened():
                        # Read a test frame to ensure it's actually working (sometimes it opens virtually but fails to read)
                        ret, test_frame = self._cap.read()
                        if ret and test_frame is not None:
                            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                            self._is_open = True
                            self.source = src  # Update active source
                            logger.info(f"Camera successfully found and opened at source {src}: {self.width}x{self.height}@{self.fps}fps")
                            return True
                        else:
                            self._cap.release()
                except Exception as e:
                    logger.error(f"Camera open error on source {src}: {e}")

            logger.warning(f"No working camera found on any source (attempt {attempt})")
            if attempt < self.max_retries:
                time.sleep(self.retry_interval)

            if attempt < self.max_retries:
                time.sleep(self.retry_interval)

        self._is_open = False
        logger.error("Failed to open camera after all retries")
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self._is_open or self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
            now = time.time()
            if self._last_frame_time > 0:
                dt = now - self._last_frame_time
                if dt > 0:
                    self._actual_fps = 0.9 * self._actual_fps + 0.1 * (1.0 / dt)
            self._last_frame_time = now
        else:
            logger.warning("Failed to read frame from camera")

        return ret, frame

    def release(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Camera released")

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def actual_fps(self) -> float:
        return round(self._actual_fps, 1)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
