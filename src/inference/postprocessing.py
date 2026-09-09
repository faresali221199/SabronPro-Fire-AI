"""Post-processing utilities for inference results."""

from typing import List, Tuple

import cv2
import numpy as np

from src.inference.engine import Detection, InferenceResult


def draw_detections(
    frame: np.ndarray,
    result: InferenceResult,
    fire_color: Tuple[int, int, int] = (0, 0, 255),
    smoke_color: Tuple[int, int, int] = (128, 128, 128),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw detection boxes and labels on frame."""
    annotated = frame.copy()

    for det in result.detections:
        color = fire_color if det.class_name == "fire" else smoke_color
        x1, y1, x2, y2 = [int(c) for c in det.bbox]

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    # Draw FPS/latency info
    info = f"Inference: {result.inference_time_ms:.1f}ms"
    cv2.putText(
        annotated, info, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    return annotated


def get_detection_centers(detections: List[Detection]) -> List[Tuple[float, float]]:
    """Get center points of detections for zone mapping."""
    centers = []
    for det in detections:
        cx = (det.bbox[0] + det.bbox[2]) / 2
        cy = (det.bbox[1] + det.bbox[3]) / 2
        centers.append((cx, cy))
    return centers


def filter_by_confidence(
    result: InferenceResult,
    fire_threshold: float = 0.5,
    smoke_threshold: float = 0.5,
) -> InferenceResult:
    """Filter detections by class-specific confidence thresholds."""
    filtered = []
    for det in result.detections:
        if det.class_name == "fire" and det.confidence >= fire_threshold:
            filtered.append(det)
        elif det.class_name == "smoke" and det.confidence >= smoke_threshold:
            filtered.append(det)

    new_result = InferenceResult(
        detections=filtered,
        inference_time_ms=result.inference_time_ms,
        model_name=result.model_name,
    )
    for det in filtered:
        if det.class_name == "fire":
            new_result.fire_max_confidence = max(new_result.fire_max_confidence, det.confidence)
        elif det.class_name == "smoke":
            new_result.smoke_max_confidence = max(new_result.smoke_max_confidence, det.confidence)

    return new_result
