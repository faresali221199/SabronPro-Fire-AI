"""Image preprocessing for inference."""

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with letterboxing to maintain aspect ratio.

    Returns:
        Tuple of (resized_image, scale_ratio, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, scale, (pad_w, pad_h)


def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 640,
    normalize: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Full preprocessing pipeline: letterbox, BGR->RGB, normalize, NCHW.

    Returns:
        Tuple of (preprocessed_tensor, scale, padding)
    """
    padded, scale, padding = letterbox(frame, target_size)

    # BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # HWC to CHW
    chw = rgb.transpose(2, 0, 1)

    # Normalize to 0-1
    if normalize:
        tensor = chw.astype(np.float32) / 255.0
    else:
        tensor = chw.astype(np.float32)

    # Add batch dimension: BCHW
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, padding


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Scale detection boxes back to original image coordinates.

    Args:
        boxes: Array of [x1, y1, x2, y2] boxes
        scale: Scale ratio from letterboxing
        padding: (pad_w, pad_h) from letterboxing
        original_shape: (height, width) of original image
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy().astype(np.float32)
    pad_w, pad_h = padding

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= scale

    # Clip to image boundaries
    h, w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    return boxes
