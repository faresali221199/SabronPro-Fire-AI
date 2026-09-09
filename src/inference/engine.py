"""Inference engine for SabronPro Fire AI.

Supports multiple backends: PyTorch, ONNX Runtime, Hailo.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 1) for c in self.bbox],
        }


@dataclass
class InferenceResult:
    """Complete inference result for a single frame."""
    detections: List[Detection] = field(default_factory=list)
    fire_max_confidence: float = 0.0
    smoke_max_confidence: float = 0.0
    inference_time_ms: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "fire_max_confidence": round(self.fire_max_confidence, 4),
            "smoke_max_confidence": round(self.smoke_max_confidence, 4),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_name": self.model_name,
        }


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        ...

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend."""

    def __init__(self, device: str = "cpu"):
        self._session = None
        self._input_name = ""
        self._device = device

    def load_model(self, model_path: str) -> bool:
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self._device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self._session = ort.InferenceSession(model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            logger.info(f"ONNX model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded")
        outputs = self._session.run(None, {self._input_name: input_tensor})
        return outputs[0]

    def is_loaded(self) -> bool:
        return self._session is not None


class PyTorchBackend(InferenceBackend):
    """PyTorch/Ultralytics inference backend (for development)."""

    def __init__(self, device: str = "cpu"):
        self._model = None
        self._device = device

    def load_model(self, model_path: str) -> bool:
        try:
            import pathlib
            orig_exists = pathlib.Path.exists
            def safe_exists(p, *args, **kwargs):
                try:
                    if 'WpSystem' in str(p):
                        return False
                    return orig_exists(p, *args, **kwargs)
                except Exception:
                    return False
            pathlib.Path.exists = safe_exists

            from ultralytics import YOLO
            self._model = YOLO(model_path)
            logger.info(f"PyTorch model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """Note: PyTorch backend uses Ultralytics which handles its own preprocessing."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        raise NotImplementedError("Use predict_frame() for PyTorch backend")

    def predict_frame(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list:
        """Predict directly from frame using Ultralytics."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        results = self._model.predict(frame, conf=conf, iou=iou, verbose=False)
        return results

    def is_loaded(self) -> bool:
        return self._model is not None


class HailoBackend(InferenceBackend):
    """Hailo inference backend for Raspberry Pi AI HAT+."""

    def __init__(self):
        self._hef = None
        self._device = None
        self._configured_network = None

    def load_model(self, model_path: str) -> bool:
        try:
            from hailo_platform import HEF, Device, ConfigureParams
            self._hef = HEF(model_path)
            devices = Device.scan()
            if not devices:
                logger.error("No Hailo devices found")
                return False
            self._device = Device(devices[0])
            configure_params = ConfigureParams.create_from_hef(self._hef, interface=None)
            self._configured_network = self._device.configure(self._hef, configure_params)
            logger.info(f"Hailo model loaded: {model_path}")
            return True
        except ImportError:
            logger.error("HailoRT not installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to load Hailo model: {e}")
            return False

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self._configured_network is None:
            raise RuntimeError("Hailo model not loaded")
        raise NotImplementedError("Hailo inference requires hardware testing")

    def is_loaded(self) -> bool:
        return self._configured_network is not None


CLASS_NAMES = {0: "fire", 1: "smoke"}


class InferenceEngine:
    """Main inference engine that manages backends and produces structured results."""

    def __init__(
        self,
        backend: str = "onnx",
        model_path: str = "",
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        device: str = "cpu",
        class_names: Optional[dict] = None,
    ):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.class_names = class_names or CLASS_NAMES
        self._backend_name = backend
        self._model_path = model_path
        self._backend: Optional[InferenceBackend] = None
        self._is_ready = False

        if backend == "onnx":
            self._backend = ONNXBackend(device=device)
        elif backend == "pytorch":
            self._backend = PyTorchBackend(device=device)
        elif backend == "hailo":
            self._backend = HailoBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def load(self) -> bool:
        """Load the model."""
        if not self._model_path:
            logger.error("No model path specified")
            return False
        if not Path(self._model_path).exists():
            logger.error(f"Model file not found: {self._model_path}")
            return False
        self._is_ready = self._backend.load_model(self._model_path)
        return self._is_ready

    def infer(self, frame: np.ndarray) -> InferenceResult:
        """Run inference on a frame and return structured results."""
        result = InferenceResult(model_name=self._backend_name)

        if not self._is_ready:
            return result

        start_time = time.perf_counter()

        try:
            if isinstance(self._backend, PyTorchBackend):
                yolo_results = self._backend.predict_frame(
                    frame, conf=self.confidence_threshold, iou=self.nms_iou_threshold
                )
                if yolo_results and len(yolo_results) > 0:
                    r = yolo_results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            raw_cls_id = int(box.cls[0])
                            cls_id = 1 - raw_cls_id  # عكس الفئة لتصحيح التبديل
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy().tolist()
                            det = Detection(
                                class_id=cls_id,
                                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                                confidence=conf,
                                bbox=xyxy,
                            )
                            result.detections.append(det)
            else:
                from src.vision.preprocessing import preprocess_frame, scale_boxes

                tensor, scale, padding = preprocess_frame(frame, self.input_size)
                raw_output = self._backend.predict(tensor)

                if raw_output.shape[-1] < raw_output.shape[-2]:
                    raw_output = raw_output.transpose(0, 2, 1)

                preds = raw_output[0]
                boxes_xywh = preds[:, :4]
                scores = preds[:, 4:]

                boxes_xyxy = np.zeros_like(boxes_xywh)
                boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
                boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
                boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
                boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

                class_ids = np.argmax(scores, axis=1)
                max_scores = np.max(scores, axis=1)

                mask = max_scores > self.confidence_threshold
                filtered_boxes = boxes_xyxy[mask]
                filtered_scores = max_scores[mask]
                filtered_classes = class_ids[mask]

                keep = self._nms(
                    filtered_boxes, filtered_scores, filtered_classes, self.nms_iou_threshold
                )

                if len(keep) > 0:
                    final_boxes = scale_boxes(
                        filtered_boxes[keep], scale, padding, frame.shape[:2]
                    )
                    for i, idx in enumerate(keep):
                        raw_cls_id = int(filtered_classes[idx])
                        cls_id = 1 - raw_cls_id  # عكس الفئة لتصحيح التبديل
                        det = Detection(
                            class_id=cls_id,
                            class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                            confidence=float(filtered_scores[idx]),
                            bbox=final_boxes[i].tolist(),
                        )
                        result.detections.append(det)

        except Exception as e:
            logger.error(f"Inference error: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000
        result.inference_time_ms = elapsed

        for det in result.detections:
            if det.class_name == "fire":
                result.fire_max_confidence = max(result.fire_max_confidence, det.confidence)
            elif det.class_name == "smoke":
                result.smoke_max_confidence = max(result.smoke_max_confidence, det.confidence)

        return result

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> list:
        if len(boxes) == 0:
            return []

        keep = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            order = cls_scores.argsort()[::-1]

            while len(order) > 0:
                i = order[0]
                keep.append(cls_indices[i])

                if len(order) == 1:
                    break

                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h

                area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
                area_rest = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * (
                    cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1]
                )
                iou = intersection / (area_i + area_rest - intersection + 1e-6)

                remaining = np.where(iou <= iou_threshold)[0]
                order = order[remaining + 1]

        return keep

    @property
    def is_ready(self) -> bool:
        return self._is_ready