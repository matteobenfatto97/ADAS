from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

from adas.perception.types import Detection


class ObjectDetector:
    """Wrapper YOLOv8 con ottimizzazioni real-time.

    Patch 6:
    - supporta modelli OpenVINO esportati per Intel Iris Xe / Intel CPU;
    - passa `classes=[...]` direttamente a YOLO, cosi' non calcola output inutili;
    - usa predict() sui backend OpenVINO, evitando tracking instabile sui backend export;
    - mantiene `half=True` solo su CUDA, mentre FP16 OpenVINO si decide in export.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 416,
        device: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        tracker: str = "bytetrack.yaml",
        enable_tracking: bool = True,
        half: bool = False,
        max_det: int = 12,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modello non trovato: {path}. Metti yolov8n.pt in models/ oppure usa --model /percorso/modello.pt"
            )
        self.is_openvino = path.is_dir() or str(path).lower().endswith("_openvino_model")
        if device is not None and str(device).lower().startswith("intel:") and not self.is_openvino:
            raise ValueError(
                "device='intel:gpu' richiede un modello esportato in OpenVINO, non un .pt. "
                "Esegui: python scripts/export_openvino.py --model models/yolov8n.pt --half "
                "e poi usa --model models/yolov8n_openvino_model"
            )
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.target_classes = set(target_classes or [])
        self.tracker = tracker
        # Gli export OpenVINO sono backend di inferenza: per stabilità usiamo predict(), non track().
        self.enable_tracking = bool(enable_tracking) and not self.is_openvino
        self.half = bool(half) and str(device).lower() not in {"cpu", "none"} and not self.is_openvino
        self.max_det = max_det
        self._tracking_failed_once = False
        self.target_class_ids = self._resolve_target_class_ids()

    def detect(self, frame) -> List[Detection]:
        result = None
        kwargs = dict(
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            classes=self.target_class_ids,
            max_det=self.max_det,
            half=self.half,
        )
        if self.device is not None:
            kwargs["device"] = self.device

        if self.enable_tracking:
            try:
                result = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    **kwargs,
                )[0]
            except Exception as exc:
                if not self._tracking_failed_once:
                    print(f"[INFO] Tracking non disponibile, uso predict(). Dettaglio: {exc}")
                    self._tracking_failed_once = True

        if result is None:
            result = self.model.predict(frame, **kwargs)[0]

        names: Dict[int, str] = self.model.names
        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes
        ids = getattr(boxes, "id", None)

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            if self.target_classes and label not in self.target_classes:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            confidence = float(box.conf[0].item())
            track_id = None
            if ids is not None and ids[idx] is not None:
                track_id = int(ids[idx].item())

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    track_id=track_id,
                )
            )
        return detections

    def fallback_to_cpu(self) -> None:
        """Switch inference to CPU after a CUDA failure without recreating the model."""
        self.device = "cpu"
        self.half = False
        self.enable_tracking = False

    def _resolve_target_class_ids(self) -> Optional[List[int]]:
        if not self.target_classes:
            return None
        ids: List[int] = []
        for class_id, name in self.model.names.items():
            if name in self.target_classes:
                ids.append(int(class_id))
        return ids or None
