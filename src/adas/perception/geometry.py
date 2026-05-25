from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

from adas.perception.types import Detection, Point


def resize_keep_aspect(frame, target_width: int | None):
    """Ridimensiona mantenendo aspect ratio. Ritorna frame, scale_x, scale_y."""
    if not target_width or target_width <= 0:
        return frame, 1.0, 1.0
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0, 1.0
    scale = target_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(target_width), h / float(new_h)


def scale_detections(detections: Iterable[Detection], scale_x: float, scale_y: float) -> List[Detection]:
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return list(detections)
    scaled: List[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        scaled.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                box=(
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                ),
                track_id=det.track_id,
            )
        )
    return scaled


def point_in_polygon(polygon: list[Point], point: Point, margin_px: int = 0) -> bool:
    if not polygon:
        return False
    poly = np.array(polygon, dtype=np.int32)
    if cv2.pointPolygonTest(poly, point, False) >= 0:
        return True
    if margin_px <= 0:
        return False
    return cv2.pointPolygonTest(poly, point, True) >= -margin_px


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
