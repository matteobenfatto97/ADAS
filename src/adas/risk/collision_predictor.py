from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from adas.perception.lane_detector import LaneDetector
from adas.perception.types import Detection, LaneState


Box = Tuple[int, int, int, int]


@dataclass
class TrackMemory:
    last_time: float
    last_scale: float
    last_center: Tuple[int, int]
    label: str
    box: Box


class CollisionPredictor:
    """Stima rischio collisione con due segnali complementari.

    1) TTC visuale: se il bounding box cresce rapidamente, l'oggetto si sta avvicinando.
    2) Proximity risk: se un veicolo è grande/basso nel frame e dentro il corridoio ego,
       viene segnalato anche senza tracking perfetto. Questo è importante con OpenVINO
       `predict()`, dove spesso non abbiamo track_id stabili.

    Non è una misura fisica certificabile: è una logica ADAS dimostrativa robusta per dashcam.
    """

    def __init__(self, cfg: dict) -> None:
        self.warning_ttc = float(cfg.get("warning_ttc_seconds", 2.0))
        self.brake_ttc = float(cfg.get("brake_ttc_seconds", 0.9))
        self.min_area_growth_rate = float(cfg.get("min_area_growth_rate", 18.0))
        self.stale_track_seconds = float(cfg.get("stale_track_seconds", 1.2))
        self.ego_lane_margin_ratio = float(cfg.get("ego_lane_margin_ratio", 0.015))
        self.fallback_center_path_ratio = float(cfg.get("fallback_center_path_ratio", 0.20))
        self.require_lane_for_brake = bool(cfg.get("require_lane_for_brake", True))
        self.require_lane_for_warning = bool(cfg.get("require_lane_for_warning", False))
        self.use_ego_corridor = bool(cfg.get("use_ego_corridor", True))
        self.min_bottom_y_ratio = float(cfg.get("min_bottom_y_ratio", 0.42))
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.0010))
        self.max_person_aspect_ratio = float(cfg.get("max_person_aspect_ratio", 0.85))
        self.allowed_risk_classes = set(cfg.get("risk_classes", ["car", "motorcycle", "bus", "truck", "bicycle"]))

        # Nuova logica patch 7: rischio visibile anche senza TTC stabile.
        self.enable_proximity_risk = bool(cfg.get("enable_proximity_risk", True))
        self.proximity_warning_bottom_y_ratio = float(cfg.get("proximity_warning_bottom_y_ratio", 0.66))
        self.proximity_brake_bottom_y_ratio = float(cfg.get("proximity_brake_bottom_y_ratio", 0.92))
        self.proximity_warning_area_ratio = float(cfg.get("proximity_warning_area_ratio", 0.010))
        self.proximity_brake_area_ratio = float(cfg.get("proximity_brake_area_ratio", 0.075))
        self.proximity_warning_score = float(cfg.get("proximity_warning_score", 0.46))
        self.proximity_brake_score = float(cfg.get("proximity_brake_score", 0.86))
        self.match_max_center_dist_ratio = float(cfg.get("match_max_center_dist_ratio", 0.18))
        self.match_min_iou = float(cfg.get("match_min_iou", 0.08))

        self.memory: Dict[str, TrackMemory] = {}
        self._next_id = 0

    def update(self, detections: List[Detection], lane_state: LaneState, frame_width: int, frame_height: int) -> List[Detection]:
        now = time.monotonic()
        margin_px = int(frame_width * self.ego_lane_margin_ratio)
        frame_area = max(1, frame_width * frame_height)

        used_keys: set[str] = set()
        for det in detections:
            det.in_ego_lane = self._is_relevant_path(det, lane_state, margin_px, frame_width)

            if not self._passes_static_filters(det, frame_area, frame_height):
                key = self._match_or_create_key(det, frame_width, frame_height, used_keys)
                used_keys.add(key)
                self.memory[key] = TrackMemory(now, math.sqrt(max(det.area, 1.0)), det.center, det.label, det.box)
                continue

            if not det.in_ego_lane:
                det.filtered_reason = "outside_ego_path"

            key = self._match_or_create_key(det, frame_width, frame_height, used_keys)
            used_keys.add(key)
            scale = math.sqrt(max(det.area, 1.0))
            previous = self.memory.get(key)

            # TTC visuale robusto: matching per track_id oppure per IoU/centro.
            if previous is not None:
                dt = max(now - previous.last_time, 1e-3)
                growth_rate = (scale - previous.last_scale) / dt
                if growth_rate > self.min_area_growth_rate:
                    det.ttc_seconds = scale / growth_rate
                else:
                    det.ttc_seconds = None

            lane_available = lane_state.has_lane
            can_warning = det.in_ego_lane and (lane_available or not self.require_lane_for_warning)
            can_brake = det.in_ego_lane and (lane_available or not self.require_lane_for_brake)

            if det.ttc_seconds is not None:
                if can_brake and det.ttc_seconds <= self.brake_ttc:
                    det.risk_level = "brake"
                    det.risk_reason = "TTC"
                elif can_warning and det.ttc_seconds <= self.warning_ttc:
                    det.risk_level = "warning"
                    det.risk_reason = "TTC"

            # Proximity risk: serve soprattutto nelle demo, perché non sempre c'è TTC stabile.
            if self.enable_proximity_risk and (can_warning or can_brake):
                score = self._proximity_score(det, frame_area, frame_height)
                det.distance_score = score
                if can_brake and score >= self.proximity_brake_score and det.risk_level != "brake":
                    det.risk_level = "brake"
                    det.risk_reason = "CLOSE_EGO_PATH"
                    det.ttc_seconds = det.ttc_seconds if det.ttc_seconds is not None else self._pseudo_ttc_from_score(score)
                elif can_warning and score >= self.proximity_warning_score and det.risk_level == "none":
                    det.risk_level = "warning"
                    det.risk_reason = "CLOSE_EGO_PATH"
                    det.ttc_seconds = det.ttc_seconds if det.ttc_seconds is not None else self._pseudo_ttc_from_score(score)

            self.memory[key] = TrackMemory(now, scale, det.center, det.label, det.box)

        self._remove_stale_tracks(now)
        return detections

    def _passes_static_filters(self, det: Detection, frame_area: int, frame_height: int) -> bool:
        det.filtered_reason = None
        if det.label not in self.allowed_risk_classes:
            det.filtered_reason = "class_not_risk"
            return False
        if det.area / frame_area < self.min_area_ratio:
            det.filtered_reason = "too_small"
            return False
        if det.bottom_center[1] < frame_height * self.min_bottom_y_ratio:
            det.filtered_reason = "too_high"
            return False
        if det.label == "person" and det.aspect_ratio > self.max_person_aspect_ratio:
            det.filtered_reason = "person_shape_invalid"
            return False
        return True

    def _match_or_create_key(self, det: Detection, frame_width: int, frame_height: int, used_keys: set[str]) -> str:
        if det.track_id is not None:
            return f"id:{det.track_id}"

        diag = math.hypot(frame_width, frame_height)
        best_key: Optional[str] = None
        best_score = -1.0
        for key, mem in self.memory.items():
            if key in used_keys or mem.label != det.label:
                continue
            iou = self._iou(det.box, mem.box)
            dist = math.hypot(det.center[0] - mem.last_center[0], det.center[1] - mem.last_center[1]) / max(1.0, diag)
            if iou >= self.match_min_iou or dist <= self.match_max_center_dist_ratio:
                score = iou * 2.0 + max(0.0, 1.0 - dist / max(1e-6, self.match_max_center_dist_ratio))
                if score > best_score:
                    best_score = score
                    best_key = key
        if best_key is not None:
            return best_key

        self._next_id += 1
        return f"auto:{self._next_id}"

    def _is_relevant_path(self, det: Detection, lane_state: LaneState, margin_px: int, frame_width: int) -> bool:
        if lane_state.has_lane:
            return LaneDetector.contains_point(
                lane_state,
                det.bottom_center,
                margin_px=margin_px,
                ego_only=self.use_ego_corridor,
            )
        cx, _ = det.bottom_center
        half = self.fallback_center_path_ratio / 2.0
        return frame_width * (0.5 - half) <= cx <= frame_width * (0.5 + half)

    def _proximity_score(self, det: Detection, frame_area: int, frame_height: int) -> float:
        bottom_ratio = det.bottom_center[1] / max(1.0, frame_height)
        area_ratio = det.area / max(1.0, frame_area)

        bottom_score = self._smoothstep(
            self.proximity_warning_bottom_y_ratio,
            self.proximity_brake_bottom_y_ratio,
            bottom_ratio,
        )
        area_score = self._smoothstep(
            self.proximity_warning_area_ratio,
            self.proximity_brake_area_ratio,
            area_ratio,
        )
        # Il bottom del box è più stabile della sola area su dashcam, ma l'area aiuta con veicoli molto vicini.
        return float(max(0.0, min(1.0, 0.62 * bottom_score + 0.38 * area_score)))

    def _pseudo_ttc_from_score(self, score: float) -> float:
        # Serve solo per comunicazione HUD, non è TTC fisico.
        s = max(0.0, min(1.0, score))
        return float(2.2 - 1.7 * s)

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        if edge1 <= edge0:
            return 1.0 if x >= edge1 else 0.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _remove_stale_tracks(self, now: float) -> None:
        stale_keys = [k for k, v in self.memory.items() if now - v.last_time > self.stale_track_seconds]
        for key in stale_keys:
            self.memory.pop(key, None)
