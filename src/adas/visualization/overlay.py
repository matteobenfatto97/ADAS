from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from adas.perception.types import ADASDecision, Detection, LaneState

Color = Tuple[int, int, int]  # BGR
Point = Tuple[int, int]


# Palette BGR pensata per OpenCV.
COLORS = {
    "panel": (22, 28, 32),
    "panel_2": (30, 38, 44),
    "cyan": (255, 215, 45),
    "cyan_soft": (220, 170, 25),
    "teal": (180, 230, 30),
    "green": (110, 245, 70),
    "amber": (0, 190, 255),
    "red": (40, 40, 255),
    "yellow": (0, 245, 255),
    "white": (245, 245, 245),
    "muted": (165, 176, 184),
    "muted_2": (105, 118, 126),
    "dark": (8, 12, 16),
}


class PremiumHUD:
    """HUD OpenCV reattivo per il prototipo ADAS.

    Reattivo qui significa che ogni elemento grafico cambia in base allo stato:
    - verde/cyan: guida normale;
    - ambra: warning collisione o lane departure;
    - rosso pulsante: emergency brake simulato;
    - intensità/label corsia basate su confidence e offset.
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.show_debug_text = bool(cfg.get("show_debug_text", False))
        self.show_fps = bool(cfg.get("show_fps", True))
        self.speed_kmh = cfg.get("speed_kmh")
        self.speed_limit_kmh = cfg.get("speed_limit_kmh")
        self.panel_alpha = float(cfg.get("panel_alpha", 0.62))
        self.road_alpha = float(cfg.get("road_alpha", 0.22))
        self.ego_alpha = float(cfg.get("ego_alpha", 0.26))
        self.glow = bool(cfg.get("glow", True))
        self.vignette = bool(cfg.get("vignette", True))
        self.micro_grid = bool(cfg.get("micro_grid", True))
        self.compact = bool(cfg.get("compact", False))
        self.render_mode = str(cfg.get("render_mode", "premium")).lower().strip()

    def draw(
        self,
        frame,
        detections: List[Detection],
        lane_state: LaneState,
        decision: ADASDecision,
        *,
        show_filtered: bool = False,
        fps: Optional[float] = None,
        frame_idx: int = 0,
    ):
        if not self.enabled:
            return draw_debug_overlay(frame, detections, lane_state, decision, show_filtered=show_filtered)

        output = frame.copy()
        if self.render_mode in {"fast", "performance", "lite"}:
            self._draw_fast_hud(output, detections, lane_state, decision, show_filtered=show_filtered, fps=fps, frame_idx=frame_idx)
            return output

        h, w = output.shape[:2]
        now = time.monotonic()

        # Leggera vignettatura opzionale. Disattivarla aiuta parecchio sugli FPS.
        if self.vignette:
            output = self._vignette(output, strength=0.18)

        self._draw_lane_ar(output, lane_state, decision, frame_idx)
        self._draw_detections(output, detections, show_filtered)
        self._draw_top_hud(output, lane_state, decision, fps=fps, now=now)
        self._draw_bottom_hud(output, lane_state, detections, decision, fps=fps)
        self._draw_alerts(output, decision, w, h, now)

        if self.show_debug_text:
            self._draw_debug_text(output, lane_state, decision)

        return output


    # ------------------------------------------------------------------
    # Fast HUD: stessa informazione, molti meno overlay full-frame.
    # ------------------------------------------------------------------

    def _draw_fast_hud(self, img, detections: List[Detection], lane_state: LaneState, decision: ADASDecision, *, show_filtered: bool, fps: Optional[float], frame_idx: int) -> None:
        h, w = img.shape[:2]
        status_text, status_color = self._system_status(decision, lane_state)

        # Road layer: un solo overlay per corsia e corridoio.
        if lane_state.has_lane:
            overlay = img.copy()
            if lane_state.polygon:
                cv2.fillPoly(overlay, [np.array(lane_state.polygon, dtype=np.int32)], (70, 120, 60))
            if lane_state.ego_polygon:
                cv2.fillPoly(overlay, [np.array(lane_state.ego_polygon, dtype=np.int32)], status_color)
            cv2.addWeighted(overlay, 0.16 if not lane_state.departure else 0.23, img, 0.84 if not lane_state.departure else 0.77, 0, dst=img)

            if lane_state.left_curve:
                cv2.polylines(img, [np.array(lane_state.left_curve, dtype=np.int32)], False, COLORS["yellow"], 3, cv2.LINE_AA)
            if lane_state.right_curve:
                cv2.polylines(img, [np.array(lane_state.right_curve, dtype=np.int32)], False, COLORS["yellow"], 3, cv2.LINE_AA)
            if lane_state.ego_polygon and len(lane_state.ego_polygon) >= 4:
                p = lane_state.ego_polygon
                half = len(p) // 2
                left_ego = p[:half]
                right_ego = list(reversed(p[half:]))
                if left_ego:
                    cv2.polylines(img, [np.array(left_ego, dtype=np.int32)], False, COLORS["cyan"], 2, cv2.LINE_AA)
                if right_ego:
                    cv2.polylines(img, [np.array(right_ego, dtype=np.int32)], False, COLORS["cyan"], 2, cv2.LINE_AA)
        else:
            center = lane_state.vehicle_center_x or w // 2
            cv2.line(img, (center, int(h * 0.58)), (center, h - 30), COLORS["muted_2"], 2, cv2.LINE_AA)

        # Detection boxes più leggere e più leggibili.
        self._draw_detections_fast(img, detections, show_filtered)

        # Top glass bar: 2 rettangoli alpha locali, niente card pesanti.
        top_h = 64
        self._fast_panel(img, 18, 16, min(430, w - 36), top_h, alpha=0.58, color=COLORS["dark"], border=status_color)
        self._put_text(img, "ADAS", (38, 44), scale=0.82, color=COLORS["white"], thickness=2)
        self._put_text(img, status_text, (132, 44), scale=0.62, color=status_color, thickness=2)
        lane_txt = "LANE OK" if lane_state.has_lane and not lane_state.departure else "LANE WARN" if lane_state.departure else "LANE SEARCH"
        off = "--" if lane_state.center_offset_ratio is None else f"{lane_state.center_offset_ratio:+.2f}"
        conf = int(max(0.0, min(1.0, lane_state.confidence)) * 100)
        self._put_text(img, f"{lane_txt} | off {off} | conf {conf}%", (230, 44), scale=0.43, color=COLORS["muted"], thickness=1)

        # Risk panel.
        urgent = self._most_urgent_detection(detections)
        rx, ry, rw, rh = w - min(420, w - 36) - 18, 16, min(420, w - 36), top_h
        risk_color = COLORS["green"]
        risk_main = "CLEAR"
        risk_sub = "ego path"
        if urgent:
            risk_color = COLORS["red"] if urgent.risk_level == "brake" else COLORS["amber"]
            risk_main = "BRAKE" if urgent.risk_level == "brake" else "COLLISION RISK"
            metric = f"TTC {urgent.ttc_seconds:.1f}s" if urgent.ttc_seconds is not None else f"score {urgent.distance_score:.2f}"
            risk_sub = f"{urgent.label.upper()} | {metric}"
        self._fast_panel(img, rx, ry, rw, rh, alpha=0.58, color=COLORS["dark"], border=risk_color)
        self._put_text(img, risk_main, (rx + 20, ry + 28), scale=0.56, color=risk_color, thickness=2)
        self._put_text(img, risk_sub, (rx + 20, ry + 52), scale=0.42, color=COLORS["white"], thickness=1)
        if self.show_fps and fps is not None:
            self._put_text(img, f"{fps:.0f} FPS", (rx + rw - 92, ry + 52), scale=0.42, color=COLORS["muted"], thickness=1)

        # Alert centrale compatto.
        if decision.brake or decision.warnings:
            text = "EMERGENCY STOP SIM" if decision.brake else decision.warnings[0][:58]
            color = COLORS["red"] if decision.brake else COLORS["amber"]
            bw = min(640, int(w * 0.55))
            bx = (w - bw) // 2
            by = 96
            self._fast_panel(img, bx, by, bw, 48, alpha=0.62, color=COLORS["dark"], border=color)
            self._put_text(img, text, (bx + 22, by + 31), scale=0.55, color=color, thickness=2)

    def _draw_detections_fast(self, img, detections: List[Detection], show_filtered: bool) -> None:
        for det in detections:
            if det.filtered_reason and not show_filtered and det.risk_level == "none":
                continue
            if det.risk_level == "brake":
                color = COLORS["red"]
            elif det.risk_level == "warning":
                color = COLORS["amber"]
            elif det.in_ego_lane:
                color = COLORS["green"]
            elif det.filtered_reason:
                color = COLORS["muted_2"]
            else:
                color = COLORS["cyan_soft"]
            x1, y1, x2, y2 = det.box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            tag = f"{det.label.upper()} {det.confidence:.2f}"
            if det.risk_level != "none":
                tag += f" {det.risk_level.upper()}"
            elif det.in_ego_lane:
                tag += " EGO"
            self._simple_label(img, tag, (x1, max(4, y1 - 22)), color)

    def _fast_panel(self, img, x: int, y: int, w: int, h: int, *, alpha: float, color: Color, border: Optional[Color] = None) -> None:
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
        roi = img[y:y+h, x:x+w]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), border or (78, 92, 101), 1, cv2.LINE_AA)

    def _simple_label(self, img, text: str, pos: Point, color: Color) -> None:
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        self._fast_panel(img, x, y, tw + 12, th + 11, alpha=0.72, color=COLORS["dark"], border=color)
        cv2.putText(img, text, (x + 6, y + th + 6), font, scale, COLORS["white"], thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Road AR layer
    # ------------------------------------------------------------------

    def _draw_lane_ar(self, img, lane_state: LaneState, decision: ADASDecision, frame_idx: int) -> None:
        h, w = img.shape[:2]

        if not lane_state.has_lane:
            # Se non abbiamo corsia, disegna solo una zona centrale di fallback molto soft.
            center = lane_state.vehicle_center_x or w // 2
            bottom_width = int(w * 0.30)
            top_width = int(w * 0.12)
            y_bottom = h
            y_top = int(h * 0.58)
            fallback = np.array([
                (center - bottom_width // 2, y_bottom),
                (center - top_width // 2, y_top),
                (center + top_width // 2, y_top),
                (center + bottom_width // 2, y_bottom),
            ], dtype=np.int32)
            self._fill_poly(img, fallback, COLORS["cyan"], alpha=0.08)
            return

        lane_poly = np.array(lane_state.polygon, dtype=np.int32)
        ego_poly = np.array(lane_state.ego_polygon or lane_state.polygon, dtype=np.int32)

        if lane_state.departure:
            lane_color = COLORS["amber"]
            ego_color = COLORS["amber"]
        elif decision.brake:
            lane_color = COLORS["red"]
            ego_color = COLORS["red"]
        else:
            lane_color = COLORS["cyan"]
            ego_color = COLORS["cyan"]

        self._fill_poly(img, lane_poly, (70, 120, 60), alpha=0.08)
        self._fill_poly(img, ego_poly, ego_color, alpha=self.ego_alpha)

        # Bordi corsia: ora supportano curve polinomiali, non solo due rette.
        left_path = lane_state.left_curve or ([lane_state.left_line[0], lane_state.left_line[1]] if lane_state.left_line else [])
        right_path = lane_state.right_curve or ([lane_state.right_line[0], lane_state.right_line[1]] if lane_state.right_line else [])
        if left_path:
            self._glow_polyline(img, left_path, COLORS["yellow"], thickness=3)
        if right_path:
            self._glow_polyline(img, right_path, COLORS["yellow"], thickness=3)

        if lane_state.ego_polygon and len(lane_state.ego_polygon) >= 4:
            p = lane_state.ego_polygon
            half = len(p) // 2
            left_ego = p[:half]
            right_ego = list(reversed(p[half:]))
            if left_ego:
                self._glow_polyline(img, left_ego, COLORS["cyan"], thickness=2)
            if right_ego:
                self._glow_polyline(img, right_ego, COLORS["cyan"], thickness=2)
            # Le animazioni usano un quad sintetico anche se la polygon reale e' curva.
            if half >= 2:
                quad = [p[0], p[half - 1], p[half], p[-1]]
                self._draw_lane_chevrons(img, quad, frame_idx, color=COLORS["cyan"])
                if self.micro_grid:
                    self._draw_lane_micro_grid(img, quad, color=COLORS["cyan_soft"])

        # Indicatore offset alla base.
        if lane_state.lane_center_x is not None and lane_state.vehicle_center_x is not None:
            vehicle_x = lane_state.vehicle_center_x
            lane_x = lane_state.lane_center_x
            y1 = h - 22
            y2 = int(h * 0.78)
            self._dashed_line(img, (lane_x, y1), (lane_x, y2), COLORS["cyan"], thickness=2, dash=12, gap=9)
            self._glow_line(img, (vehicle_x, y1), (vehicle_x, int(h * 0.84)), COLORS["white"], thickness=1)
            self._draw_offset_bubble(img, lane_state)

    def _draw_lane_chevrons(self, img, poly: List[Point], frame_idx: int, color: Color) -> None:
        if len(poly) != 4:
            return
        lb, lt, rt, rb = poly
        # Chevrons prospettici dentro il corridoio.
        phases = [0.30, 0.45, 0.60]
        anim = (frame_idx % 40) / 40.0
        for i, t in enumerate(phases):
            # t=0 bottom, t=1 top
            t2 = min(0.95, t + 0.035)
            left = self._interp(lb, lt, t)
            right = self._interp(rb, rt, t)
            left2 = self._interp(lb, lt, t2)
            right2 = self._interp(rb, rt, t2)
            center = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
            center2 = ((left2[0] + right2[0]) // 2, (left2[1] + right2[1]) // 2)
            width = max(18, abs(right[0] - left[0]) // 6)
            alpha = 0.20 + 0.18 * (1.0 - abs(anim - i / 3.0))
            overlay = img.copy()
            cv2.line(overlay, (center[0] - width, center[1]), center2, color, 2, cv2.LINE_AA)
            cv2.line(overlay, (center[0] + width, center[1]), center2, color, 2, cv2.LINE_AA)
            self._blend(img, overlay, alpha)

    def _draw_lane_micro_grid(self, img, poly: List[Point], color: Color) -> None:
        if len(poly) != 4:
            return
        lb, lt, rt, rb = poly
        overlay = img.copy()
        for t in np.linspace(0.10, 0.92, 9):
            left = self._interp(lb, lt, float(t))
            right = self._interp(rb, rt, float(t))
            cv2.line(overlay, left, right, color, 1, cv2.LINE_AA)
        self._blend(img, overlay, 0.09)

    def _draw_offset_bubble(self, img, lane_state: LaneState) -> None:
        h, w = img.shape[:2]
        x = lane_state.vehicle_center_x or w // 2
        y = h - 78
        offset = lane_state.center_offset_ratio or 0.0
        color = COLORS["amber"] if lane_state.departure else COLORS["cyan"]
        # Card circolare glass.
        self._circle_panel(img, (x, y), 44, border_color=color, alpha=0.50)
        text = f"{offset:+.2f}"
        self._put_text(img, text, (x - 34, y + 8), scale=0.62, color=COLORS["white"], thickness=2)
        self._put_text(img, "OFFSET", (x - 26, y + 30), scale=0.32, color=COLORS["muted"], thickness=1)

    # ------------------------------------------------------------------
    # Detection layer
    # ------------------------------------------------------------------

    def _draw_detections(self, img, detections: List[Detection], show_filtered: bool) -> None:
        for det in detections:
            if det.filtered_reason and not show_filtered:
                continue
            if det.risk_level == "brake":
                color = COLORS["red"]
            elif det.risk_level == "warning":
                color = COLORS["amber"]
            elif det.in_ego_lane:
                color = COLORS["green"]
            elif det.filtered_reason:
                color = COLORS["muted_2"]
            else:
                color = COLORS["cyan_soft"]

            x1, y1, x2, y2 = det.box
            self._corner_box(img, (x1, y1), (x2, y2), color, thickness=2)

            label = f"{det.label.upper()} {det.confidence:.2f}"
            if det.ttc_seconds is not None:
                label += f"  TTC {det.ttc_seconds:.1f}s"
            elif det.risk_level != "none":
                label += f"  RISK {det.distance_score:.2f}"
            if det.track_id is not None:
                label += f"  #{det.track_id}"
            if det.filtered_reason:
                label += f"  {det.filtered_reason}"
            self._label_chip(img, label, (x1, max(8, y1 - 30)), color)

    # ------------------------------------------------------------------
    # Dashboard UI
    # ------------------------------------------------------------------

    def _draw_top_hud(self, img, lane_state: LaneState, decision: ADASDecision, *, fps: Optional[float], now: float) -> None:
        h, w = img.shape[:2]
        margin = max(18, int(w * 0.015))
        top = 18
        card_h = 76 if self.compact else 92

        status_text, status_color = self._system_status(decision, lane_state)

        # Left brand/status card.
        left_w = min(440, int(w * 0.30))
        self._rounded_panel(img, margin, top, left_w, card_h, alpha=self.panel_alpha, border_color=status_color)
        self._draw_lane_icon(img, margin + 22, top + 18, status_color)
        self._put_text(img, "ADAS MONITORING", (margin + 84, top + 43), scale=0.82, color=COLORS["white"], thickness=2)
        self._status_chip(img, margin + left_w - 118, top + 28, status_text, status_color)

        # Center lane metrics card.
        center_w = min(760, int(w * 0.48))
        center_x = (w - center_w) // 2
        self._rounded_panel(img, center_x, top, center_w, card_h, alpha=self.panel_alpha)
        self._draw_lane_metrics(img, center_x, top, center_w, card_h, lane_state)

        # Right health/fps card.
        right_w = min(220, int(w * 0.15))
        right_x = w - right_w - margin
        self._rounded_panel(img, right_x, top, right_w, card_h, alpha=self.panel_alpha)
        health = int(max(0, min(99, 55 + lane_state.confidence * 35 + (10 if not decision.brake else -25))))
        dot_color = status_color
        cv2.circle(img, (right_x + 26, top + 29), 6, dot_color, -1, cv2.LINE_AA)
        self._put_text(img, "SYSTEM", (right_x + 46, top + 34), scale=0.42, color=COLORS["muted"], thickness=1)
        self._put_text(img, f"{health}%", (right_x + 32, top + 70), scale=0.86, color=COLORS["white"], thickness=2)
        if self.show_fps and fps is not None:
            self._put_text(img, f"{fps:.0f} FPS", (right_x + 112, top + 70), scale=0.42, color=COLORS["muted"], thickness=1)

    def _draw_lane_metrics(self, img, x: int, y: int, width: int, height: int, lane_state: LaneState) -> None:
        third = width // 3
        sep_color = (70, 83, 92)
        cv2.line(img, (x + third, y + 18), (x + third, y + height - 18), sep_color, 1, cv2.LINE_AA)
        cv2.line(img, (x + 2 * third, y + 18), (x + 2 * third, y + height - 18), sep_color, 1, cv2.LINE_AA)

        lane_color = COLORS["green"] if lane_state.has_lane else COLORS["amber"]
        lane_label = "DETECTED" if lane_state.has_lane else "SEARCHING"
        if lane_state.departure:
            lane_color = COLORS["amber"]
            lane_label = f"DEPART {lane_state.departure.upper()}"

        self._put_text(img, "LANE STATUS", (x + 72, y + 30), scale=0.38, color=COLORS["muted"], thickness=1)
        self._draw_lane_icon(img, x + 26, y + 30, lane_color, small=True)
        self._put_text(img, lane_label, (x + 72, y + 58), scale=0.50, color=lane_color, thickness=2)

        offset = lane_state.center_offset_ratio
        offset_text = "--" if offset is None else f"{offset:+.2f}"
        off_color = COLORS["amber"] if lane_state.departure else COLORS["cyan"]
        self._put_text(img, "LANE OFFSET", (x + third + 70, y + 30), scale=0.38, color=COLORS["muted"], thickness=1)
        self._put_text(img, offset_text, (x + third + 88, y + 64), scale=0.78, color=off_color, thickness=2)
        self._put_text(img, "ratio", (x + third + 166, y + 64), scale=0.34, color=COLORS["muted"], thickness=1)
        self._draw_offset_scale(img, x + third + 52, y + height - 18, third - 100, offset or 0.0, off_color)

        conf = lane_state.confidence
        conf_label = "HIGH" if conf >= 0.70 else "MED" if conf >= 0.35 else "LOW"
        conf_color = COLORS["green"] if conf >= 0.70 else COLORS["amber"] if conf >= 0.35 else COLORS["red"]
        self._put_text(img, "LINE CONFIDENCE", (x + 2 * third + 56, y + 30), scale=0.38, color=COLORS["muted"], thickness=1)
        self._put_text(img, conf_label, (x + 2 * third + 84, y + 58), scale=0.52, color=conf_color, thickness=2)
        self._segmented_bar(img, x + 2 * third + 56, y + height - 26, third - 106, 8, conf, conf_color)

    def _draw_bottom_hud(self, img, lane_state: LaneState, detections: List[Detection], decision: ADASDecision, *, fps: Optional[float]) -> None:
        h, w = img.shape[:2]
        margin = max(18, int(w * 0.015))
        bottom = h - 96

        # Speed card solo se abbiamo un valore configurato; altrimenti mostriamo modalità e FPS.
        left_w = 300 if self.speed_kmh is not None else 240
        self._rounded_panel(img, margin, bottom, left_w, 72, alpha=0.58)
        self._speedometer_icon(img, margin + 44, bottom + 38, COLORS["white"])
        if self.speed_kmh is not None:
            self._put_text(img, str(int(self.speed_kmh)), (margin + 92, bottom + 44), scale=1.05, color=COLORS["white"], thickness=2)
            self._put_text(img, "KM/H", (margin + 94, bottom + 64), scale=0.36, color=COLORS["muted"], thickness=1)
        else:
            self._put_text(img, "SIM MODE", (margin + 92, bottom + 36), scale=0.56, color=COLORS["white"], thickness=2)
            self._put_text(img, "NO VEHICLE CONTROL", (margin + 92, bottom + 58), scale=0.32, color=COLORS["muted"], thickness=1)

        if self.speed_limit_kmh is not None:
            cx = margin + left_w - 44
            cy = bottom + 38
            cv2.circle(img, (cx, cy), 25, COLORS["white"], -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 25, COLORS["red"], 5, cv2.LINE_AA)
            self._put_text(img, str(int(self.speed_limit_kmh)), (cx - 14, cy + 8), scale=0.50, color=(20, 20, 20), thickness=2)

        # TTC / risk card.
        urgent = self._most_urgent_detection(detections)
        card_w = 300
        card_x = w - card_w - margin
        self._rounded_panel(img, card_x, bottom, card_w, 72, alpha=0.58)
        if urgent:
            color = COLORS["red"] if urgent.risk_level == "brake" else COLORS["amber"]
            metric = f"{urgent.ttc_seconds:.1f}s" if urgent.ttc_seconds is not None else f"{urgent.distance_score:.2f}"
            metric_label = "TTC" if urgent.ttc_seconds is not None else "SCORE"
            self._put_text(img, "COLLISION RISK", (card_x + 24, bottom + 27), scale=0.40, color=COLORS["muted"], thickness=1)
            self._put_text(img, metric, (card_x + 24, bottom + 60), scale=0.86, color=color, thickness=2)
            self._put_text(img, metric_label, (card_x + 112, bottom + 59), scale=0.34, color=COLORS["muted"], thickness=1)
            self._put_text(img, urgent.label.upper(), (card_x + 160, bottom + 57), scale=0.48, color=COLORS["white"], thickness=2)
        else:
            self._put_text(img, "COLLISION RISK", (card_x + 24, bottom + 27), scale=0.40, color=COLORS["muted"], thickness=1)
            self._put_text(img, "CLEAR", (card_x + 24, bottom + 58), scale=0.70, color=COLORS["green"], thickness=2)
            self._put_text(img, "EGO PATH", (card_x + 126, bottom + 57), scale=0.42, color=COLORS["muted"], thickness=1)

    def _draw_alerts(self, img, decision: ADASDecision, w: int, h: int, now: float) -> None:
        if not decision.brake and not decision.warnings:
            return

        if decision.brake:
            pulse = 0.45 + 0.25 * (0.5 + 0.5 * math.sin(now * 9.0))
            alert = "EMERGENCY STOP SIMULATED"
            sub = decision.reason[:70]
            color = COLORS["red"]
        else:
            pulse = 0.46
            alert = "ADAS WARNING"
            sub = decision.warnings[0][:70]
            color = COLORS["amber"]

        box_w = min(680, int(w * 0.54))
        box_h = 74
        x = (w - box_w) // 2
        y = 122
        self._rounded_panel(img, x, y, box_w, box_h, alpha=pulse, border_color=color, fill_color=COLORS["dark"])
        cv2.circle(img, (x + 36, y + 37), 14, color, 2, cv2.LINE_AA)
        self._put_text(img, "!", (x + 31, y + 44), scale=0.58, color=color, thickness=2)
        self._put_text(img, alert, (x + 64, y + 30), scale=0.58, color=COLORS["white"], thickness=2)
        self._put_text(img, sub, (x + 64, y + 56), scale=0.38, color=COLORS["muted"], thickness=1)

    def _draw_debug_text(self, img, lane_state: LaneState, decision: ADASDecision) -> None:
        y = 122
        self._put_text(img, f"DEBUG: {lane_state.debug}", (24, y), scale=0.42, color=COLORS["muted"], thickness=1)
        if decision.reason:
            self._put_text(img, decision.reason[:110], (24, y + 22), scale=0.42, color=COLORS["muted"], thickness=1)

    # ------------------------------------------------------------------
    # Small widgets / primitives
    # ------------------------------------------------------------------

    def _system_status(self, decision: ADASDecision, lane_state: LaneState) -> Tuple[str, Color]:
        if decision.brake:
            return "BRAKE", COLORS["red"]
        if decision.warnings or lane_state.departure:
            return "WARN", COLORS["amber"]
        return "ACTIVE", COLORS["teal"]

    def _draw_lane_icon(self, img, x: int, y: int, color: Color, small: bool = False) -> None:
        s = 38 if not small else 26
        if not small:
            self._rounded_panel(img, x, y, s + 14, s + 14, alpha=0.35, border_color=color, radius=10)
            ox, oy = x + 7, y + 7
        else:
            ox, oy = x, y
        # Icona strada stilizzata.
        cv2.line(img, (ox + int(s * 0.35), oy + s), (ox + int(s * 0.47), oy), color, 2, cv2.LINE_AA)
        cv2.line(img, (ox + int(s * 0.65), oy + s), (ox + int(s * 0.53), oy), color, 2, cv2.LINE_AA)
        self._dashed_line(img, (ox + s // 2, oy + s), (ox + s // 2, oy), color, thickness=1, dash=5, gap=4)

    def _status_chip(self, img, x: int, y: int, text: str, color: Color) -> None:
        width = 92 if len(text) <= 6 else 110
        self._rounded_panel(img, x, y, width, 34, alpha=0.42, border_color=color, fill_color=COLORS["panel_2"], radius=11)
        self._put_text(img, text, (x + 18, y + 23), scale=0.45, color=color, thickness=2)

    def _draw_offset_scale(self, img, x: int, y: int, width: int, value: float, color: Color) -> None:
        width = max(90, width)
        cv2.line(img, (x, y), (x + width, y), COLORS["muted_2"], 1, cv2.LINE_AA)
        for i in range(5):
            tx = x + int(width * i / 4)
            cv2.line(img, (tx, y - 4), (tx, y + 4), COLORS["muted_2"], 1, cv2.LINE_AA)
        v = max(-0.5, min(0.5, value))
        px = x + int((v + 0.5) / 1.0 * width)
        cv2.line(img, (px, y - 10), (px, y + 10), color, 2, cv2.LINE_AA)

    def _segmented_bar(self, img, x: int, y: int, width: int, height: int, value: float, color: Color) -> None:
        segments = 4
        gap = 7
        seg_w = max(16, (width - gap * (segments - 1)) // segments)
        active = int(round(max(0.0, min(1.0, value)) * segments))
        for i in range(segments):
            sx = x + i * (seg_w + gap)
            c = color if i < active else COLORS["muted_2"]
            self._rounded_rect(img, sx, y, seg_w, height, radius=3, color=c, alpha=0.92 if i < active else 0.40)

    def _speedometer_icon(self, img, cx: int, cy: int, color: Color) -> None:
        cv2.ellipse(img, (cx, cy), (34, 34), 0, 200, 340, color, 2, cv2.LINE_AA)
        for angle in range(210, 331, 30):
            rad = math.radians(angle)
            x1 = int(cx + math.cos(rad) * 25)
            y1 = int(cy + math.sin(rad) * 25)
            x2 = int(cx + math.cos(rad) * 31)
            y2 = int(cy + math.sin(rad) * 31)
            cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        needle_angle = math.radians(315)
        cv2.line(img, (cx, cy), (int(cx + math.cos(needle_angle) * 23), int(cy + math.sin(needle_angle) * 23)), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)

    def _most_urgent_detection(self, detections: List[Detection]) -> Optional[Detection]:
        risky = [d for d in detections if d.risk_level in {"warning", "brake"}]
        if not risky:
            return None
        def score(d: Detection) -> float:
            level = 0.0 if d.risk_level == "brake" else 1.0
            ttc = d.ttc_seconds if d.ttc_seconds is not None else (3.0 - d.distance_score)
            return level * 10.0 + ttc
        return min(risky, key=score)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _vignette(self, img, strength: float = 0.16):
        h, w = img.shape[:2]
        overlay = np.zeros_like(img)
        cv2.rectangle(overlay, (0, 0), (w, h), COLORS["dark"], -1)
        # Più scuro ai bordi e appena in alto/basso, leggibile ma non invasivo.
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (w, int(h * 0.18)), strength + 0.10, -1)
        cv2.rectangle(mask, (0, int(h * 0.82)), (w, h), strength + 0.08, -1)
        # Bordo laterale soft.
        mask[:, : int(w * 0.08)] += strength * 0.5
        mask[:, int(w * 0.92) :] += strength * 0.5
        mask = np.clip(mask, 0.0, 0.35)
        out = img.astype(np.float32) * (1 - mask[..., None]) + overlay.astype(np.float32) * mask[..., None]
        return out.astype(np.uint8)

    def _rounded_panel(
        self,
        img,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        alpha: float = 0.60,
        border_color: Optional[Color] = None,
        fill_color: Optional[Color] = None,
        radius: int = 18,
    ) -> None:
        fill = fill_color or COLORS["panel"]
        self._rounded_rect(img, x, y, w, h, radius=radius, color=fill, alpha=alpha)
        # Highlight superiore tipo glass.
        overlay = img.copy()
        cv2.line(overlay, (x + radius, y + 1), (x + w - radius, y + 1), (120, 145, 155), 1, cv2.LINE_AA)
        self._blend(img, overlay, 0.22)
        if border_color is None:
            border_color = (78, 92, 101)
        self._rounded_rect_outline(img, x, y, w, h, radius=radius, color=border_color, thickness=1)

    def _rounded_rect(self, img, x: int, y: int, w: int, h: int, *, radius: int, color: Color, alpha: float = 1.0) -> None:
        overlay = img.copy()
        self._rounded_rect_on(overlay, x, y, w, h, radius, color, thickness=-1)
        self._blend(img, overlay, alpha)

    def _rounded_rect_outline(self, img, x: int, y: int, w: int, h: int, *, radius: int, color: Color, thickness: int = 1) -> None:
        self._rounded_rect_on(img, x, y, w, h, radius, color, thickness=thickness)

    @staticmethod
    def _rounded_rect_on(img, x: int, y: int, w: int, h: int, radius: int, color: Color, thickness: int) -> None:
        r = min(radius, w // 2, h // 2)
        if thickness < 0:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
            cv2.circle(img, (x + r, y + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x + w - r, y + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x + r, y + h - r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x + w - r, y + h - r), r, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _circle_panel(self, img, center: Point, radius: int, *, border_color: Color, alpha: float = 0.5) -> None:
        overlay = img.copy()
        cv2.circle(overlay, center, radius, COLORS["panel"], -1, cv2.LINE_AA)
        cv2.circle(overlay, center, radius, border_color, 2, cv2.LINE_AA)
        cv2.circle(overlay, center, radius + 8, border_color, 1, cv2.LINE_AA)
        self._blend(img, overlay, alpha)

    def _fill_poly(self, img, poly: np.ndarray, color: Color, *, alpha: float) -> None:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], color)
        self._blend(img, overlay, alpha)

    def _glow_line(self, img, p1: Point, p2: Point, color: Color, *, thickness: int = 2) -> None:
        if self.glow:
            for t, a in [(12, 0.08), (7, 0.14), (4, 0.18)]:
                overlay = img.copy()
                cv2.line(overlay, p1, p2, color, max(thickness, t), cv2.LINE_AA)
                self._blend(img, overlay, a)
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    def _glow_polyline(self, img, points: List[Point], color: Color, *, thickness: int = 2) -> None:
        if len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        if self.glow:
            for t, a in [(12, 0.06), (7, 0.10), (4, 0.14)]:
                overlay = img.copy()
                cv2.polylines(overlay, [pts], False, color, max(thickness, t), cv2.LINE_AA)
                self._blend(img, overlay, a)
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

    def _dashed_line(self, img, p1: Point, p2: Point, color: Color, *, thickness: int = 1, dash: int = 8, gap: int = 6) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = int(math.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        dist = 0
        while dist < length:
            start = dist
            end = min(dist + dash, length)
            sx = int(x1 + dx * start)
            sy = int(y1 + dy * start)
            ex = int(x1 + dx * end)
            ey = int(y1 + dy * end)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            dist += dash + gap

    def _corner_box(self, img, p1: Point, p2: Point, color: Color, *, thickness: int = 2) -> None:
        x1, y1 = p1
        x2, y2 = p2
        l = max(16, min(34, (x2 - x1) // 4, (y2 - y1) // 4))
        self._glow_line(img, (x1, y1), (x1 + l, y1), color, thickness=thickness)
        self._glow_line(img, (x1, y1), (x1, y1 + l), color, thickness=thickness)
        self._glow_line(img, (x2, y1), (x2 - l, y1), color, thickness=thickness)
        self._glow_line(img, (x2, y1), (x2, y1 + l), color, thickness=thickness)
        self._glow_line(img, (x1, y2), (x1 + l, y2), color, thickness=thickness)
        self._glow_line(img, (x1, y2), (x1, y2 - l), color, thickness=thickness)
        self._glow_line(img, (x2, y2), (x2 - l, y2), color, thickness=thickness)
        self._glow_line(img, (x2, y2), (x2, y2 - l), color, thickness=thickness)

    def _label_chip(self, img, text: str, pos: Point, color: Color) -> None:
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.43
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        self._rounded_panel(img, x, y, tw + 18, th + 16, alpha=0.68, border_color=color, fill_color=COLORS["dark"], radius=8)
        self._put_text(img, text, (x + 9, y + th + 8), scale=scale, color=COLORS["white"], thickness=thickness)

    def _put_text(self, img, text: str, org: Point, *, scale: float, color: Color, thickness: int = 1) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _blend(base, overlay, alpha: float) -> None:
        cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0, dst=base)

    @staticmethod
    def _interp(p1: Point, p2: Point, t: float) -> Point:
        return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


# ----------------------------------------------------------------------
# API pubblica usata da main.py
# ----------------------------------------------------------------------


def draw_overlay(
    frame,
    detections: List[Detection],
    lane_state: LaneState,
    decision: ADASDecision,
    show_filtered: bool = False,
    hud_config: Optional[dict] = None,
    fps: Optional[float] = None,
    frame_idx: int = 0,
):
    hud = PremiumHUD(hud_config)
    return hud.draw(
        frame,
        detections,
        lane_state,
        decision,
        show_filtered=show_filtered,
        fps=fps,
        frame_idx=frame_idx,
    )


def draw_debug_overlay(frame, detections: List[Detection], lane_state: LaneState, decision: ADASDecision, show_filtered: bool = False):
    """Vecchia UI debug, utile se vuoi confrontare prestazioni/leggibilità."""
    output = frame.copy()
    h, w = output.shape[:2]

    if lane_state.left_curve:
        cv2.polylines(output, [np.array(lane_state.left_curve, dtype=np.int32)], False, (0, 255, 255), 4, cv2.LINE_AA)
    elif lane_state.left_line:
        cv2.line(output, lane_state.left_line[0], lane_state.left_line[1], (0, 255, 255), 4)
    if lane_state.right_curve:
        cv2.polylines(output, [np.array(lane_state.right_curve, dtype=np.int32)], False, (0, 255, 255), 4, cv2.LINE_AA)
    elif lane_state.right_line:
        cv2.line(output, lane_state.right_line[0], lane_state.right_line[1], (0, 255, 255), 4)
    if lane_state.has_lane:
        poly = np.array(lane_state.polygon, dtype=np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [poly], (0, 120, 0))
        output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)
        if lane_state.ego_polygon:
            ego_poly = np.array(lane_state.ego_polygon, dtype=np.int32)
            overlay = output.copy()
            cv2.fillPoly(overlay, [ego_poly], (255, 120, 0))
            output = cv2.addWeighted(overlay, 0.20, output, 0.80, 0)

    for det in detections:
        if det.filtered_reason and not show_filtered:
            continue
        if det.risk_level == "brake":
            color = (0, 0, 255)
        elif det.risk_level == "warning":
            color = (0, 165, 255)
        elif det.in_ego_lane:
            color = (0, 255, 0)
        elif det.filtered_reason:
            color = (90, 90, 90)
        else:
            color = (180, 180, 180)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    panel_color = (0, 0, 255) if decision.brake else (35, 35, 35)
    cv2.rectangle(output, (0, 0), (w, 112), panel_color, -1)
    status = "EMERGENCY STOP SIMULATO" if decision.brake else "ADAS MONITORING"
    cv2.putText(output, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    lane_text = f"LANE: {lane_state.debug}"
    if lane_state.center_offset_ratio is not None:
        lane_text += f" | offset={lane_state.center_offset_ratio:+.2f}"
    cv2.putText(output, lane_text[:110], (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    return output
