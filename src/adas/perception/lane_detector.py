from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from adas.perception.geometry import point_in_polygon
from adas.perception.types import LaneState, Line, Point

Fit = Tuple[float, float, float]


class LaneDetector:
    """Lane detector OpenCV con curve polinomiali.

    Patch 4 cambia il modello mentale: non disegniamo piu' due rette debug.
    Cerchiamo pixel di corsia, tracciamo le lane con sliding windows e fittiamo:

        x = a*y^2 + b*y + c

    In questo modo l'HUD puo' seguire curve, tornanti e strade non perfettamente dritte.
    Resta un detector classico, non una rete neurale di lane segmentation: e' veloce e didattico,
    ma su notte/pioggia/ombre forti avra' comunque limiti.
    """

    def __init__(self, cfg: dict) -> None:
        self.mode = str(cfg.get("mode", "poly"))
        self.processing_width = int(cfg.get("processing_width", 640))
        self.roi_top_ratio = float(cfg.get("roi_top_ratio", 0.48))
        # Compatibilità con i preset precedenti: roi_half_width_* erano mezze larghezze.
        self.roi_bottom_width_ratio = float(cfg.get("roi_bottom_width_ratio", cfg.get("roi_half_width_bottom_ratio", 0.49) * 2.0))
        self.roi_top_width_ratio = float(cfg.get("roi_top_width_ratio", cfg.get("roi_half_width_top_ratio", 0.21) * 2.0))
        self.vehicle_center_ratio = float(cfg.get("vehicle_center_ratio", 0.50))
        self.departure_offset_ratio = float(cfg.get("departure_offset_ratio", 0.13))
        self.line_invasion_margin_ratio = float(cfg.get("line_invasion_margin_ratio", 0.055))
        self.ego_path_width_ratio = float(cfg.get("ego_path_width_ratio", 0.52))
        self.expected_lane_width_ratio = float(cfg.get("expected_lane_width_ratio", 0.55))
        self.lane_width_tolerance_ratio = float(cfg.get("lane_width_tolerance_ratio", 0.35))
        self.fallback_infer_missing_line = bool(cfg.get("fallback_infer_missing_line", True))
        self.smoothing_frames = int(cfg.get("smoothing_frames", 7))
        self.curve_points = int(cfg.get("curve_points", 48))
        self.y_eval_ratio = float(cfg.get("y_eval_ratio", 0.86))
        # Punto più basso usato per invasione corsia. Più basso = più sensibile quando tocchi la linea vicino al cofano.
        self.departure_y_eval_ratio = float(cfg.get("departure_y_eval_ratio", max(self.y_eval_ratio, 0.91)))

        # Maschera corsie.
        self.white_v_min = int(cfg.get("white_v_min", 155))
        self.white_s_max = int(cfg.get("white_s_max", 95))
        self.yellow_h_low = int(cfg.get("yellow_h_low", 15))
        self.yellow_h_high = int(cfg.get("yellow_h_high", 40))
        self.yellow_s_min = int(cfg.get("yellow_s_min", 45))
        self.yellow_v_min = int(cfg.get("yellow_v_min", 80))
        self.sobel_threshold = int(cfg.get("sobel_threshold", 45))
        self.morph_kernel = int(cfg.get("morph_kernel", 5))

        # Sliding windows.
        self.sliding_windows = int(cfg.get("sliding_windows", 9))
        self.window_margin = int(cfg.get("window_margin", 70))
        self.minpix = int(cfg.get("minpix", 45))
        self.min_lane_pixels = int(cfg.get("min_lane_pixels", 260))
        self.histogram_y_min_ratio = float(cfg.get("histogram_y_min_ratio", 0.56))
        self.histogram_y_max_ratio = float(cfg.get("histogram_y_max_ratio", 0.94))
        self.dead_zone_ratio = float(cfg.get("dead_zone_ratio", 0.06))

        self._left_fit_history: Deque[Fit] = deque(maxlen=max(1, self.smoothing_frames))
        self._right_fit_history: Deque[Fit] = deque(maxlen=max(1, self.smoothing_frames))
        self._lane_width_history: Deque[float] = deque(maxlen=max(3, self.smoothing_frames * 2))

    def detect(self, frame) -> LaneState:
        orig_h, orig_w = frame.shape[:2]
        vehicle_center_x = int(orig_w * self.vehicle_center_ratio)

        proc, sx, sy = self._resize_for_processing(frame)
        h, w = proc.shape[:2]
        vehicle_center_proc = int(w * self.vehicle_center_ratio)
        roi_top = int(h * self.roi_top_ratio)
        y_bottom = int(h * 0.98)
        y_eval = int(h * self.y_eval_ratio)

        binary = self._binary_lane_mask(proc)
        binary = self._apply_roi(binary)

        left_fit, right_fit, left_count, right_count, windows_hit = self._sliding_window_polyfit(
            binary, vehicle_center_proc, roi_top, y_bottom
        )

        source = "poly"
        synthetic_side = None
        expected_width = self._expected_lane_width(w)

        if left_fit is None and right_fit is not None and self.fallback_infer_missing_line:
            left_fit = self._shift_fit(right_fit, -expected_width)
            synthetic_side = "left"
            source = "partial"
        elif right_fit is None and left_fit is not None and self.fallback_infer_missing_line:
            right_fit = self._shift_fit(left_fit, expected_width)
            synthetic_side = "right"
            source = "partial"

        # Salva solo fit reali/sani nella history. Le linee sintetiche non devono trascinare la stima.
        if left_fit is not None and synthetic_side != "left":
            self._left_fit_history.append(left_fit)
        if right_fit is not None and synthetic_side != "right":
            self._right_fit_history.append(right_fit)

        left_fit = self._smooth_fit(self._left_fit_history) if self._left_fit_history and synthetic_side != "left" else left_fit
        right_fit = self._smooth_fit(self._right_fit_history) if self._right_fit_history and synthetic_side != "right" else right_fit

        state = LaneState(vehicle_center_x=vehicle_center_x, lane_source=source)

        if left_fit is None or right_fit is None:
            state.confidence = 0.0
            state.debug = f"no lane | pixels L:{left_count} R:{right_count} | win:{windows_hit}"
            return state

        # Sanity width: se la larghezza e' assurda, prova con history/expected width.
        left_x_eval = self._eval_fit(left_fit, y_eval)
        right_x_eval = self._eval_fit(right_fit, y_eval)
        lane_width = right_x_eval - left_x_eval
        expected = self._expected_lane_width(w)
        min_w = expected * (1.0 - self.lane_width_tolerance_ratio)
        max_w = expected * (1.0 + self.lane_width_tolerance_ratio)
        width_sane = min_w <= lane_width <= max_w

        if lane_width <= 20 or not width_sane:
            # Prova a riparare usando il lato piu' forte e una larghezza attesa/storica.
            fixed_width = expected
            if len(self._lane_width_history) > 0:
                fixed_width = float(np.median(np.array(self._lane_width_history)))
            if left_count >= right_count and left_fit is not None:
                right_fit = self._shift_fit(left_fit, fixed_width)
            elif right_fit is not None:
                left_fit = self._shift_fit(right_fit, -fixed_width)
            source = "fallback"
            left_x_eval = self._eval_fit(left_fit, y_eval)
            right_x_eval = self._eval_fit(right_fit, y_eval)
            lane_width = right_x_eval - left_x_eval

        if lane_width > 20:
            self._lane_width_history.append(float(lane_width))

        # Campiona le curve nel frame processato e poi scala al frame originale.
        y_values = np.linspace(y_bottom, roi_top, self.curve_points)
        left_curve_proc = [(int(self._eval_fit(left_fit, y)), int(y)) for y in y_values]
        right_curve_proc = [(int(self._eval_fit(right_fit, y)), int(y)) for y in y_values]

        left_curve = self._scale_points(left_curve_proc, sx, sy, orig_w, orig_h)
        right_curve = self._scale_points(right_curve_proc, sx, sy, orig_w, orig_h)

        state.left_curve = left_curve
        state.right_curve = right_curve
        state.left_line = (left_curve[0], left_curve[-1])
        state.right_line = (right_curve[0], right_curve[-1])
        state.polygon = left_curve + list(reversed(right_curve))
        state.ego_polygon = self._build_ego_corridor(left_curve, right_curve)
        state.left_fit = self._scale_fit_to_original(left_fit, sx, sy)
        state.right_fit = self._scale_fit_to_original(right_fit, sx, sy)
        state.lane_source = source

        # Offset: valore stabile a metà-avanti per HUD.
        y_eval_orig = int(orig_h * self.y_eval_ratio)
        left_eval_orig = self._x_at_y_from_curve(left_curve, y_eval_orig)
        right_eval_orig = self._x_at_y_from_curve(right_curve, y_eval_orig)
        lane_width_orig = max(1.0, right_eval_orig - left_eval_orig)
        lane_center_orig = int((left_eval_orig + right_eval_orig) / 2.0)
        offset_px = vehicle_center_x - lane_center_orig

        state.lane_center_x = lane_center_orig
        state.center_offset_px = int(offset_px)
        state.center_offset_ratio = float(offset_px / lane_width_orig)
        state.lane_width_px = float(lane_width_orig)

        # Invasione corsia: usa un punto più vicino al veicolo, altrimenti il warning arriva tardi.
        y_depart_orig = int(orig_h * self.departure_y_eval_ratio)
        left_depart_orig = self._x_at_y_from_curve(left_curve, y_depart_orig)
        right_depart_orig = self._x_at_y_from_curve(right_curve, y_depart_orig)
        lane_width_depart = max(1.0, right_depart_orig - left_depart_orig)
        dist_left = vehicle_center_x - left_depart_orig
        dist_right = right_depart_orig - vehicle_center_x
        invasion_margin = lane_width_depart * self.line_invasion_margin_ratio
        if dist_left < invasion_margin:
            state.departure = "left"
        elif dist_right < invasion_margin:
            state.departure = "right"
        elif state.center_offset_ratio < -self.departure_offset_ratio:
            state.departure = "left"
        elif state.center_offset_ratio > self.departure_offset_ratio:
            state.departure = "right"

        # Confidence piu' utile: non dipende piu' dal numero di segmenti Hough.
        pixel_score = min(1.0, (left_count + right_count) / max(1.0, self.min_lane_pixels * 4.0))
        window_score = min(1.0, windows_hit / max(1.0, self.sliding_windows * 1.4))
        base = 0.68 if synthetic_side is None and width_sane else 0.48
        if source == "fallback":
            base -= 0.12
        state.confidence = float(np.clip(base + 0.18 * pixel_score + 0.16 * window_score, 0.0, 0.98))
        state.debug = (
            f"curve {source} | off:{state.center_offset_ratio:+.2f} | "
            f"conf:{state.confidence:.2f} | px L:{left_count} R:{right_count} | w:{lane_width_orig:.0f}"
        )
        return state

    # ------------------------------------------------------------------
    # Binary mask / ROI
    # ------------------------------------------------------------------

    def _resize_for_processing(self, frame):
        h, w = frame.shape[:2]
        if self.processing_width <= 0 or w <= self.processing_width:
            return frame, 1.0, 1.0
        scale = self.processing_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx = w / float(new_w)
        sy = h / float(new_h)
        return resized, sx, sy

    def _binary_lane_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, self.white_v_min]), np.array([180, self.white_s_max, 255]))
        yellow = cv2.inRange(
            hsv,
            np.array([self.yellow_h_low, self.yellow_s_min, self.yellow_v_min]),
            np.array([self.yellow_h_high, 255, 255]),
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.absolute(sobel_x)
        max_sobel = np.max(abs_sobel)
        if max_sobel > 0:
            scaled = np.uint8(255 * abs_sobel / max_sobel)
        else:
            scaled = np.zeros_like(gray)
        sobel_binary = cv2.inRange(scaled, self.sobel_threshold, 255)

        binary = cv2.bitwise_or(cv2.bitwise_or(white, yellow), sobel_binary)
        k = max(3, self.morph_kernel | 1)
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return binary

    def _apply_roi(self, binary):
        h, w = binary.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        bottom_half = self.roi_bottom_width_ratio / 2.0
        top_half = self.roi_top_width_ratio / 2.0
        polygon = np.array([[
            (int(w * (0.5 - bottom_half)), h),
            (int(w * (0.5 - top_half)), roi_top),
            (int(w * (0.5 + top_half)), roi_top),
            (int(w * (0.5 + bottom_half)), h),
        ]], dtype=np.int32)
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(binary, mask)

    # ------------------------------------------------------------------
    # Sliding windows + polynomial fit
    # ------------------------------------------------------------------

    def _sliding_window_polyfit(self, binary, vehicle_center_x: int, roi_top: int, y_bottom: int):
        h, w = binary.shape[:2]
        hist_y1 = int(h * self.histogram_y_min_ratio)
        hist_y2 = int(h * self.histogram_y_max_ratio)
        histogram = np.sum(binary[hist_y1:hist_y2, :], axis=0).astype(np.float32)
        if histogram.size > 15:
            kernel = np.ones(15, dtype=np.float32) / 15.0
            histogram = np.convolve(histogram, kernel, mode="same")

        dead = int(w * self.dead_zone_ratio)
        min_sep = max(dead, int(w * 0.06))
        left_region = histogram[int(w * 0.04): max(int(w * 0.05), vehicle_center_x - min_sep)]
        right_region = histogram[min(w - 1, vehicle_center_x + min_sep): int(w * 0.96)]

        left_base = None
        right_base = None
        min_peak = max(60.0, float(np.max(histogram)) * 0.10) if histogram.size else 60.0
        if left_region.size and float(np.max(left_region)) > min_peak:
            left_base = int(np.argmax(left_region) + int(w * 0.04))
        if right_region.size and float(np.max(right_region)) > min_peak:
            right_base = int(np.argmax(right_region) + min(w - 1, vehicle_center_x + min_sep))

        # Se l'histogramma non trova un picco, usa history per non perdere subito la corsia.
        if left_base is None and self._left_fit_history:
            left_base = int(self._eval_fit(self._smooth_fit(self._left_fit_history), y_bottom))
        if right_base is None and self._right_fit_history:
            right_base = int(self._eval_fit(self._smooth_fit(self._right_fit_history), y_bottom))

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        window_height = max(1, int((y_bottom - roi_top) / max(1, self.sliding_windows)))

        left_lane_inds: List[np.ndarray] = []
        right_lane_inds: List[np.ndarray] = []
        left_current = left_base
        right_current = right_base
        windows_hit = 0

        for window in range(self.sliding_windows):
            win_y_low = y_bottom - (window + 1) * window_height
            win_y_high = y_bottom - window * window_height
            if win_y_high < roi_top:
                continue

            if left_current is not None:
                good_left = (
                    (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= left_current - self.window_margin) &
                    (nonzerox < left_current + self.window_margin)
                ).nonzero()[0]
                if len(good_left) > 0:
                    left_lane_inds.append(good_left)
                    if len(good_left) > self.minpix:
                        left_current = int(np.mean(nonzerox[good_left]))
                    windows_hit += 1

            if right_current is not None:
                good_right = (
                    (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= right_current - self.window_margin) &
                    (nonzerox < right_current + self.window_margin)
                ).nonzero()[0]
                if len(good_right) > 0:
                    right_lane_inds.append(good_right)
                    if len(good_right) > self.minpix:
                        right_current = int(np.mean(nonzerox[good_right]))
                    windows_hit += 1

        left_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int64)
        right_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int64)

        left_fit = self._fit_poly(nonzerox[left_inds], nonzeroy[left_inds]) if len(left_inds) >= self.min_lane_pixels else None
        right_fit = self._fit_poly(nonzerox[right_inds], nonzeroy[right_inds]) if len(right_inds) >= self.min_lane_pixels else None
        return left_fit, right_fit, int(len(left_inds)), int(len(right_inds)), int(windows_hit)

    @staticmethod
    def _fit_poly(xs: np.ndarray, ys: np.ndarray) -> Optional[Fit]:
        """Fit x = ay^2 + by + c with normalized y for numerical stability.

        Raw dashcam y coordinates can make np.polyfit poorly conditioned, especially
        when points occupy a narrow vertical band. We normalize y, fit, then convert
        the coefficients back to the original pixel coordinate system.
        """
        if len(xs) < 8:
            return None
        ys_f = ys.astype(np.float64)
        xs_f = xs.astype(np.float64)
        if float(np.ptp(ys_f)) < 24.0:
            return None
        y_mean = float(np.mean(ys_f))
        y_scale = float(np.std(ys_f))
        if y_scale < 1e-6:
            return None
        z = (ys_f - y_mean) / y_scale
        try:
            qa, qb, qc = np.polyfit(z, xs_f, 2)
            # x = qa*((y-y_mean)/y_scale)^2 + qb*((y-y_mean)/y_scale) + qc
            a = qa / (y_scale * y_scale)
            b = qb / y_scale - 2.0 * qa * y_mean / (y_scale * y_scale)
            c = qa * y_mean * y_mean / (y_scale * y_scale) - qb * y_mean / y_scale + qc
            if not np.all(np.isfinite([a, b, c])):
                return None
            return float(a), float(b), float(c)
        except Exception:
            return None

    @staticmethod
    def _eval_fit(fit: Fit, y: float) -> float:
        a, b, c = fit
        return a * y * y + b * y + c

    @staticmethod
    def _shift_fit(fit: Fit, dx: float) -> Fit:
        a, b, c = fit
        return a, b, c + dx

    @staticmethod
    def _smooth_fit(history: Deque[Fit]) -> Fit:
        arr = np.array(history, dtype=np.float64)
        avg = arr.mean(axis=0)
        return float(avg[0]), float(avg[1]), float(avg[2])

    def _expected_lane_width(self, width: int) -> float:
        if self._lane_width_history:
            return float(np.median(np.array(self._lane_width_history)))
        return float(width * self.expected_lane_width_ratio)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_points(points: List[Point], sx: float, sy: float, max_w: int, max_h: int) -> List[Point]:
        scaled: List[Point] = []
        for x, y in points:
            xx = int(np.clip(round(x * sx), 0, max_w - 1))
            yy = int(np.clip(round(y * sy), 0, max_h - 1))
            scaled.append((xx, yy))
        return scaled

    @staticmethod
    def _scale_fit_to_original(fit: Fit, sx: float, sy: float) -> Fit:
        # x_orig = sx * (a*(y_orig/sy)^2 + b*(y_orig/sy) + c)
        a, b, c = fit
        return float(sx * a / (sy * sy)), float(sx * b / sy), float(sx * c)

    @staticmethod
    def _x_at_y_from_curve(curve: List[Point], y: int) -> float:
        if not curve:
            return 0.0
        pts = sorted(curve, key=lambda p: p[1])
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        return float(np.interp(y, ys, xs))

    def _build_ego_corridor(self, left_curve: List[Point], right_curve: List[Point]) -> List[Point]:
        ratio = max(0.1, min(1.0, self.ego_path_width_ratio))
        shrink = (1.0 - ratio) / 2.0
        left_ego: List[Point] = []
        right_ego: List[Point] = []
        for lp, rp in zip(left_curve, right_curve):
            lx, y = lp
            rx, _ = rp
            ex_l = int(lx + (rx - lx) * shrink)
            ex_r = int(rx - (rx - lx) * shrink)
            left_ego.append((ex_l, y))
            right_ego.append((ex_r, y))
        return left_ego + list(reversed(right_ego))

    @staticmethod
    def contains_point(lane_state: LaneState, point: Point, margin_px: int = 0, ego_only: bool = False) -> bool:
        polygon = lane_state.ego_polygon if ego_only else lane_state.polygon
        return point_in_polygon(polygon, point, margin_px=margin_px)
