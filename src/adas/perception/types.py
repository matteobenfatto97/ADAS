from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[int, int]
Line = Tuple[Point, Point]
Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box
    track_id: Optional[int] = None
    in_ego_lane: bool = False
    ttc_seconds: Optional[float] = None
    risk_level: str = "none"  # none | warning | brake
    filtered_reason: Optional[str] = None
    distance_score: float = 0.0  # 0..1: quanto è vicino nel corridoio ego, stima monocamera euristica
    risk_reason: Optional[str] = None  # TTC | CLOSE_EGO_PATH | ...

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bottom_center(self) -> Point:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.box
        return max(0, x2 - x1)

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.box
        return max(0, y2 - y1)

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0
        return self.width / self.height


@dataclass
class LaneState:
    left_line: Optional[Line] = None
    right_line: Optional[Line] = None
    lane_center_x: Optional[int] = None
    vehicle_center_x: Optional[int] = None
    center_offset_px: Optional[int] = None
    center_offset_ratio: Optional[float] = None
    departure: Optional[str] = None  # left | right | None
    polygon: List[Point] = field(default_factory=list)
    ego_polygon: List[Point] = field(default_factory=list)
    confidence: float = 0.0
    debug: str = "NO_LANE"

    # Patch 4: geometria curva. Le vecchie left_line/right_line restano per compatibilita'.
    left_curve: List[Point] = field(default_factory=list)
    right_curve: List[Point] = field(default_factory=list)
    left_fit: Optional[Tuple[float, float, float]] = None   # x = ay^2 + by + c
    right_fit: Optional[Tuple[float, float, float]] = None
    lane_width_px: Optional[float] = None
    lane_source: str = "none"  # poly | hough | fallback | partial

    @property
    def has_lane(self) -> bool:
        return self.left_line is not None and self.right_line is not None

    @property
    def has_curves(self) -> bool:
        return bool(self.left_curve) and bool(self.right_curve)


@dataclass
class ADASDecision:
    brake: bool = False
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
