from __future__ import annotations

import time
from typing import List

from adas.perception.types import ADASDecision, Detection, LaneState


class ADASController:
    """Decision layer.

    In questa versione non comanda un veicolo reale: produce solo decisioni e
    segnali simulati. È la separazione corretta per studiare ADAS senza rischi.
    """

    def __init__(self, cfg: dict) -> None:
        self.simulate_brake_only = bool(cfg.get("simulate_brake_only", True))
        self.emergency_stop_latch_seconds = float(cfg.get("emergency_stop_latch_seconds", cfg.get("brake_latch_seconds", 1.0)))
        self._brake_until = 0.0

    def decide(self, detections: List[Detection], lane_state: LaneState) -> ADASDecision:
        decision = ADASDecision()
        now = time.monotonic()

        brake_objects = [d for d in detections if d.risk_level == "brake"]
        warning_objects = [d for d in detections if d.risk_level == "warning"]

        if brake_objects:
            most_urgent = min(brake_objects, key=lambda d: d.ttc_seconds or 999)
            decision.brake = True
            if most_urgent.ttc_seconds is not None:
                decision.reason = f"Collisione imminente con {most_urgent.label}, TTC={most_urgent.ttc_seconds:.2f}s"
            else:
                decision.reason = f"Collisione imminente con {most_urgent.label}: distanza critica"
            self._brake_until = now + self.emergency_stop_latch_seconds
        elif now < self._brake_until:
            decision.brake = True
            decision.reason = "Emergency stop mantenuto per latch temporale"

        for obj in warning_objects:
            if obj.ttc_seconds is not None:
                decision.warnings.append(f"Rischio collisione: {obj.label}, TTC={obj.ttc_seconds:.2f}s")
            else:
                decision.warnings.append(f"Rischio collisione: {obj.label} nel corridoio ego")

        if lane_state.departure == "left":
            decision.warnings.append("Lane departure: stai uscendo verso sinistra")
        elif lane_state.departure == "right":
            decision.warnings.append("Lane departure: stai uscendo verso destra")

        return decision
