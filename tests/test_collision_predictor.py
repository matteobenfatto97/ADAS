from adas.control.adas_controller import ADASController
from adas.perception.types import Detection, LaneState


def test_controller_brakes_on_brake_risk():
    det = Detection(label="car", confidence=0.9, box=(100, 100, 200, 250), in_ego_lane=True, ttc_seconds=0.8, risk_level="brake")
    controller = ADASController({"simulate_brake_only": True})
    decision = controller.decide([det], LaneState())
    assert decision.brake is True
    assert "Collisione imminente" in decision.reason
