import pytest

from train_model import load_zone_data, build_transitions
from app import generate_rotation_strategy, inside_circle


def test_historical_prediction_integrates_with_route_and_strategy(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    transitions, _ = build_transitions(observations)
    current = transitions.loc[(transitions.match_id == "TEST-ONLY-1") & (transitions.zone_number == 3)].iloc[0]
    result = generate_rotation_strategy(
        transitions,
        {"team": {"x": current.next_x, "y": current.next_y},
         "zone": {"x": current.current_x, "y": current.current_y,
                  "r": current.current_radius, "placed": True}},
        vehicle_available=True, movement_mode="auto",
        planning_inputs={"vehicle_available": True, "movement_mode": "auto"},
        zone_number=3, time_budget=600,
    )
    prediction = result["prediction"]
    assert prediction["source_match_id"] == "TEST-ONLY-1"
    assert result["team"] == (current.next_x, current.next_y)
    assert result["routes"]
    assert result["strategy"]["feasible"]
    primary = result["strategy"]["primary"]
    assert inside_circle(primary["destination"],
        (prediction["center_x"], prediction["center_y"]), prediction["radius"])
    assert all(inside_circle(route["destination"],
        (prediction["center_x"], prediction["center_y"]), prediction["radius"])
        for route in result["routes"])
    assert prediction["estimate_match_count"] == 1
    assert prediction["contributing_matches"][0]["match_id"] == prediction["source_match_id"]
    assert result["candidates"][0]["source_match_id"] == prediction["source_match_id"]


def test_missing_team_and_unselected_zone_still_block_planning(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    transitions, _ = build_transitions(observations)
    with pytest.raises(ValueError, match="place your team"):
        generate_rotation_strategy(transitions, {"team": None, "zone": {"placed": True}},
            vehicle_available=False, movement_mode="walk", planning_inputs={})
    with pytest.raises(ValueError, match="set the current zone"):
        generate_rotation_strategy(transitions, {"team": {"x": 1, "y": 1}, "zone": {"placed": False}},
            vehicle_available=False, movement_mode="walk", planning_inputs={})
