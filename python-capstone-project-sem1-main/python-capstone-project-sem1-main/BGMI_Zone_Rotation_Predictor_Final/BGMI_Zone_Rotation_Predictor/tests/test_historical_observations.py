import pytest
from train_model import load_zone_data, build_transitions, predict_next_zone, evaluate_historical_matches


def test_consecutive_transitions_are_same_match_only(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    transitions, rejected = build_transitions(observations)
    assert len(transitions) == 12
    assert rejected == []
    assert set(transitions.match_id) == {f"TEST-ONLY-{i}" for i in range(1, 5)}
    assert all(row.next_phase == row.zone_number + 1 for row in rejected) is True if rejected else True


def test_phase_gap_is_flagged_and_never_bridged(local_zone_csv, tmp_path):
    import pandas as pd
    frame = pd.read_csv(local_zone_csv)
    frame = frame.loc[~((frame.Match_ID == "TEST-ONLY-1") & (frame.Phase == 2))]
    path = tmp_path / "phase_gap.csv"
    frame.to_csv(path, index=False)
    observations = load_zone_data(path)
    transitions, rejected = build_transitions(observations)
    assert not ((transitions.match_id == "TEST-ONLY-1") & (transitions.zone_number == 1)).any()
    assert any(row["reason"] == "non_consecutive_phase" for row in rejected)


def test_matching_returns_an_actual_observed_next_circle(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    transitions, _ = build_transitions(observations)
    query = transitions.loc[(transitions.match_id == "TEST-ONLY-1") & (transitions.zone_number == 3)].iloc[0]
    result = predict_next_zone(transitions, zone_number=3, current_x=query.current_x,
        current_y=query.current_y, current_radius=query.current_radius)
    assert result["source_match_id"] == "TEST-ONLY-1"
    assert result["center_x"] == pytest.approx(query.next_x)
    assert result["center_y"] == pytest.approx(query.next_y)
    assert result["radius"] == pytest.approx(query.next_radius)
    assert len(result["candidates"]) == 3
    assert result["comparable_matches"] == 4


def test_no_phase_support_returns_clear_error(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    transitions, _ = build_transitions(observations)
    with pytest.raises(ValueError, match="at least 2 required"):
        predict_next_zone(transitions, zone_number=4, current_x=500, current_y=500, current_radius=50)


def test_leave_one_match_out_evaluation_groups_entire_matches(local_zone_csv, monkeypatch):
    import train_model
    observations = load_zone_data(local_zone_csv)
    transitions, rejected = build_transitions(observations)
    by_query = {
        (int(row.zone_number), float(row.current_x), float(row.current_y), float(row.current_radius)): row.match_id
        for row in transitions.itertuples()
    }
    original = train_model.predict_next_zone
    checked = []
    def check_group_holdout(training, **query):
        key = (int(query["zone_number"]), float(query["current_x"]), float(query["current_y"]), float(query["current_radius"]))
        held_out_match = by_query[key]
        assert held_out_match not in set(training.match_id)
        checked.append(held_out_match)
        return original(training, **query)
    monkeypatch.setattr(train_model, "predict_next_zone", check_group_holdout)
    report = train_model.evaluate_historical_matches(observations, transitions, rejected)
    evaluation = report["evaluation"]
    assert report["source_matches"] == 4
    assert report["usable_transitions"] == 12
    assert evaluation["held_out_matches_with_predictions"] == 4
    assert evaluation["held_out_transitions"] == 12
    assert len(checked) == 12
    assert evaluation["metrics"]["euclidean_center_error_canonical_units"] >= 0
