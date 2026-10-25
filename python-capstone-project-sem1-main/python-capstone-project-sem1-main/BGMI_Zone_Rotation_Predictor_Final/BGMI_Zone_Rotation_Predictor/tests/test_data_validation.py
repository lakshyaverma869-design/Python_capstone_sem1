import pandas as pd
import pytest

from train_model import COORDINATE_SCALE, load_zone_data, build_transitions, validate_zone


def test_source_schema_loads_and_converts_reference_pixels(local_zone_csv):
    observations = load_zone_data(local_zone_csv)
    assert len(observations) == 16
    first = observations.iloc[0]
    assert first.Match_ID == "TEST-ONLY-1"
    assert first.center_x == pytest.approx(500 * COORDINATE_SCALE)
    assert first.center_y == pytest.approx(500 * COORDINATE_SCALE)
    assert first.radius == pytest.approx(300 * COORDINATE_SCALE)
    assert first.game == "PUBG per source project"
    assert first.platform == "Not identified in source"


def test_duplicate_phase_is_rejected(local_zone_csv, tmp_path):
    frame = pd.read_csv(local_zone_csv)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate Match_ID/Phase"):
        load_zone_data(path)


def test_invalid_coordinates_radius_and_order_are_rejected(local_zone_csv, tmp_path):
    frame = pd.read_csv(local_zone_csv)
    frame.loc[0, "White_X"] = 1200
    path = tmp_path / "coordinate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="White_X"):
        load_zone_data(path)
    frame = pd.read_csv(local_zone_csv)
    frame.loc[0, "White_R"] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="White_R"):
        load_zone_data(path)
    frame = pd.read_csv(local_zone_csv)
    frame.loc[[0, 1], "Phase"] = [2, 1]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="out of order"):
        load_zone_data(path)


def test_non_decreasing_zone_radius_is_rejected(local_zone_csv, tmp_path):
    frame = pd.read_csv(local_zone_csv)
    frame.loc[1, "White_R"] = 310
    path = tmp_path / "radius_order.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="radius must decrease"):
        load_zone_data(path)


def test_geometrically_invalid_pair_is_flagged_not_used(local_zone_csv, tmp_path):
    frame = pd.read_csv(local_zone_csv)
    frame.loc[1, "White_X"] = 900
    path = tmp_path / "bad_nesting.csv"
    frame.to_csv(path, index=False)
    observations = load_zone_data(path)
    transitions, rejected = build_transitions(observations)
    bad_pairs = [row for row in rejected if row["reason"] == "next_circle_not_contained_in_current"]
    assert bad_pairs
    assert all("match_id" in row and "phase" in row and "next_phase" in row for row in bad_pairs)
    assert all(row["containment_excess_px"] > 12 for row in bad_pairs)
    assert len(transitions) < 12


def test_missing_local_csv_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthetic fallback is disabled"):
        load_zone_data(tmp_path / "absent.csv")


def test_public_source_is_loaded_in_memory_only_when_local_file_is_absent(local_zone_csv, tmp_path, monkeypatch):
    import train_model
    payload = local_zone_csv.read_bytes()

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return payload

    monkeypatch.setattr(train_model, "urlopen", lambda request, timeout: Response())
    missing = tmp_path / "historical_zone_data.csv"
    observations = load_zone_data(missing, fetch_if_missing=True)
    assert len(observations) == 16
    assert observations.iloc[0].Match_ID == "TEST-ONLY-1"
    assert observations.iloc[0].source_reference.startswith("transient public source")
    assert not missing.exists()


def test_local_csv_takes_precedence_over_public_fetch(local_zone_csv, monkeypatch):
    import train_model
    monkeypatch.setattr(train_model, "urlopen", lambda *args, **kwargs: pytest.fail("remote fetch must not run"))
    observations = load_zone_data(local_zone_csv, fetch_if_missing=True)
    assert len(observations) == 16
    assert str(local_zone_csv.name) in observations.iloc[0].source_reference


def test_public_fetch_failure_has_actionable_missing_data_message(tmp_path, monkeypatch):
    import train_model
    from urllib.error import URLError
    monkeypatch.setattr(train_model, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(FileNotFoundError, match="public source could not be fetched"):
        load_zone_data(tmp_path / "absent.csv", fetch_if_missing=True)


def test_candidate_circle_must_fit_reference_map_bounds():
    with pytest.raises(ValueError, match="extends outside"):
        validate_zone({"center_x": 20, "center_y": 100, "radius": 40}, require_circle_in_map=True)
    assert validate_zone({"center_x": 20, "center_y": 100, "radius": 40})["radius"] == 40


def test_processed_export_retains_raw_and_canonical_rows(local_zone_csv, tmp_path):
    from train_model import export_processed_observations
    observations = load_zone_data(local_zone_csv)
    transitions, rejected = build_transitions(observations)
    output = tmp_path / "historical_zone_data.csv"
    export_processed_observations(observations, transitions, rejected, output)
    frame = pd.read_csv(output)
    assert len(frame) == len(observations)
    assert {"Match_ID", "Phase", "White_X", "White_Y", "White_R",
            "source_center_x_px", "converted_center_x", "converted_center_y",
            "converted_radius", "source_reference", "coordinate_system",
            "validation_status", "game", "platform"}.issubset(frame.columns)
    assert frame.loc[0, "converted_center_x"] == pytest.approx(frame.loc[0, "White_X"] * COORDINATE_SCALE)
    assert frame.loc[0, "converted_center_y"] == pytest.approx(frame.loc[0, "White_Y"] * COORDINATE_SCALE)
    assert frame.loc[0, "converted_radius"] == pytest.approx(frame.loc[0, "White_R"] * COORDINATE_SCALE)


def test_processed_csv_reimport_recomputes_from_raw_without_double_scaling(local_zone_csv, tmp_path):
    from train_model import COORDINATE_SCALE, export_processed_observations
    first_pass = load_zone_data(local_zone_csv)
    transitions, rejected = build_transitions(first_pass)
    processed = tmp_path / "processed.csv"
    export_processed_observations(first_pass, transitions, rejected, processed)
    second_pass = load_zone_data(processed)
    assert second_pass.loc[0, "center_x"] == pytest.approx(
        second_pass.loc[0, "White_X"] * COORDINATE_SCALE
    )
    assert second_pass.loc[0, "center_y"] == pytest.approx(
        second_pass.loc[0, "White_Y"] * COORDINATE_SCALE
    )
    assert second_pass.loc[0, "radius"] == pytest.approx(
        second_pass.loc[0, "White_R"] * COORDINATE_SCALE
    )
    assert second_pass.loc[0, "platform"] == "Not identified in source"


def test_out_of_bounds_next_circle_is_excluded(local_zone_csv, tmp_path):
    frame = pd.read_csv(local_zone_csv)
    frame.loc[(frame.Match_ID == "TEST-ONLY-1") & (frame.Phase == 4), "White_X"] = 1170
    path = tmp_path / "out_of_bounds_candidate.csv"
    frame.to_csv(path, index=False)
    transitions, rejected = build_transitions(load_zone_data(path))
    excluded = [row for row in rejected if row.get("reason") == "next_circle_out_of_bounds"]
    assert excluded
    assert excluded[0]["match_id"] == "TEST-ONLY-1"
    assert excluded[0]["phase"] == 3
    assert not ((transitions.match_id == "TEST-ONLY-1") & (transitions.zone_number == 3)).any()
