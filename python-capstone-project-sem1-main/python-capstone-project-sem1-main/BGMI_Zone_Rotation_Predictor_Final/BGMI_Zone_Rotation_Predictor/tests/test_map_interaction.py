import pytest

from app import MapCoordinateTransform, validate_zone
from map_coordinates import COORDINATE_SCALE, LANDMARK_COUNT, LANDMARK_RMS_ERROR_PX, LANDMARK_MAX_ERROR_PX, LANDMARK_PAIRS
from app import (TEAM, CURRENT, PREVIOUS, extract_map_click,
    apply_map_click, reset_map_state)
from app import render_tactical_map
from app import (MAP_IMAGE, SUPPORTED_MAP, TRANSFORMS,
    decode_component_value, sync_component_state)


def test_coordinate_transform_round_trip_and_resize_consistency():
    transform = MapCoordinateTransform()
    point = (237.0, 814.0)
    for width, height in ((474, 429), (948, 858), (1422, 1287)):
        pixel = transform.model_to_pixel(*point, width, height)
        assert transform.pixel_to_model(*pixel, width, height) == pytest.approx(point)
    assert transform.display_to_model(*transform.model_to_display(*point)) == pytest.approx(point)


def test_map_click_filter_and_mode_specific_state_persists():
    click = {"selection":{"points":[{"customdata":"map-click-target","x":240,"y":780,"point_number":42}]}}
    assert extract_map_click(click) == (240.0, 780.0)
    assert extract_map_click({"selection":{"points":[{"customdata":"zone","x":1,"y":2}]}}) is None
    state = {"current_radius":290, "zone_number":4}
    apply_map_click(state, extract_map_click(click), TEAM)
    apply_map_click(state, (330, 640), CURRENT)
    apply_map_click(state, (410, 580), PREVIOUS)
    assert state["selected_team"] == (240, 780)
    assert state["selected_current"] == (330, 640)
    assert state["selected_previous"] == (410, 580)
    assert state["current_radius"] == 290 and state["zone_number"] == 4
    reset_map_state(state)
    assert "selected_team" not in state and "selected_current" not in state and "latest_run" not in state
    assert state["current_radius"] == 290 and state["zone_number"] == 4


def test_zone_validation_rejects_bad_prediction_instead_of_clipping():
    with pytest.raises(ValueError, match="outside"):
        validate_zone({"center_x":1200,"center_y":300,"radius":50})
    with pytest.raises(ValueError, match="radius"):
        validate_zone({"center_x":300,"center_y":300,"radius":0})
    with pytest.raises(ValueError, match="non-finite"):
        validate_zone({"center_x":float("nan"),"center_y":300,"radius":50})


def test_zone_and_route_overlays_render_once_and_stay_aligned():
    current = {"center_x":300,"center_y":400,"radius":180}
    predicted = {"center_x":600,"center_y":650,"radius":120}
    route = {"waypoints":[(100,150),(500,600)],"destination":(500,600),"mode":"walk",
             "travel_seconds":20,"score":8,"risk":10}
    fig = render_tactical_map(current=current, predicted=predicted, team=(100,150), routes=[route], click_spacing=200)
    circles = [shape for shape in fig.layout.shapes if shape.type == "circle"]
    assert len(circles) == 2
    assert (circles[0].x0, circles[0].x1) == (120, 480)
    assert (circles[1].x0, circles[1].x1) == (480, 720)
    assert len(fig.layout.annotations) >= 1  # route direction arrow and schematic label
    target = next(trace for trace in fig.data if getattr(trace, "name", "") == "Click map to place selection")
    assert len(target.x) == len(target.y) and len(target.x) > 1


def test_uploaded_erangel_image_transform_and_drag_component():
    assert SUPPORTED_MAP in TRANSFORMS
    transform = TRANSFORMS[SUPPORTED_MAP]
    image = MAP_IMAGE.read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = int.from_bytes(image[16:20], "big"), int.from_bytes(image[20:24], "big")
    assert (width, height) == (474, 429)
    assert (transform.image_width, transform.image_height) == (474, 429)
    model_point = (237.0, 814.0)
    pixel = transform.model_to_pixel(*model_point, width, height)
    assert transform.pixel_to_model(*pixel, width, height) == pytest.approx(model_point)
    assert transform.model_radius_to_ellipse(200, width, height) == pytest.approx((200*transform.x_scale, 200*transform.y_scale))
    assert transform.model_radius_to_pixel(200, width, height) == pytest.approx((200*transform.x_scale*200*transform.y_scale)**0.5)
    with pytest.raises(ValueError, match="playable image bounds"):
        transform.pixel_to_model(width + 1, 0, width, height)

    component = (MAP_IMAGE.parents[1] / "map_component" / "index.html").read_text(encoding="utf-8")
    assert "style.aspectRatio" in component and "preserveAspectRatio=\"none\"" in component
    for event in ("pointerdown", "pointermove", "pointerup", "wheel"):
        assert f"'{event}'" in component
    assert "resizeHandle" in component and "resizeHit" in component and "teamNode" in component
    assert "modelPoint" in component and "imagePoint" in component and "circlePath" in component
    assert "source_to_image" in component and "radius_scale" not in component
    assert "zone-pending" in component and "interaction_id" in component
    assert "Erangel_Main_No_Text_Low_Res.png" not in component


def test_component_v1_json_string_is_decoded_for_streamlit_state():
    payload = {"team": {"x": 150, "y": 800}, "zone": {"x": 500, "y": 500, "r": 200, "placed": True}}
    import json
    assert decode_component_value(json.dumps(payload)) == payload
    assert decode_component_value(payload) == payload
    assert decode_component_value("not-json") is None


def test_component_selection_state_survives_partial_updates_and_reruns():
    import json
    state = {"team": None, "zone": {"x": 500, "y": 500, "r": 240, "placed": False},
             "view_box": [0, 0, 819, 819]}
    select = {"team": {"x": 220, "y": 710},
              "zone": {"x": 540, "y": 460, "r": 195, "placed": True},
              "view_box": [0, 0, 819, 819], "interaction_id": 1}
    assert sync_component_state(state, json.dumps(select))
    assert state["team"] == select["team"] and state["zone"] == select["zone"]
    assert not sync_component_state(state, None)
    assert state["team"] == select["team"] and state["zone"] == select["zone"]
    assert sync_component_state(state, {"view_box": [10, 20, 400, 400], "interaction_id": 2}) is False
    assert state["view_box"] == [10, 20, 400, 400]
    assert state["team"] == select["team"] and state["zone"] == select["zone"]


def test_reference_to_uploaded_map_round_trip_and_landmark_fit():
    transform = MapCoordinateTransform()
    source_points = [(0, 0), (130, 188), (560, 422), (1014, 484), (642, 1014), (1181, 1133)]
    for source in source_points:
        canonical = (source[0] * COORDINATE_SCALE, source[1] * COORDINATE_SCALE)
        image = transform.model_to_display(*canonical)
        recovered_canonical = transform.display_to_model(*image)
        recovered_source = (recovered_canonical[0] / COORDINATE_SCALE, recovered_canonical[1] / COORDINATE_SCALE)
        assert recovered_source == pytest.approx(source, abs=1e-9)
    errors = transform.landmark_errors()
    assert len(errors) == LANDMARK_COUNT == len(LANDMARK_PAIRS)
    rms = (sum(error**2 for _, error in errors) / len(errors))**0.5
    assert rms == pytest.approx(LANDMARK_RMS_ERROR_PX, abs=0.02)
    assert max(error for _, error in errors) == pytest.approx(LANDMARK_MAX_ERROR_PX, abs=0.02)
    assert rms < 16 and max(error for _, error in errors) < 26
