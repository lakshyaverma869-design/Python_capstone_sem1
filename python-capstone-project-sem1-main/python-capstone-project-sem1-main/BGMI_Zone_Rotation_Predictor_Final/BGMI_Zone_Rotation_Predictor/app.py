"""One-screen Erangel rotation strategy application and map/route logic."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from train_model import (
    DATA_PATH, MAP_MIN, MAP_MAX, MAP_WIDTH, MAP_HEIGHT, MAP_SHORT_SIDE,
    load_zone_data, build_transitions, predict_next_zone, validate_zone,
)
from map_coordinates import MapCoordinateTransform

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "prediction_history.sqlite3"
MAP_IMAGE_PATH = ROOT / "assets" / "maps" / "erangel_user_upload.png"
MAP_IMAGE_WIDTH, MAP_IMAGE_HEIGHT = 474.0, 429.0
MAP_TRANSFORM = MapCoordinateTransform()
MAP_NAMES = ("Erangel",)


@st.cache_data(ttl=300, show_spinner=False)
def load_app_history(csv_path: str, local_file_signature: tuple[int, int] | None):
    """Cache successful history or a brief failure so map interactions do not refetch."""
    try:
        observations = load_zone_data(csv_path, fetch_if_missing=True)
        transitions, rejected = build_transitions(observations)
        if transitions.empty:
            return observations, transitions, rejected, "The selected CSV has no validated adjacent zone transitions."
        return observations, transitions, rejected, None
    except (FileNotFoundError, ValueError, OSError) as exc:
        return pd.DataFrame(), pd.DataFrame(), [], str(exc)


def extract_map_click(event):
    if not event:
        return None
    selection = event.get("selection", {}) if hasattr(event, "get") else {}
    points = selection.get("points", []) if hasattr(selection, "get") else []
    for point in reversed(points):
        if point.get("customdata") in ("map-click-target", ["map-click-target"]):
            try:
                x, y = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                return None
            if MapCoordinateTransform().contains(x, y):
                return (x, y)
    return None


TEAM, CURRENT, PREVIOUS = "Team position", "Current zone", "Previous zone"

def apply_map_click(state, point, mode):
    if point is None:
        return False
    x, y = map(float, point)
    key = {TEAM: "selected_team", CURRENT: "selected_current", PREVIOUS: "selected_previous"}.get(mode)
    if key is None:
        raise ValueError(f"Unknown map interaction mode: {mode}")
    state[key] = (x, y)
    return True


def reset_map_state(state):
    for key in ("selected_team", "selected_current", "selected_previous", "latest_run", "last_map_click"):
        state.pop(key, None)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def inside_circle(point, center, radius):
    return _distance(point, center) <= radius


def estimate_risk(start, end, enemies=None):
    enemies = enemies or []
    length = _distance(start, end)
    if not enemies:
        return min(100.0, 18.0 + length / 18.0), "Geometric exposure proxy; enemy and terrain data unavailable"
    nearest = min(_distance(end, enemy) for enemy in enemies)
    score = min(100.0, 18 + length / 22 + max(0, 120-nearest) * .42)
    return score, f"Geometric exposure proxy; nearest supplied enemy is {nearest:.0f} map units away"


def plan_routes(team, predicted, seconds_remaining, vehicle_available=False, enemies=None,
                walking_speed=4.0, vehicle_speed=16.0, movement_mode="auto", candidate_count=8,
                travel_weight=.55, risk_weight=.45):
    """Rank straight-line geometric routes that fit the selected movement options."""
    if seconds_remaining < 0 or walking_speed <= 0 or vehicle_speed <= 0:
        raise ValueError("Time must be nonnegative and movement speeds must be positive")
    if travel_weight < 0 or risk_weight < 0 or travel_weight+risk_weight <= 0:
        raise ValueError("Route score weights must be nonnegative and at least one must be positive")
    total = travel_weight+risk_weight
    travel_weight, risk_weight = travel_weight/total, risk_weight/total
    center = (float(predicted["center_x"]), float(predicted["center_y"]))
    radius = float(predicted["radius"])
    if radius <= 0:
        raise ValueError("Predicted radius must be positive")
    modes = ["walk"] if movement_mode == "walk" else (["vehicle"] if movement_mode == "vehicle" else ["walk", "vehicle"])
    routes = []
    for i in range(candidate_count):
        angle = 2*math.pi*i/candidate_count
        fraction = (.25, .58, .82)[i % 3]
        destination = (center[0]+math.cos(angle)*radius*fraction, center[1]+math.sin(angle)*radius*fraction)
        if not inside_circle(destination, center, radius) or not (MAP_MIN <= destination[0] <= MAP_WIDTH and MAP_MIN <= destination[1] <= MAP_HEIGHT):
            continue
        distance = _distance(tuple(map(float, team)), destination)
        risk, basis = estimate_risk(tuple(map(float, team)), destination, enemies)
        for mode in modes:
            if mode == "vehicle" and not vehicle_available:
                continue
            travel = distance / (vehicle_speed if mode == "vehicle" else walking_speed)
            if travel > seconds_remaining:
                continue
            score = travel_weight*(travel/max(seconds_remaining, 1.0)*100) + risk_weight*risk
            routes.append({"destination": destination, "distance": distance, "travel_seconds": travel,
                "mode": mode, "risk": risk, "risk_basis": basis, "score": score,
                "weights": {"travel_time": travel_weight, "route_risk": risk_weight},
                "waypoints": [tuple(map(float, team)), destination], "inside_predicted_zone": True,
                "time_feasible": True, "route_type": "straight-line geometric approximation"})
    return sorted(routes, key=lambda route: route["score"])


DEFAULT_WEIGHTS = {"travel_time": .55, "route_risk": .45}
def recommend_strategy(routes, team, predicted, seconds_remaining, data_source="unknown"):
    if not routes:
        return {"feasible": False, "label": "No feasible route", "reason": "No candidate can reach a destination inside the predicted safe zone before the configured deadline."}
    best = routes[0]
    alternative = routes[1] if len(routes) > 1 else None
    if best["travel_seconds"] > seconds_remaining*.75:
        label = "Rotate now"
    elif best["mode"] == "vehicle":
        label = "Use a vehicle"
    elif best["risk"] <= 30:
        label = "Move to a lower estimated-risk position"
    else:
        label = "Take the fastest feasible route"
    source_note = "PUBG historical analog estimate; not BGMI-specific"
    weights = best.get("weights", DEFAULT_WEIGHTS)
    explanation = [f"Current team position: ({team[0]:.1f}, {team[1]:.1f}).",
        f"Historical estimate center: ({predicted['center_x']:.1f}, {predicted['center_y']:.1f}) canonical reference units; radius {predicted['radius']:.1f} canonical reference units.",
        f"Route destination: ({best['destination'][0]:.1f}, {best['destination'][1]:.1f}) canonical reference units; inside the reported historical estimate.",
        f"Primary route: {best['mode']} over {best['distance']:.1f} units; estimated {best['travel_seconds']:.0f} seconds.",
        (f"Alternative route: {alternative['mode']} to ({alternative['destination'][0]:.1f}, {alternative['destination'][1]:.1f}), estimated {alternative['travel_seconds']:.0f} seconds." if alternative else "Alternative route: none available."),
        f"Risk score: {best['risk']:.0f}/100. {best['risk_basis']}.",
        f"Selected by the configured score ({weights['travel_time']:.0%} deadline-normalized travel time, {weights['route_risk']:.0%} risk proxy); {best['travel_seconds']:.0f}s fits the {seconds_remaining:.0f}s deadline.",
        f"Assumptions: {best['route_type']}; movement speeds are user-configured; terrain and cover are unavailable. {source_note}; not BGMI telemetry."]
    return {"feasible": True, "label": label, "primary": best, "alternative": alternative,
            "explanation": "\n".join(explanation), "weights": weights}


def save_run(inputs, prediction, strategy, path=DB_PATH):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS prediction_runs (id INTEGER PRIMARY KEY, created_at TEXT, inputs_json TEXT, prediction_json TEXT, strategy_json TEXT, model_name TEXT)")
        cursor = connection.execute("INSERT INTO prediction_runs(created_at,inputs_json,prediction_json,strategy_json,model_name) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(inputs), json.dumps(prediction), json.dumps(strategy), prediction.get("model_name", "unknown")))
        return int(cursor.lastrowid)


def list_runs(path=DB_PATH, limit=50):
    path = Path(path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute("SELECT * FROM prediction_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            return []
    return [{**dict(row), "inputs": json.loads(row["inputs_json"]), "prediction": json.loads(row["prediction_json"]), "strategy": json.loads(row["strategy_json"])} for row in rows]


def render_map(current=None, predicted=None, team=None, routes=None, enemies=None):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title="Schematic tactical map · source-canonical reference units",
        xaxis=dict(range=[MAP_MIN, MAP_WIDTH], title="Canonical X (reference units)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[MAP_MIN, MAP_HEIGHT], title="Canonical Y (reference units)"), height=600,
        legend=dict(orientation="h", y=-.15), annotations=[dict(text="SCHEMATIC · landmarks and terrain are not represented", xref="paper", yref="paper", x=.5, y=1.08, showarrow=False, font=dict(color="#f6c85f"))])
    for zone, color, label in ((current, "#45d483", "Current safe zone"), (predicted, "#58a6ff", "Predicted next zone")):
        if zone and float(zone.get("radius", 0)) > 0:
            fig.add_shape(type="circle", x0=zone["center_x"]-zone["radius"], y0=zone["center_y"]-zone["radius"], x1=zone["center_x"]+zone["radius"], y1=zone["center_y"]+zone["radius"], line=dict(color=color, width=3, dash="solid" if label.startswith("Current") else "dash"), name=label)
            fig.add_trace(go.Scatter(x=[zone["center_x"]], y=[zone["center_y"]], mode="markers", name=label+" center", marker=dict(color=color, size=9)))
    if routes:
        for index, route in enumerate(routes[:6]):
            xs, ys = zip(*route["waypoints"]); color = "#f6c85f" if index == 0 else ("#bd8cff" if index % 2 else "#8b949e")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=("Recommended route" if index == 0 else f"Alternative {index}"), line=dict(color=color, width=4 if index == 0 else 2, dash="solid" if index == 0 else "dot")))
            if len(xs) >= 2:
                fig.add_annotation(x=xs[-1], y=ys[-1], ax=xs[-2], ay=ys[-2], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=3 if index == 0 else 1.5, arrowcolor=color)
            fig.add_trace(go.Scatter(x=[route["destination"][0]], y=[route["destination"][1]], mode="markers", showlegend=False, marker=dict(size=8, color="#f6c85f" if index == 0 else "#8b949e", symbol="diamond")))
    if team:
        fig.add_trace(go.Scatter(x=[team[0]], y=[team[1]], mode="markers+text", text=["Team"], textposition="top center", name="Team position", marker=dict(color="#ff6b6b", size=14, symbol="star")))
    if enemies:
        fig.add_trace(go.Scatter(x=[point[0] for point in enemies], y=[point[1] for point in enemies], mode="markers", name="Known enemies", marker=dict(color="#ff4d4d", size=10, symbol="x")))
    return fig


def render_tactical_map(current=None, previous=None, predicted=None, team=None, routes=None, enemies=None, click_spacing=20):
    fig = render_map(current, predicted, team, routes, enemies)
    if previous:
        fig.add_shape(type="circle", x0=previous["center_x"]-previous["radius"], y0=previous["center_y"]-previous["radius"], x1=previous["center_x"]+previous["radius"], y1=previous["center_y"]+previous["radius"], line=dict(color="#a78bfa", width=1.5, dash="dot"))
        fig.add_trace(go.Scatter(x=[previous["center_x"]], y=[previous["center_y"]], mode="markers", name="Previous zone center", marker=dict(color="#a78bfa", size=7, symbol="circle-open")))
    if click_spacing < 1:
        raise ValueError("click_spacing must be at least one map unit")
    xs = np.clip(np.arange(MAP_MIN, MAP_WIDTH + click_spacing, click_spacing, dtype=float), MAP_MIN, MAP_WIDTH)
    ys = np.clip(np.arange(MAP_MIN, MAP_HEIGHT + click_spacing, click_spacing, dtype=float), MAP_MIN, MAP_HEIGHT)
    gx, gy = np.meshgrid(xs, ys)
    fig.add_trace(go.Scattergl(x=gx.ravel(), y=gy.ravel(), mode="markers", name="Click map to place selection", customdata=["map-click-target"]*gx.size, marker=dict(size=10, color="rgba(170,190,215,0.025)", line=dict(width=0)), hovertemplate="Select here · (%{x:.0f}, %{y:.0f})<extra></extra>", showlegend=True, selected=dict(marker=dict(color="rgba(255,255,255,0.5)")), unselected=dict(marker=dict(opacity=.1))))
    fig.update_layout(title="Erangel · schematic (not geographically accurate)", height=650, clickmode="event+select", dragmode="pan", legend=dict(orientation="h", y=-.13, x=0), margin=dict(l=25, r=20, t=65, b=65))
    fig.update_xaxes(fixedrange=False, constrain="domain"); fig.update_yaxes(fixedrange=False, scaleanchor="x", scaleratio=1)
    return fig


MAP_IMAGE = MAP_IMAGE_PATH
SUPPORTED_MAP = "Erangel"
TRANSFORMS = {SUPPORTED_MAP: MapCoordinateTransform()}
_COMPONENT_DIR = ROOT / "assets" / "map_component"
_component = components.declare_component("bgmi_tactical_map", path=str(_COMPONENT_DIR))

def tactical_map(*, map_name=SUPPORTED_MAP, team=None, zone=None, predicted=None, routes=None, mode="team", view_box=None, key="tactical_map"):
    if map_name not in TRANSFORMS:
        raise ValueError(f"No visual asset and coordinate configuration for {map_name!r}")
    transform = TRANSFORMS[map_name]
    image_uri = "data:image/png;base64," + base64.b64encode(MAP_IMAGE.read_bytes()).decode("ascii")
    initial_view_box = view_box or [0, 0, transform.image_width, transform.image_height]
    config = {
        "image_width": transform.image_width, "image_height": transform.image_height,
        "reference_width": MAP_WIDTH, "reference_height": MAP_HEIGHT,
        "source_to_image": [transform.x_scale, 0.0, transform.x_offset, 0.0, transform.y_scale, transform.y_offset],
    }
    return _component(map_image=image_uri, map_config=config, map_name=map_name, team=team, zone=zone,
        predicted=predicted, routes=routes or [], mode=mode, view_box=initial_view_box,
        default={"team": team, "zone": zone, "view_box": initial_view_box}, height=730, key=key)


def decode_component_value(value):
    if isinstance(value, str):
        try: value = json.loads(value)
        except (json.JSONDecodeError, TypeError): return None
    return value if isinstance(value, dict) else None


def sync_component_state(state, value):
    payload = decode_component_value(value)
    if not payload: return False
    changed = False
    if "team" in payload and payload["team"] != state.get("team"):
        state["team"] = payload["team"]; changed = True
    if isinstance(payload.get("zone"), dict) and payload["zone"] != state.get("zone"):
        state["zone"] = payload["zone"]; changed = True
    if isinstance(payload.get("view_box"), list): state["view_box"] = payload["view_box"]
    if payload.get("interaction_id") is not None: state["interaction_id"] = payload["interaction_id"]
    return changed


def generate_rotation_strategy(transitions, map_state, *, vehicle_available,
                               movement_mode, planning_inputs, zone_number=3, time_budget=180):
    """Use exact observed next-zone circles and the existing route planner."""
    team = map_state.get("team")
    zone = map_state.get("zone") or {}
    if team is None:
        raise ValueError("Click or drag on the map to place your team marker.")
    if not zone.get("placed"):
        raise ValueError("Drag the safe-zone circle to set the current zone.")
    if transitions is None or transitions.empty:
        raise ValueError("No suitable historical zone records are loaded.")
    prediction = predict_next_zone(
        transitions, zone_number=zone_number,
        current_x=float(zone["x"]), current_y=float(zone["y"]),
        current_radius=float(zone["r"]), minimum_matches=2, candidate_limit=3,
    )
    candidates = prediction["candidates"]
    team_point = (float(team["x"]), float(team["y"])) if isinstance(team, dict) else tuple(map(float, team))
    # All route alternatives target the one reported nearest-history estimate.
    routes = plan_routes(team_point, prediction, seconds_remaining=time_budget,
        vehicle_available=vehicle_available, movement_mode=movement_mode, walking_speed=4.0)
    strategy = recommend_strategy(routes, team_point, prediction, time_budget, prediction["data_source"])
    return {"prediction": prediction, "candidates": candidates, "routes": routes,
            "strategy": strategy, "team": team_point, "zone_number": zone_number,
            "time_budget": time_budget, "planning_inputs": planning_inputs}


def main():
    st.set_page_config(
        page_title="BGMI Rotation Strategy",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""
    <style>
    .stApp{background:radial-gradient(ellipse at 52% 0%,#1b3039 0,#0d171e 52%,#081116 100%);color:#e9f1ed}
    .stMainBlockContainer{max-width:1500px;padding-top:1rem;padding-bottom:2rem}
    header,[data-testid="stSidebar"],footer,#MainMenu{display:none!important}
    h1,h2,h3{letter-spacing:.02em;color:#eff6ef!important}
    .eyebrow{color:#68d8aa;font-size:12px;letter-spacing:.22em;font-weight:800}
    .strategy{background:linear-gradient(140deg,#162a31,#101b22);border:1px solid #35515a;border-radius:16px;padding:20px;margin-top:16px}
    .muted{color:#a9bfbb}
    div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#e6b945,#f2d36e);color:#152027;border:0;border-radius:10px;font-weight:900;letter-spacing:.06em;min-height:3.3rem}
    @media(max-width:900px){div.stButton>button[kind="primary"]{font-size:.72rem;letter-spacing:0;white-space:normal}}
    </style>
    """, unsafe_allow_html=True)


    try:
        try:
            local_stat = DATA_PATH.stat()
            local_file_signature = (local_stat.st_mtime_ns, local_stat.st_size)
        except FileNotFoundError:
            local_file_signature = None
        (historical_observations, historical_transitions,
         rejected_transitions, historical_error) = load_app_history(str(DATA_PATH), local_file_signature)
    except (FileNotFoundError, ValueError, OSError) as exc:
        historical_observations = pd.DataFrame()
        historical_transitions = pd.DataFrame()
        rejected_transitions = []
        historical_error = str(exc)

    if "map_state" not in st.session_state:
        st.session_state.map_state = {
            "team": None,
            "zone": {"x": MAP_WIDTH / 2, "y": MAP_HEIGHT / 2, "r": 240.0, "placed": False},
            "view_box": [0, 0, MAP_IMAGE_WIDTH, MAP_IMAGE_HEIGHT],
        }
    st.session_state.setdefault("squad_size", "Squad")
    st.session_state.setdefault("members_alive", 4)
    st.session_state.setdefault("vehicle_available", False)
    st.session_state.setdefault("movement_mode", "Walk")

    st.markdown('<div class="eyebrow">TACTICAL DECISION SUPPORT</div>', unsafe_allow_html=True)
    st.title("BGMI · Rotation Strategy")
    if not historical_transitions.empty:
        history_access = (
            "Public CSV read into memory; not saved"
            if str(historical_observations.iloc[0].source_reference).startswith("transient public source")
            else "Using local CSV"
        )
        st.caption(
            f"Historical PUBG Erangel analogs - {historical_observations.Match_ID.nunique()} source matches - "
            f"{len(historical_transitions)} validated adjacent transitions. {history_access}. "
            "Source does not establish BGMI or tournament provenance."
        )
        if rejected_transitions:
            st.caption(f"{len(rejected_transitions)} phase pairs failed adjacency or safe-circle nesting checks and are excluded.")
    else:
        st.warning(
            "Historical matching is unavailable. Connect to the stated public CSV source or place a permitted "
            "zone_data.csv at data/historical_zone_data.csv. No synthetic data or model fallback is used."
        )
        if historical_error:
            st.caption(historical_error)

    map_col, strategy_col = st.columns([1.45, 1], gap="large")
    with map_col:
        st.markdown(f"### {SUPPORTED_MAP}")
        st.caption("Drag the safe-zone circle; pull its bright edge handle to resize. Click or drag the map to place your team marker. Shift+drag to pan; scroll to zoom.")
        map_state = st.session_state.map_state
        latest = st.session_state.get("latest_strategy") or {}
        component_value = tactical_map(
            map_name=SUPPORTED_MAP,
            team=map_state.get("team"),
            zone=map_state.get("zone"),
            predicted=latest.get("prediction"),
            routes=latest.get("routes"),
            mode="team",
            view_box=map_state.get("view_box"),
            key="interactive_map",
        )
        if sync_component_state(map_state, component_value):
            st.session_state.pop("latest_strategy", None)
        if map_state.get("team") is not None:
            st.caption("Team position selected · marker is visible on the map")
        if map_state.get("zone", {}).get("placed"):
            st.caption("Current safe zone selected · drag the circle or resize handle to adjust")

    with strategy_col:
        st.markdown("### Plan the next rotation")
        setup_col, movement_col = st.columns(2)
        with setup_col:
            squad_size_label = st.selectbox(
                "Squad size",
                ["Solo", "Duo", "Trio", "Squad"],
                key="squad_size",
            )
            squad_size = ["Solo", "Duo", "Trio", "Squad"].index(squad_size_label) + 1
            if st.session_state.members_alive > squad_size:
                st.session_state.members_alive = squad_size
            members_alive = st.number_input(
                "Members still alive",
                min_value=1,
                max_value=squad_size,
                step=1,
                key="members_alive",
            )
        with movement_col:
            vehicle_available = st.checkbox("Vehicle available", key="vehicle_available")
            movement_options = ["Walk", "Auto"]
            if vehicle_available:
                movement_options.append("Vehicle")
            if st.session_state.movement_mode not in movement_options:
                st.session_state.movement_mode = "Walk"
            movement_label = st.selectbox(
                "Movement preference",
                movement_options,
                key="movement_mode",
                help="Auto considers all available movement modes and chooses the lowest-scoring feasible route.",
            )
        movement_mode = movement_label.lower()
        planning_inputs = {
            "squad_size": squad_size,
            "members_alive": int(members_alive),
            "vehicle_available": vehicle_available,
            "movement_mode": movement_mode,
        }
        previous_plan = st.session_state.get("latest_strategy")
        if previous_plan and previous_plan.get("planning_inputs") != planning_inputs:
            st.session_state.pop("latest_strategy", None)
            st.rerun()
        st.caption("Squad count is shown for context; route estimates use the same assumed movement speeds per player.")
        # Keep the existing phase-3 workflow while using observed historical transitions.
        zone_number = 3
        if st.button("GENERATE ROTATION STRATEGY", type="primary", use_container_width=True,
                     disabled=historical_transitions.empty):
            try:
                st.session_state.latest_strategy = generate_rotation_strategy(
                    historical_transitions, map_state,
                    vehicle_available=vehicle_available, movement_mode=movement_mode,
                    planning_inputs=planning_inputs, zone_number=zone_number, time_budget=180,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to generate a strategy: {exc}")

        latest = st.session_state.get("latest_strategy")
        if latest:
            prediction, strategy = latest["prediction"], latest["strategy"]
            st.markdown('<div class="strategy">', unsafe_allow_html=True)
            st.markdown("### Rotation strategy")
            st.markdown("**A · Historical zone estimate**")
            st.write(
                f"Observed next circle: center ({prediction['center_x']:.0f}, {prediction['center_y']:.0f}) "
                f"and radius {prediction['radius']:.0f} canonical reference units."
            )
            st.markdown("**B · Historical evidence**")
            evidence = ", ".join(
                f"{item['source_match_id']} phase {item['source_phase']}→{item['source_next_phase']}"
                for item in prediction["candidates"]
            )
            st.caption(
                f"Estimate uses the single nearest observed transition: "
                f"{prediction['source_match_id']} phase {prediction['source_phase']}→{prediction['source_next_phase']}. "
                f"Other distinct-match same-phase analogs: {evidence}. "
                f"{prediction['comparable_matches']} distinct matches have valid transitions for this phase; "
                "the estimate is not averaged."
            )
            if strategy["feasible"]:
                primary = strategy["primary"]
                tx, ty = latest["team"]
                dx, dy = primary["destination"]
                bearing = math.degrees(math.atan2(ty - dy, dx - tx))
                direction = (
                    "E" if -22.5 <= bearing < 22.5 else
                    "NE" if 22.5 <= bearing < 67.5 else
                    "N" if 67.5 <= bearing < 112.5 else
                    "NW" if 112.5 <= bearing < 157.5 else
                    "W" if bearing >= 157.5 or bearing < -157.5 else
                    "SW" if -157.5 <= bearing < -112.5 else
                    "S" if -112.5 <= bearing < -67.5 else "SE"
                )
                st.markdown("**C · Recommended rotation**")
                st.markdown(f"**Move {direction} to ({dx:.0f}, {dy:.0f}) · {strategy['label']}**")
                st.write(
                    f"{primary['mode'].capitalize()} about {primary['distance']:.0f} canonical reference units "
                    f"({primary['travel_seconds']:.0f} sec), inside the displayed historical estimate. "
                    f"Estimated risk proxy: {primary['risk']:.0f}/100."
                )
                if strategy.get("alternative"):
                    alt = strategy["alternative"]
                    st.caption(
                        f"Route alternative in the same estimated circle: {alt['travel_seconds']:.0f} sec · "
                        f"risk proxy {alt['risk']:.0f}/100."
                    )
                st.markdown("**D · Strategy and limitations**")
                st.caption(
                    f"Squad: {latest['planning_inputs']['members_alive']} of "
                    f"{latest['planning_inputs']['squad_size']} members alive · "
                    f"Movement preference: {latest['planning_inputs']['movement_mode']} · "
                    f"Vehicle available: {'yes' if latest['planning_inputs']['vehicle_available'] else 'no'}."
                )
                st.write(strategy["explanation"])
            else:
                st.markdown("**C · Recommended rotation**")
                st.error(strategy["reason"])
                st.caption("The historical estimate is shown above, but the route planner found no destination inside it that meets the configured movement/deadline constraints.")
                st.markdown("**D · Strategy and limitations**")
                st.caption("No route is recommended. Historical evidence remains a small PUBG source sample, not validated BGMI data; terrain, roads, cover and live opponents are not modeled.")
            st.caption(
                "Source-reported PUBG Erangel observations; platform, tournament and BGMI provenance are unverified. "
                "Only six matches are available. Straight-line travel does not model terrain, roads, cover or live opponents."
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="muted">Place your team, move the current zone, then generate a historical estimate and ranked route.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
